import os
import logging
from datetime import datetime
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from dataset import SoccerNetClips, SoccerNetClipsTesting
from model import ContextAwareModel
from train import trainer, test
from loss import ContextAwareLoss, SpottingLoss

torch.manual_seed(0)
np.random.seed(0)


def get_game_list(dataset_path):
    """Walk league/season/game folders and find all games with required files."""
    games = []
    if not os.path.isdir(dataset_path):
        return games
    
    for league in os.listdir(dataset_path):
        league_path = os.path.join(dataset_path, league)
        if not os.path.isdir(league_path):
            continue
        for season in os.listdir(league_path):
            season_path = os.path.join(league_path, season)
            if not os.path.isdir(season_path):
                continue
            for game in os.listdir(season_path):
                game_path = os.path.join(season_path, game)
                if not os.path.isdir(game_path):
                    continue
                if (os.path.exists(os.path.join(game_path, "1_ResNET_TF2_PCA512.npy")) and
                    os.path.exists(os.path.join(game_path, "2_ResNET_TF2_PCA512.npy")) and
                    os.path.exists(os.path.join(game_path, "Labels-v2.json"))):
                    rel_path = os.path.join(league, season, game)
                    games.append(rel_path)
    return games


def main(args):
    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(25) + " : " + str(getattr(args, arg)))
    
    device = torch.device("cuda" if (torch.cuda.is_available() and args.GPU >= 0) else "cpu")
    logging.info(f"Using device: {device}")
    
    # ---- Find games ----
    all_games = get_game_list(args.dataset_path) if args.dataset_path else []
    logging.info(f"Found {len(all_games)} games in {args.dataset_path}")
    
    if len(all_games) == 0 and not args.test_only:
        raise RuntimeError(
            "No games found. Check --dataset_path. It should be the parent of "
            "league folders (e.g., parent of 'england_epl')."
        )
    
    if not args.test_only:
        # Train/Val split
        np.random.seed(42)
        np.random.shuffle(all_games)
        split_idx = max(1, int(0.8 * len(all_games)))
        train_games = all_games[:split_idx]
        val_games = all_games[split_idx:] if split_idx < len(all_games) else all_games[-1:]
        
        logging.info(f"Train: {len(train_games)} games | Val: {len(val_games)} games")
        
        train_dataset = SoccerNetClips(
            path=args.dataset_path,
            list_games=train_games,
            framerate=args.framerate,
            chunk_size=args.chunk_size * args.framerate,
            receptive_field=args.receptive_field * args.framerate,
            chunks_per_epoch=args.chunks_per_epoch
        )
        val_dataset = SoccerNetClips(
            path=args.dataset_path,
            list_games=val_games,
            framerate=args.framerate,
            chunk_size=args.chunk_size * args.framerate,
            receptive_field=args.receptive_field * args.framerate,
            chunks_per_epoch=max(100, args.chunks_per_epoch // 10)
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=False
        )
        
        # Build model
        model = ContextAwareModel(
            weights=None,
            input_size=args.num_features,
            num_classes=17,
            chunk_size=args.chunk_size * args.framerate,
            dim_capsule=args.dim_capsule,
            receptive_field=args.receptive_field * args.framerate,
            num_detections=15,
            framerate=args.framerate
        ).to(device)
        
        # Load pre-trained
        if args.pretrained_weights and os.path.exists(args.pretrained_weights):
            logging.info(f"Loading pre-trained weights: {args.pretrained_weights}")
            ckpt = torch.load(args.pretrained_weights, map_location=device)
            model.load_state_dict(ckpt['state_dict'])
            logging.info("Pre-trained weights loaded")
        else:
            logging.warning("No pre-trained weights found — training from scratch")
        
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Trainable parameters: {n_params}")
        
        # Optimizer / scheduler / loss
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=args.patience
        )
        criterion_seg = ContextAwareLoss(framerate=args.framerate)
        criterion_spot = SpottingLoss(
            lambda_coord=args.lambda_coord,
            lambda_noobj=args.lambda_noobj
        )
        
        # Train!
        trainer(
            train_loader, val_loader, model, optimizer, scheduler,
            criterion_seg, criterion_spot,
            loss_weight_segmentation=args.loss_weight_segmentation,
            loss_weight_detection=args.loss_weight_detection,
            model_name=args.model_name,
            max_epochs=args.max_epochs,
            evaluation_frequency=args.evaluation_frequency
        )
    
    # ---- Optional: inference after training (or with --test_only) ----
    if args.test_video_path is not None:
        logging.info(f"Running inference on {args.test_video_path}")
        test_dataset = SoccerNetClipsTesting(
            path=args.test_video_path,
            features=args.features,
            framerate=args.framerate,
            chunk_size=args.chunk_size * args.framerate,
            receptive_field=args.receptive_field * args.framerate
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        
        model = ContextAwareModel(
            input_size=args.num_features, num_classes=17,
            chunk_size=args.chunk_size * args.framerate,
            dim_capsule=args.dim_capsule,
            receptive_field=args.receptive_field * args.framerate,
            num_detections=15, framerate=args.framerate
        ).to(device)
        
        ckpt_path = os.path.join("models", args.model_name, "model.pth.tar")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Trained model not found: {ckpt_path}\n"
                f"Train first or set --pretrained_weights properly."
            )
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        
        test(test_loader, model=model, model_name=args.model_name,
             save_predictions=True, output_path=args.output_path)


if __name__ == '__main__':
    parser = ArgumentParser(description='CALF Fine-Tuning', formatter_class=ArgumentDefaultsHelpFormatter)
    
    # Dataset
    parser.add_argument('--dataset_path', required=False, type=str, default=None,
                        help='Root folder containing league/season/game structure')
    parser.add_argument('--features', required=False, type=str, default="1_ResNET_TF2_PCA512.npy")
    
    # Pre-trained
    parser.add_argument('--pretrained_weights', required=False, type=str, default="model.pth.tar")
    
    # Training
    parser.add_argument('--model_name', required=False, type=str, default="CALF_finetuned")
    parser.add_argument('--max_epochs', required=False, type=int, default=20)
    parser.add_argument('--test_only', required=False, action='store_true')
    
    # Inference
    parser.add_argument('--test_video_path', required=False, type=str, default=None)
    parser.add_argument('--output_path', required=False, type=str, default=None)
    
    # Model
    parser.add_argument('--num_features', required=False, type=int, default=512)
    parser.add_argument('--dim_capsule', required=False, type=int, default=16)
    parser.add_argument('--framerate', required=False, type=int, default=2)
    parser.add_argument('--chunk_size', required=False, type=int, default=120)
    parser.add_argument('--receptive_field', required=False, type=int, default=40)
    
    # Loss
    parser.add_argument('--lambda_coord', required=False, type=float, default=5.0)
    parser.add_argument('--lambda_noobj', required=False, type=float, default=0.5)
    parser.add_argument('--loss_weight_segmentation', required=False, type=float, default=0.000367)
    parser.add_argument('--loss_weight_detection', required=False, type=float, default=1.0)
    
    # Optimization
    parser.add_argument('--chunks_per_epoch', required=False, type=int, default=2000)
    parser.add_argument('--evaluation_frequency', required=False, type=int, default=2)
    parser.add_argument('--batch_size', required=False, type=int, default=8)
    parser.add_argument('--LR', required=False, type=float, default=1e-04)
    parser.add_argument('--patience', required=False, type=int, default=3)
    
    parser.add_argument('--GPU', required=False, type=int, default=-1)
    parser.add_argument('--loglevel', required=False, type=str, default='INFO')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.dataset_path is None and not args.test_only:
        parser.error("--dataset_path is required for training. Use --test_only to skip training.")
    
    if args.test_only and args.test_video_path is None:
        parser.error("--test_video_path is required when --test_only is set.")
    
    # Setup logging
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    
    os.makedirs(os.path.join("models", args.model_name), exist_ok=True)
    log_path = os.path.join(
        "models", args.model_name,
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log')
    )
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    # GPU setup
    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    
    # Run
    start = time.time()
    logging.info('Starting fine-tuning')
    main(args)
    logging.info(f'Total time: {time.time() - start:.1f}s')