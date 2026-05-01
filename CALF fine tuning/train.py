import os
import time
from tqdm import tqdm
import torch
import numpy as np
from preprocessing import batch2long, timestamps2long, NMS
from json_io import predictions2json


def trainer(train_loader, val_loader, model, optimizer, scheduler,
            criterion_seg, criterion_spot,
            loss_weight_segmentation=0.000367, loss_weight_detection=1.0,
            model_name="CALF_finetuned", max_epochs=20, evaluation_frequency=2):
    
    best_loss = float('inf')
    
    for epoch in range(max_epochs):
        print(f"\n{'='*60}\nEpoch {epoch+1}/{max_epochs}\n{'='*60}")
        
        # ---- TRAIN ----
        model.train()
        train_loss = run_one_epoch(
            train_loader, model, optimizer,
            criterion_seg, criterion_spot,
            loss_weight_segmentation, loss_weight_detection,
            train=True
        )
        print(f"Train loss: {train_loss:.6f}")
        
        # ---- VAL ----
        if (epoch + 1) % evaluation_frequency == 0 or epoch == max_epochs - 1:
            model.eval()
            with torch.no_grad():
                val_loss = run_one_epoch(
                    val_loader, model, optimizer,
                    criterion_seg, criterion_spot,
                    loss_weight_segmentation, loss_weight_detection,
                    train=False
                )
            print(f"Val loss: {val_loss:.6f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                save_path = os.path.join("models", model_name, "model.pth.tar")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict(),
                }, save_path)
                print(f"Saved best model to {save_path} (val loss: {best_loss:.6f})")
            
            scheduler.step(val_loss)
        else:
            scheduler.step(train_loss)
    
    print(f"\nTraining complete! Best val loss: {best_loss:.6f}")


def run_one_epoch(loader, model, optimizer, criterion_seg, criterion_spot,
                  w_seg, w_spot, train=True):
    losses = []
    device = next(model.parameters()).device
    
    pbar = tqdm(loader, desc="Train" if train else "Val ", ncols=120)
    for batch in pbar:
        feats, seg_labels, spot_labels = batch
        feats = feats.to(device).unsqueeze(1)  # (B, 1, T, D)
        seg_labels = seg_labels.to(device)
        spot_labels = spot_labels.to(device)
        
        output_seg, output_spot = model(feats)
        
        loss_seg = criterion_seg(seg_labels, output_seg)
        loss_spot = criterion_spot(spot_labels, output_spot)
        loss = w_seg * loss_seg + w_spot * loss_spot
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        losses.append(loss.item())
        pbar.set_postfix({
            "loss": f"{np.mean(losses):.4f}",
            "seg": f"{loss_seg.item():.4f}",
            "spot": f"{loss_spot.item():.4f}"
        })
    
    return np.mean(losses)


def test(dataloader, model, model_name, save_predictions=False, output_path=None):
    spotting_predictions = list()
    segmentation_predictions = list()
    
    chunk_size = model.chunk_size
    receptive_field = model.receptive_field
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
        for i, (feat_half1, size) in t:
            feat_half1 = feat_half1.to(device).squeeze(0).unsqueeze(1)
            
            with torch.no_grad():
                output_seg, output_spot = model(feat_half1)
            
            ts = timestamps2long(output_spot.cpu().detach(), size, chunk_size, receptive_field)
            seg = batch2long(output_seg.cpu().detach(), size, chunk_size, receptive_field)
            spotting_predictions.append(ts)
            segmentation_predictions.append(seg)
    
    detections_numpy = []
    segmentation_numpy = []
    for seg, det in zip(segmentation_predictions, spotting_predictions):
        segmentation_numpy.append(seg.numpy())
        detections_numpy.append(NMS(det.numpy(), 20 * model.framerate))
    
    if output_path is None:
        output_path = "outputs"
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Saving predictions to: {output_path}")
    predictions2json(detections_numpy[0], output_path, model.framerate)