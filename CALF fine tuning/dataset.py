from torch.utils.data import Dataset
import numpy as np
import random
import os
import torch
import json
from tqdm import tqdm
from config.classes import EVENT_DICTIONARY_V2


def feats2clip(feats, stride, clip_length, padding="replicate_last", off=0):
    """Convert features into overlapping clips."""
    idx = torch.arange(start=0, end=feats.shape[0] - 1, step=stride)
    idxs = []
    for i in torch.arange(-off, clip_length - off):
        idxs.append(idx + i)
    idx = torch.stack(idxs, dim=1)
    if padding == "replicate_last":
        idx = idx.clamp(0, feats.shape[0] - 1)
    return feats[idx, :]


class SoccerNetClips(Dataset):
    """Training dataset — loads features + labels from multiple games."""
    
    def __init__(self, path, list_games,
                 features="1_ResNET_TF2_PCA512.npy",
                 features2="2_ResNET_TF2_PCA512.npy",
                 framerate=2, chunk_size=240, receptive_field=80,
                 chunks_per_epoch=2000, num_detections=15):
        self.path = path
        self.list_games = list_games
        self.features = features
        self.features2 = features2
        self.framerate = framerate
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.chunks_per_epoch = chunks_per_epoch
        self.num_classes = 17
        self.num_detections = num_detections
        self.dict_event = EVENT_DICTIONARY_V2
        
        print(f"Loading {len(list_games)} games...")
        
        self.game_feats = []  # list of (T, 512) feature tensors per half
        self.game_events = []  # list of [(frame, class), ...] per half
        
        loaded = 0
        for game in tqdm(list_games):
            for half_idx, feat_file in enumerate([features, features2]):
                feat_path = os.path.join(self.path, game, feat_file)
                label_path = os.path.join(self.path, game, "Labels-v2.json")
                
                if not (os.path.exists(feat_path) and os.path.exists(label_path)):
                    continue
                
                feat = np.load(feat_path)
                self.game_feats.append(feat)
                
                # Parse events for this half
                with open(label_path) as f:
                    labels = json.load(f)
                
                events = []
                for ann in labels.get("annotations", []):
                    try:
                        time_str = ann["gameTime"]
                        event = ann["label"]
                        half = int(time_str[0])
                        if half != half_idx + 1:
                            continue
                        if event not in self.dict_event:
                            continue
                        time_part = time_str.split(" ")[-1]
                        minutes, seconds = int(time_part.split(":")[0]), int(time_part.split(":")[1])
                        frame = framerate * (seconds + 60 * minutes)
                        if 0 <= frame < feat.shape[0]:
                            events.append((frame, self.dict_event[event]))
                    except Exception:
                        continue
                
                self.game_events.append(events)
            loaded += 1
        
        print(f"Loaded {loaded} games, {len(self.game_feats)} halves total")
        
        # Build chunk index for sampling
        self.chunk_index = []
        stride = chunk_size - receptive_field
        for half_idx, feat in enumerate(self.game_feats):
            n_chunks = max(1, (feat.shape[0] - chunk_size) // stride + 1)
            for c in range(n_chunks):
                self.chunk_index.append((half_idx, c * stride))
        
        print(f"Total chunks available: {len(self.chunk_index)}")
    
    def __getitem__(self, index):
        # Random chunk
        half_idx, start_frame = random.choice(self.chunk_index)
        feat = self.game_feats[half_idx]
        events = self.game_events[half_idx]
        
        end_frame = start_frame + self.chunk_size
        
        # Pad if necessary
        if end_frame > feat.shape[0]:
            chunk = np.zeros((self.chunk_size, feat.shape[1]), dtype=np.float32)
            valid = feat.shape[0] - start_frame
            chunk[:valid] = feat[start_frame:feat.shape[0]]
        else:
            chunk = feat[start_frame:end_frame].astype(np.float32)
        
        # Build segmentation labels: signed distance (in frames) to closest event per class
        # Default: very far (so loss treats them as "no event")
        BIG = 1e6
        seg_labels = np.full((self.chunk_size, self.num_classes), BIG, dtype=np.float32)
        
        # Build spotting anchors: (num_detections, 3) = [is_event, time_in_chunk(0..1), class]
        spot_labels = np.zeros((self.num_detections, 3), dtype=np.float32)
        anchor_count = 0
        
        for ev_frame, ev_class in events:
            # Distance from each frame in chunk to this event (signed)
            for t in range(self.chunk_size):
                d = (start_frame + t) - ev_frame  # signed distance
                if abs(d) < abs(seg_labels[t, ev_class]):
                    seg_labels[t, ev_class] = float(d)
            
            # Add as spotting anchor if event is inside this chunk
            if start_frame <= ev_frame < end_frame and anchor_count < self.num_detections:
                spot_labels[anchor_count] = [
                    1.0,
                    (ev_frame - start_frame) / self.chunk_size,
                    float(ev_class)
                ]
                anchor_count += 1
        
        return (
            torch.from_numpy(chunk).float(),
            torch.from_numpy(seg_labels).float(),
            torch.from_numpy(spot_labels).float()
        )
    
    def __len__(self):
        return self.chunks_per_epoch


class SoccerNetClipsTesting(Dataset):
    """Inference dataset — loads features for one game half."""
    
    def __init__(self, path, features="1_ResNET_TF2_PCA512.npy",
                 framerate=2, chunk_size=240, receptive_field=80):
        self.path = path
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.framerate = framerate
        self.num_classes = 17
        self.num_detections = 15
        
        self.features_path = os.path.join(path, features)
        if not os.path.exists(self.features_path):
            raise FileNotFoundError(f"Features file not found: {self.features_path}")
        print(f"Using pre-extracted features: {self.features_path}")
    
    def __getitem__(self, index):
        feat_half1 = np.load(self.features_path)
        print("Shape half 1: ", feat_half1.shape)
        size = feat_half1.shape[0]
        
        feat_half1 = feats2clip(
            torch.from_numpy(feat_half1),
            stride=self.chunk_size - self.receptive_field,
            clip_length=self.chunk_size
        )
        return feat_half1, size
    
    def __len__(self):
        return 1