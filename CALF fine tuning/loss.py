import torch
import torch.nn as nn
import torch.nn.functional as F
from config.classes import K_V2


class ContextAwareLoss(nn.Module):
    """
    The Context-Aware Loss for segmentation, using the K_V2 time bands.
    Implements the loss from the CALF paper:
    https://arxiv.org/abs/1912.01326
    """
    def __init__(self, K_V2=K_V2, framerate=2, hit_radius=0.5):
        super(ContextAwareLoss, self).__init__()
        self.K_V2 = K_V2 * framerate  # Convert seconds to frames
        self.framerate = framerate
        self.hit_radius = hit_radius
    
    def forward(self, labels, output):
        """
        labels: (B, T, num_classes) - target distance to nearest event (in frames)
                                       0 = at event, larger = far from event
        output: (B, T, num_classes) - predicted segmentation (0..1)
        """
        device = output.device
        K = self.K_V2.to(device)
        
        loss = torch.zeros(1, device=device)
        eps = 1e-7
        
        for c in range(output.size(-1)):
            d = labels[:, :, c]                  # (B, T) - signed distance in frames
            y = output[:, :, c]                  # (B, T) - prediction
            y = torch.clamp(y, eps, 1 - eps)
            
            k1, k2, k3, k4 = K[0, c], K[1, c], K[2, c], K[3, c]
            
            # Region 1: too far before — encourage 0
            mask_far_before = (d <= k1).float()
            # Region 2: approaching — encourage rising
            mask_before = ((d > k1) & (d <= k2)).float()
            # Region 3: at the event — encourage 1
            mask_at = ((d > k2) & (d <= k3)).float()
            # Region 4: leaving — encourage falling
            mask_after = ((d > k3) & (d <= k4)).float()
            # Region 5: too far after — encourage 0
            mask_far_after = (d > k4).float()
            
            # Encourage low predictions in 'far' regions
            loss_far = -(mask_far_before + mask_far_after) * torch.log(1 - y)
            # Encourage high predictions at the event
            loss_at = -mask_at * torch.log(y)
            # Smooth transitions in approach/leave regions
            loss_before = -mask_before * torch.log(y)  
            loss_after = -mask_after * torch.log(y)
            
            loss = loss + (loss_far + loss_at + loss_before + loss_after).mean()
        
        return loss / output.size(-1)


class SpottingLoss(nn.Module):
    """YOLO-style detection loss for action spotting."""
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super(SpottingLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
    
    def forward(self, labels, output):
        """
        labels: (B, num_anchors, 3) - [is_event, time_in_chunk, class_idx]
        output: (B, num_detections, 2 + num_classes) - [conf, time, ...class_probs]
        """
        device = output.device
        eps = 1e-7
        
        confidences = output[:, :, 0]              # (B, D)
        times = output[:, :, 1]                    # (B, D)
        class_probs = output[:, :, 2:]             # (B, D, C)
        
        # Mask of valid anchor labels (is_event > 0)
        obj_mask = (labels[:, :, 0] > 0).float()   # (B, A)
        
        if obj_mask.sum() == 0:
            # No events in this batch -> only no-object loss
            noobj_loss = -torch.log(1 - torch.clamp(confidences, eps, 1 - eps)).mean()
            return self.lambda_noobj * noobj_loss
        
        # For each anchor, find the closest detection
        target_times = labels[:, :, 1]             # (B, A) in [0,1]
        target_classes = labels[:, :, 2].long()    # (B, A)
        
        B, A, _ = labels.shape
        D = output.size(1)
        
        # Time differences: (B, A, D)
        time_diff = (times.unsqueeze(1) - target_times.unsqueeze(2)).abs()
        # Pick the closest detection for each anchor
        best_det = time_diff.argmin(dim=2)         # (B, A)
        
        loss_total = torch.zeros(1, device=device)
        n_obj = 0
        
        for b in range(B):
            for a in range(A):
                if obj_mask[b, a] < 0.5:
                    continue
                d = best_det[b, a]
                # Coordinate loss (time)
                loss_coord = (times[b, d] - target_times[b, a]) ** 2
                # Confidence loss (should be 1)
                loss_conf = -torch.log(torch.clamp(confidences[b, d], eps, 1 - eps))
                # Class loss (cross-entropy)
                cls_logit = class_probs[b, d]
                loss_cls = -torch.log(torch.clamp(cls_logit[target_classes[b, a]], eps, 1 - eps))
                
                loss_total = loss_total + self.lambda_coord * loss_coord + loss_conf + loss_cls
                n_obj += 1
        
        # No-object loss (for detections that aren't matched)
        noobj_loss = -torch.log(1 - torch.clamp(confidences, eps, 1 - eps)).mean()
        
        return loss_total / max(n_obj, 1) + self.lambda_noobj * noobj_loss