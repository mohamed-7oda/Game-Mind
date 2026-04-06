# inference.py

import numpy as np
import torch

CHUNK_SIZE = 240
STRIDE = 120
NUM_CLASSES = 17
NUM_DETECTIONS = 15


# =====================================================
# TEMPORAL NON-MAXIMUM SUPPRESSION
# =====================================================
def non_maximum_suppression(detections, delta=10):
    """
    Apply temporal NMS per class.
    """
    remaining = np.copy(detections)
    suppressed = np.full_like(remaining, -1)

    for c in range(NUM_CLASSES):

        while np.max(remaining[:, c]) > 0:

            idx = np.argmax(remaining[:, c])
            val = remaining[idx, c]

            suppressed[idx, c] = val

            start = max(0, idx - delta // 2)
            end = min(remaining.shape[0], idx + delta // 2)

            remaining[start:end, c] = -1

    return suppressed


# =====================================================
# MODEL INFERENCE
# =====================================================
def run_inference(model, features, device, conf_threshold=0.3):
    """
    Run sliding window inference on full video features.

    Args:
        model: trained spotting model
        features: numpy array (num_frames, 512)
        device: cpu or cuda
        conf_threshold: minimum confidence

    Returns:
        detections (num_frames, NUM_CLASSES)
    """

    model.eval()

    video_length = features.shape[0]
    detections = np.zeros((video_length, NUM_CLASSES))

    with torch.no_grad():

        for start in range(0, video_length, STRIDE):

            end = min(start + CHUNK_SIZE, video_length)
            clip = features[start:end]

            # Pad if last chunk smaller than CHUNK_SIZE
            if clip.shape[0] < CHUNK_SIZE:
                pad = np.zeros((CHUNK_SIZE - clip.shape[0], clip.shape[1]))
                clip = np.vstack((clip, pad))

            clip_tensor = (
                torch.from_numpy(clip)
                .float()
                .unsqueeze(0)      # batch dim
                .unsqueeze(1)      # channel dim
                .to(device)
            )

            _, spotting = model(clip_tensor)
            spotting = spotting.squeeze(0).cpu().numpy()

            for d in range(NUM_DETECTIONS):

                conf = spotting[d, 0]

                if conf < conf_threshold:
                    continue

                time_norm = spotting[d, 1]
                cls = np.argmax(spotting[d, 2:])

                frame_local = int(time_norm * CHUNK_SIZE)
                frame_global = start + frame_local

                if 0 <= frame_global < video_length:
                    detections[frame_global, cls] = max(
                        detections[frame_global, cls],
                        conf
                    )

    return non_maximum_suppression(detections)