import torch
import numpy as np

from feature_extractor import VideoFeatureExtractor, PCAReducer
from inference import run_inference
from model import ContextAwareModel

# ==============================
# CONFIG
# ==============================
FEATURE_PATH = "features.npy"

PCA_FILE = "pca_512_TF2.pkl"
SCALER_FILE = "average_512_TF2.pkl"
MODEL_PATH = "context_aware_spotting_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# EVENT DICTIONARY
# ==============================
EVENT_DICTIONARY = {
    0: "Penalty",
    1: "Kick-off",
    2: "Goal",
    3: "Substitution",
    4: "Offside",
    5: "Shots on target",
    6: "Shots off target",
    7: "Clearance",
    8: "Ball out of play",
    9: "Throw-in",
    10: "Foul",
    11: "Indirect free-kick",
    12: "Direct free-kick",
    13: "Corner",
    14: "Yellow card",
    15: "Red card",
    16: "Yellow->red card",
}

# ==============================
# MAIN FUNCTION
# ==============================
def run_event_detection(video_path):
    print("Extracting features...")

    extractor = VideoFeatureExtractor(fps=2.0)
    extractor.extract(video_path, FEATURE_PATH)

    print("Applying PCA...")
    reducer = PCAReducer(PCA_FILE, SCALER_FILE)
    reducer.reduce(FEATURE_PATH)

    features = np.load(FEATURE_PATH)

    print("Loading model...")
    model = ContextAwareModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("Running inference...")
    detections = run_inference(model, features, DEVICE, conf_threshold=0.05)

    frame_indices, class_indices = np.where(detections > 0)

    events = []

    for frame_idx, class_idx in zip(frame_indices, class_indices):
        confidence = detections[frame_idx, class_idx]

        seconds_total = frame_idx // 2
        minutes = seconds_total // 60
        seconds = seconds_total % 60

        events.append({
            "time": f"{minutes}:{seconds:02d}",
            "event": EVENT_DICTIONARY[class_idx],
            "confidence": float(confidence)
        })

    return events


# ==============================
# TEST ONLY
# ==============================
if __name__ == "__main__":
    events = run_event_detection("test.mp4")
    print(events)