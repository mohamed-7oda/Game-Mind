# feature_extractor.py

import os
import cv2
import numpy as np
import pickle as pkl

from keras.applications.resnet_v2 import ResNet152V2, preprocess_input
from keras.models import Model


# ==========================================
# VIDEO FEATURE EXTRACTOR (NO SoccerNet)
# ==========================================
class VideoFeatureExtractor:
    """
    Extract frame-level features using ResNet152V2 (ImageNet pretrained)
    Uses OpenCV for frame extraction
    """

    def __init__(self, fps=2.0, batch_size=32):

        self.fps = fps
        self.batch_size = batch_size

        # Load pretrained ResNet152V2
        base_model = ResNet152V2(
            include_top=True,
            weights="imagenet"
        )

        # Use global average pooling layer (2048-dim output)
        self.model = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer("avg_pool").output
        )

        self.model.trainable = False


    def extract(self, video_path, output_path):
        """
        Extract features from video and save as .npy
        """

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)

        # Some videos return 0 FPS → fix it
        if original_fps is None or original_fps <= 0:
            print("Warning: Could not read FPS. Using default 25 FPS.")
            original_fps = 25

        frame_interval = max(int(original_fps / self.fps), 1)

        frames = []
        frame_id = 0
        features_list = []

        print("Reading video frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)

                # Process in batches (memory safe)
                if len(frames) == self.batch_size:
                    batch = np.array(frames, dtype=np.float32)
                    batch = preprocess_input(batch)

                    batch_features = self.model.predict(batch, verbose=0)
                    features_list.append(batch_features)

                    frames = []

            frame_id += 1

        cap.release()

        # Process remaining frames
        if len(frames) > 0:
            batch = np.array(frames, dtype=np.float32)
            batch = preprocess_input(batch)
            batch_features = self.model.predict(batch, verbose=0)
            features_list.append(batch_features)

        if len(features_list) == 0:
            raise Exception("No frames extracted from video.")

        features = np.vstack(features_list)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.save(output_path, features)

        print(f"Features saved to: {output_path}")
        print(f"Feature shape: {features.shape}")

        return output_path


# ==========================================
# PCA REDUCER
# ==========================================
class PCAReducer:
    """
    Apply PCA + Mean subtraction
    """

    def __init__(self, pca_file, scaler_file):

        if not os.path.exists(pca_file):
            raise FileNotFoundError(f"PCA file not found: {pca_file}")

        if not os.path.exists(scaler_file):
            raise FileNotFoundError(f"Scaler file not found: {scaler_file}")

        with open(pca_file, "rb") as f:
            self.pca = pkl.load(f)

        with open(scaler_file, "rb") as f:
            self.average = pkl.load(f)


    def reduce(self, feature_path):
        """
        Apply mean subtraction + PCA transform
        """

        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        features = np.load(feature_path)

        features = features - self.average
        features = self.pca.transform(features)

        np.save(feature_path, features)

        print("PCA reduction complete.")
        print(f"Reduced feature shape: {features.shape}")

        return feature_path


if __name__ == "__main__":

    # 🔹 Change this to your real video path
    video_path = r"C:\Users\mohamed mahmoud emam\OneDrive\Desktop\GradProject Website\test.mp4"

    # Features file (2048-dim first)
    output_path = "features.npy"

    # 1️⃣ Extract ResNet features
    extractor = VideoFeatureExtractor(fps=2.0)
    extractor.extract(video_path, output_path)

    print("Applying PCA reduction...")

    # 🔹 Put your real PCA file paths here
    pca_file = "pca_512_TF2.pkl"
    scaler_file = "average_512_TF2.pkl"

    # 2️⃣ Apply PCA → 512 dims
    reducer = PCAReducer(pca_file, scaler_file)
    reducer.reduce(output_path)

    # 3️⃣ Check final shape
    reduced = np.load(output_path)
    print("Reduced feature shape:", reduced.shape)

    print("Feature extraction + PCA finished successfully.")
