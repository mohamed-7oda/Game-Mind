# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

CHUNK_SIZE = 240
NUM_DETECTIONS = 15
NUM_CLASSES = 17

class ContextAwareModel(nn.Module):

    def __init__(self):
        super(ContextAwareModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 128, kernel_size=(1, 512))
        self.conv2 = nn.Conv2d(128, 32, kernel_size=(1, 1))

        # IMPORTANT: Use old names
        self.seg_head = nn.Conv2d(
            32,
            NUM_CLASSES,
            kernel_size=(3, 1),
            padding=(1, 0)
        )

        self.fc_spot = nn.Linear(
            32 * CHUNK_SIZE,
            NUM_DETECTIONS * (2 + NUM_CLASSES)
        )

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # segmentation
        seg = torch.sigmoid(self.seg_head(x))
        seg = seg.squeeze(3).permute(0, 2, 1)

        # spotting
        flat = x.reshape(x.size(0), -1)
        spot = self.fc_spot(flat)
        spot = spot.view(
            x.size(0),
            NUM_DETECTIONS,
            2 + NUM_CLASSES
        )

        return seg, spot
