from os import PathLike
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from numpy import linalg
from torch import nn
from torchvision import transforms, models

class FeatureExtractor(object):
    def __init__(self):
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = models.googlenet(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.model = self.model.cuda().eval()

    def run(self, img: np.ndarray) -> np.ndarray:
        img = Image.fromarray(img)
        img = self.preprocess(img)
        batch = img.unsqueeze(0)
        with torch.no_grad():
            feat = self.model(batch.cuda())
            feat = feat.squeeze().cpu().numpy()

        assert feat.shape == (1024,), f'Invalid feature shape {feat.shape}: expected 1024'
        # normalize frame features
        feat /= linalg.norm(feat) + 1e-10
        return feat


class VideoPreprocessor(object):
    def __init__(self, sample_rate: int, *, segmenter=None) -> None:
        self.model = FeatureExtractor()
        self.sample_rate = sample_rate
        self.segmenter = segmenter

    def get_features(self, video_path: PathLike):
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        assert cap is not None, f'Cannot open video: {video_path}'

        features = []
        n_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if n_frames % self.sample_rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                feat = self.model.run(frame)
                features.append(feat)

            n_frames += 1

        cap.release()

        features = np.array(features)
        return n_frames, features

    def kts(self, n_frames, features):
        seq_len = len(features)
        picks = np.arange(0, seq_len) * self.sample_rate

        if self.segmenter is None:
            from temporal_segments import TemporalSegmenter
            self.segmenter = TemporalSegmenter()
        result = self.segmenter.segment(features, picks, n_frames)
        return np.asarray(result['frame_ranges_inclusive']), np.asarray(result['frame_lengths']), picks

    def run(self, video_path: PathLike):
        n_frames, features = self.get_features(video_path)
        cps, nfps, picks = self.kts(n_frames, features)
        return n_frames, features, cps, nfps, picks
