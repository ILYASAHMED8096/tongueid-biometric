from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models


@dataclass
class EmbedConfig:
    model_name: str = "resnet18"   # resnet18 is fast + strong baseline
    device: str = "cpu"            # change to "cuda" if you have GPU
    input_size: int = 224          # ResNet expects 224x224


class ResNetEmbedder:
    def __init__(self, cfg: EmbedConfig = EmbedConfig()):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        if cfg.model_name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            out_dim = 512
        elif cfg.model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            out_dim = 2048
        else:
            raise ValueError("Unsupported model_name. Use resnet18 or resnet50.")

        # Remove final classification head -> output embedding
        model.fc = nn.Identity()
        model.eval()
        model.to(self.device)

        self.model = model
        self.out_dim = out_dim

        # Normalization used by ImageNet pretrained models
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        # Convert BGR -> RGB
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.cfg.input_size, self.cfg.input_size), interpolation=cv2.INTER_AREA)
        x = rgb.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        x = torch.from_numpy(x).unsqueeze(0)  # add batch dim
        return x.to(self.device)

    @torch.no_grad()
    def embed(self, img_bgr: np.ndarray) -> np.ndarray:
        x = self._preprocess(img_bgr)
        z = self.model(x).squeeze(0)  # (D,)
        z = z.detach().cpu().numpy().astype(np.float32)

        # L2 normalize for cosine similarity
        z = z / (np.linalg.norm(z) + 1e-12)
        return z
