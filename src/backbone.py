"""The frozen ResNet-18 feature extractor (same as step 05), as a reusable helper."""
import numpy as np
import torch
import torchvision


class Backbone:
    """photo (BGR, MODEL_W x MODEL_H) -> 256 x 22 x 40 features. Frozen, pretrained on ImageNet."""

    def __init__(self, dev):
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.net = torch.nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3).eval().to(dev)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
        self.dev = dev

    @torch.no_grad()
    def __call__(self, imgs_bgr):
        x = torch.from_numpy(np.stack(imgs_bgr)[..., ::-1].copy()).to(self.dev)
        x = (x.permute(0, 3, 1, 2).float() / 255.0 - self.mean) / self.std
        return self.net(x)                                   # (B, 256, 22, 40) on the GPU
