import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def rgb_to_xyz(rgb: torch.Tensor) -> torch.Tensor:
    """
    rgb: [B, 3, H, W] in [0, 1]
    returns xyz: [B, 3, H, W]
    """
    rgb = torch.clamp(rgb, 0.0, 1.0)

    mask = (rgb > 0.04045).float()
    rgb_lin = ((rgb + 0.055) / 1.055) ** 2.4 * mask + (rgb / 12.92) * (1.0 - mask)

    r = rgb_lin[:, 0:1]
    g = rgb_lin[:, 1:2]
    b = rgb_lin[:, 2:3]

    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    return torch.cat([x, y, z], dim=1)


def xyz_to_lab(xyz: torch.Tensor) -> torch.Tensor:
    """
    xyz: [B, 3, H, W]
    returns lab: [B, 3, H, W]
    """
    # D65 white point
    xn = 0.95047
    yn = 1.00000
    zn = 1.08883

    x = xyz[:, 0:1] / xn
    y = xyz[:, 1:2] / yn
    z = xyz[:, 2:3] / zn

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def f(t):
        return torch.where(t > eps, t ** (1.0 / 3.0), (kappa * t + 16.0) / 116.0)

    fx = f(x)
    fy = f(y)
    fz = f(z)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return torch.cat([L, a, b], dim=1)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    xyz = rgb_to_xyz(rgb)
    lab = xyz_to_lab(xyz)
    return lab


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize: bool = False):
        super().__init__()

        weights = models.VGG16_Weights.IMAGENET1K_V1
        vgg_features = models.vgg16(weights=weights).features

        self.blocks = nn.ModuleList([
            vgg_features[:4].eval(),
            vgg_features[4:9].eval(),
            vgg_features[9:16].eval(),
        ])

        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad = False

        self.resize = resize

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        pred = torch.clamp(pred.float(), 0.0, 1.0)
        target = torch.clamp(target.float(), 0.0, 1.0)

        pred = self.normalize(pred)
        target = self.normalize(target)

        if self.resize:
            pred = F.interpolate(pred, size=(224, 224), mode="bilinear", align_corners=False)
            target = F.interpolate(target, size=(224, 224), mode="bilinear", align_corners=False)

        loss = 0.0
        x = pred
        y = target

        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss = loss + F.l1_loss(x, y)

        return loss


class L1PerceptualColorLoss(nn.Module):
    def __init__(
        self,
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        color_weight: float = 0.3,
        resize_vgg: bool = False,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.color_weight = color_weight

        self.l1 = nn.L1Loss()
        self.perc = VGGPerceptualLoss(resize=resize_vgg)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # pred is logits
        pred = torch.sigmoid(pred.float())
        target = torch.clamp(target.float(), 0.0, 1.0)

        # RGB L1
        l1 = self.l1(pred, target)

        # perceptual
        perc = self.perc(pred, target)

        # Lab color loss on a,b only
        pred_lab = rgb_to_lab(pred)
        target_lab = rgb_to_lab(target)

        pred_ab = pred_lab[:, 1:3]
        target_ab = target_lab[:, 1:3]

        color = self.l1(pred_ab, target_ab)

        loss = (
            self.l1_weight * l1
            + self.perceptual_weight * perc
            + self.color_weight * color
        )
        loss = torch.nan_to_num(loss, nan=1.0, posinf=1.0, neginf=1.0)

        return loss, {
            "l1": float(l1.detach().cpu()),
            "perc": float(perc.detach().cpu()),
            "color": float(color.detach().cpu()),
            "total": float(loss.detach().cpu()),
        }