import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_keep_ratio_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    y0 = (nh - size) // 2
    x0 = (nw - size) // 2
    return img[y0:y0 + size, x0:x0 + size]


def random_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h < size or w < size:
        return resize_keep_ratio_crop(img, size)
    y = random.randint(0, h - size)
    x = random.randint(0, w - size)
    return img[y:y + size, x:x + size]


def robust_normalize(x: np.ndarray, low_q: float = 1.0, high_q: float = 99.0) -> np.ndarray:
    lo = np.percentile(x, low_q)
    hi = np.percentile(x, high_q)
    x = (x - lo) / (hi - lo + 1e-6)
    return np.clip(x, 0.0, 1.0)


def add_multiplicative_speckle(x: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.randn(*x.shape).astype(np.float32)
    out = x * (1.0 + sigma * noise)
    return np.clip(out, 0.0, 1.0)


def add_gamma_speckle(x: np.ndarray, looks: float = 2.0) -> np.ndarray:
    # Простейшая SAR-подобная модель: intensity * gamma noise
    # mean = 1, var = 1/looks
    noise = np.random.gamma(shape=looks, scale=1.0 / looks, size=x.shape).astype(np.float32)
    out = x * noise
    return np.clip(out, 0.0, 1.0)


def add_scatterers(x: np.ndarray, prob: float = 0.001, strength=(0.6, 1.0)) -> np.ndarray:
    h, w = x.shape
    mask = (np.random.rand(h, w) < prob).astype(np.float32)
    bright = np.random.uniform(strength[0], strength[1], size=(h, w)).astype(np.float32)
    out = x + mask * bright
    return np.clip(out, 0.0, 1.0)


def add_streaky_reflections(x: np.ndarray, prob: float = 0.15) -> np.ndarray:
    # Редкие "звёздочки"/полосы вокруг ярких точек
    if random.random() > prob:
        return x

    h, w = x.shape
    out = x.copy()

    n = random.randint(2, 8)
    ys = np.random.randint(0, h, size=n)
    xs = np.random.randint(0, w, size=n)

    for cy, cx in zip(ys, xs):
        val = random.uniform(0.5, 1.0)
        length = random.randint(5, 20)
        out[max(0, cy - length):min(h, cy + length + 1), cx] += val * 0.5
        out[cy, max(0, cx - length):min(w, cx + length + 1)] += val * 0.5

    return np.clip(out, 0.0, 1.0)


def pseudo_sar_from_rgb(img_rgb: np.ndarray) -> np.ndarray:
    img = img_rgb.astype(np.float32) / 255.0

    # 1. Базовая интенсивность
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2. Подавляем часть мелкой текстуры
    k = random.choice([3, 5, 7])
    base = cv2.GaussianBlur(gray, (k, k), 0)

    # 3. Добавляем локальную структурную информацию через градиенты
    gx = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad = grad / (grad.max() + 1e-6)

    x = 0.75 * base + 0.25 * grad
    x = robust_normalize(x)

    # 4. Нелинейность, чтобы статистика была менее "фото"-подобной
    gamma = random.uniform(0.6, 1.4)
    x = np.power(np.clip(x, 1e-6, 1.0), gamma)

    # 5. Имитация SAR-speckle
    if random.random() < 0.5:
        x = add_multiplicative_speckle(x, sigma=random.uniform(0.08, 0.22))
    else:
        x = add_gamma_speckle(x, looks=random.uniform(1.5, 4.0))

    # 6. Яркие scatterers
    x = add_scatterers(
        x,
        prob=random.uniform(0.0003, 0.0020),
        strength=(0.5, 1.0),
    )

    # 7. Редкие streak-like reflections
    x = add_streaky_reflections(x, prob=0.2)

    # 8. Локальные насыщения / контраст
    alpha = random.uniform(0.9, 1.25)
    beta = random.uniform(-0.05, 0.05)
    x = np.clip(alpha * x + beta, 0.0, 1.0)

    # 9. Робастная нормализация
    x = robust_normalize(x)

    return x.astype(np.float32)


class RadarColorizationDataset(Dataset):
    def __init__(self, root: str, image_size: int = 256, train: bool = True):
        self.root = Path(root)
        self.paths = sorted(
            [p for p in self.root.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]]
        )
        self.image_size = image_size
        self.train = train

        if not self.paths:
            raise ValueError(f"No images found in {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = str(self.paths[idx])
        img = read_image(path)

        if self.train:
            img = random_crop(img, self.image_size)

            # rotate by 0, 90, 180, 270 degrees
            k = random.randint(0, 3)
            if k > 0:
                img = np.ascontiguousarray(np.rot90(img, k))

            if random.random() < 0.5:
                img = np.ascontiguousarray(np.fliplr(img))
            if random.random() < 0.5:
                img = np.ascontiguousarray(np.flipud(img))
        else:
            img = resize_keep_ratio_crop(img, self.image_size)

        pseudo_sar = pseudo_sar_from_rgb(img)

        rgb = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        sar = torch.from_numpy(pseudo_sar).unsqueeze(0)

        return {
            "input": sar,
            "target": rgb,
            "path": path,
        }