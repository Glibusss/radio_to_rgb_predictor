import random
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = x.min()
    mx = x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    x = normalize01(x)
    return (x * 255.0).clip(0, 255).astype(np.uint8)


def read_rgb_image(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def extract_patches(
    image: np.ndarray,
    patch_size: int = 256,
    stride: int = 128,
    min_std: float = 10.0
):
    h, w, _ = image.shape
    patches = []
    if h < patch_size or w < patch_size:
        return patches

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            if gray.std() >= min_std:
                patches.append(patch)

    return patches


def gradient_magnitude(gray_u8: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def corner_response(gray_u8: np.ndarray) -> np.ndarray:
    dst = cv2.cornerHarris(gray_u8.astype(np.float32), blockSize=2, ksize=3, k=0.04)
    dst = cv2.GaussianBlur(dst, (5, 5), 0)
    dst = np.maximum(dst, 0)
    return dst


def local_maxima_mask(arr: np.ndarray, thresh: float, ksize: int = 5) -> np.ndarray:
    dil = cv2.dilate(arr, np.ones((ksize, ksize), np.uint8))
    mask = (arr >= dil - 1e-8) & (arr >= thresh)
    return mask.astype(np.uint8)


def local_variance_map(gray_f: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    mean = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mean_sq = cv2.GaussianBlur(gray_f * gray_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    var = mean_sq - mean * mean
    var = np.maximum(var, 0.0)
    return normalize01(var)


def surface_reflectivity_map(rgb_patch: np.ndarray) -> np.ndarray:
    """
    Слабая карта фоновой отражательной способности поверхности.

    Идея:
    - texture выше -> лес/кустарник/шероховатые поверхности
    - smooth dark ниже -> гладкие слабые поверхности
    - bright hard немного выше -> светлые жесткие поверхности
    """
    gray = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32) / 255.0

    texture = local_variance_map(gray_f, sigma=3.0)
    smooth = normalize01(1.0 - texture)

    bright_hard = np.power(gray_f, 1.7) * (0.35 + 0.65 * smooth)
    bright_hard = normalize01(bright_hard)

    smooth_dark = (1.0 - gray_f) * smooth
    smooth_dark = normalize01(smooth_dark)

    surf = (
        0.55 * texture +
        0.25 * bright_hard +
        0.10 * gray_f -
        0.20 * smooth_dark
    )
    surf = np.clip(surf, 0.0, None)
    surf = normalize01(surf)
    return surf.astype(np.float32)


def build_sparse_scatterer_map(
    rgb_patch: np.ndarray,
    max_points: int = 180,
    edge_weight: float = 0.8,
    corner_weight: float = 1.3
) -> np.ndarray:
    """
    Ключевой принцип:
    - яркие точки выбираются ТОЛЬКО по геометрии (edges + corners)
    - поверхность влияет ТОЛЬКО на слабый фон
    """
    gray = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32) / 255.0

    edges = normalize01(gradient_magnitude(gray))
    corners = normalize01(corner_response(gray))

    score = (
        0.10 * gray_f +
        edge_weight * edges +
        corner_weight * corners
    )
    score = normalize01(score)

    thr = np.quantile(score, 0.94)
    maxima = local_maxima_mask(score, thresh=thr, ksize=5)

    ys, xs = np.where(maxima > 0)
    vals = score[ys, xs]

    scatter = np.zeros_like(score, dtype=np.float32)

    if len(vals) > 0:
        order = np.argsort(-vals)
        ys = ys[order]
        xs = xs[order]
        vals = vals[order]

        n = min(max_points, len(vals))
        ys = ys[:n]
        xs = xs[:n]
        vals = vals[:n]

        for y, x, v in zip(ys, xs, vals):
            amp = 0.50 + 3.50 * float(v)
            scatter[y, x] += amp

    surface = surface_reflectivity_map(rgb_patch)

    background = (
        0.012 * surface +
        0.004 * gray_f
    )
    scatter += background

    return scatter.astype(np.float32)


def sinc_np(x: np.ndarray) -> np.ndarray:
    return np.sinc(x)


def raised_cosine_window(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones((n,), dtype=np.float32)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    return w.astype(np.float32)


def rotate_coords(xx: np.ndarray, yy: np.ndarray, angle_deg: float):
    th = np.deg2rad(angle_deg)
    xr = xx * np.cos(th) + yy * np.sin(th)
    yr = -xx * np.sin(th) + yy * np.cos(th)
    return xr, yr


def build_af_psf(
    size: int = 101,
    range_scale: float = 6.0,
    azimuth_scale: float = 12.0,
    angle_deg: float = 0.0,
    curvature: float = 0.015,
    sidelobe_decay: float = 1.0,
    use_window: bool = True
) -> np.ndarray:
    """
    PSF с параболической кривизной:
    xr_curved = xr - curvature * yr^2
    """
    assert size % 2 == 1
    c = size // 2

    yy, xx = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    xr, yr = rotate_coords(xx, yy, angle_deg)

    xr_curved = xr - curvature * (yr ** 2)

    hr = sinc_np(xr_curved / range_scale)
    ha = sinc_np(yr / azimuth_scale)

    h = hr * ha

    rr = np.sqrt(
        (xr_curved / (range_scale + 1e-6)) ** 2 +
        (yr / (azimuth_scale + 1e-6)) ** 2
    )
    env = 1.0 / (1.0 + sidelobe_decay * 0.15 * rr ** 2)
    h = h * env

    if use_window:
        w = raised_cosine_window(size)
        win2d = np.outer(w, w).astype(np.float32)
        h = h * win2d

    h = np.abs(h).astype(np.float32)

    s = h.sum()
    if s > 1e-8:
        h /= s

    return h

def add_speckle(image: np.ndarray, looks: float = 1.5) -> np.ndarray:
    noise = np.random.gamma(
        shape=looks,
        scale=1.0 / looks,
        size=image.shape
    ).astype(np.float32)
    return image * noise


def degrade_resolution(
    image: np.ndarray,
    out_w_range=(16, 28),
    out_h_range=(14, 24),
    restore_mode=cv2.INTER_LINEAR
) -> np.ndarray:
    h, w = image.shape
    out_w = random.randint(*out_w_range)
    out_h = random.randint(*out_h_range)

    low = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(low, (w, h), interpolation=restore_mode)
    return restored.astype(np.float32)


def log_compression(x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    x = np.maximum(x, 0.0)
    return np.log1p(gamma * x)


def synthesize_radar_like_af(
    rgb_patch: np.ndarray,
    psf_size: int = 101,
    range_scale_range=(5.0, 8.0),
    azimuth_scale_range=(10.0, 18.0),
    angle_range=(-20.0, 20.0),
    curvature_range=(0.008, 0.025),
    looks_range=(1.2, 2.5),
    background_noise_std=0.01
) -> np.ndarray:
    scatter = build_sparse_scatterer_map(rgb_patch)

    range_scale = random.uniform(*range_scale_range)
    azimuth_scale = random.uniform(*azimuth_scale_range)
    angle_deg = random.uniform(*angle_range)
    curvature = random.uniform(*curvature_range)

    psf = build_af_psf(
        size=psf_size,
        range_scale=range_scale,
        azimuth_scale=azimuth_scale,
        angle_deg=angle_deg,
        curvature=curvature,
        sidelobe_decay=1.0,
        use_window=True
    )

    radar_amp = cv2.filter2D(scatter, -1, psf, borderType=cv2.BORDER_REFLECT)

    radar_amp = radar_amp + np.random.normal(
        0.0, background_noise_std, radar_amp.shape
    ).astype(np.float32)
    radar_amp = np.maximum(radar_amp, 0.0)

    radar_amp = np.power(radar_amp + 1e-8, 0.65)

    looks = random.uniform(*looks_range)
    radar_amp = add_speckle(radar_amp, looks=looks)

    radar_amp = degrade_resolution(
        radar_amp,
        out_w_range=(16, 28),
        out_h_range=(14, 24),
        restore_mode=cv2.INTER_LINEAR
    )

    radar_log = log_compression(radar_amp, gamma=10.0)
    radar_u8 = normalize_to_uint8(radar_log)
    return radar_u8


def split_train_val(items, val_ratio=0.1, seed=42):
    rnd = random.Random(seed)
    items = items.copy()
    rnd.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio))
    return items[n_val:], items[:n_val]


def save_pair(
    rgb_patch: np.ndarray,
    radar_patch: np.ndarray,
    rgb_path: Path,
    radar_path: Path
):
    rgb_bgr = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(rgb_path), rgb_bgr)
    cv2.imwrite(str(radar_path), radar_patch)


def generate_dataset(
    raw_dir="data/raw_rgb",
    out_dir="data",
    patch_size=256,
    stride=128,
    val_ratio=0.1,
    augment_per_patch=2,
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)

    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    train_radar = out_dir / "train" / "radar"
    train_rgb = out_dir / "train" / "rgb"
    val_radar = out_dir / "val" / "radar"
    val_rgb = out_dir / "val" / "rgb"

    for d in [train_radar, train_rgb, val_radar, val_rgb]:
        ensure_dir(d)

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
        image_paths.extend(raw_dir.glob(ext))
        image_paths.extend(raw_dir.glob(ext.upper()))

    if not image_paths:
        raise RuntimeError(f"No images found in {raw_dir}")

    patches = []
    for img_path in image_paths:
        try:
            img = read_rgb_image(str(img_path))
            ps = extract_patches(img, patch_size=patch_size, stride=stride, min_std=10.0)
            patches.extend(ps)
        except Exception as e:
            print(f"[WARN] skip {img_path}: {e}")

    if len(patches) == 0:
        raise RuntimeError("No valid patches extracted")

    train_patches, val_patches = split_train_val(
        patches,
        val_ratio=val_ratio,
        seed=seed
    )

    print("Total patches:", len(patches))
    print("Train patches:", len(train_patches))
    print("Val patches:", len(val_patches))

    idx = 0
    for patch in train_patches:
        for _ in range(augment_per_patch):
            radar = synthesize_radar_like_af(patch)
            name = f"{idx:06d}.png"
            save_pair(patch, radar, train_rgb / name, train_radar / name)
            idx += 1

    idx_val = 0
    for patch in val_patches:
        radar = synthesize_radar_like_af(patch)
        name = f"{idx_val:06d}.png"
        save_pair(patch, radar, val_rgb / name, val_radar / name)
        idx_val += 1


if __name__ == "__main__":
    generate_dataset(
        raw_dir="data/raw_rgb",
        out_dir="data",
        patch_size=256,
        stride=128,
        val_ratio=0.1,
        augment_per_patch=2,
        seed=42
    )