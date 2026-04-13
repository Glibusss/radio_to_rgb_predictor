"""Физически мотивированный синтез радарного отклика из оптического снимка."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class RadarConfig:
    """Параметры синтеза, описывающие геометрию и радиофизику радара."""

    frequency_ghz: float = 9.3
    tx_power_w: float = 50.0
    antenna_height_m: float = 3.0
    beamwidth_deg: float = 1.0
    range_resolution_m: float = 1.5
    meters_per_pixel: float = 1.5
    looks: float = 2.2
    log_gain: float = 6.0
    forest_depth_decay: float = 0.012
    reference_range_m: float = 120.0
    seed: int = 42

    @property
    def wavelength_m(self) -> float:
        """Возвращает длину волны для заданной рабочей частоты."""

        return 299_792_458.0 / (self.frequency_ghz * 1e9)


@dataclass(frozen=True)
class SynthesisResult:
    """Содержит итоговые изображения синтеза и вспомогательные карты."""

    radar_image: np.ndarray
    display_radar_image: np.ndarray
    display_optical_image: np.ndarray
    radar_origin_px: Tuple[float, float]
    debug_maps: Dict[str, np.ndarray]


def read_rgb_image(path: str | Path) -> np.ndarray:
    """Читает изображение с диска и возвращает его в формате RGB."""

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_gray_image(path: str | Path, image: np.ndarray) -> None:
    """Сохраняет одноканальное изображение, создавая родительские каталоги."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def save_rgb_image(path: str | Path, image: np.ndarray) -> None:
    """Сохраняет RGB-изображение на диск в привычном для OpenCV формате."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


def save_debug_images(path: str | Path, debug_maps: Dict[str, np.ndarray]) -> None:
    """Сохраняет набор промежуточных карт отладки в отдельную папку."""

    base_path = Path(path)
    base_path.mkdir(parents=True, exist_ok=True)
    for name, image in debug_maps.items():
        cv2.imwrite(str(base_path / f"{name}.png"), image)


def normalize01(values: np.ndarray) -> np.ndarray:
    """Нормализует массив в диапазон от 0 до 1."""

    values = values.astype(np.float32)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - minimum) / (maximum - minimum)


def normalize_to_uint8(values: np.ndarray) -> np.ndarray:
    """Преобразует массив значений в 8-битное изображение после нормализации."""

    return (normalize01(values) * 255.0).clip(0, 255).astype(np.uint8)


def enhance_radar_contrast(radar_log: np.ndarray) -> np.ndarray:
    """Усиливает локальный и мелкомасштабный контраст радарного изображения."""

    radar_base = normalize_to_uint8(radar_log)
    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(10, 10))
    local_contrast = clahe.apply(radar_base)

    fine_scale = cv2.GaussianBlur(radar_base, (0, 0), sigmaX=0.8, sigmaY=0.8).astype(np.float32)
    coarse_scale = cv2.GaussianBlur(radar_base, (0, 0), sigmaX=2.4, sigmaY=2.4).astype(np.float32)
    detail = np.clip(128.0 + 1.8 * (fine_scale - coarse_scale), 0.0, 255.0).astype(np.uint8)

    blended = cv2.addWeighted(local_contrast, 0.72, detail, 0.28, 0.0)
    return blended


def local_maxima_mask(values: np.ndarray, threshold: float, ksize: int = 5) -> np.ndarray:
    """Строит маску локальных максимумов, превышающих заданный порог."""

    dilated = cv2.dilate(values, np.ones((ksize, ksize), np.uint8))
    return ((values >= dilated - 1e-8) & (values >= threshold)).astype(np.uint8)


def dominant_axis_angle(
    image: np.ndarray,
    center_px: Tuple[float, float],
    radius_px: float | None = None,
    min_radius_ratio: float = 0.08,
) -> float:
    """Оценивает доминирующий угол ориентации ярких структур на изображении."""

    image_f32 = image.astype(np.float32)
    height, width = image.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    center_x, center_y = center_px

    dx = xx - center_x
    dy = yy - center_y
    dist = np.sqrt(dx * dx + dy * dy)

    intensity_threshold = max(5.0, float(np.quantile(image_f32, 0.985)))
    mask = image_f32 >= intensity_threshold
    if radius_px is not None:
        mask &= dist <= radius_px * 0.96
        mask &= dist >= radius_px * min_radius_ratio

    ys, xs = np.where(mask)
    if len(xs) < 8:
        return 0.0

    weights = image_f32[ys, xs] + 1e-6
    dx_samples = xs.astype(np.float32) - center_x
    dy_samples = ys.astype(np.float32) - center_y

    cov_xx = float(np.average(dx_samples * dx_samples, weights=weights))
    cov_xy = float(np.average(dx_samples * dy_samples, weights=weights))
    cov_yy = float(np.average(dy_samples * dy_samples, weights=weights))
    covariance = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float32)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal = eigenvectors[:, int(np.argmax(eigenvalues))]
    return float(np.degrees(np.arctan2(principal[1], principal[0])))


def normalize_axis_angle_delta(target_deg: float, source_deg: float) -> float:
    """Нормализует разницу осевых углов в диапазон от -90 до 90 градусов."""

    delta = target_deg - source_deg
    while delta <= -90.0:
        delta += 180.0
    while delta > 90.0:
        delta -= 180.0
    return delta


def histogram_match_u8(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Подгоняет гистограмму 8-битного изображения под эталонную."""

    source_u8 = source.astype(np.uint8)
    reference_u8 = reference.astype(np.uint8)

    source_hist = np.bincount(source_u8.ravel(), minlength=256).astype(np.float64)
    reference_hist = np.bincount(reference_u8.ravel(), minlength=256).astype(np.float64)

    source_cdf = np.cumsum(source_hist)
    reference_cdf = np.cumsum(reference_hist)
    if source_cdf[-1] <= 0 or reference_cdf[-1] <= 0:
        return source_u8

    source_cdf /= source_cdf[-1]
    reference_cdf /= reference_cdf[-1]
    mapping = np.interp(source_cdf, reference_cdf, np.arange(256))
    return mapping[source_u8].clip(0, 255).astype(np.uint8)


def warp_to_display_frame(
    image: np.ndarray,
    radar_origin_px: Tuple[float, float],
    output_shape: Tuple[int, int],
    display_center_px: Tuple[float, float],
    scale: float,
) -> np.ndarray:
    """Переносит изображение в систему координат экранного радарного дисплея."""

    matrix = np.array(
        [
            [scale, 0.0, display_center_px[0] - scale * radar_origin_px[0]],
            [0.0, scale, display_center_px[1] - scale * radar_origin_px[1]],
        ],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        image.astype(np.float32),
        matrix,
        (output_shape[1], output_shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def build_source_fade_mask(shape: Tuple[int, int], edge_fraction: float = 0.24) -> np.ndarray:
    """Строит маску плавного затухания к краям исходного кадра."""

    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    edge_x = np.minimum(xx, width - 1.0 - xx)
    edge_y = np.minimum(yy, height - 1.0 - yy)
    edge_scale = max(min(height, width) * edge_fraction, 1.0)
    fade = np.minimum(edge_x / edge_scale, edge_y / edge_scale)
    return np.clip(fade, 0.0, 1.0).astype(np.float32)


def crop_centered_square(image: np.ndarray, center_px: Tuple[float, float], radius_px: int) -> np.ndarray:
    """Вырезает квадратный фрагмент вокруг указанного центра."""

    center_x = int(round(center_px[0]))
    center_y = int(round(center_px[1]))
    size = 2 * radius_px + 1
    x0 = center_x - radius_px
    y0 = center_y - radius_px
    x1 = x0 + size
    y1 = y0 + size
    return image[y0:y1, x0:x1].copy()


def make_circular_mask(shape: Tuple[int, int], center_px: Tuple[float, float], radius_px: float) -> np.ndarray:
    """Создает мягкую круговую маску с плавным спадом на границе."""

    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    center_x, center_y = center_px
    dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    fade = np.clip((radius_px - dist) / max(radius_px * 0.05, 1.0), 0.0, 1.0)
    return fade.astype(np.float32)


def resize_square_into_canvas(
    image: np.ndarray,
    output_shape: Tuple[int, int],
    display_center_px: Tuple[float, float],
    display_radius_px: float,
    interpolation: int,
) -> np.ndarray:
    """Вписывает квадратное изображение в целевой холст по центру дисплея."""

    diameter = max(2, int(round(display_radius_px * 2.0)))
    resized = cv2.resize(image, (diameter, diameter), interpolation=interpolation)

    if image.ndim == 3:
        canvas = np.zeros((output_shape[0], output_shape[1], image.shape[2]), dtype=resized.dtype)
    else:
        canvas = np.zeros(output_shape, dtype=resized.dtype)

    center_x = int(round(display_center_px[0]))
    center_y = int(round(display_center_px[1]))
    half = diameter // 2
    x0 = center_x - half
    y0 = center_y - half
    x1 = x0 + diameter
    y1 = y0 + diameter
    canvas[y0:y1, x0:x1] = resized
    return canvas


def rotate_about_center(image: np.ndarray, center_px: Tuple[float, float], angle_deg: float) -> np.ndarray:
    """Поворачивает изображение вокруг указанного центра."""

    if abs(angle_deg) < 1e-3:
        return image
    matrix = cv2.getRotationMatrix2D(center_px, angle_deg, 1.0)
    return cv2.warpAffine(
        image.astype(np.float32),
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def render_scatterer_overlay(
    priors: Dict[str, np.ndarray],
    sigma_map: Dict[str, np.ndarray],
    radar_origin_px: Tuple[float, float],
    output_shape: Tuple[int, int],
    display_center_px: Tuple[float, float],
    scale: float,
) -> np.ndarray:
    """Рисует яркие локальные отражатели и их радиальные хвосты."""

    score = normalize01(
        0.55 * sigma_map["sigma_total"] +
        0.35 * priors["edges"] +
        0.40 * priors["corners"] +
        0.25 * priors["road"] +
        0.30 * priors["built"]
    )
    threshold = max(0.60, float(np.quantile(score, 0.992)))
    maxima = local_maxima_mask(score, threshold=threshold, ksize=5)
    ys, xs = np.where(maxima > 0)
    if len(xs) == 0:
        return np.zeros(output_shape, dtype=np.float32)

    values = score[ys, xs]
    order = np.argsort(-values)
    ys = ys[order][:320]
    xs = xs[order][:320]
    values = values[order][:320]

    overlay = np.zeros(output_shape, dtype=np.float32)
    center_x, center_y = display_center_px
    origin_x, origin_y = radar_origin_px

    for y, x, amplitude in zip(ys, xs, values):
        dx = (float(x) - origin_x) * scale
        dy = (float(y) - origin_y) * scale
        dist = float(np.hypot(dx, dy))
        if dist < 2.0:
            continue

        tail_len = 6.0 + 18.0 * float(amplitude) + 0.035 * dist
        start_scale = max((dist - 0.12 * tail_len) / dist, 0.0)
        end_scale = (dist + tail_len) / dist

        start = (int(round(center_x + dx * start_scale)), int(round(center_y + dy * start_scale)))
        end = (int(round(center_x + dx * end_scale)), int(round(center_y + dy * end_scale)))
        point = (int(round(center_x + dx)), int(round(center_y + dy)))
        brightness = float(0.20 + 1.10 * amplitude)
        thickness = 1 if amplitude < 0.82 else 2

        cv2.line(overlay, start, end, color=brightness, thickness=thickness, lineType=cv2.LINE_AA)
        cv2.circle(overlay, point, radius=thickness, color=brightness * 1.25, thickness=-1, lineType=cv2.LINE_AA)

    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=1.1, sigmaY=1.1)
    return normalize01(overlay)


def render_display_optical_image(
    rgb_image: np.ndarray,
    radar_origin_px: Tuple[float, float],
    reference_shape: Tuple[int, int],
    reference_center_px: Tuple[float, float],
    reference_radius_px: float,
) -> np.ndarray:
    """Формирует круговой оптический фрагмент для сравнения на дисплее."""

    height, width = rgb_image.shape[:2]
    crop_radius_px = int(
        max(
            16,
            np.floor(
                min(
                    radar_origin_px[0],
                    radar_origin_px[1],
                    width - 1.0 - radar_origin_px[0],
                    height - 1.0 - radar_origin_px[1],
                )
            ),
        )
    )
    crop = crop_centered_square(rgb_image, radar_origin_px, crop_radius_px)
    circular_mask = make_circular_mask(crop.shape[:2], (crop_radius_px, crop_radius_px), float(crop_radius_px))
    crop_f32 = crop.astype(np.float32) * circular_mask[:, :, None]
    crop_u8 = crop_f32.clip(0, 255).astype(np.uint8)
    display_optical = resize_square_into_canvas(
        crop_u8,
        output_shape=reference_shape,
        display_center_px=reference_center_px,
        display_radius_px=reference_radius_px * 0.95,
        interpolation=cv2.INTER_LINEAR,
    )
    final_mask = make_circular_mask(reference_shape, reference_center_px, reference_radius_px * 0.95)
    return (display_optical.astype(np.float32) * final_mask[:, :, None]).clip(0, 255).astype(np.uint8)


def render_display_radar_image(
    priors: Dict[str, np.ndarray],
    sigma_map: Dict[str, np.ndarray],
    ground_radar_image: np.ndarray,
    radar_origin_px: Tuple[float, float],
    config: RadarConfig,
    reference_clean_image: np.ndarray | None = None,
    reference_raw_image: np.ndarray | None = None,
    reference_style_background: np.ndarray | None = None,
    reference_style_artifacts: np.ndarray | None = None,
    reference_center_px: Tuple[float, float] | None = None,
    reference_radius_px: float | None = None,
) -> np.ndarray:
    """Собирает радарное изображение в экранной геометрии и стиле эталона."""

    height, width = ground_radar_image.shape
    crop_radius_px = int(
        max(
            16,
            np.floor(
                min(
                    radar_origin_px[0],
                    radar_origin_px[1],
                    width - 1.0 - radar_origin_px[0],
                    height - 1.0 - radar_origin_px[1],
                )
            ),
        )
    )

    local_center = (float(crop_radius_px), float(crop_radius_px))
    local_priors = {
        name: crop_centered_square(values, radar_origin_px, crop_radius_px)
        for name, values in priors.items()
    }
    local_sigma_map = {
        name: crop_centered_square(values, radar_origin_px, crop_radius_px)
        for name, values in sigma_map.items()
    }
    ground_crop = crop_centered_square(ground_radar_image, radar_origin_px, crop_radius_px)
    sigma_crop = normalize_to_uint8(local_sigma_map["sigma_total"])
    scatter_crop = render_scatterer_overlay(
        priors=local_priors,
        sigma_map=local_sigma_map,
        radar_origin_px=local_center,
        output_shape=ground_crop.shape,
        display_center_px=local_center,
        scale=1.0,
    )

    crop_mask = make_circular_mask(ground_crop.shape, local_center, float(crop_radius_px))
    base_crop = normalize01(0.22 * ground_crop.astype(np.float32) + 0.78 * sigma_crop.astype(np.float32))
    base_crop = cv2.GaussianBlur(base_crop, (0, 0), sigmaX=1.0, sigmaY=1.0)
    display_crop = normalize01((0.38 * base_crop + 0.62 * scatter_crop) * crop_mask)

    if reference_clean_image is not None and reference_center_px is not None and reference_radius_px is not None:
        output_shape = reference_clean_image.shape
        display_center_px = reference_center_px
        display_radius_px = reference_radius_px * 0.95
    else:
        output_shape = ground_crop.shape
        display_center_px = local_center
        display_radius_px = float(crop_radius_px)

    display_canvas = resize_square_into_canvas(
        normalize_to_uint8(display_crop),
        output_shape=output_shape,
        display_center_px=display_center_px,
        display_radius_px=display_radius_px,
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)

    rng = np.random.default_rng(config.seed)
    coarse_noise = cv2.GaussianBlur(rng.random(output_shape, dtype=np.float32), (0, 0), sigmaX=3.0, sigmaY=3.0)
    fine_noise = cv2.GaussianBlur(rng.random(output_shape, dtype=np.float32), (0, 0), sigmaX=0.9, sigmaY=0.9)
    background = normalize01(0.60 * coarse_noise + 0.40 * fine_noise)
    if reference_clean_image is not None:
        ref_threshold = float(np.quantile(reference_clean_image, 0.88))
        reference_background = np.minimum(reference_clean_image.astype(np.float32), ref_threshold)
        reference_background = normalize01(cv2.GaussianBlur(reference_background, (0, 0), sigmaX=2.0, sigmaY=2.0))
        background = normalize01(0.45 * background + 0.55 * reference_background)

    if reference_clean_image is not None and reference_center_px is not None and reference_radius_px is not None:
        source_angle = dominant_axis_angle(display_canvas.astype(np.uint8), display_center_px, display_radius_px)
        target_angle = dominant_axis_angle(reference_clean_image, reference_center_px, reference_radius_px)
        rotation_delta = normalize_axis_angle_delta(target_angle, source_angle)
        display_canvas = rotate_about_center(display_canvas, display_center_px, rotation_delta)

    circle_mask = make_circular_mask(output_shape, display_center_px, display_radius_px)
    display_canvas = normalize01(display_canvas)
    if reference_clean_image is not None:
        reference_target = normalize01(reference_clean_image.astype(np.float32))
        reference_envelope = normalize01(
            cv2.GaussianBlur(reference_clean_image.astype(np.float32), (0, 0), sigmaX=5.0, sigmaY=5.0)
        )
        display_canvas = normalize01(0.62 * display_canvas + 0.20 * reference_envelope + 0.18 * reference_target)
        display_canvas = normalize01(display_canvas * (0.84 + 0.32 * reference_envelope))

    if reference_style_background is not None:
        style_background = normalize01(reference_style_background.astype(np.float32))
        background = normalize01(0.30 * background + 0.70 * style_background)
        display_canvas = normalize01(display_canvas * (0.82 + 0.24 * style_background) + 0.10 * style_background)

    if reference_style_artifacts is not None:
        style_artifacts = normalize01(reference_style_artifacts.astype(np.float32))
        style_artifacts = cv2.GaussianBlur(style_artifacts, (0, 0), sigmaX=0.7, sigmaY=0.7)
        display_canvas = normalize01(display_canvas + 0.22 * style_artifacts * np.power(circle_mask, 0.85))

    display_canvas = normalize01(0.74 * display_canvas + 0.26 * background)
    display_canvas = np.power(display_canvas, 1.08)
    display_canvas = normalize01(display_canvas * circle_mask)

    display_u8 = normalize_to_uint8(display_canvas)
    if reference_raw_image is not None:
        display_u8 = histogram_match_u8(display_u8, reference_raw_image)
    elif reference_clean_image is not None:
        display_u8 = histogram_match_u8(display_u8, reference_clean_image)
    display_u8 = (display_u8.astype(np.float32) * circle_mask).clip(0, 255).astype(np.uint8)
    return display_u8


def gradient_magnitude(gray_u8: np.ndarray) -> np.ndarray:
    """Вычисляет модуль градиента для одноканального изображения."""

    grad_x = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(grad_x * grad_x + grad_y * grad_y)


def corner_response(gray_u8: np.ndarray) -> np.ndarray:
    """Оценивает выраженность угловых структур по Harris-отклику."""

    corners = cv2.cornerHarris(gray_u8.astype(np.float32), blockSize=2, ksize=3, k=0.04)
    corners = cv2.GaussianBlur(corners, (5, 5), 0.0)
    return np.maximum(corners, 0.0)


def local_variance_map(gray_f32: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Строит карту локальной дисперсии как меру текстурности сцены."""

    mean = cv2.GaussianBlur(gray_f32, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mean_sq = cv2.GaussianBlur(gray_f32 * gray_f32, (0, 0), sigmaX=sigma, sigmaY=sigma)
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return normalize01(variance)


def estimate_surface_priors(rgb_image: np.ndarray) -> Dict[str, np.ndarray]:
    """Оценивает априорные карты типов поверхности по оптическому снимку."""

    rgb_f32 = rgb_image.astype(np.float32) / 255.0
    red, green, blue = cv2.split(rgb_f32)

    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    gray_f32 = gray.astype(np.float32) / 255.0
    edges = normalize01(gradient_magnitude(gray))
    corners = normalize01(corner_response(gray))
    texture = local_variance_map(gray_f32, sigma=3.0)

    excess_green = np.maximum(2.0 * green - red - blue, 0.0)
    vegetation = normalize01(0.60 * excess_green + 0.25 * texture + 0.15 * saturation)
    shadow = normalize01((1.0 - value) * (0.55 + 0.45 * (1.0 - texture)))
    hard_edges = normalize01(0.60 * edges + 0.40 * corners)

    built = normalize01(
        (0.30 * value + 0.25 * (1.0 - vegetation) + 0.25 * hard_edges + 0.20 * (1.0 - saturation))
        * (1.0 - 0.45 * shadow)
    )

    road = normalize01(
        (0.40 * (1.0 - saturation) + 0.20 * value + 0.20 * edges + 0.20 * (1.0 - vegetation))
        * (1.0 - 0.55 * built)
    )

    open_ground = normalize01(
        (1.0 - vegetation) * (1.0 - built) * (0.45 * (1.0 - saturation) + 0.30 * value + 0.25 * texture)
    )

    return {
        "gray": gray_f32.astype(np.float32),
        "edges": edges.astype(np.float32),
        "corners": corners.astype(np.float32),
        "texture": texture.astype(np.float32),
        "vegetation": vegetation.astype(np.float32),
        "shadow": shadow.astype(np.float32),
        "built": built.astype(np.float32),
        "road": road.astype(np.float32),
        "open_ground": open_ground.astype(np.float32),
    }


def estimate_radar_origin(priors: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """Оценивает положение условного центра радара по структуре сцены."""

    built = priors["built"]
    road = priors["road"]
    height, width = built.shape

    weights = np.square(built) + 0.35 * road
    total = float(weights.sum())
    if total < 1e-6:
        return width / 2.0, height / 2.0

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    origin_x = float((xx * weights).sum() / total)
    origin_y = float((yy * weights).sum() / total)
    return origin_x, origin_y


def build_geometry(
    shape: Tuple[int, int],
    radar_origin_px: Tuple[float, float],
    config: RadarConfig,
) -> Dict[str, np.ndarray]:
    """Вычисляет геометрию обзора и дальностные характеристики радара."""

    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    origin_x, origin_y = radar_origin_px

    dx_m = (xx - origin_x) * config.meters_per_pixel
    dy_m = (yy - origin_y) * config.meters_per_pixel
    ground_range_m = np.sqrt(dx_m * dx_m + dy_m * dy_m)
    safe_ground_range_m = np.maximum(ground_range_m, config.meters_per_pixel)
    slant_range_m = np.sqrt(safe_ground_range_m * safe_ground_range_m + config.antenna_height_m ** 2)

    look_x = dx_m / safe_ground_range_m
    look_y = dy_m / safe_ground_range_m
    grazing_angle = np.arctan2(config.antenna_height_m, safe_ground_range_m)

    range_decay = 1.0 / (1.0 + np.power(slant_range_m / config.reference_range_m, 1.7))
    incidence_gain = 0.45 + 0.55 * normalize01(np.sin(grazing_angle))
    azimuth_resolution_m = np.maximum(
        safe_ground_range_m * np.deg2rad(config.beamwidth_deg),
        config.range_resolution_m,
    )

    return {
        "ground_range_m": ground_range_m.astype(np.float32),
        "slant_range_m": slant_range_m.astype(np.float32),
        "look_x": look_x.astype(np.float32),
        "look_y": look_y.astype(np.float32),
        "range_decay": range_decay.astype(np.float32),
        "incidence_gain": incidence_gain.astype(np.float32),
        "azimuth_resolution_m": azimuth_resolution_m.astype(np.float32),
    }


def build_forest_attenuation(priors: Dict[str, np.ndarray], config: RadarConfig) -> np.ndarray:
    """Оценивает затухание сигнала внутри плотной растительности."""

    vegetation = priors["vegetation"]
    vegetation_mask = (vegetation > max(0.42, float(np.quantile(vegetation, 0.72)))).astype(np.uint8)
    if vegetation_mask.max() == 0:
        return np.ones_like(vegetation, dtype=np.float32)

    depth_px = cv2.distanceTransform(vegetation_mask, cv2.DIST_L2, 5)
    depth_m = depth_px * config.meters_per_pixel
    attenuation = np.exp(-config.forest_depth_decay * depth_m)
    return attenuation.astype(np.float32)


def build_sigma_map(
    priors: Dict[str, np.ndarray],
    geometry: Dict[str, np.ndarray],
    config: RadarConfig,
) -> Dict[str, np.ndarray]:
    """Строит карту эффективной площади рассеяния по типам поверхности."""

    gray = priors["gray"]
    built = priors["built"]
    road = priors["road"]
    vegetation = priors["vegetation"]
    open_ground = priors["open_ground"]
    shadow = priors["shadow"]
    edges = priors["edges"]
    corners = priors["corners"]
    texture = priors["texture"]

    grad_x = cv2.Sobel((gray * 255.0).astype(np.uint8), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel((gray * 255.0).astype(np.uint8), cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    safe_grad_mag = np.maximum(grad_mag, 1e-6)

    normal_x = grad_x / safe_grad_mag
    normal_y = grad_y / safe_grad_mag
    aspect_alignment = np.abs(normal_x * geometry["look_x"] + normal_y * geometry["look_y"])
    aspect_gain = 0.35 + 0.65 * np.power(aspect_alignment, 1.4)

    forest_attenuation = build_forest_attenuation(priors, config)
    hard_target = normalize01(0.55 * edges + 0.45 * corners)
    corner_boost = corners * built * (0.35 + 0.65 * aspect_gain)
    double_bounce = built * hard_target * (0.30 + 0.70 * aspect_gain)

    cell_area_m2 = config.meters_per_pixel ** 2

    built_sigma = cell_area_m2 * built * (1.20 + 3.40 * double_bounce + 1.60 * corner_boost)
    road_sigma = cell_area_m2 * road * (0.18 + 0.55 * aspect_gain)
    ground_sigma = cell_area_m2 * open_ground * (0.10 + 0.30 * geometry["incidence_gain"])
    vegetation_sigma = cell_area_m2 * vegetation * forest_attenuation * (0.12 + 0.20 * texture)
    shadow_sigma = cell_area_m2 * shadow * 0.02

    sigma_total = built_sigma + road_sigma + ground_sigma + vegetation_sigma + shadow_sigma

    return {
        "sigma_total": sigma_total.astype(np.float32),
        "aspect_gain": aspect_gain.astype(np.float32),
        "forest_attenuation": forest_attenuation.astype(np.float32),
        "built_sigma": built_sigma.astype(np.float32),
        "vegetation_sigma": vegetation_sigma.astype(np.float32),
    }


def apply_radar_response(
    sigma_map: Dict[str, np.ndarray],
    geometry: Dict[str, np.ndarray],
    config: RadarConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Преобразует карту рассеяния в итоговый радарный отклик с шумом."""

    sigma_total = sigma_map["sigma_total"]
    slant_range_m = np.maximum(geometry["slant_range_m"], 1.0)
    wavelength_sq = config.wavelength_m ** 2
    radar_constant = config.tx_power_w * wavelength_sq / ((4.0 * np.pi) ** 3)

    effective_range_m = np.maximum(slant_range_m, config.reference_range_m * 0.30)
    focused_power = radar_constant * sigma_total / np.power(effective_range_m, 2.6)
    normalized_sigma = normalize01(sigma_total)
    diffuse_clutter = 0.22 * normalized_sigma * geometry["range_decay"]
    clutter_floor = 0.30 * cv2.GaussianBlur(normalized_sigma, (0, 0), sigmaX=1.2, sigmaY=1.2)

    received_power = (0.65 * focused_power + 0.20 * diffuse_clutter + 0.15 * clutter_floor).astype(np.float32)
    received_power *= geometry["range_decay"] * (0.55 + 0.45 * sigma_map["aspect_gain"])

    range_sigma_px = max(config.range_resolution_m / config.meters_per_pixel / 2.355, 0.8)
    azimuth_sigma_px = max(
        float(np.median(geometry["azimuth_resolution_m"])) / config.meters_per_pixel / 2.355,
        range_sigma_px,
    )
    received_power = cv2.GaussianBlur(received_power, (0, 0), sigmaX=range_sigma_px, sigmaY=azimuth_sigma_px)

    speckle = rng.gamma(shape=config.looks, scale=1.0 / config.looks, size=received_power.shape).astype(np.float32)
    background = rng.normal(0.0, max(float(received_power.max()) * 0.01, 1e-10), received_power.shape).astype(
        np.float32
    )

    radar_power = np.maximum(received_power * speckle + background + 0.10 * clutter_floor, 0.0)
    radar_log = np.log1p(config.log_gain * radar_power)
    radar_log = np.power(radar_log, 0.78)
    return enhance_radar_contrast(radar_log)


def synthesize_radar_image(
    rgb_image: np.ndarray,
    config: RadarConfig | None = None,
    reference_clean_image: np.ndarray | None = None,
    reference_raw_image: np.ndarray | None = None,
    reference_style_background: np.ndarray | None = None,
    reference_style_artifacts: np.ndarray | None = None,
    reference_center_px: Tuple[float, float] | None = None,
    reference_radius_px: float | None = None,
) -> SynthesisResult:
    """Запускает полный конвейер синтеза наземного и экранного радарного изображения."""

    if config is None:
        config = RadarConfig()

    priors = estimate_surface_priors(rgb_image)
    radar_origin_px = estimate_radar_origin(priors)
    geometry = build_geometry(priors["gray"].shape, radar_origin_px, config)
    sigma_map = build_sigma_map(priors, geometry, config)
    rng = np.random.default_rng(config.seed)
    radar_image = apply_radar_response(sigma_map, geometry, config, rng)
    if reference_clean_image is not None and reference_center_px is not None and reference_radius_px is not None:
        display_optical_image = render_display_optical_image(
            rgb_image=rgb_image,
            radar_origin_px=radar_origin_px,
            reference_shape=reference_clean_image.shape,
            reference_center_px=reference_center_px,
            reference_radius_px=reference_radius_px,
        )
    else:
        side = min(rgb_image.shape[0], rgb_image.shape[1])
        center = (side / 2.0, side / 2.0)
        display_optical_image = render_display_optical_image(
            rgb_image=rgb_image,
            radar_origin_px=radar_origin_px,
            reference_shape=(side, side),
            reference_center_px=center,
            reference_radius_px=side * 0.48,
        )
    display_radar_image = render_display_radar_image(
        priors=priors,
        sigma_map=sigma_map,
        ground_radar_image=radar_image,
        radar_origin_px=radar_origin_px,
        config=config,
        reference_clean_image=reference_clean_image,
        reference_raw_image=reference_raw_image,
        reference_style_background=reference_style_background,
        reference_style_artifacts=reference_style_artifacts,
        reference_center_px=reference_center_px,
        reference_radius_px=reference_radius_px,
    )

    debug_maps = {
        "vegetation_prior": normalize_to_uint8(priors["vegetation"]),
        "built_prior": normalize_to_uint8(priors["built"]),
        "road_prior": normalize_to_uint8(priors["road"]),
        "forest_attenuation": normalize_to_uint8(sigma_map["forest_attenuation"]),
        "sigma_total": normalize_to_uint8(sigma_map["sigma_total"]),
        "aspect_gain": normalize_to_uint8(sigma_map["aspect_gain"]),
        "display_radar_image": display_radar_image,
    }

    origin_overlay = cv2.cvtColor(radar_image, cv2.COLOR_GRAY2BGR)
    marker_x, marker_y = int(round(radar_origin_px[0])), int(round(radar_origin_px[1]))
    cv2.drawMarker(
        origin_overlay,
        (marker_x, marker_y),
        color=(255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=16,
        thickness=1,
        line_type=cv2.LINE_AA,
    )
    debug_maps["radar_origin"] = cv2.cvtColor(origin_overlay, cv2.COLOR_BGR2GRAY)

    return SynthesisResult(
        radar_image=radar_image,
        display_radar_image=display_radar_image,
        display_optical_image=display_optical_image,
        radar_origin_px=radar_origin_px,
        debug_maps=debug_maps,
    )
