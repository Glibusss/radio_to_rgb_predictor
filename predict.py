import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.utils import save_image

from model import UNetColorizer


def read_input_image(path: str, image_size: int) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0

    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return x


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)

    ckpt_args = checkpoint.get("args", {})
    base_channels = ckpt_args.get("base_channels", 64)

    model = UNetColorizer(
        in_channels=1,
        out_channels=3,
        base=base_channels,
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    x = read_input_image(args.input, args.image_size).to(device)

    with torch.no_grad():
        pred = model(x)

    pred = torch.clamp(pred[0].cpu(), 0.0, 1.0)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(pred, str(out_path))

    print(f"Saved result to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to grayscale/radio image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt or last.pt")
    parser.add_argument("--output", type=str, default="prediction.png", help="Output image path")
    parser.add_argument("--image_size", type=int, default=256, help="Input size expected by model")
    args = parser.parse_args()

    main(args)