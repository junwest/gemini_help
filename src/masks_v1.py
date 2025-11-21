
#!/usr/bin/env python3
"""
data/images/ 안의 영상을 미리 학습된 UNet(lane_detect.pth)으로 추론해
ROI(data/masks/masked.png)의 흰 영역 안에서만 차선을 활성화한 마스크를
data/masks/ 및 data/answer/에 저장하는 스크립트.

실행 예시:
    python src/masks_v1.py
    python src/masks_v1.py --threshold 0.45 --limit 20
    python src/masks_v1.py --copy-to-answer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from colab_model import UNet


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "images"
DEFAULT_MASK_DIR = PROJECT_ROOT / "data" / "masks"
DEFAULT_ANSWER_DIR = PROJECT_ROOT / "data" / "answer"
DEFAULT_WEIGHT_PATH = PROJECT_ROOT / "model" / "lane_detect.pth"
DEFAULT_ROI_MASK = DEFAULT_MASK_DIR / "masked.png"

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UNet 기반 차선 추론 결과를 ROI 내 마스크로 저장합니다."
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--answer-dir", type=Path, default=DEFAULT_ANSWER_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHT_PATH)
    parser.add_argument("--roi-mask", type=Path, default=DEFAULT_ROI_MASK)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="시그모이드 확률을 마스크로 이진화할 기준 (0~1)")
    parser.add_argument("--device", default=None,
                        help="cuda 또는 cpu. 기본은 자동 감지")
    parser.add_argument("--limit", type=int, default=None,
                        help="처리할 최대 이미지 수 (디버깅용)")
    parser.add_argument("--copy-to-answer", action="store_true",
                        help="결과를 data/answer/에도 복사")
    return parser.parse_args()


def collect_image_paths(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"이미지 경로가 존재하지 않습니다: {image_dir}")
    paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"이미지를 찾을 수 없습니다: {image_dir}")
    return paths


def prepare_transform(target_hw: Sequence[int]) -> transforms.Compose:
    height, width = target_hw
    return transforms.Compose([
        transforms.Resize((height, width), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_roi_mask(roi_path: Path,
                  target_size: Sequence[int] | None = None) -> np.ndarray:
    if not roi_path.exists():
        raise FileNotFoundError(f"ROI 마스크가 존재하지 않습니다: {roi_path}")
    roi_img = Image.open(roi_path).convert("L")
    if target_size is None:
        tgt_h, tgt_w = roi_img.height, roi_img.width
    else:
        tgt_h, tgt_w = target_size
    if (roi_img.height, roi_img.width) != (tgt_h, tgt_w):
        roi_img = roi_img.resize((tgt_w, tgt_h), Image.NEAREST)
    roi_array = np.array(roi_img)
    roi_mask = (roi_array > 127).astype(np.uint8)
    return roi_mask


def load_model(weight_path: Path, device: torch.device) -> UNet:
    if not weight_path.exists():
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {weight_path}")
    model = UNet(n_channels=3, n_classes=1, bilinear=True).to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    # state_dict가 module. 로 시작하면 제거
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def run_inference(model: UNet, tensor: torch.Tensor,
                  device: torch.device) -> np.ndarray:
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)
    pred = probs.squeeze().cpu().numpy()
    return pred


def save_mask(mask_array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_array).save(path)


def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[INFO] Device: {device}")

    image_paths = collect_image_paths(args.image_dir)
    if args.limit is not None:
        image_paths = image_paths[:args.limit]
        print(f"[INFO] 제한 적용: {len(image_paths)}장만 처리합니다.")

    roi_mask = load_roi_mask(args.roi_mask)
    target_hw = roi_mask.shape  # (H, W)
    transform = prepare_transform(target_hw)
    model = load_model(args.weights, device)

    print(f"[INFO] 총 {len(image_paths)}장의 이미지를 처리합니다.")
    for idx, img_path in enumerate(image_paths, start=1):
        image = Image.open(img_path).convert("RGB")
        tensor = transform(image)
        pred = run_inference(model, tensor, device)
        binary_mask = (pred >= args.threshold).astype(np.uint8) * 255
        if binary_mask.shape != roi_mask.shape:
            binary_mask = np.array(
                Image.fromarray(binary_mask).resize(
                    (roi_mask.shape[1], roi_mask.shape[0]),
                    Image.NEAREST
                )
            )
        final_mask = (binary_mask // 255) * roi_mask * 255

        mask_filename = f"{img_path.stem}_mask.png"
        out_path = (args.output_dir / mask_filename).resolve()
        save_mask(final_mask.astype(np.uint8), out_path)

        if args.copy_to_answer:
            answer_path = (args.answer_dir / mask_filename).resolve()
            save_mask(final_mask.astype(np.uint8), answer_path)

        coverage = final_mask.sum() / (255 * final_mask.size)
        print(f"[{idx:04d}/{len(image_paths):04d}] {img_path.name} -> "
              f"{mask_filename} (coverage {coverage:.4f})")

    print("[INFO] 모든 처리가 완료되었습니다.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\n[WARN] 사용자가 작업을 중단했습니다.")

