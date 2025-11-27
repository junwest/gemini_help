#!/usr/bin/env python3
"""
데이터셋 이미지(1초 간격 캡처)를 차선탐지 모델로 분석해
오버레이 이미지를 이벤트 단위로 outputs/img_summary 아래에 저장한다.

기능 요약
---------
1. data/images의 파일을 이벤트(예: original_1-1_...)별로 묶음.
2. 각 이미지를 LaneDepartureAnalyzer로 추론하여 차선 오버레이 생성.
3. 이벤트별 폴더에 오버레이 저장 + 대표 썸네일(collage) 생성.
4. 전체/이벤트 요약 통계를 CSV(JSON)로 저장.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from make_masks import LaneDepartureAnalyzer, load_roi_mask  # noqa: E402


DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "images"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "img_summary"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "lane_detect.pth"
DEFAULT_ROI_PATH = PROJECT_ROOT / "data" / "masks" / "masked.png"


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


@dataclass
class FrameMeta:
    event_id: str
    frame_idx: int
    source_path: Path


@dataclass
class FrameData:
    frame_idx: int
    label: str
    image_rgb: np.ndarray
    source_path: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="이벤트 단위 차선탐지 요약 이미지 생성기")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--roi-mask", type=Path, default=DEFAULT_ROI_PATH)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--threshold", type=float, default=0.5, help="차선 이탈 판별 임계값")
    parser.add_argument("--bottom-ratio", type=float, default=0.3, help="하단 차단 비율 (0~1)")
    parser.add_argument("--limit", type=int, default=None, help="처리할 전체 프레임 수 제한")
    parser.add_argument("--max-frames-per-event", type=int, default=None, help="이벤트별 처리 프레임 상한")
    parser.add_argument("--collage-columns", type=int, default=3, help="이벤트 썸네일 열 개수")
    parser.add_argument("--collage-max-tiles", type=int, default=6, help="썸네일에 배치할 최대 이미지 수")
    parser.add_argument("--disable-collage", action="store_true", help="이벤트 썸네일 생성을 건너뜀")
    parser.add_argument("--disable-roi", action="store_true", help="ROI 마스크 적용 안 함")
    parser.add_argument("--disable-images", action="store_true", help="이미지 입력 처리를 건너뜀")
    parser.add_argument("--summary-json", type=Path, default=None, help="이벤트 요약 JSON 저장 경로")
    parser.add_argument("--summary-csv", type=Path, default=None, help="이벤트 요약 CSV 저장 경로")
    parser.add_argument("--video-dir", type=Path, default=None, help="영상 파일 디렉터리 (옵션)")
    parser.add_argument("--video-step-sec", type=float, default=1.0, help="영상에서 샘플링할 간격(초)")
    parser.add_argument("--video-max-samples", type=int, default=None, help="영상별 최대 샘플 프레임 수")
    parser.add_argument("--video-limit", type=int, default=None, help="처리할 최대 영상 수")
    return parser.parse_args()


def collect_image_paths(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"이미지 경로가 존재하지 않습니다: {image_dir}")
    paths = sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"이미지를 찾을 수 없습니다: {image_dir}")
    return paths


def extract_event_and_frame(stem: str) -> Tuple[str, int]:
    """
    파일명 규칙: original_<event>_<res>_<frame>
    예) original_1-1_960x540_00009
    """
    tokens = stem.split("_")
    event_id = tokens[1] if len(tokens) >= 2 else "unknown"
    frame_token = tokens[-1] if tokens else stem
    digits = "".join(ch for ch in frame_token if ch.isdigit())
    frame_idx = int(digits) if digits else 0
    return event_id, frame_idx


def sanitize_label(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "frame"


def group_by_event(
    image_paths: Sequence[Path],
    limit: int | None = None,
    max_per_event: int | None = None,
) -> Dict[str, List[FrameMeta]]:
    grouped: Dict[str, List[FrameMeta]] = defaultdict(list)
    picked = 0
    for path in image_paths:
        if limit is not None and picked >= limit:
            break
        event_id, frame_idx = extract_event_and_frame(path.stem)
        if max_per_event is not None and len(grouped[event_id]) >= max_per_event:
            continue
        grouped[event_id].append(FrameMeta(event_id, frame_idx, path))
        picked += 1
    for frames in grouped.values():
        frames.sort(key=lambda meta: meta.frame_idx)
    return dict(sorted(grouped.items(), key=lambda kv: kv[0]))


def ensure_output_dirs(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)


def save_overlay_image(overlay_rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))


def path_for_report(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_collage(
    images: Sequence[Path],
    output_path: Path,
    columns: int,
    max_tiles: int,
) -> Path | None:
    if not images:
        return None
    tiles = images[:max_tiles]
    pil_images: List[Image.Image] = []
    try:
        for tile in tiles:
            pil_images.append(Image.open(tile).convert("RGB"))
        widths, heights = zip(*(img.size for img in pil_images))
        tile_w = max(widths)
        tile_h = max(heights)
        rows = math.ceil(len(tiles) / columns)
        canvas = Image.new("RGB", (tile_w * columns, tile_h * rows), color=(0, 0, 0))
        for idx, img in enumerate(pil_images):
            r = idx // columns
            c = idx % columns
            offset = (c * tile_w, r * tile_h)
            canvas.paste(img.resize((tile_w, tile_h)), offset)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
        return output_path
    finally:
        for img in pil_images:
            img.close()


def collect_video_paths(video_dir: Path) -> List[Path]:
    if not video_dir.exists():
        raise FileNotFoundError(f"영상 경로가 존재하지 않습니다: {video_dir}")
    paths = sorted(
        p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"영상 파일을 찾을 수 없습니다: {video_dir}")
    return paths


def event_statistics(entries: Iterable[dict]) -> dict:
    entries_list = list(entries)
    total = len(entries_list)
    if total == 0:
        return {"frames": 0}
    departed = sum(1 for e in entries_list if e["departed"])
    left_departed = sum(1 for e in entries_list if e["left_departed"])
    right_departed = sum(1 for e in entries_list if e["right_departed"])
    left_ratio = np.mean([e["left_ratio"] for e in entries_list]) if entries_list else 0.0
    right_ratio = np.mean([e["right_ratio"] for e in entries_list]) if entries_list else 0.0
    return {
        "frames": total,
        "departed_rate": departed / total * 100.0,
        "left_departed_rate": left_departed / total * 100.0,
        "right_departed_rate": right_departed / total * 100.0,
        "avg_left_ratio": float(left_ratio),
        "avg_right_ratio": float(right_ratio),
    }


def save_event_summaries_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def iter_image_frames(frames: Sequence[FrameMeta]) -> Iterable[FrameData]:
    for meta in frames:
        img_bgr = cv2.imread(str(meta.source_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] 이미지를 읽을 수 없어 건너뜁니다: {meta.source_path}")
            continue
        yield FrameData(
            frame_idx=meta.frame_idx,
            label=meta.source_path.stem,
            image_rgb=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
            source_path=meta.source_path,
        )


def iter_video_frames(
    video_path: Path,
    step_sec: float,
    max_samples: int | None,
) -> Iterable[FrameData]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] 영상을 열 수 없어 건너뜁니다: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0
    step_frames = max(1, int(round(max(step_sec, 1e-3) * fps)))
    sample_idx = 0
    frame_idx = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_idx % step_frames == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                label = f"{video_path.stem}_{sample_idx:05d}"
                yield FrameData(
                    frame_idx=sample_idx,
                    label=label,
                    image_rgb=frame_rgb,
                    source_path=None,
                )
                sample_idx += 1
                if max_samples is not None and sample_idx >= max_samples:
                    break
            frame_idx += 1
    finally:
        cap.release()


def process_event(
    event_id: str,
    frame_iter: Iterable[FrameData],
    event_dir: Path,
    analyzer: LaneDepartureAnalyzer,
    args: argparse.Namespace,
    roi_mask: np.ndarray | None,
) -> dict | None:
    ensure_output_dirs(event_dir)
    frame_records: list[dict] = []
    overlay_paths: list[Path] = []
    processed = 0

    for frame in frame_iter:
        processed += 1
        result = analyzer.analyze_rgb(
            frame.image_rgb,
            depart_thr=args.threshold,
            bottom_ratio=args.bottom_ratio,
            roi_mask=roi_mask,
        )
        overlay_rgb = analyzer.draw_overlay(
            image_rgb=result["image_rgb"],
            mask=result["mask"],
            lane_center_x=result["lane_center_x"],
            lane_width=result["lane_width"],
            left_x=result["left_x"],
            right_x=result["right_x"],
            norm_offset=result["norm_offset"],
            departed=result["departed"],
            left_departed=result["left_departed"],
            right_departed=result["right_departed"],
            car_center_x=result["vehicle_center_x"],
            per_frame_left_ratio=result["left_ratio"],
            per_frame_right_ratio=result["right_ratio"],
            window_stats=None,
            left_departure_rate=None,
            right_departure_rate=None,
            total_departure_rate=None,
        )
        overlay_name = f"{sanitize_label(frame.label)}_overlay.jpg"
        overlay_path = event_dir / overlay_name
        save_overlay_image(overlay_rgb, overlay_path)
        overlay_paths.append(overlay_path)

        frame_records.append(
            {
                "event_id": event_id,
                "frame_idx": frame.frame_idx,
                "image_name": frame.source_path.name if frame.source_path else frame.label,
                "overlay_path": path_for_report(overlay_path),
                "departed": bool(result["departed"]),
                "left_departed": bool(result["left_departed"]),
                "right_departed": bool(result["right_departed"]),
                "left_ratio": float(result["left_ratio"]),
                "right_ratio": float(result["right_ratio"]),
                "norm_offset": float(result["norm_offset"]) if result["norm_offset"] is not None else None,
            }
        )

    if processed == 0:
        print(f"[WARN] {event_id} 이벤트에서 처리된 프레임이 없습니다.")
        return None

    stats = event_statistics(frame_records)
    collage_path = None
    if not args.disable_collage:
        collage_path = build_collage(
            images=overlay_paths,
            output_path=event_dir / f"{sanitize_label(event_id)}_summary.jpg",
            columns=max(1, args.collage_columns),
            max_tiles=max(1, args.collage_max_tiles),
        )

    summary = {
        "event_id": event_id,
        "frames": stats.get("frames", 0),
        "departed_rate": round(stats.get("departed_rate", 0.0), 2),
        "left_departed_rate": round(stats.get("left_departed_rate", 0.0), 2),
        "right_departed_rate": round(stats.get("right_departed_rate", 0.0), 2),
        "avg_left_ratio": round(stats.get("avg_left_ratio", 0.0), 2),
        "avg_right_ratio": round(stats.get("avg_right_ratio", 0.0), 2),
        "overlay_dir": path_for_report(event_dir),
        "collage": path_for_report(collage_path) if collage_path else None,
    }

    with (event_dir / "frames.json").open("w", encoding="utf-8") as fp:
        json.dump(frame_records, fp, indent=2, ensure_ascii=False)

    return summary


def process_image_events(
    args: argparse.Namespace,
    analyzer: LaneDepartureAnalyzer,
    roi_mask: np.ndarray | None,
) -> List[dict]:
    if args.disable_images or args.image_dir is None:
        return []
    if not args.image_dir.exists():
        print(f"[WARN] 이미지 경로가 없어 건너뜁니다: {args.image_dir}")
        return []

    image_paths = collect_image_paths(args.image_dir)
    grouped = group_by_event(
        image_paths=image_paths,
        limit=args.limit,
        max_per_event=args.max_frames_per_event,
    )

    print(f"[INFO] 이미지 이벤트 {len(grouped)}개 처리 예정")
    summaries: list[dict] = []
    for event_id, frames in grouped.items():
        event_dir = args.output_dir / event_id
        print(f"[EVENT] {event_id} - {len(frames)} frames")
        summary = process_event(
            event_id=event_id,
            frame_iter=iter_image_frames(frames),
            event_dir=event_dir,
            analyzer=analyzer,
            args=args,
            roi_mask=roi_mask,
        )
        if summary:
            summaries.append(summary)
    return summaries


def process_video_events(
    args: argparse.Namespace,
    analyzer: LaneDepartureAnalyzer,
    roi_mask: np.ndarray | None,
) -> List[dict]:
    if args.video_dir is None:
        return []
    if not args.video_dir.exists():
        print(f"[WARN] 영상 경로가 없어 건너뜁니다: {args.video_dir}")
        return []

    video_paths = collect_video_paths(args.video_dir)
    if args.video_limit is not None:
        video_paths = video_paths[: args.video_limit]

    print(f"[INFO] 영상 이벤트 {len(video_paths)}개 처리 예정")
    summaries: list[dict] = []
    for idx, video_path in enumerate(video_paths, start=1):
        event_id = video_path.stem
        event_dir = args.output_dir / event_id
        print(f"[VIDEO {idx}/{len(video_paths)}] {video_path.name}")
        summary = process_event(
            event_id=event_id,
            frame_iter=iter_video_frames(
                video_path=video_path,
                step_sec=args.video_step_sec,
                max_samples=args.video_max_samples,
            ),
            event_dir=event_dir,
            analyzer=analyzer,
            args=args,
            roi_mask=roi_mask,
        )
        if summary:
            summaries.append(summary)
    return summaries


def main() -> None:
    args = parse_args()

    ensure_output_dirs(args.output_dir)

    roi_mask = None
    if not args.disable_roi and args.roi_mask is not None and args.roi_mask.exists():
        roi_mask = load_roi_mask(args.roi_mask)

    analyzer = LaneDepartureAnalyzer(
        model_path=str(args.model_path.resolve()),
        device=args.device,
        threshold=args.threshold,
        use_resnet=False,
        vehicle_center_x=None,
    )

    summary_rows: list[dict] = []
    summary_rows.extend(process_image_events(args, analyzer, roi_mask))
    summary_rows.extend(process_video_events(args, analyzer, roi_mask))
    if not summary_rows:
        raise RuntimeError("처리할 이미지 또는 영상이 없습니다.")

    summary_json_path = args.summary_json or (args.output_dir / "event_summary.json")
    summary_csv_path = args.summary_csv or (args.output_dir / "event_summary.csv")

    with summary_json_path.open("w", encoding="utf-8") as fp:
        json.dump(summary_rows, fp, indent=2, ensure_ascii=False)

    save_event_summaries_csv(summary_csv_path, summary_rows)

    print(f"[DONE] 요약 JSON: {summary_json_path}")
    print(f"[DONE] 요약 CSV:  {summary_csv_path}")


if __name__ == "__main__":
    main()
