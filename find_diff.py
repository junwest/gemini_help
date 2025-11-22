import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_SRC_POINTS: Tuple[Tuple[float, float], ...] = (
    (250.0, 680.0),   # 좌하
    (1050.0, 680.0),  # 우하
    (800.0, 450.0),   # 우상
    (480.0, 450.0),   # 좌상
)

DEFAULT_DST_POINTS: Tuple[Tuple[float, float], ...] = (
    (150.0, 800.0),  # 좌하
    (450.0, 800.0),  # 우하
    (450.0, 0.0),    # 우상
    (150.0, 0.0),    # 좌상
)


@dataclass
class DepartureMeasurement:
    left_x: Optional[float]
    right_x: Optional[float]
    lane_center_x: Optional[float]
    lane_width: Optional[float]
    left_ratio: float
    right_ratio: float
    offset_px: Optional[float]
    car_center_bev_x: float
    departed: bool
    bev_image: np.ndarray


@dataclass
class DepartureRatioSample:
    """윈도우 통계를 계산하기 위해 필요한 최소 정보만 담는 샘플."""

    left_ratio: float
    right_ratio: float
    departed: bool

    @classmethod
    def from_measurement(cls, measurement: DepartureMeasurement) -> "DepartureRatioSample":
        return cls(
            left_ratio=float(measurement.left_ratio),
            right_ratio=float(measurement.right_ratio),
            departed=measurement.departed,
        )


@dataclass
class DepartureWindowStats:
    """여러 프레임을 기반으로 계산된 차선 이탈 통계."""

    sample_count: int
    avg_left_ratio: float
    avg_right_ratio: float
    departure_rate: float
    dominant_side: Optional[str]

    def as_dict(self) -> dict:
        return {
            "sample_count": self.sample_count,
            "avg_left_ratio": self.avg_left_ratio,
            "avg_right_ratio": self.avg_right_ratio,
            "departure_rate": self.departure_rate,
            "dominant_side": self.dominant_side,
        }


def summarize_departure_samples(samples: Iterable[DepartureRatioSample]) -> DepartureWindowStats:
    """
    주어진 샘플 집합에서 평균 좌/우 이탈률과 전체 이탈 비율을 계산한다.
    """

    sample_list = list(samples)
    count = len(sample_list)
    if count == 0:
        return DepartureWindowStats(0, 0.0, 0.0, 0.0, None)

    avg_left = sum(sample.left_ratio for sample in sample_list) / count
    avg_right = sum(sample.right_ratio for sample in sample_list) / count
    departure_rate = sum(1 for sample in sample_list if sample.departed) / count * 100.0

    if max(avg_left, avg_right) < 1e-3:
        dominant = None
    else:
        dominant = "LEFT" if avg_left >= avg_right else "RIGHT"

    return DepartureWindowStats(count, avg_left, avg_right, departure_rate, dominant)


def _to_float32(points: Sequence[Tuple[float, float]]) -> np.ndarray:
    return np.array(points, dtype=np.float32)


class LaneDepartureRatioCalculator:
    """
    차선 이탈률(왼/오른쪽)을 계산하기 위한 헬퍼 클래스.
    입력 마스크를 BEV(Perspective Transform)로 변환한 뒤 히스토그램 피크를 활용합니다.
    """

    def __init__(
        self,
        car_center_x: float = 620.0,
        car_center_y_ratio: float = 0.92,
        bev_size: Tuple[int, int] = (600, 800),
        src_points: Sequence[Tuple[float, float]] = DEFAULT_SRC_POINTS,
        dst_points: Sequence[Tuple[float, float]] = DEFAULT_DST_POINTS,
        histogram_threshold: float = 100.0,
        depart_ratio_threshold: float = 50.0,
    ) -> None:
        self.car_center_x = float(car_center_x)
        self.car_center_y_ratio = float(np.clip(car_center_y_ratio, 0.0, 1.0))
        self.bev_size = bev_size
        self.histogram_threshold = histogram_threshold
        self.depart_ratio_threshold = depart_ratio_threshold

        self.src_points = _to_float32(src_points)
        self.dst_points = _to_float32(dst_points)
        self.transform = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

    def _ensure_binary(self, mask: np.ndarray) -> np.ndarray:
        if mask is None:
            raise ValueError("마스크 이미지를 읽을 수 없습니다.")
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return (mask > 127).astype(np.uint8) * 255

    def _warp_to_bev(self, mask: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(mask, self.transform, self.bev_size)

    def _histogram_lane_positions(self, bev_img: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        if bev_img.sum() == 0:
            return None, None
        bottom_half = bev_img[bev_img.shape[0] // 2:, :]
        histogram = bottom_half.sum(axis=0)
        midpoint = histogram.shape[0] // 2

        left_hist = histogram[:midpoint]
        right_hist = histogram[midpoint:]

        if left_hist.size == 0 or right_hist.size == 0:
            return None, None

        left_x = int(np.argmax(left_hist))
        right_x = int(np.argmax(right_hist)) + midpoint

        if histogram[left_x] < self.histogram_threshold or histogram[right_x] < self.histogram_threshold:
            return None, None
        if right_x - left_x <= 0:
            return None, None

        return left_x, right_x

    def _transform_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        src = np.array([[point]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.transform)
        return float(dst[0][0][0]), float(dst[0][0][1])

    def measure(self, mask: np.ndarray) -> DepartureMeasurement:
        binary = self._ensure_binary(mask)
        bev = self._warp_to_bev(binary)
        left_x, right_x = self._histogram_lane_positions(bev)

        lane_center = None
        lane_width = None
        offset_px = None
        left_ratio = 0.0
        right_ratio = 0.0
        departed = False

        if left_x is not None and right_x is not None:
            lane_center = (left_x + right_x) / 2.0
            lane_width = float(right_x - left_x)
            half_width = max(1.0, lane_width / 2.0)

            car_center_y = int(round(self.car_center_y_ratio * (binary.shape[0] - 1)))
            car_center_y = np.clip(car_center_y, 0, binary.shape[0] - 1)
            car_center_x = np.clip(self.car_center_x, 0, binary.shape[1] - 1)

            car_center_bev_x, _ = self._transform_point((car_center_x, car_center_y))
            offset_px = car_center_bev_x - lane_center
            ratio_pct = abs(offset_px) / half_width * 100.0

            if offset_px < 0:
                left_ratio = ratio_pct
            elif offset_px > 0:
                right_ratio = ratio_pct

            departed = ratio_pct >= self.depart_ratio_threshold
        else:
            car_center_y = int(round(self.car_center_y_ratio * (binary.shape[0] - 1)))
            car_center_y = np.clip(car_center_y, 0, binary.shape[0] - 1)
            car_center_x = np.clip(self.car_center_x, 0, binary.shape[1] - 1)
            car_center_bev_x, _ = self._transform_point((car_center_x, car_center_y))

        return DepartureMeasurement(
            left_x=left_x,
            right_x=right_x,
            lane_center_x=lane_center,
            lane_width=lane_width,
            left_ratio=left_ratio,
            right_ratio=right_ratio,
            offset_px=offset_px,
            car_center_bev_x=car_center_bev_x,
            departed=departed,
            bev_image=bev,
        )

    def draw_debug_overlay(self, measurement: DepartureMeasurement) -> np.ndarray:
        canvas = cv2.cvtColor(measurement.bev_image, cv2.COLOR_GRAY2BGR)
        h, _ = canvas.shape

        if measurement.left_x is not None:
            cv2.line(canvas, (int(measurement.left_x), 0), (int(measurement.left_x), h - 1), (255, 0, 0), 2)
        if measurement.right_x is not None:
            cv2.line(canvas, (int(measurement.right_x), 0), (int(measurement.right_x), h - 1), (255, 0, 0), 2)
        if measurement.lane_center_x is not None:
            cv2.line(
                canvas,
                (int(measurement.lane_center_x), 0),
                (int(measurement.lane_center_x), h - 1),
                (0, 255, 0),
                1,
            )

        cv2.line(
            canvas,
            (int(measurement.car_center_bev_x), 0),
            (int(measurement.car_center_bev_x), h - 1),
            (0, 0, 255),
            3,
        )

        text_lines = [
            f"left ratio:  {measurement.left_ratio:6.2f}%",
            f"right ratio: {measurement.right_ratio:6.2f}%",
            f"offset(px):  {measurement.offset_px if measurement.offset_px is not None else 'N/A'}",
            f"departed?:  {'YES' if measurement.departed else 'NO'}",
        ]
        y = 25
        for line in text_lines:
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            y += 25

        return canvas


def _parse_points(value: str) -> Sequence[Tuple[float, float]]:
    chunks = [chunk.strip() for chunk in value.replace(";", " ").split() if chunk.strip()]
    if len(chunks) != 4:
        raise argparse.ArgumentTypeError("좌표는 'x,y x,y x,y x,y' 형식으로 4개를 입력해야 합니다.")
    points: list[Tuple[float, float]] = []
    for chunk in chunks:
        try:
            x_str, y_str = chunk.split(",")
            points.append((float(x_str), float(y_str)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"좌표 파싱 실패: {chunk}") from exc
    return points


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BEV 기반 차선 이탈률 계산 도구")
    parser.add_argument("--mask-path", type=Path, required=True, help="이진 마스크 이미지 경로")
    parser.add_argument("--car-center-x", type=float, default=620.0, help="원본 이미지에서 차량 중심 X 좌표")
    parser.add_argument(
        "--car-center-y-ratio",
        type=float,
        default=0.92,
        help="차량 중심 Y 비율(0~1, 1은 이미지 하단)",
    )
    parser.add_argument("--bev-width", type=int, default=600, help="BEV 출력 가로 크기")
    parser.add_argument("--bev-height", type=int, default=800, help="BEV 출력 세로 크기")
    parser.add_argument(
        "--src-points",
        type=str,
        default=None,
        help="원근 변환 소스 좌표. 예) '250,680 1050,680 800,450 480,450'",
    )
    parser.add_argument(
        "--dst-points",
        type=str,
        default=None,
        help="원근 변환 목적 좌표. 형식은 src와 동일",
    )
    parser.add_argument("--hist-threshold", type=float, default=100.0, help="히스토그램 최소 값")
    parser.add_argument("--depart-threshold", type=float, default=50.0, help="이탈 판단 퍼센트(%)")
    parser.add_argument("--save-overlay", type=Path, default=None, help="디버그 이미지를 저장할 경로")
    parser.add_argument("--show", action="store_true", help="matplotlib으로 결과 표시")
    return parser


def analyze_mask(args: argparse.Namespace) -> None:
    mask = cv2.imread(str(args.mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {args.mask_path}")

    calculator = LaneDepartureRatioCalculator(
        car_center_x=args.car_center_x,
        car_center_y_ratio=args.car_center_y_ratio,
        bev_size=(args.bev_width, args.bev_height),
        src_points=_parse_points(args.src_points) if args.src_points else DEFAULT_SRC_POINTS,
        dst_points=_parse_points(args.dst_points) if args.dst_points else DEFAULT_DST_POINTS,
        histogram_threshold=args.hist_threshold,
        depart_ratio_threshold=args.depart_threshold,
    )
    measurement = calculator.measure(mask)
    overlay = calculator.draw_debug_overlay(measurement)

    print("=" * 50)
    print(f"왼쪽 차선 X: {measurement.left_x}")
    print(f"오른쪽 차선 X: {measurement.right_x}")
    print(f"차선 폭(px): {measurement.lane_width}")
    print(f"차량 중심(BEV) X: {measurement.car_center_bev_x:.2f}")
    print(f"왼쪽 이탈률: {measurement.left_ratio:.2f}%")
    print(f"오른쪽 이탈률: {measurement.right_ratio:.2f}%")
    print(f"이탈 여부: {'YES' if measurement.departed else 'NO'}")
    print("=" * 50)

    if args.save_overlay:
        args.save_overlay.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_overlay), overlay)
        print(f"[INFO] 디버그 이미지를 저장했습니다: {args.save_overlay}")

    if args.show:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARN] matplotlib을 찾을 수 없어 시각화를 건너뜁니다.")
        else:
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.title("BEV Lane Departure Analysis")
            plt.axis("off")
    plt.show()


if __name__ == "__main__":
    analyze_mask(build_arg_parser().parse_args())