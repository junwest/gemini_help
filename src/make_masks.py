import argparse
import csv
import glob
import os
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
for p in {PROJECT_ROOT, SRC_DIR}:
    if str(p) not in sys.path:
        sys.path.append(str(p))

from colab_model import UNet  # 기존 lane_model 대체
from find_diff import DepartureRatioSample, DepartureWindowStats, summarize_departure_samples


class ResNetSegmentation(nn.Module):
    """Wide ResNet-101 백본을 사용한 세그멘테이션 모델"""
    def __init__(self, num_classes=1):
        super(ResNetSegmentation, self).__init__()
        # Wide ResNet-101 백본
        resnet = models.wide_resnet101_2(pretrained=False)
        
        # 백본 레이어
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # 디코더 (간단한 FPN 스타일)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        # 인코더
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 디코더
        x = self.decoder(x)
        return x


class LaneDepartureAnalyzer:
    def __init__(
        self,
        model_path,
        device='auto',
        threshold=0.5,
        use_resnet=False,
        vehicle_center_x: float | None = 620.0,
    ):
        """
        Args:
            model_path: 모델 파일 경로
            device: 디바이스 ('cuda', 'mps', 'cpu', 또는 'auto')
            threshold: 이진화 임계값
            use_resnet: True면 Wide ResNet 백본 사용, False면 UNet 사용
        """
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # 모델 로드
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 모델 타입 자동 감지 (Wide ResNet인지 확인)
        is_resnet_model = False
        if isinstance(checkpoint, dict):
            keys = list(checkpoint.keys())
            # Wide ResNet 특징 키 확인
            if any('layer1' in k or 'layer2' in k or 'layer3' in k or 'layer4' in k for k in keys):
                is_resnet_model = True
        elif isinstance(checkpoint, torch.nn.Module):
            # 모델 구조 확인
            if hasattr(checkpoint, 'layer1') or hasattr(checkpoint, 'layer4'):
                is_resnet_model = True
        
        # 모델 생성
        if is_resnet_model or use_resnet:
            print("Wide ResNet-101 백본 모델 사용")
            self.model = ResNetSegmentation(num_classes=1)
            
            # 체크포인트 로드
            if isinstance(checkpoint, dict):
                # 백본 가중치만 로드 (세그멘테이션 헤더는 제외)
                model_dict = self.model.state_dict()
                pretrained_dict = {}
                
                for k, v in checkpoint.items():
                    # 백본 레이어만 필터링
                    if 'decoder' not in k and 'fc' not in k:
                        # 키 이름이 일치하는 경우
                        if k in model_dict:
                            pretrained_dict[k] = v
                        # conv1, bn1 등 직접 매칭
                        elif k.startswith('conv1.') or k.startswith('bn1.'):
                            pretrained_dict[k] = v
                        elif any(k.startswith(f'{layer}.') for layer in ['layer1', 'layer2', 'layer3', 'layer4']):
                            pretrained_dict[k] = v
                
                # 가중치 로드
                model_dict.update(pretrained_dict)
                try:
                    self.model.load_state_dict(model_dict, strict=False)
                    print(f"백본 가중치 로드 완료 ({len(pretrained_dict)}개 레이어)")
                except Exception as e:
                    print(f"백본 가중치 로드 중 일부 실패 (계속 진행): {e}")
                    # 부분 로드 시도
                    try:
                        self.model.load_state_dict(pretrained_dict, strict=False)
                    except:
                        print("백본 가중치 로드 실패, 랜덤 초기화된 가중치 사용")
            else:
                print("체크포인트가 dict 형식이 아닙니다. 모델 구조 확인 필요.")
        else:
            print("UNet 모델 사용")
            self.model = UNet(n_channels=3, n_classes=1, bilinear=True)
            
            # 체크포인트 형식 확인 및 로드
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'model_state' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    try:
                        self.model.load_state_dict(checkpoint)
                    except Exception as e:
                        print(f"모델 로드 실패 (state_dict 형식 불일치): {e}")
                        print(f"체크포인트 키: {list(checkpoint.keys())[:10] if len(checkpoint.keys()) > 0 else 'empty'}")
                        raise
            else:
                if isinstance(checkpoint, torch.nn.Module):
                    self.model = checkpoint
                else:
                    print(f"알 수 없는 체크포인트 형식: {type(checkpoint)}")
                    raise ValueError("모델 파일 형식을 인식할 수 없습니다.")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 이미지 변환
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.threshold = threshold
        self.vehicle_center_x = vehicle_center_x
        
        print(f"모델 로드 완료: {model_path}")
        print(f"디바이스: {self.device}")
        print(f"임계값: {threshold}")
    
    @torch.no_grad()
    def predict_mask(self, image_rgb, threshold=None):
        """
        이미지에서 차선 마스크 예측
        
        Args:
            image_rgb: RGB 형식의 numpy array
            threshold: 이진화 임계값 (None이면 self.threshold 사용)
            
        Returns:
            mask: 차선 마스크 (0 또는 255)
        """
        if threshold is None:
            threshold = self.threshold
            
        pil_image = Image.fromarray(image_rgb)
        h, w = image_rgb.shape[:2]
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        output = self.model(image_tensor)
        prediction = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # 출력 크기가 원본과 다를 수 있으므로 리사이즈
        if prediction.shape != (h, w):
            prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 이진화
        mask = (prediction > threshold).astype(np.uint8) * 255
        
        return mask
    
    def postprocess_mask(self, mask: np.ndarray, bottom_ratio: float = 0.0,
                         roi_mask: np.ndarray | None = None) -> np.ndarray:
        """예측된 마스크에 하단 제거 및 ROI 마스크를 적용한다."""
        processed = mask
        if bottom_ratio > 0:
            processed = self.remove_bottom_region(processed, bottom_ratio=bottom_ratio)
        if roi_mask is not None:
            processed = apply_roi_mask(processed, roi_mask)
        return processed
    
    def analyze_rgb(self, image_rgb: np.ndarray, depart_thr: float = None,
                    bottom_ratio: float = 0.0,
                    roi_mask: np.ndarray | None = None) -> dict:
        """RGB 이미지 배열 기반 차선 분석."""
        if depart_thr is None:
            depart_thr = self.threshold
        mask = self.predict_mask(image_rgb)
        processed_mask = self.postprocess_mask(mask, bottom_ratio=bottom_ratio, roi_mask=roi_mask)
        return self._collect_metrics(image_rgb, processed_mask, depart_thr)
    
    def _collect_metrics(self, image_rgb: np.ndarray, mask: np.ndarray,
                         depart_thr: float) -> dict:
        h, w = image_rgb.shape[:2]
        lane_center_x, lane_width, left_x, right_x, both_sides = self.compute_lane_center_and_width(mask)
        
        if self.vehicle_center_x is None:
            vehicle_center_x = w / 2.0
        else:
            vehicle_center_x = float(np.clip(self.vehicle_center_x, 0.0, w - 1))
        vehicle_center_x = float(vehicle_center_x)
        
        norm_offset = None
        departed = False
        left_departed = False
        right_departed = False
        left_ratio = 0.0
        right_ratio = 0.0
        offset_px = None
        
        if lane_center_x is not None:
            offset_px = lane_center_x - vehicle_center_x
            if lane_width is None:
                denom = max(1.0, (w / 2.0))
            else:
                denom = max(1.0, lane_width / 2.0)
            norm_offset = float(abs(offset_px) / denom)
            departed = norm_offset > depart_thr
            ratio_pct = norm_offset * 100.0
            if offset_px < 0:
                left_ratio = ratio_pct
            elif offset_px > 0:
                right_ratio = ratio_pct
        
        if left_x is not None:
            left_departed = vehicle_center_x < left_x
        if right_x is not None:
            right_departed = vehicle_center_x > right_x
        
        return {
            'image_rgb': image_rgb,
            'mask': mask,
            'width': w,
            'height': h,
            'lane_center_x': lane_center_x,
            'lane_width': lane_width,
            'left_x': left_x,
            'right_x': right_x,
            'norm_offset': norm_offset,
            'departed': departed,
            'left_departed': left_departed,
            'right_departed': right_departed,
            'both_sides_detected': both_sides,
            'left_ratio': left_ratio,
            'right_ratio': right_ratio,
            'vehicle_center_x': vehicle_center_x,
            'offset_px': offset_px,
        }
    
    def remove_bottom_region(self, mask, bottom_ratio=0.3):
        """
        마스크의 아래 영역을 제거 (차량이 차선으로 인식되는 것을 방지)
        
        Args:
            mask: 차선 마스크
            bottom_ratio: 제거할 아래 영역 비율 (0.3 = 30%)
            
        Returns:
            mask: 아래 영역이 제거된 마스크
        """
        h, w = mask.shape
        bottom_start = int(h * (1 - bottom_ratio))
        
        # 아래 30% 영역을 0으로 설정
        mask_cleaned = mask.copy()
        mask_cleaned[bottom_start:, :] = 0
        
        return mask_cleaned
    
    def compute_lane_center_and_width(self, mask, roi_y_ratio=0.7, band_px=30):
        """
        이진 마스크에서 차선 중심과 폭 계산
        
        Args:
            mask: 차선 마스크
            roi_y_ratio: ROI y 비율 (0.7 = 상위 70% 영역 사용, 아래 30%는 이미 제거됨)
            band_px: 분석할 밴드 픽셀 수
            
        Returns:
            lane_center_x: 차선 중심 x좌표 (없으면 None)
            lane_width: 차선 폭 (없으면 None)
            left_x: 왼쪽 차선 x좌표 (없으면 None)
            right_x: 오른쪽 차선 x좌표 (없으면 None)
            both_sides: 좌우 차선이 모두 검출되었는지 여부
        """
        h, w = mask.shape
        # ROI는 상위 70% 영역에서만 (아래 30%는 이미 제거됨)
        y = int(h * roi_y_ratio)
        y0 = max(0, y - band_px)
        y1 = min(h - 1, y + band_px)
        
        band = (mask[y0:y1 + 1] > 0).astype(np.uint8)
        if band.sum() == 0:
            return None, None, None, None, False
        
        # x별로 픽셀 카운트
        x_hist = band.sum(axis=0)
        mid = w // 2
        left_hist = x_hist[:mid]
        right_hist = x_hist[mid:]
        
        left_present = left_hist.sum() > 0
        right_present = right_hist.sum() > 0
        
        left_x = None
        right_x = None
        if left_present:
            xs = np.arange(0, mid)
            left_x = float((xs * left_hist).sum() / max(1, left_hist.sum()))
        if right_present:
            xs = np.arange(mid, w)
            right_x = float((xs * right_hist).sum() / max(1, right_hist.sum()))
        
        if left_present and right_present:
            lane_center_x = (left_x + right_x) / 2.0
            lane_width = max(1.0, right_x - left_x)
            return lane_center_x, lane_width, left_x, right_x, True
        
        # 한쪽만 있을 때
        if left_present or right_present:
            xs = np.arange(0, w)
            lane_center_x = float((xs * x_hist).sum() / max(1, x_hist.sum()))
            lane_width = None
            return lane_center_x, lane_width, left_x, right_x, False
        
        return None, None, None, None, False
    
    def analyze_image(self, image_path, depart_thr=0.5, bottom_ratio=0.3,
                      roi_mask: np.ndarray | None = None):
        """
        단일 이미지 분석
        
        Args:
            image_path: 이미지 파일 경로
            depart_thr: 차선 이탈 판단 임계값 (norm_offset > depart_thr이면 이탈)
            
        Returns:
            dict: 분석 결과
        """
        # 이미지 로드
        image_path = Path(image_path)
        image_path_str = str(image_path)
        img_bgr = cv2.imread(image_path_str, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        result = self.analyze_rgb(
            img_rgb,
            depart_thr=depart_thr,
            bottom_ratio=bottom_ratio,
            roi_mask=roi_mask,
        )
        result.update({
            'image_path': image_path_str,
            'image_name': image_path.name,
        })
        return result
    
    def draw_overlay(self, image_rgb, mask, lane_center_x, lane_width, left_x, right_x, 
                     norm_offset, departed, left_departed, right_departed,
                     left_departure_rate=None, right_departure_rate=None, total_departure_rate=None,
                     car_center_x: float | None = None,
                     per_frame_left_ratio: float | None = None,
                     per_frame_right_ratio: float | None = None,
                     window_stats: DepartureWindowStats | None = None):
        """
        오버레이 이미지 생성 (차선 표시 및 통계 정보)
        
        Args:
            image_rgb: 원본 RGB 이미지
            mask: 차선 마스크
            lane_center_x: 차선 중심 x좌표
            lane_width: 차선 폭
            left_x: 왼쪽 차선 x좌표
            right_x: 오른쪽 차선 x좌표
            norm_offset: 정규화된 오프셋
            departed: 전체 이탈 여부
            left_departed: 왼쪽 이탈 여부
            right_departed: 오른쪽 이탈 여부
            left_departure_rate: 왼쪽 차선 이탈률 (전체 통계용)
            right_departure_rate: 오른쪽 차선 이탈률 (전체 통계용)
            total_departure_rate: 전체 차선 이탈률 (전체 통계용)
            
        Returns:
            overlay: 오버레이된 이미지 (RGB)
        """
        h, w = image_rgb.shape[:2]
        # RGB를 BGR로 변환 (OpenCV 함수 사용을 위해)
        overlay = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
        
        # 마스크 오버레이 (빨간색) - 검출된 차선 (BGR 순서: 0, 0, 255)
        # 차선 탐지 시각화를 더 명확하게 하기 위해 약간 더 진하게 표시
        mask_bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
        color_mask = np.zeros_like(mask_bgr)
        color_mask[mask > 0] = [0, 0, 255]  # BGR에서 빨간색
        overlay = cv2.addWeighted(overlay, 0.65, color_mask, 0.35, 0)  # 차선을 더 명확하게 표시
        
        # 차량 중심선 (노란색) - 사용자가 지정한 x좌표
        if car_center_x is None:
            cx_vehicle = w // 2
        else:
            cx_vehicle = int(np.clip(round(car_center_x), 0, w - 1))
        cv2.line(overlay, (cx_vehicle, int(h * 0.6)), (cx_vehicle, h - 1), (0, 255, 255), 3)
        cv2.putText(
            overlay,
            f"Vehicle Center ({cx_vehicle})",
            (max(0, cx_vehicle - 110), int(h * 0.58)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        
        # 차선 중심선 (초록색) (BGR 순서: 0, 255, 0)
        if lane_center_x is not None:
            cx = int(round(lane_center_x))
            cv2.line(overlay, (cx, int(h * 0.6)), (cx, h - 1), (0, 255, 0), 3)
            cv2.putText(overlay, "Lane Center", (cx - 70, int(h * 0.58)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        # 왼쪽 차선 (파란색) (BGR 순서: 255, 0, 0)
        if left_x is not None:
            lx = int(round(left_x))
            cv2.line(overlay, (lx, int(h * 0.6)), (lx, h - 1), (255, 0, 0), 2)
            if left_departed:
                cv2.putText(overlay, "LEFT DEPARTED", (lx - 90, int(h * 0.55)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        
        # 오른쪽 차선 (파란색) (BGR 순서: 255, 0, 0)
        if right_x is not None:
            rx = int(round(right_x))
            cv2.line(overlay, (rx, int(h * 0.6)), (rx, h - 1), (255, 0, 0), 2)
            if right_departed:
                cv2.putText(overlay, "RIGHT DEPARTED", (rx - 100, int(h * 0.55)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        
        # 정보 패널 (오른쪽 상단)
        info_lines: list[str] = []

        def add_blank():
            if info_lines and info_lines[-1] != "":
                info_lines.append("")
        
        summary_lines = []
        if total_departure_rate is not None:
            summary_lines.append(f"Total: {total_departure_rate:.1f}%")
        if left_departure_rate is not None:
            summary_lines.append(f"Left: {left_departure_rate:.1f}%")
        if right_departure_rate is not None:
            summary_lines.append(f"Right: {right_departure_rate:.1f}%")
        if summary_lines:
            info_lines.extend(summary_lines)
            add_blank()

        if window_stats and window_stats.sample_count > 0:
            info_lines.append(f"Window: {window_stats.sample_count}f")
            info_lines.append(
                f" Avg L: {window_stats.avg_left_ratio:.1f}% | Avg R: {window_stats.avg_right_ratio:.1f}%"
            )
            depart_line = f" Depart: {window_stats.departure_rate:.1f}%"
            if window_stats.dominant_side:
                depart_line += f" ({window_stats.dominant_side})"
            info_lines.append(depart_line)
            add_blank()
        
        # 이탈 상태 표시
        status_parts = []
        if left_departed:
            status_parts.append("LEFT DEPARTED")
        if right_departed:
            status_parts.append("RIGHT DEPARTED")
        if not left_departed and not right_departed:
            if departed:
                status_parts.append("DEPARTED")
            else:
                status_parts.append("IN-LANE")
        
        if status_parts:
            info_lines.append(" | ".join(status_parts))
        
        # 추가 정보
        if norm_offset is not None:
            info_lines.append(f"Offset: {norm_offset:.3f}")
        if lane_width is not None:
            info_lines.append(f"Width: {lane_width:.0f}px")
        
        if per_frame_left_ratio is not None or per_frame_right_ratio is not None:
            add_blank()
            info_lines.append(f"Frame L: {0.0 if per_frame_left_ratio is None else per_frame_left_ratio:.1f}%")
            info_lines.append(f"Frame R: {0.0 if per_frame_right_ratio is None else per_frame_right_ratio:.1f}%")
        
        # 배경 박스 (오른쪽 상단)
        text_height = 22
        box_width = 250
        box_height = len([l for l in info_lines if l]) * text_height + 15
        box_x = w - box_width - 10
        box_y = 10
        
        cv2.rectangle(overlay, (box_x, box_y), (w - 10, box_y + box_height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (box_x, box_y), (w - 10, box_y + box_height), (255, 255, 255), 2)
        
        # 텍스트 출력
        y_offset = box_y + 25
        for i, line in enumerate(info_lines):
            if line == "":
                y_offset += text_height // 2
                continue
            if "Total:" in line or "Left:" in line or "Right:" in line:
                color = (0, 255, 255)  # 노란색 (BGR) - 이탈률 강조
                font_scale = 0.55
                thickness = 2
            elif "DEPARTED" in line:
                color = (0, 0, 255)  # 빨간색 (BGR)
                font_scale = 0.5
                thickness = 2
            elif "IN-LANE" in line:
                color = (0, 255, 0)  # 초록색 (BGR)
                font_scale = 0.5
                thickness = 2
            else:
                color = (255, 255, 255)  # 흰색 (BGR)
                font_scale = 0.45
                thickness = 1
            cv2.putText(overlay, line, (box_x + 10, y_offset + i * text_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        
        # 범례 (하단)
        legend_y = h - 100
        cv2.rectangle(overlay, (10, legend_y), (450, h - 10), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, legend_y), (450, h - 10), (255, 255, 255), 2)
        cv2.putText(overlay, "Legend:", (20, legend_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(overlay, (25, legend_y + 30), (65, legend_y + 30), (0, 255, 255), 2)  # 노란색 (BGR)
        cv2.putText(overlay, "Vehicle Center", (70, legend_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(overlay, (25, legend_y + 50), (65, legend_y + 50), (0, 255, 0), 2)  # 초록색 (BGR)
        cv2.putText(overlay, "Lane Center", (70, legend_y + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(overlay, (25, legend_y + 70), (65, legend_y + 70), (255, 0, 0), 2)  # 파란색 (BGR)
        cv2.putText(overlay, "Left/Right Lane", (70, legend_y + 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # BGR을 RGB로 다시 변환하여 반환
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def load_roi_mask(mask_path: Path | None,
                  target_hw: tuple[int, int] | None = None) -> np.ndarray | None:
    if mask_path is None:
        return None
    if not mask_path.exists():
        raise FileNotFoundError(f"ROI 마스크를 찾을 수 없습니다: {mask_path}")
    roi_img = Image.open(mask_path).convert("L")
    if target_hw is not None:
        h, w = target_hw
        if (roi_img.height, roi_img.width) != (h, w):
            roi_img = roi_img.resize((w, h), Image.NEAREST)
    roi_arr = (np.array(roi_img) > 127).astype(np.uint8) * 255
    return roi_arr


def apply_roi_mask(mask: np.ndarray, roi_mask: np.ndarray | None) -> np.ndarray:
    """ROI 마스크를 차선 마스크에 적용."""
    if roi_mask is None:
        return mask
    if roi_mask.shape != mask.shape:
        resized_roi = cv2.resize(
            roi_mask,
            (mask.shape[1], mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        resized_roi = roi_mask
    return cv2.bitwise_and(mask, mask, mask=resized_roi)


def collect_image_files(image_dir: Path, limit: int | None = None) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"이미지 디렉터리가 없습니다: {image_dir}")
    paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        raise RuntimeError(f"이미지 파일을 찾을 수 없습니다: {image_dir}")
    if limit is not None:
        paths = paths[:limit]
    return paths


def generate_masks_from_dataset(args, analyzer: LaneDepartureAnalyzer,
                                roi_mask: np.ndarray | None) -> None:
    image_dir: Path = args.image_dir.resolve()
    mask_dir: Path = args.mask_dir.resolve()
    answer_dir: Path = args.answer_dir.resolve()

    mask_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_to_answer:
        answer_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_files(image_dir, args.limit)
    print(f"[INFO] 총 {len(image_paths)}장의 이미지를 처리합니다.")

    for idx, img_path in enumerate(tqdm(image_paths, desc="마스크 생성"), start=1):
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] 이미지를 열 수 없습니다: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = analyzer.predict_mask(img_rgb)
        mask = analyzer.postprocess_mask(
            mask,
            bottom_ratio=args.bottom_ratio,
            roi_mask=roi_mask,
        )

        mask_filename = f"{img_path.stem}_mask.png"
        mask_output = mask_dir / mask_filename
        Image.fromarray(mask.astype(np.uint8)).save(mask_output)
        if args.copy_to_answer:
            Image.fromarray(mask.astype(np.uint8)).save(answer_dir / mask_filename)

        coverage = mask.sum() / (255 * mask.size)
        print(f"[{idx:04d}/{len(image_paths):04d}] {img_path.name} -> "
              f"{mask_filename} (coverage {coverage:.4f})")

    print("[INFO] 마스크 생성을 마쳤습니다.")


def generate_overlay_video(args, analyzer: LaneDepartureAnalyzer,
                           roi_mask: np.ndarray | None) -> Path:
    video_path: Path = args.video_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

    if args.video_output is None:
        output_path = PROJECT_ROOT / "outputs" / f"{video_path.stem}_overlay.mp4"
    else:
        output_path = args.video_output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = args.video_fps or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError("영상 해상도를 확인할 수 없습니다.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"출력 영상을 생성할 수 없습니다: {output_path}")

    window_size = max(1, int(round(fps * args.video_window_sec)))
    ratio_window: deque[DepartureRatioSample] = deque(maxlen=window_size)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_limit = args.video_max_frames if args.video_max_frames else None
    progress_total = frame_limit or (total_frames if total_frames > 0 else None)
    processed = 0

    progress = tqdm(total=progress_total, desc="영상 처리", unit="frame")
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_result = analyzer.analyze_rgb(
                frame_rgb,
                depart_thr=args.threshold,
                bottom_ratio=args.bottom_ratio,
                roi_mask=roi_mask,
            )

            ratio_window.append(
                DepartureRatioSample(
                    left_ratio=frame_result['left_ratio'],
                    right_ratio=frame_result['right_ratio'],
                    departed=frame_result['departed'],
                )
            )
            window_stats = summarize_departure_samples(ratio_window)

            overlay_rgb = analyzer.draw_overlay(
                image_rgb=frame_result['image_rgb'],
                mask=frame_result['mask'],
                lane_center_x=frame_result['lane_center_x'],
                lane_width=frame_result['lane_width'],
                left_x=frame_result['left_x'],
                right_x=frame_result['right_x'],
                norm_offset=frame_result['norm_offset'],
                departed=frame_result['departed'],
                left_departed=frame_result['left_departed'],
                right_departed=frame_result['right_departed'],
                left_departure_rate=None,
                right_departure_rate=None,
                total_departure_rate=None,
                car_center_x=frame_result['vehicle_center_x'],
                per_frame_left_ratio=frame_result['left_ratio'],
                per_frame_right_ratio=frame_result['right_ratio'],
                window_stats=window_stats,
            )

            writer.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

            processed += 1
            progress.update(1)
            if frame_limit and processed >= frame_limit:
                break
    finally:
        progress.close()
        cap.release()
        writer.release()

    print(f"[INFO] 영상 처리 완료 ({processed} frames) → {output_path}")
    return output_path


def parse_mask_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="make_masks.py 기반으로 data/images → data/masks(+answer) 마스크 생성"
    )
    parser.add_argument("--image-dir", type=Path, default=PROJECT_ROOT / "data" / "images")
    parser.add_argument("--mask-dir", type=Path, default=PROJECT_ROOT / "data" / "masks")
    parser.add_argument("--answer-dir", type=Path, default=PROJECT_ROOT / "data" / "answer")
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "model" / "lane_detect.pth")
    parser.add_argument("--roi-mask", type=Path, default=PROJECT_ROOT / "data" / "masks" / "masked.png")
    parser.add_argument("--disable-roi", action="store_true", help="ROI 마스크 적용 안 함")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bottom-ratio", type=float, default=0.3,
                        help="아래쪽 제거 비율 (0이면 미적용)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--limit", type=int, default=None, help="처리할 이미지 수 제한")
    parser.add_argument("--copy-to-answer", action="store_true",
                        help="생성된 마스크를 answer 디렉터리에도 복사")
    parser.add_argument("--use-resnet", action="store_true",
                        help="Wide ResNet 세그멘테이션 모델 로드 시도")
    parser.add_argument("--car-center-x", type=float, default=620.0,
                        help="오버레이에 사용할 차량 중심 X 좌표")
    parser.add_argument("--video-path", type=Path, default=None,
                        help="차선 이탈률 오버레이를 생성할 입력 영상 경로")
    parser.add_argument("--video-output", type=Path, default=None,
                        help="오버레이 결과 영상을 저장할 경로 (기본: outputs/<이름>_overlay.mp4)")
    parser.add_argument("--video-window-sec", type=float, default=2.0,
                        help="실시간 시나리오를 위한 차선 이탈률 윈도 길이(초)")
    parser.add_argument("--video-max-frames", type=int, default=None,
                        help="디버깅용: 처리할 최대 프레임 수")
    parser.add_argument("--video-fps", type=float, default=None,
                        help="영상 메타데이터에 FPS가 없을 경우 사용할 값")
    parser.add_argument("--video-only", action="store_true",
                        help="마스크 생성은 건너뛰고 영상 오버레이만 생성")
    return parser.parse_args()


def mask_generation_cli():
    args = parse_mask_cli_args()
    roi_mask = None
    if not args.disable_roi and args.roi_mask is not None:
        roi_mask = load_roi_mask(args.roi_mask, None)

    analyzer = LaneDepartureAnalyzer(
        model_path=str(args.model_path.resolve()),
        device=args.device,
        threshold=args.threshold,
        use_resnet=args.use_resnet,
        vehicle_center_x=args.car_center_x,
    )

    if not args.video_only:
        generate_masks_from_dataset(args, analyzer, roi_mask)
    if args.video_path:
        args.video_path = args.video_path.expanduser().resolve()
        if args.video_output:
            args.video_output = args.video_output.expanduser()
        generate_overlay_video(args, analyzer, roi_mask)
    elif args.video_only:
        print("[WARN] --video-only 옵션이 지정되었지만 --video-path 가 없어 실행할 작업이 없습니다.")


def lane_departure_analysis_report():
    # 설정
    image_dir = "/Users/joonseokim/Desktop/캡스톤영상_11_16/drive-download-20251117T145515Z-1-001/drive-download-20251117T145515Z-1-001_frames"
    model_path = "/Users/joonseokim/Downloads/wide_resnet101_2-32ee1156.pth"
    output_dir = "./11_16_lane_detection_results"
    
    # 모델 파일 확인
    if not os.path.exists(model_path):
        print(f"에러: 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    # 이미지 디렉토리 확인
    if not os.path.exists(image_dir):
        print(f"에러: 이미지 디렉토리를 찾을 수 없습니다: {image_dir}")
        return
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 파일 목록 수집
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) + 
                        glob.glob(os.path.join(image_dir, "*.png")))
    
    if not image_files:
        print(f"에러: 이미지 파일을 찾을 수 없습니다: {image_dir}")
        return
    
    print(f"총 {len(image_files)}개의 이미지 파일을 찾았습니다.")
    print(f"모델: {model_path}")
    print(f"이미지 디렉토리: {image_dir}")
    print(f"출력 디렉토리: {output_dir}")
    print("-" * 60)
    
    # 분석기 생성
    analyzer = LaneDepartureAnalyzer(model_path, device='auto', threshold=0.5)
    
    # detected 폴더 생성
    detected_dir = os.path.join(output_dir, 'detected')
    os.makedirs(detected_dir, exist_ok=True)
    
    # 결과 저장용
    results = []
    departed_count = 0
    left_departed_count = 0
    right_departed_count = 0
    images_with_lanes = []  # 차선이 탐지된 이미지들
    
    # 첫 번째 패스: 이미지 분석
    print("\n1단계: 이미지 분석 중...")
    for img_path in tqdm(image_files, desc="처리 중"):
        try:
            result = analyzer.analyze_image(img_path, depart_thr=0.5)
            
            if result['departed']:
                departed_count += 1
            if result['left_departed']:
                left_departed_count += 1
            if result['right_departed']:
                right_departed_count += 1
            
            # 차선이 탐지된 이미지만 저장
            if result['lane_center_x'] is not None:
                images_with_lanes.append(result)
            
            results.append({
                'image_name': result['image_name'],
                'image_path': result['image_path'],
                'width': result['width'],
                'height': result['height'],
                'lane_center_x': '' if result['lane_center_x'] is None else f"{result['lane_center_x']:.2f}",
                'lane_width_px': '' if result['lane_width'] is None else f"{result['lane_width']:.1f}",
                'left_x': '' if result['left_x'] is None else f"{result['left_x']:.2f}",
                'right_x': '' if result['right_x'] is None else f"{result['right_x']:.2f}",
                'norm_offset': '' if result['norm_offset'] is None else f"{result['norm_offset']:.3f}",
                'departed': int(result['departed']),
                'left_departed': int(result['left_departed']),
                'right_departed': int(result['right_departed']),
                'both_sides_detected': int(result['both_sides_detected']),
            })
        except Exception as e:
            print(f"\n에러 발생 ({os.path.basename(img_path)}): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 통계 계산
    total_images = len(results)
    departure_rate = departed_count / total_images if total_images > 0 else 0.0
    
    # 왼쪽/오른쪽 이탈률 계산 (차선이 탐지된 이미지만 대상)
    left_departed_lane_count = sum(1 for r in images_with_lanes if r['left_departed'])
    right_departed_lane_count = sum(1 for r in images_with_lanes if r['right_departed'])
    total_with_lanes = len(images_with_lanes)
    
    left_departure_rate = (left_departed_lane_count / total_with_lanes * 100) if total_with_lanes > 0 else 0.0
    right_departure_rate = (right_departed_lane_count / total_with_lanes * 100) if total_with_lanes > 0 else 0.0
    total_departure_rate = departure_rate * 100
    
    # norm_offset 통계
    norm_offsets = [float(r['norm_offset']) for r in results if r['norm_offset'] and r['norm_offset'] != '']
    avg_offset = sum(norm_offsets) / len(norm_offsets) if norm_offsets else 0.0
    max_offset = max(norm_offsets) if norm_offsets else 0.0
    min_offset = min(norm_offsets) if norm_offsets else 0.0
    
    # 두 번째 패스: 오버레이 이미지 생성 및 저장
    print(f"\n2단계: 차선 탐지된 이미지 오버레이 생성 및 원본 이미지 저장 중... ({len(images_with_lanes)}개)")
    for result in tqdm(images_with_lanes, desc="이미지 저장 중"):
        try:
            # 오버레이 이미지 생성
            overlay = analyzer.draw_overlay(
                result['image_rgb'],
                result['mask'],
                result['lane_center_x'],
                result['lane_width'],
                result['left_x'],
                result['right_x'],
                result['norm_offset'],
                result['departed'],
                result['left_departed'],
                result['right_departed'],
                left_departure_rate,
                right_departure_rate,
                total_departure_rate
            )
            
            # 오버레이 이미지 저장
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            overlay_path = os.path.join(detected_dir, f"overlay_{result['image_name']}")
            cv2.imwrite(overlay_path, overlay_bgr)
            
            # 원본 이미지 저장
            original_bgr = cv2.cvtColor(result['image_rgb'], cv2.COLOR_RGB2BGR)
            original_path = os.path.join(detected_dir, f"original_{result['image_name']}")
            cv2.imwrite(original_path, original_bgr)
        except Exception as e:
            print(f"\n이미지 저장 에러 ({result['image_name']}): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # CSV 저장
    csv_path = os.path.join(output_dir, 'lane_departure_results.csv')
    if results:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    # 결과 출력
    print("\n" + "=" * 70)
    print(" " * 20 + "차선 이탈률 분석 결과")
    print("=" * 70)
    print(f"  📊 총 분석 이미지:       {total_images}개")
    print(f"  🔍 차선 탐지된 이미지:   {total_with_lanes}개")
    print()
    print(f"  📈 전체 차선 이탈률:     {total_departure_rate:.2f}%")
    print(f"     - 차선 내 주행:       {total_images - departed_count}개 ({(total_images - departed_count)/total_images*100:.1f}%)")
    print(f"     - 차선 이탈:           {departed_count}개 ({departed_count/total_images*100:.1f}%)")
    print()
    print(f"  ⬅️  왼쪽 차선 이탈률:     {left_departure_rate:.2f}%")
    print(f"     - 왼쪽 이탈:           {left_departed_lane_count}개 / {total_with_lanes}개")
    print()
    print(f"  ➡️  오른쪽 차선 이탈률:   {right_departure_rate:.2f}%")
    print(f"     - 오른쪽 이탈:         {right_departed_lane_count}개 / {total_with_lanes}개")
    print()
    if norm_offsets:
        print(f"  📏 norm_offset 통계:")
        print(f"     - 평균: {avg_offset:.3f}")
        print(f"     - 최소: {min_offset:.3f}")
        print(f"     - 최대: {max_offset:.3f}")
    print()
    print(f"  📁 결과 파일:")
    print(f"     - CSV 결과:        {csv_path}")
    print(f"     - 원본 이미지:     {detected_dir}/original_*.jpg (총 {len(images_with_lanes)}개)")
    print(f"     - 오버레이 이미지: {detected_dir}/overlay_*.jpg (총 {len(images_with_lanes)}개)")
    print()
    print("  📌 주요 지표 설명:")
    print("     - 전체 차선 이탈률: norm_offset 기준 이탈 이미지 비율")
    print("     - 왼쪽/오른쪽 이탈률: 각 차선 기준 이탈 이미지 비율 (차선 탐지된 이미지만 대상)")
    print("     - norm_offset: 차선 중심과 차량 중심의 정규화된 거리")
    print("       * 0.0에 가까울수록 차선 중앙에 위치")
    print("       * 0.5 이상이면 이탈로 판단")
    print("     - 아래 30% 영역은 차선 탐지에서 제외됨 (차량 방지)")
    print("=" * 70)
    
    # 이미지별 상세 결과 출력 (처음 10개, 차선 탐지된 것만)
    detected_results = [r for r in results if r['lane_center_x']]
    print(f"\n차선 탐지된 이미지별 상세 결과 (처음 10개 / 총 {len(detected_results)}개):")
    print("-" * 90)
    print(f"{'이미지명':<35} {'norm_offset':<12} {'왼쪽':<8} {'오른쪽':<8} {'전체':<8}")
    print("-" * 90)
    for i, r in enumerate(detected_results[:10]):
        offset_str = r['norm_offset'] if r['norm_offset'] else "N/A"
        left_status = "이탈" if r['left_departed'] else "정상"
        right_status = "이탈" if r['right_departed'] else "정상"
        total_status = "이탈" if r['departed'] else "정상"
        print(f"{r['image_name']:<35} {offset_str:<12} {left_status:<8} {right_status:<8} {total_status:<8}")
    
    if len(detected_results) > 10:
        print(f"... 외 {len(detected_results) - 10}개 이미지 (전체 결과는 CSV 파일 참조)")


if __name__ == '__main__':
    mask_generation_cli()

