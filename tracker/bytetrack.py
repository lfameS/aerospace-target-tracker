from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from .kalman import ConstantAccelerationKalmanFilter, TargetTelemetry


@dataclass
class Detection:
    """Single-frame detector output (YOLO / RT-DETR / Electro-Optical sensor)."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    label: str = "AERIAL_TARGET"


def compute_iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    area_a = max(1e-6, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1e-6, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return float(inter / (area_a + area_b - inter))


@dataclass
class Track:
    track_id: int
    kf: ConstantAccelerationKalmanFilter
    label: str
    confidence: float
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    history: List[Tuple[float, float]] = field(default_factory=list)

    def register_point(self, max_history: int = 45) -> None:
        cx, cy = float(self.kf.x[0]), float(self.kf.x[1])
        self.history.append((cx, cy))
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]


class ByteTracker:
    """
    Two-Stage ByteTrack + Mahalanobis Kinematic Association Engine
    tailored for high-speed aerospace & defense targets.

    Stage 1: Associates high-confidence detections (`conf >= high_thresh`)
             with active and coasting tracks using hybrid IoU + Mahalanobis gating.
    Stage 2: Associates remaining low-confidence detections (`low_thresh <= conf < high_thresh`)
             with unmatched tracks to preserve Track IDs through cloud occlusion,
             plume interference, or long-range small-pixel degradation.
    """

    def __init__(
        self,
        high_thresh: float = 0.55,
        low_thresh: float = 0.15,
        max_time_lost: int = 18,
        min_hits: int = 2,
        process_noise: float = 1.0,
        measurement_noise: float = 1.0,
        fps: float = 30.0,
        prediction_horizon: int = 25,
    ) -> None:
        self.high_thresh = float(high_thresh)
        self.low_thresh = float(low_thresh)
        self.max_time_lost = int(max_time_lost)
        self.min_hits = int(min_hits)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.fps = max(1.0, float(fps))
        self.prediction_horizon = int(prediction_horizon)

        self._next_id: int = 1
        self.tracks: List[Track] = []

    def _associate(
        self,
        tracks: List[Track],
        detections: List[Detection],
        max_cost: float = 0.82,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        cost_matrix = np.ones((len(tracks), len(detections)), dtype=np.float64) * 1e6
        for t_idx, trk in enumerate(tracks):
            pred_box = trk.kf.current_bbox
            for d_idx, det in enumerate(detections):
                iou = compute_iou(pred_box, det.bbox)
                maha = trk.kf.mahalanobis_distance(det.bbox)
                # Normalized kinematic distance gate (critical for fast rockets/jets where boxes don't overlap)
                maha_norm = min(1.0, maha / 45.0)
                cost = 0.45 * (1.0 - iou) + 0.55 * maha_norm
                cost_matrix[t_idx, d_idx] = cost

        matches: List[Tuple[int, int]] = []
        used_tracks = set()
        used_dets = set()

        # Greedy minimum-cost bipartite matching
        flat_indices = np.argsort(cost_matrix, axis=None)
        for flat_idx in flat_indices:
            t_idx = int(flat_idx // len(detections))
            d_idx = int(flat_idx % len(detections))
            if t_idx in used_tracks or d_idx in used_dets:
                continue
            if cost_matrix[t_idx, d_idx] <= max_cost:
                matches.append((t_idx, d_idx))
                used_tracks.add(t_idx)
                used_dets.add(d_idx)

        unmatched_tracks = [i for i in range(len(tracks)) if i not in used_tracks]
        unmatched_dets = [j for j in range(len(detections)) if j not in used_dets]
        return matches, unmatched_tracks, unmatched_dets

    def update(self, detections: List[Detection]) -> List[TargetTelemetry]:
        # 1. Predict all existing tracks forward
        for trk in self.tracks:
            trk.kf.predict()
            trk.age += 1
            trk.time_since_update += 1

        # 2. Split detections into high and low confidence pools
        high_dets = [d for d in detections if d.confidence >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]

        # 3. Stage 1 association (High confidence detections)
        matches_1, unmatched_trk_idx, unmatched_high_idx = self._associate(
            self.tracks, high_dets, max_cost=0.82
        )
        for t_idx, d_idx in matches_1:
            trk = self.tracks[t_idx]
            det = high_dets[d_idx]
            trk.kf.update(det.bbox)
            trk.confidence = det.confidence
            trk.label = det.label
            trk.hits += 1
            trk.time_since_update = 0
            trk.register_point()

        # 4. Stage 2 association (Low confidence detections -> recover occluded targets)
        remaining_tracks = [self.tracks[i] for i in unmatched_trk_idx]
        matches_2, still_unmatched_sub, _ = self._associate(
            remaining_tracks, low_dets, max_cost=0.78
        )
        for sub_t_idx, d_idx in matches_2:
            trk = remaining_tracks[sub_t_idx]
            det = low_dets[d_idx]
            trk.kf.update(det.bbox)
            trk.confidence = det.confidence
            trk.hits += 1
            trk.time_since_update = 0
            trk.register_point()

        # Coasting tracks still record their Kalman-extrapolated center
        for sub_t_idx in still_unmatched_sub:
            trk = remaining_tracks[sub_t_idx]
            trk.register_point()

        # 5. Spawn new tracks from unmatched high-confidence detections
        for d_idx in unmatched_high_idx:
            det = high_dets[d_idx]
            kf = ConstantAccelerationKalmanFilter(
                det.bbox,
                process_noise_scale=self.process_noise,
                measurement_noise_scale=self.measurement_noise,
            )
            new_trk = Track(
                track_id=self._next_id,
                kf=kf,
                label=det.label,
                confidence=det.confidence,
            )
            new_trk.register_point()
            self._next_id += 1
            self.tracks.append(new_trk)

        # 6. Prune expired tracks
        self.tracks = [
            t for t in self.tracks if t.time_since_update <= self.max_time_lost
        ]

        # 7. Build telemetry output
        telemetry_list: List[TargetTelemetry] = []
        for trk in self.tracks:
            if trk.hits < self.min_hits and trk.age > 2:
                continue

            if trk.time_since_update == 0 and trk.hits >= self.min_hits:
                status = "CONFIRMED"
            elif trk.time_since_update > 0:
                status = "COASTING"
            else:
                status = "TENTATIVE"

            vx, vy = float(trk.kf.x[2]), float(trk.kf.x[3])
            ax, ay = float(trk.kf.x[4]), float(trk.kf.x[5])
            speed_px_s = math.hypot(vx, vy) * self.fps
            heading_deg = (math.degrees(math.atan2(vy, vx)) + 360.0) % 360.0
            accel_mag = math.hypot(ax, ay) * (self.fps ** 2)
            g_load = round(min(18.0, 1.0 + accel_mag / 250.0), 2)

            telemetry_list.append(
                TargetTelemetry(
                    track_id=trk.track_id,
                    label=trk.label,
                    bbox=trk.kf.current_bbox,
                    center=(float(trk.kf.x[0]), float(trk.kf.x[1])),
                    velocity=(round(vx, 2), round(vy, 2)),
                    acceleration=(round(ax, 3), round(ay, 3)),
                    speed_px_s=round(speed_px_s, 1),
                    heading_deg=round(heading_deg, 1),
                    g_load_est=g_load,
                    confidence=round(float(trk.confidence), 3),
                    state_status=status,
                    age=trk.age,
                    hits=trk.hits,
                    time_since_update=trk.time_since_update,
                    predicted_path=trk.kf.predict_trajectory(self.prediction_horizon),
                )
            )
        return telemetry_list
