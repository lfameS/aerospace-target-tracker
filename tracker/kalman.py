from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class TargetTelemetry:
    """Kinematic telemetry extracted from the 8-state Constant-Acceleration Kalman Filter."""
    track_id: int
    label: str
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: Tuple[float, float]
    velocity: Tuple[float, float]            # (vx, vy) px/frame
    acceleration: Tuple[float, float]        # (ax, ay) px/frame^2
    speed_px_s: float
    heading_deg: float
    g_load_est: float
    confidence: float
    state_status: str                        # "CONFIRMED" | "COASTING" | "TENTATIVE"
    age: int
    hits: int
    time_since_update: int
    predicted_path: List[Tuple[float, float]] = field(default_factory=list)


class ConstantAccelerationKalmanFilter:
    """
    8-State Constant-Acceleration Kalman Filter for high-velocity aerial targets
    (UAVs, fixed-wing aircraft, cruise missiles, and ballistic rockets).

    State vector x (8x1):
        [cx, cy, vx, vy, ax, ay, w, h]^T
    Measurement vector z (4x1):
        [cx, cy, w, h]^T
    """

    def __init__(
        self,
        initial_bbox: Tuple[float, float, float, float],
        process_noise_scale: float = 1.0,
        measurement_noise_scale: float = 1.0,
        dt: float = 1.0,
    ) -> None:
        self.dt = float(dt)
        x1, y1, x2, y2 = initial_bbox
        w = max(4.0, float(x2 - x1))
        h = max(4.0, float(y2 - y1))
        cx = float(x1 + 0.5 * w)
        cy = float(y1 + 0.5 * h)

        # State vector: [cx, cy, vx, vy, ax, ay, w, h]
        self.x = np.array([cx, cy, 0.0, 0.0, 0.0, 0.0, w, h], dtype=np.float64)

        # State transition matrix F (Constant Acceleration kinematic model)
        half_dt2 = 0.5 * (self.dt ** 2)
        self.F = np.eye(8, dtype=np.float64)
        self.F[0, 2] = self.dt
        self.F[1, 3] = self.dt
        self.F[0, 4] = half_dt2
        self.F[1, 5] = half_dt2
        self.F[2, 4] = self.dt
        self.F[3, 5] = self.dt

        # Measurement matrix H (observes [cx, cy, w, h])
        self.H = np.zeros((4, 8), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 6] = 1.0
        self.H[3, 7] = 1.0

        # Initial state covariance P
        self.P = np.diag([
            10.0, 10.0,      # position
            100.0, 100.0,    # velocity (high initial uncertainty)
            25.0, 25.0,      # acceleration
            16.0, 16.0,      # size
        ]).astype(np.float64)

        # Process noise covariance Q
        q = max(1e-3, float(process_noise_scale))
        self.Q = np.diag([
            0.5 * q, 0.5 * q,
            2.0 * q, 2.0 * q,
            1.2 * q, 1.2 * q,
            0.2 * q, 0.2 * q,
        ]).astype(np.float64)

        # Measurement noise covariance R
        r = max(1e-3, float(measurement_noise_scale))
        self.R = np.diag([
            2.5 * r, 2.5 * r,
            4.0 * r, 4.0 * r,
        ]).astype(np.float64)

    def predict(self) -> np.ndarray:
        """Advance state by one time step dt using kinematic motion equations."""
        self.x = self.F @ self.x
        # Enforce damped acceleration during coasting to avoid runaway divergence
        self.x[4] *= 0.92
        self.x[5] *= 0.92
        self.x[6] = max(4.0, self.x[6])
        self.x[7] = max(4.0, self.x[7])
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """Correct state estimate with a bounding-box observation (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = bbox
        w = max(4.0, float(x2 - x1))
        h = max(4.0, float(y2 - y1))
        z = np.array([x1 + 0.5 * w, y1 + 0.5 * h, w, h], dtype=np.float64)

        y_innov = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y_innov)
        self.x[6] = max(4.0, self.x[6])
        self.x[7] = max(4.0, self.x[7])

        I = np.eye(8, dtype=np.float64)
        I_KH = I - (K @ self.H)
        # Joseph stabilized covariance update
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.x.copy()

    def mahalanobis_distance(self, bbox: Tuple[float, float, float, float]) -> float:
        """Compute squared Mahalanobis distance between measurement and predicted state."""
        x1, y1, x2, y2 = bbox
        w = max(4.0, float(x2 - x1))
        h = max(4.0, float(y2 - y1))
        z = np.array([x1 + 0.5 * w, y1 + 0.5 * h, w, h], dtype=np.float64)
        y_innov = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        try:
            inv_S = np.linalg.inv(S)
            dist2 = float(y_innov.T @ inv_S @ y_innov)
            return max(0.0, math.sqrt(max(0.0, dist2)))
        except np.linalg.LinAlgError:
            return float(np.linalg.norm(y_innov[:2]))

    def predict_trajectory(self, horizon_steps: int = 25) -> List[Tuple[float, float]]:
        """
        Extrapolate future target trajectory over `horizon_steps` frames
        without mutating the filter state.
        """
        sim_x = self.x.copy()
        points: List[Tuple[float, float]] = []
        for _ in range(max(1, int(horizon_steps))):
            sim_x = self.F @ sim_x
            sim_x[4] *= 0.95
            sim_x[5] *= 0.95
            points.append((float(sim_x[0]), float(sim_x[1])))
        return points

    @property
    def current_bbox(self) -> Tuple[float, float, float, float]:
        cx, cy = float(self.x[0]), float(self.x[1])
        w, h = max(4.0, float(self.x[6])), max(4.0, float(self.x[7]))
        return (cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h)
