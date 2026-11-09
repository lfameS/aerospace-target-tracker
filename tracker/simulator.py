from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .bytetrack import ByteTracker, Detection, TargetTelemetry


@dataclass
class SimTargetSpec:
    label: str
    x0: float
    y0: float
    vx: float
    vy: float
    ax: float
    ay: float
    w: float
    h: float
    turn_omega: float = 0.0
    occlusion_frames: Tuple[int, int] = (-1, -1)


SCENARIOS: Dict[str, List[SimTargetSpec]] = {
    "ballistic_intercept": [
        SimTargetSpec(
            label="BALLISTIC_ROCKET",
            x0=95.0, y0=490.0, vx=9.2, vy=-7.8, ax=0.14, ay=0.09,
            w=24.0, h=14.0, turn_omega=0.0, occlusion_frames=(22, 31)
        ),
        SimTargetSpec(
            label="FIXED_WING_JET",
            x0=840.0, y0=150.0, vx=-7.4, vy=1.8, ax=-0.05, ay=0.08,
            w=38.0, h=20.0, turn_omega=0.025, occlusion_frames=(26, 33)
        ),
        SimTargetSpec(
            label="LOITER_UAV",
            x0=420.0, y0=280.0, vx=3.6, vy=2.4, ax=0.0, ay=0.0,
            w=22.0, h=18.0, turn_omega=0.06, occlusion_frames=(-1, -1)
        ),
    ],
    "high_g_evasive_fighter": [
        SimTargetSpec(
            label="SUPERSONIC_FIGHTER",
            x0=120.0, y0=220.0, vx=10.5, vy=1.2, ax=0.08, ay=0.22,
            w=40.0, h=22.0, turn_omega=0.055, occlusion_frames=(18, 25)
        ),
        SimTargetSpec(
            label="CRUISE_TARGET",
            x0=880.0, y0=410.0, vx=-9.8, vy=-2.1, ax=-0.06, ay=-0.02,
            w=26.0, h=12.0, turn_omega=0.0, occlusion_frames=(28, 35)
        ),
    ],
    "uav_swarm_occlusion": [
        SimTargetSpec(
            label="ROTARY_UAV_01",
            x0=180.0, y0=160.0, vx=5.5, vy=2.8, ax=0.02, ay=0.01,
            w=20.0, h=18.0, turn_omega=0.03, occlusion_frames=(20, 29)
        ),
        SimTargetSpec(
            label="ROTARY_UAV_02",
            x0=150.0, y0=210.0, vx=5.8, vy=2.5, ax=0.02, ay=0.02,
            w=20.0, h=18.0, turn_omega=0.03, occlusion_frames=(22, 31)
        ),
        SimTargetSpec(
            label="ROTARY_UAV_03",
            x0=210.0, y0=250.0, vx=5.3, vy=3.0, ax=0.01, ay=-0.02,
            w=20.0, h=18.0, turn_omega=-0.025, occlusion_frames=(24, 32)
        ),
        SimTargetSpec(
            label="BALLISTIC_ROCKET",
            x0=860.0, y0=500.0, vx=-10.2, vy=-6.8, ax=-0.12, ay=0.06,
            w=24.0, h=14.0, turn_omega=0.0, occlusion_frames=(-1, -1)
        ),
    ],
}


def step_scenario_detections(
    specs: List[SimTargetSpec],
    frame_idx: int,
    rng: np.random.Generator,
    noise_px: float = 1.8,
) -> Tuple[List[Detection], List[Tuple[float, float]]]:
    """Generate noisy sensor detections and ground-truth centers at `frame_idx`."""
    dets: List[Detection] = []
    gt_centers: List[Tuple[float, float]] = []

    t = float(frame_idx)
    for spec in specs:
        if spec.turn_omega != 0.0:
            angle = spec.turn_omega * t
            cx = spec.x0 + spec.vx * t + 38.0 * math.sin(angle)
            cy = spec.y0 + spec.vy * t + 28.0 * math.cos(angle)
        else:
            cx = spec.x0 + spec.vx * t + 0.5 * spec.ax * (t ** 2)
            cy = spec.y0 + spec.vy * t + 0.5 * spec.ay * (t ** 2)

        gt_centers.append((cx, cy))

        occ_start, occ_end = spec.occlusion_frames
        if occ_start <= frame_idx <= occ_end:
            # During partial cloud/thermal occlusion, emit either a low-conf detection or complete drop
            if frame_idx % 2 == 0:
                nx = cx + float(rng.normal(0.0, noise_px * 2.2))
                ny = cy + float(rng.normal(0.0, noise_px * 2.2))
                dets.append(
                    Detection(
                        bbox=(nx - spec.w / 2, ny - spec.h / 2, nx + spec.w / 2, ny + spec.h / 2),
                        confidence=float(rng.uniform(0.18, 0.42)),
                        label=spec.label,
                    )
                )
            continue

        nx = cx + float(rng.normal(0.0, noise_px))
        ny = cy + float(rng.normal(0.0, noise_px))
        conf = float(rng.uniform(0.76, 0.97))
        dets.append(
            Detection(
                bbox=(nx - spec.w / 2, ny - spec.h / 2, nx + spec.w / 2, ny + spec.h / 2),
                confidence=conf,
                label=spec.label,
            )
        )

    return dets, gt_centers


def render_tactical_hud(
    tracker: ByteTracker,
    telemetry: List[TargetTelemetry],
    scenario_name: str,
    frame_idx: int,
    width: int = 960,
    height: int = 560,
) -> Image.Image:
    """Render an Electro-Optical / Thermal IR Defense Tactical HUD frame."""
    img = Image.new("RGB", (width, height), (7, 13, 22))
    draw = ImageDraw.Draw(img, "RGBA")

    # 1. Tactical radar / EO-IR grid overlay
    for x in range(0, width, 60):
        draw.line([(x, 0), (x, height)], fill=(18, 38, 58, 180), width=1)
    for y in range(0, height, 60):
        draw.line([(0, y), (width, y)], fill=(18, 38, 58, 180), width=1)

    # Center boresight crosshair & range rings
    cx_mid, cy_mid = width // 2, height // 2
    for r in (90, 180, 260):
        draw.ellipse(
            [cx_mid - r, cy_mid - r, cx_mid + r, cy_mid + r],
            outline=(24, 62, 88, 140),
            width=1,
        )
    draw.line([(cx_mid - 25, cy_mid), (cx_mid + 25, cy_mid)], fill=(0, 210, 255, 130), width=1)
    draw.line([(cx_mid, cy_mid - 25), (cx_mid, cy_mid + 25)], fill=(0, 210, 255, 130), width=1)

    # Simulated cloud / thermal occlusion zone indicator
    draw.rectangle(
        [310, 170, 560, 360],
        fill=(30, 50, 75, 48),
        outline=(90, 145, 195, 110),
        width=1,
    )
    draw.text((318, 176), "CLOUD / IR OCCLUSION SECTOR", fill=(120, 175, 220, 200))

    # 2. Draw track histories, bounding boxes, and Kalman predicted trajectories
    track_map = {t.track_id: t for t in tracker.tracks}
    for item in telemetry:
        trk = track_map.get(item.track_id)
        is_rocket = "ROCKET" in item.label or "CRUISE" in item.label
        is_coasting = item.state_status == "COASTING"

        if is_coasting:
            color = (255, 190, 40, 255)
        elif is_rocket:
            color = (255, 72, 88, 255)
        else:
            color = (0, 240, 180, 255)

        # Past trajectory trail
        if trk and len(trk.history) > 1:
            for i in range(1, len(trk.history)):
                alpha = int(50 + 180 * (i / len(trk.history)))
                p1 = trk.history[i - 1]
                p2 = trk.history[i]
                draw.line([p1, p2], fill=(color[0], color[1], color[2], alpha), width=2)

        # Future Kalman predicted trajectory (dashed lookahead + uncertainty cone)
        if item.predicted_path:
            prev = item.center
            for step_i, pt in enumerate(item.predicted_path):
                if step_i % 2 == 0:
                    draw.line([prev, pt], fill=(0, 215, 255, 220), width=2)
                prev = pt
            end_x, end_y = item.predicted_path[-1]
            r_cep = 6 + int(len(item.predicted_path) * 0.35)
            draw.ellipse(
                [end_x - r_cep, end_y - r_cep, end_x + r_cep, end_y + r_cep],
                outline=(0, 215, 255, 180),
                width=1,
            )

        # Bounding box with corner brackets
        x1, y1, x2, y2 = item.bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        tag = (
            f"ID:{item.track_id:02d} {item.label} [{item.state_status}] "
            f"v={item.speed_px_s:.0f}px/s {item.g_load_est:.1f}G"
        )
        draw.rectangle([x1, y1 - 18, x1 + len(tag) * 6 + 8, y1], fill=(8, 16, 28, 220))
        draw.text((x1 + 4, y1 - 15), tag, fill=color)

    # 3. Top & Bottom Tactical Telemetry Bar
    draw.rectangle([0, 0, width, 34], fill=(5, 10, 18, 235))
    header = (
        f"AEROSPACE & DEFENSE EO/IR TRACKER  |  SCENARIO: {scenario_name.upper()}  |  "
        f"FRAME: {frame_idx:03d}  |  ACTIVE TRACKS: {len(telemetry)}  |  FILTER: 8-DOF EKF + BYTETRACK"
    )
    draw.text((14, 10), header, fill=(0, 230, 200, 255))

    return img


def run_simulation(
    scenario_name: str = "ballistic_intercept",
    frames: int = 45,
    process_noise: float = 1.0,
    measurement_noise: float = 1.0,
    high_thresh: float = 0.55,
    low_thresh: float = 0.15,
    prediction_horizon: int = 25,
    seed: int = 42,
) -> Tuple[Image.Image, List[TargetTelemetry]]:
    specs = SCENARIOS.get(scenario_name, SCENARIOS["ballistic_intercept"])
    rng = np.random.default_rng(seed)
    tracker = ByteTracker(
        high_thresh=high_thresh,
        low_thresh=low_thresh,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        prediction_horizon=prediction_horizon,
    )

    last_telemetry: List[TargetTelemetry] = []
    for f_idx in range(1, max(2, int(frames)) + 1):
        dets, _ = step_scenario_detections(specs, f_idx, rng=rng)
        last_telemetry = tracker.update(dets)

    hud_img = render_tactical_hud(
        tracker=tracker,
        telemetry=last_telemetry,
        scenario_name=scenario_name,
        frame_idx=int(frames),
    )
    return hud_img, last_telemetry
