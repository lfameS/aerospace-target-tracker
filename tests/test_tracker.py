from __future__ import annotations

import math
from tracker.kalman import ConstantAccelerationKalmanFilter
from tracker.bytetrack import ByteTracker, Detection, compute_iou
from tracker.simulator import run_simulation


def test_kalman_constant_acceleration_convergence() -> None:
    kf = ConstantAccelerationKalmanFilter((100.0, 100.0, 120.0, 114.0))
    # Simulate an accelerating rocket target
    x, y = 110.0, 107.0
    vx, vy = 8.0, -5.0
    ax, ay = 0.4, -0.2
    for step in range(1, 20):
        kf.predict()
        cx = x + vx * step + 0.5 * ax * (step ** 2)
        cy = y + vy * step + 0.5 * ay * (step ** 2)
        kf.update((cx - 10.0, cy - 7.0, cx + 10.0, cy + 7.0))

    traj = kf.predict_trajectory(horizon_steps=10)
    assert len(traj) == 10
    # Verify predicted trajectory continues moving right (+x) and up (-y)
    assert traj[-1][0] > kf.x[0]
    assert traj[-1][1] < kf.x[1]


def test_bytetrack_preserves_id_across_cloud_occlusion() -> None:
    tracker = ByteTracker(high_thresh=0.55, low_thresh=0.15, max_time_lost=15, min_hits=2)

    assigned_id = None
    for frame in range(1, 30):
        cx = 80.0 + 7.5 * frame
        cy = 400.0 - 4.2 * frame
        bbox = (cx - 12.0, cy - 8.0, cx + 12.0, cy + 8.0)

        if 12 <= frame <= 16:
            # Stage-2 low-confidence detection (partial cloud occlusion)
            dets = [Detection(bbox=bbox, confidence=0.24, label="BALLISTIC_ROCKET")]
        elif 17 <= frame <= 19:
            # Total sensor blackout (coasting purely on 8-DOF Kalman Filter)
            dets = []
        else:
            dets = [Detection(bbox=bbox, confidence=0.91, label="BALLISTIC_ROCKET")]

        telem = tracker.update(dets)
        if frame >= 3:
            assert len(telem) == 1
            if assigned_id is None:
                assigned_id = telem[0].track_id
            else:
                # Zero ID switches across low-conf + full blackout frames!
                assert telem[0].track_id == assigned_id


def test_tactical_simulation_and_hud_render() -> None:
    img, telemetry = run_simulation(scenario_name="ballistic_intercept", frames=35)
    assert img.size == (960, 560)
    assert len(telemetry) >= 3
    labels = {t.label for t in telemetry}
    assert "BALLISTIC_ROCKET" in labels
