from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tracker.simulator import SCENARIOS, run_simulation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aerospace & Defense Multi-Object Target Tracker CLI (8-DOF Kalman + ByteTrack)"
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default="ballistic_intercept",
        help="Tactical simulation scenario to execute.",
    )
    parser.add_argument("--frames", type=int, default=38, help="Number of frames to simulate.")
    parser.add_argument("--horizon", type=int, default=25, help="Trajectory prediction lookahead steps.")
    parser.add_argument("--render", type=str, default="assets/hud_preview.png", help="Output HUD PNG path.")
    parser.add_argument("--out", type=str, default="", help="Optional output telemetry JSON path.")
    args = parser.parse_args()

    hud_img, telemetry = run_simulation(
        scenario_name=args.scenario,
        frames=args.frames,
        prediction_horizon=args.horizon,
    )

    if args.render:
        out_img = Path(args.render)
        out_img.parent.mkdir(parents=True, exist_ok=True)
        hud_img.save(out_img)
        print(f"[OK] Saved Tactical EO/IR HUD frame -> {out_img}")

    payload = [asdict(t) for t in telemetry]
    if args.out:
        out_json = Path(args.out)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[OK] Saved telemetry JSON -> {out_json}")

    for t in telemetry:
        print(
            f"  Track #{t.track_id:02d} | {t.label:<18} | {t.state_status:<9} | "
            f"v={t.speed_px_s:6.1f} px/s | hdg={t.heading_deg:5.1f} deg | {t.g_load_est:4.2f}G"
        )


if __name__ == "__main__":
    main()
