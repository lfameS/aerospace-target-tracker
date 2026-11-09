from __future__ import annotations

import json
import gradio as gr
from tracker.simulator import SCENARIOS, run_simulation


def run_demo(
    scenario_name: str,
    frames: int,
    process_noise: float,
    measurement_noise: float,
    high_thresh: float,
    low_thresh: float,
    prediction_horizon: int,
):
    hud_img, telemetry = run_simulation(
        scenario_name=scenario_name,
        frames=int(frames),
        process_noise=float(process_noise),
        measurement_noise=float(measurement_noise),
        high_thresh=float(high_thresh),
        low_thresh=float(low_thresh),
        prediction_horizon=int(prediction_horizon),
    )

    rows = []
    json_payload = []
    for t in telemetry:
        intercept_pt = t.predicted_path[-1] if t.predicted_path else t.center
        rows.append([
            f"TRK-{t.track_id:02d}",
            t.label,
            t.state_status,
            f"{t.speed_px_s:.1f} px/s",
            f"{t.heading_deg:.1f}°",
            f"{t.g_load_est:.2f} G",
            f"{t.confidence:.2f}",
            f"({intercept_pt[0]:.1f}, {intercept_pt[1]:.1f})",
        ])
        json_payload.append({
            "track_id": t.track_id,
            "class": t.label,
            "status": t.state_status,
            "center": [round(t.center[0], 1), round(t.center[1], 1)],
            "velocity_px_frame": list(t.velocity),
            "speed_px_s": t.speed_px_s,
            "heading_deg": t.heading_deg,
            "g_load_est": t.g_load_est,
            "confidence": t.confidence,
            "predicted_intercept_xy": [round(intercept_pt[0], 1), round(intercept_pt[1], 1)],
        })

    return hud_img, rows, json.dumps(json_payload, indent=2)


with gr.Blocks(title="Aerospace & Defense Target Tracker (8-DOF EKF + ByteTrack)") as demo:
    gr.Markdown(
        """
        # Aerospace & Defense Multi-Object Target Tracker
        **Real-Time High-Velocity Aerial Target Detection, Occlusion Recovery & Kinematic Trajectory Prediction**
        *8-State Constant-Acceleration Kalman Filter + Two-Stage ByteTrack + Mahalanobis Gating for Aircraft, UAV Swarms & Ballistic Rockets.*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            scenario = gr.Dropdown(
                choices=list(SCENARIOS.keys()),
                value="ballistic_intercept",
                label="Tactical Engagement Scenario",
            )
            frames = gr.Slider(10, 65, value=38, step=1, label="Simulation Frame Index")
            prediction_horizon = gr.Slider(
                10, 45, value=25, step=1, label="Kalman Trajectory Lookahead (Frames)"
            )
            process_noise = gr.Slider(
                0.2, 3.0, value=1.0, step=0.1, label="Kalman Process Noise Scale (Q)"
            )
            measurement_noise = gr.Slider(
                0.2, 3.0, value=1.0, step=0.1, label="Sensor Measurement Noise Scale (R)"
            )
            high_thresh = gr.Slider(
                0.35, 0.85, value=0.55, step=0.05, label="ByteTrack Stage-1 High Conf Threshold"
            )
            low_thresh = gr.Slider(
                0.05, 0.35, value=0.15, step=0.05, label="ByteTrack Stage-2 Occlusion Recovery Threshold"
            )
            run_btn = gr.Button("Run Tactical Tracker & Predict Trajectories", variant="primary")

        with gr.Column(scale=2):
            hud_output = gr.Image(type="pil", label="Electro-Optical / Thermal IR Tactical HUD")
            table_output = gr.Dataframe(
                headers=[
                    "Track ID",
                    "Target Class",
                    "Track State",
                    "Velocity",
                    "Heading",
                    "Est. G-Load",
                    "Confidence",
                    "Predicted Intercept (X, Y)",
                ],
                label="Real-Time Kinematic Telemetry",
            )
            json_output = gr.Code(language="json", label="Raw Fire-Control / Telemetry JSON Stream")

    run_btn.click(
        fn=run_demo,
        inputs=[
            scenario,
            frames,
            process_noise,
            measurement_noise,
            high_thresh,
            low_thresh,
            prediction_horizon,
        ],
        outputs=[hud_output, table_output, json_output],
    )
    demo.load(
        fn=run_demo,
        inputs=[
            scenario,
            frames,
            process_noise,
            measurement_noise,
            high_thresh,
            low_thresh,
            prediction_horizon,
        ],
        outputs=[hud_output, table_output, json_output],
    )


if __name__ == "__main__":
    demo.launch()
