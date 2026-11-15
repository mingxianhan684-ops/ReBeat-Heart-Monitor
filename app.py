
# ============================================================
# ReBeat v2.0 - Two-Point PPG Monitoring Dashboard
# GitHub Codespaces + Gradio + Matplotlib
# Research demonstration only - NOT a medical device
# ============================================================

import os
import tempfile
from datetime import datetime, timezone

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

# ---------------- CONFIGURATION ----------------

FS = 125
DURATION = 300
DELAY_MS = 85
WINDOW = 8

rng = np.random.default_rng(2026)

# ---------------- GENERATE SIMULATED PPG ----------------

t = np.arange(0, DURATION, 1 / FS)
chest = np.zeros(len(t))

beat = 0.5

while beat < DURATION - 0.6:
    bpm = 76 + 5 * np.sin(2 * np.pi * beat / 60)
    interval = 60 / bpm

    x = t - beat

    chest += np.exp(
        -0.5 * ((x - 0.18) / 0.065) ** 2
    )

    chest += 0.20 * np.exp(
        -0.5 * ((x - 0.37) / 0.075) ** 2
    )

    beat += interval + rng.normal(0, 0.01)

chest += 0.018 * rng.normal(size=len(t))

wrist = np.interp(
    t - DELAY_MS / 1000,
    t,
    chest,
    left=0,
    right=0
)

wrist += 0.025 * rng.normal(size=len(t))

# ---------------- SIGNAL ANALYSIS ----------------

smooth = savgol_filter(chest, 15, 3)

d1 = np.gradient(smooth, 1 / FS)
d2 = np.gradient(d1, 1 / FS)

peaks, _ = find_peaks(
    smooth,
    distance=int(0.45 * FS),
    prominence=0.35
)

peak_times = t[peaks]
intervals = np.diff(peak_times)

valid = (intervals > 0.4) & (intervals < 1.8)

hr_times = peak_times[1:][valid]
heart_rates = 60 / intervals[valid]

# ---------------- PLOTTING ----------------

def style_plot(ax, title, ylabel):
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (seconds)")
    ax.grid(alpha=0.2)


def waveform_chart(position):
    end = max(WINDOW, float(position))
    start = end - WINDOW

    mask = (t >= start) & (t <= end)

    fig, ax = plt.subplots(figsize=(10, 3.5))

    ax.plot(
        t[mask],
        chest[mask],
        label="Chest PPG - DEMO",
        color="#12B886",
        linewidth=1.6
    )

    ax.plot(
        t[mask],
        wrist[mask],
        label="Wrist PPG - DEMO",
        color="#339AF0",
        linewidth=1.4
    )

    style_plot(
        ax,
        "Two-Point PPG Waveforms",
        "Normalized amplitude"
    )

    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    return fig


def derivative_chart(position):
    end = max(WINDOW, float(position))
    start = end - 3

    mask = (t >= start) & (t <= end)

    fig, axes = plt.subplots(
        3, 1,
        figsize=(10, 7),
        sharex=True
    )

    series = [
        (smooth, "PPG", "#12B886"),
        (d1, "First Derivative", "#F59F00"),
        (d2, "Second Derivative", "#845EF7")
    ]

    for ax, (values, title, color) in zip(axes, series):
        ax.plot(t[mask], values[mask], color=color)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Time (seconds)")
    fig.tight_layout()

    return fig


def trend_chart(position):
    mask = hr_times <= float(position)

    fig, ax = plt.subplots(figsize=(10, 3.5))

    ax.plot(
        hr_times[mask],
        heart_rates[mask],
        color="#FA5252",
        linewidth=1.6
    )

    style_plot(ax, "Heart Rate Trend", "BPM")
    ax.set_ylim(40, 130)
    fig.tight_layout()

    return fig


def delay_chart(position):
    # This is a simulated delay, not measured clinical PTT.
    mask = hr_times <= float(position)

    times = hr_times[mask]
    delays = DELAY_MS + 4 * np.sin(
        2 * np.pi * times / 50
    )

    fig, ax = plt.subplots(figsize=(10, 3.5))

    ax.plot(
        times,
        delays,
        color="#9775FA"
    )

    style_plot(
        ax,
        "Simulated Inter-Sensor Delay",
        "Delay (ms)"
    )

    ax.set_ylim(60, 110)
    fig.tight_layout()

    return fig


def distribution_chart(position):
    values = heart_rates[hr_times <= float(position)]

    fig, ax = plt.subplots(figsize=(10, 3.5))

    ax.hist(
        values,
        bins=15,
        color="#4DABF7",
        edgecolor="white"
    )

    ax.set_title("Heart Rate Distribution")
    ax.set_xlabel("BPM")
    ax.set_ylabel("Beat count")
    ax.grid(alpha=0.2)

    fig.tight_layout()

    return fig


# ---------------- DASHBOARD LOGIC ----------------

def render(position):
    position = max(WINDOW, min(DURATION, float(position)))

    mask = (
        (hr_times >= position - WINDOW)
        & (hr_times <= position)
    )

    recent = heart_rates[mask]

    if len(recent):
        bpm = float(np.mean(recent))
    else:
        bpm = float("nan")

    if len(recent) >= 2:
        recent_intervals = 60 / recent
        prv = float(
            np.std(recent_intervals, ddof=1) * 1000
        )
        prv_text = f"{prv:.1f} ms"
    else:
        prv_text = "Insufficient data"

    bpm_text = (
        f"{bpm:.1f} BPM"
        if np.isfinite(bpm)
        else "Insufficient data"
    )

    return (
        bpm_text,
        f"{DELAY_MS} ms - simulated",
        prv_text,
        "DEMO: Sensors not connected",
        waveform_chart(position),
        derivative_chart(position),
        trend_chart(position),
        delay_chart(position),
        distribution_chart(position),
        f"Simulation time: {position:.1f} / {DURATION} s"
    )


def start_monitor(position):
    return True, "RUNNING - SIMULATION"


def pause_monitor(position):
    return False, "PAUSED"


def reset_monitor():
    position = float(WINDOW)
    return (
        False,
        position,
        "READY - SIMULATION",
        *render(position)
    )


def advance(running, position):
    if not running:
        return (position, *render(position))

    position = min(DURATION, float(position) + 1)

    return (position, *render(position))


# ---------------- SMS DRY RUN ----------------

def test_sms():
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")

    return (
        f"[{timestamp}]\n"
        "SMS MODE: DRY RUN\n"
        "Recipient: Demo caregiver\n"
        "Message: ReBeat research test notification.\n"
        "No real SMS was sent.\n"
        "No emergency service was contacted."
    )


# ---------------- CSV EXPORT ----------------

def export_csv(position):
    mask = t <= float(position)

    df = pd.DataFrame({
        "time_s": t[mask],
        "chest_ppg_simulated": chest[mask],
        "wrist_ppg_simulated": wrist[mask]
    })

    path = os.path.join(
        tempfile.gettempdir(),
        "ReBeat_Demo_Export.csv"
    )

    df.to_csv(path, index=False)

    return path


# ---------------- USER INTERFACE ----------------

with gr.Blocks(title="ReBeat Heart Monitor") as demo:

    gr.Markdown("""
    # ❤️ ReBeat
    ### Two-Point Wearable PPG Monitoring System

    **RESEARCH DEMONSTRATION — NOT FOR MEDICAL USE**

    Chest sensor + Wrist sensor

    All signals are simulated. No real hardware is connected.
    """)

    running = gr.State(False)
    position = gr.State(float(WINDOW))

    with gr.Row():
        bpm_box = gr.Textbox(
            label="Heart Rate",
            interactive=False
        )

        delay_box = gr.Textbox(
            label="Pulse Delay",
            interactive=False
        )

        prv_box = gr.Textbox(
            label="Pulse Rate Variability",
            interactive=False
        )

    sensor_box = gr.Textbox(
        label="Sensor Status",
        interactive=False
    )

    status_box = gr.Textbox(
        value="READY - SIMULATION",
        label="Monitoring Status",
        interactive=False
    )

    with gr.Row():
        start_btn = gr.Button(
            "▶ Start",
            variant="primary"
        )

        pause_btn = gr.Button("⏸ Pause")
        reset_btn = gr.Button("↺ Reset")

    progress_box = gr.Textbox(
        label="Simulation Progress",
        interactive=False
    )

    gr.Markdown("## Live-Style PPG Monitoring")

    waveform_plot = gr.Plot()

    gr.Markdown("## Research Analysis")

    with gr.Tab("PPG Derivatives"):
        derivative_plot = gr.Plot()

    with gr.Tab("Heart Rate Trend"):
        hr_plot = gr.Plot()

    with gr.Tab("Pulse Delay"):
        delay_plot = gr.Plot()

    with gr.Tab("Distribution"):
        distribution_plot = gr.Plot()

    gr.Markdown("## Caregiver Notifications")

    gr.Markdown(
        "Test mode only. Real SMS and 911 calling are disabled."
    )

    sms_btn = gr.Button("Test Caregiver SMS")

    sms_log = gr.Textbox(
        label="Notification Log",
        lines=6
    )

    sms_btn.click(
        fn=test_sms,
        outputs=sms_log
    )

    gr.Markdown("## Export Research Data")

    export_btn = gr.Button("Download CSV Data")
    export_file = gr.File(label="Research Data")

    export_btn.click(
        fn=export_csv,
        inputs=position,
        outputs=export_file
    )

    chart_outputs = [
        bpm_box,
        delay_box,
        prv_box,
        sensor_box,
        waveform_plot,
        derivative_plot,
        hr_plot,
        delay_plot,
        distribution_plot,
        progress_box
    ]

    start_btn.click(
        fn=start_monitor,
        inputs=position,
        outputs=[running, status_box]
    )

    pause_btn.click(
        fn=pause_monitor,
        inputs=position,
        outputs=[running, status_box]
    )

    reset_btn.click(
        fn=reset_monitor,
        outputs=[
            running,
            position,
            status_box,
            *chart_outputs
        ]
    )

    # Automatically advance one simulated second
    # every second while the page is open.
    timer = gr.Timer(value=1.0)

    timer.tick(
        fn=advance,
        inputs=[running, position],
        outputs=[position, *chart_outputs],
        show_progress="hidden",
        concurrency_limit=1
    )

    demo.load(
        fn=lambda: render(WINDOW),
        outputs=chart_outputs
    )

    gr.Markdown("""
    ---
    **Safety:** This prototype does not detect, predict,
    or diagnose heart failure. Simulated measurements
    must not trigger medical decisions.

    For an actual emergency in the United States,
    call 911 manually.
    """)

# ---------------- START SERVER ----------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
