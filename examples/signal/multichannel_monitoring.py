"""Multi-sensor online health monitoring with alignment and event lifecycle."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mastermlx.signal import MultiChannelSignalMonitor


rng = np.random.default_rng(7)
duration = 14.0
motor_rate = 200.0
housing_rate = 100.0
target_rate = 100.0
motor_time = np.arange(int(duration * motor_rate)) / motor_rate
housing_time = np.arange(int(duration * housing_rate)) / housing_rate
motor = np.sin(2.0 * np.pi * 18.0 * motor_time) + 0.04 * rng.normal(size=motor_time.size)
housing = 0.5 * np.sin(2.0 * np.pi * 18.0 * housing_time + 0.2) + 0.03 * rng.normal(size=housing_time.size)
fault_motor = motor_time >= 8.0
fault_housing = housing_time >= 8.0
motor[fault_motor] += 1.0 * np.sin(2.0 * np.pi * 43.0 * motor_time[fault_motor])
housing[fault_housing] += 0.65 * np.sin(2.0 * np.pi * 43.0 * housing_time[fault_housing])

monitor = MultiChannelSignalMonitor(
    ["motor", "housing"],
    sample_rate=target_rate,
    source_sample_rates={"motor": motor_rate, "housing": housing_rate},
    frame_length=200,
    hop_length=100,
    baseline_frames=5,
    confirmation_frames=2,
    recovery_frames=3,
    adaptation_rate=0.0,
    warning_threshold=75.0,
    critical_threshold=45.0,
)

health_times = []
health_scores = []
statuses = []
events = []
for second in range(int(duration)):
    motor_chunk = motor[int(second * motor_rate) : int((second + 1) * motor_rate)]
    housing_chunk = housing[int(second * housing_rate) : int((second + 1) * housing_rate)]
    result = monitor.push({"motor": motor_chunk, "housing": housing_chunk})
    health_times.extend(result["timestamps"])
    health_scores.extend(result["health_scores"])
    statuses.extend(result["statuses"])
    events.extend(result["events"])

print("final_state:", monitor.state())
for event in events:
    print("event:", dict(event))

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
axes[0].plot(motor_time, motor, label="motor accelerometer", linewidth=0.8)
axes[0].plot(housing_time, housing, label="housing accelerometer", linewidth=0.8, alpha=0.8)
axes[0].axvspan(8.0, duration, color="tab:red", alpha=0.1, label="injected fault")
axes[0].set_ylabel("amplitude")
axes[0].set_title("Aligned multi-channel condition monitoring")
axes[0].legend(loc="upper right", ncol=3)

health_times = np.asarray(health_times)
health_scores = np.asarray(health_scores)
axes[1].plot(health_times, health_scores, marker="o", markersize=3, label="fused health score")
axes[1].axhline(75.0, color="tab:orange", linestyle="--", label="warning threshold")
axes[1].axhline(45.0, color="tab:red", linestyle="--", label="critical threshold")
for event in events:
    color = "tab:green" if event["type"] == "recovered" else "tab:red"
    axes[1].axvline(event["timestamp"], color=color, alpha=0.65)
axes[1].set_ylim(-5, 105)
axes[1].set_ylabel("health score")
axes[1].legend(loc="upper right")

status_level = {"initializing": 0, "healthy": 1, "warning": 2, "critical": 3}
axes[2].step(health_times, [status_level[item] for item in statuses], where="post", color="tab:purple")
axes[2].set_yticks(list(status_level.values()), list(status_level))
axes[2].set_xlabel("time (s)")
axes[2].set_ylabel("lifecycle state")

output = Path(__file__).resolve().parents[1] / "outputs" / "signal" / "multichannel_monitoring.png"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=160)
print("plot:", output)
