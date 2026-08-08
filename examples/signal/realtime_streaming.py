"""Real-time IIR + STFT processing with explicit packet-loss semantics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mastermlx.signal import StreamingIIRFilter, StreamingSTFT, butterworth


rng = np.random.default_rng(11)
sample_rate = 500.0
duration = 10.0
time = np.arange(int(sample_rate * duration)) / sample_rate
raw = np.sin(2.0 * np.pi * 18.0 * time) + 0.45 * np.sin(2.0 * np.pi * 120.0 * time)
raw += 0.08 * rng.normal(size=time.size)

b, a = butterworth(order=4, cutoff=35.0, sample_rate=sample_rate, btype="lowpass")
filter_stream = StreamingIIRFilter(b, a, sample_rate=sample_rate)
stft_stream = StreamingSTFT(
    frame_length=128,
    hop_length=64,
    n_fft=256,
    sample_rate=sample_rate,
    pad_end=True,
)

gap_start = int(5.0 * sample_rate)
gap_length = int(0.3 * sample_rate)
segments = [(0, gap_start, 0), (gap_start + gap_length, raw.size, gap_length)]
filtered_time = []
filtered_values = []
spectra = []
frame_times = []
for start, stop, gap_samples in segments:
    cursor = start
    first_chunk = True
    while cursor < stop:
        next_cursor = min(cursor + 173, stop)
        gap = gap_samples if first_chunk else 0
        filtered = filter_stream.push(raw[cursor:next_cursor], gap_samples=gap)
        result = stft_stream.push(filtered, gap_samples=gap)
        sample_indices = np.arange(cursor, next_cursor)
        filtered_time.append(sample_indices / sample_rate)
        filtered_values.append(filtered)
        if result["spectrogram"].size:
            spectra.append(result["spectrogram"])
            frame_times.append(result["frame_end_times"])
        cursor = next_cursor
        first_chunk = False

tail = stft_stream.flush()
if tail["spectrogram"].size:
    spectra.append(tail["spectrogram"])
    frame_times.append(tail["frame_end_times"])

filtered_time = np.concatenate(filtered_time)
filtered_values = np.concatenate(filtered_values)
spectrogram = np.vstack(spectra)
frame_times = np.concatenate(frame_times)
frequencies = np.fft.rfftfreq(stft_stream.n_fft, d=1.0 / sample_rate)

print("filter_state:", filter_stream.state())
print("stft_state_after_flush:", stft_stream.state())

fig, axes = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True, sharex=True)
axes[0].plot(time, raw, linewidth=0.65, color="0.45", label="raw sensor signal")
axes[0].plot(filtered_time, filtered_values, linewidth=1.0, color="tab:blue", label="causal IIR output")
axes[0].axvspan(gap_start / sample_rate, (gap_start + gap_length) / sample_rate, color="tab:red", alpha=0.16)
axes[0].set_title("Streaming IIR and STFT do not bridge declared packet loss")
axes[0].set_ylabel("amplitude")
axes[0].legend(loc="upper right")

axes[1].plot(frame_times, np.max(np.abs(spectrogram), axis=1), marker="o", markersize=2)
axes[1].axvspan(gap_start / sample_rate, (gap_start + gap_length) / sample_rate, color="tab:red", alpha=0.16)
axes[1].set_ylabel("max STFT magnitude")

image = axes[2].pcolormesh(frame_times, frequencies, 20.0 * np.log10(np.maximum(np.abs(spectrogram.T), 1e-8)), shading="auto")
axes[2].axvspan(gap_start / sample_rate, (gap_start + gap_length) / sample_rate, color="white", alpha=0.45)
axes[2].set_ylim(0.0, 180.0)
axes[2].set_xlabel("frame end time (s)")
axes[2].set_ylabel("frequency (Hz)")
fig.colorbar(image, ax=axes[2], label="magnitude (dB)")

output = Path(__file__).resolve().parents[1] / "outputs" / "signal" / "realtime_streaming.png"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=160)
print("plot:", output)
