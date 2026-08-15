# Signal Examples

This folder contains signal-processing demos for the `mastermlx.signal` package.

The examples target `mastermlx` 0.1.15. From the repository root, start with:

```bash
MPLBACKEND=Agg python examples/signal/fourier_demo.py
python examples/signal/experiment_demo.py
```

Low-level functions accept NumPy arrays and return arrays or explicit result
objects. Streaming objects retain state between chunks; create a new instance
for an independent stream. High-level experiments expose reports compatible
with the shared interface in [`docs/workflows.md`](../../docs/workflows.md).

## Intended demos

- waveform normalization and pre-emphasis
- STFT, mel-spectrogram, and MFCC extraction
- Fourier-domain analysis, dominant frequency detection, and band-energy summaries
- unified STFT + FFT feature vectors for downstream models
- streaming feature extraction on chunked input
- stateful streaming monitoring with feature extraction and event detection
- multi-channel time alignment, quality-aware health fusion, and alert lifecycle events
- stateful IIR/SOS filtering and chunk-invariant STFT with packet-loss semantics
- event detection with `CUSUMDetector` and threshold-based detectors
- high-level signal experiments with `SignalExperiment`

## Good example stories

- a speech-like feature extraction pipeline
- a compact Fourier demo that surfaces peaks and reconstructs the waveform
- a spectral feature demo that combines STFT summaries with FFT features
- anomaly detection over a sensor stream
- a compact business signal workflow that goes from raw samples to detection output
- a supervised signal experiment that wraps feature extraction and a classifier

## Benchmark link

The corresponding smoke benchmark lives in [`benchmarks/bench_signal.py`](../../benchmarks/bench_signal.py).

## Demo

The high-level experiment demo lives in [`experiment_demo.py`](experiment_demo.py).
The Fourier-focused demo lives in [`fourier_demo.py`](fourier_demo.py).
The multi-sensor online monitoring demo lives in [`multichannel_monitoring.py`](multichannel_monitoring.py)
and writes a health-score plot to `examples/outputs/signal/`.
The real-time filter and STFT demo lives in [`realtime_streaming.py`](realtime_streaming.py)
and writes a time/frequency plot with its packet-loss interval marked.

See the [`signal API overview`](../API_REFERENCE.md#anomaly-detection-signal-and-time-series)
for import guidance.
