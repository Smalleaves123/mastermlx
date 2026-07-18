# Signal Examples

This folder contains signal-processing demos for the `mastermlx.signal` package.

## Intended demos

- waveform normalization and pre-emphasis
- STFT, mel-spectrogram, and MFCC extraction
- Fourier-domain analysis, dominant frequency detection, and band-energy summaries
- unified STFT + FFT feature vectors for downstream models
- streaming feature extraction on chunked input
- stateful streaming monitoring with feature extraction and event detection
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
