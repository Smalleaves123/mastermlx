from __future__ import annotations

import numpy as np

from ..base import BaseTransformer
from ..utils.estimator import clone
from .core import istft, stft
from .features import mfcc, mel_spectrogram
from .filters import bandpass_filter, normalize_signal, pre_emphasis


class SignalTransformer(BaseTransformer):
    """Base class for signal transforms that follow fit/transform semantics."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        raise NotImplementedError("Subclasses must implement transform()")


class SignalPipeline(SignalTransformer):
    """Chain signal transforms together with optional fitting support."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.steps_ = None

    def _validate_steps(self):
        if not self.steps:
            raise ValueError("steps must be non-empty")
        names = [name for name, _ in self.steps]
        if len(names) != len(set(names)):
            raise ValueError("step names must be unique")
        return self.steps

    def fit(self, X, y=None):
        steps = self._validate_steps()
        Xt = np.asarray(X)
        self.steps_ = []
        for name, step in steps:
            obj = clone(step)
            if hasattr(obj, "fit"):
                obj.fit(Xt, y)
            self.steps_.append((name, obj))
            if hasattr(obj, "transform"):
                Xt = obj.transform(Xt)
        return self

    def transform(self, X):
        if self.steps_ is None:
            raise RuntimeError("SignalPipeline has not been fit yet")
        Xt = np.asarray(X)
        for _, step in self.steps_:
            if not hasattr(step, "transform"):
                raise TypeError("All SignalPipeline steps must define transform()")
            Xt = step.transform(Xt)
        return Xt

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    @property
    def named_steps(self):
        if self.steps_ is None:
            return {name: step for name, step in self.steps}
        return {name: step for name, step in self.steps_}

    def get_params(self, deep=True):
        params = {"steps": self.steps}
        for name, step in self.steps:
            params[name] = step
            if deep and hasattr(step, "get_params"):
                for key, value in step.get_params().items():
                    params[f"{name}__{key}"] = value
            elif deep and hasattr(step, "__dict__"):
                for key, value in step.__dict__.items():
                    if key.endswith("_") or key.startswith("_") or callable(value):
                        continue
                    params[f"{name}__{key}"] = value
        return params

    def set_params(self, **params):
        steps = list(self.steps)
        name_to_idx = {name: idx for idx, (name, _) in enumerate(steps)}
        for key, value in params.items():
            if "__" not in key:
                if key == "steps":
                    steps = list(value)
                    name_to_idx = {name: idx for idx, (name, _) in enumerate(steps)}
                elif key in name_to_idx:
                    steps[name_to_idx[key]] = (key, value)
                else:
                    setattr(self, key, value)
                continue
            name, subkey = key.split("__", 1)
            if name not in name_to_idx:
                raise ValueError(f"Unknown step '{name}'")
            step_name, step = steps[name_to_idx[name]]
            if hasattr(step, "set_params"):
                step.set_params(**{subkey: value})
            else:
                setattr(step, subkey, value)
            steps[name_to_idx[name]] = (step_name, step)
        self.steps = steps
        return self


class NormalizeSignalTransformer(SignalTransformer):
    def __init__(self, eps=1e-12):
        self.eps = float(eps)

    def transform(self, X):
        return normalize_signal(X, eps=self.eps)


class PreEmphasisTransformer(SignalTransformer):
    def __init__(self, coef=0.97):
        self.coef = float(coef)

    def transform(self, X):
        return pre_emphasis(X, coef=self.coef)


class BandpassFilterTransformer(SignalTransformer):
    def __init__(self, low_cutoff, high_cutoff, sample_rate, num_taps=101, window="hann"):
        self.low_cutoff = float(low_cutoff)
        self.high_cutoff = float(high_cutoff)
        self.sample_rate = float(sample_rate)
        self.num_taps = int(num_taps)
        self.window = window

    def transform(self, X):
        return bandpass_filter(
            X,
            low_cutoff=self.low_cutoff,
            high_cutoff=self.high_cutoff,
            sample_rate=self.sample_rate,
            num_taps=self.num_taps,
            window=self.window,
        )


class STFTTransformer(SignalTransformer):
    def __init__(self, frame_length=256, hop_length=None, window="hann", n_fft=None, pad_end=True):
        self.frame_length = int(frame_length)
        self.hop_length = None if hop_length is None else int(hop_length)
        self.window = window
        self.n_fft = None if n_fft is None else int(n_fft)
        self.pad_end = bool(pad_end)

    def transform(self, X):
        return stft(
            X,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            window=self.window,
            n_fft=self.n_fft,
            pad_end=self.pad_end,
        )


class ISTFTTransformer(SignalTransformer):
    def __init__(self, frame_length=256, hop_length=None, window="hann", length=None):
        self.frame_length = int(frame_length)
        self.hop_length = None if hop_length is None else int(hop_length)
        self.window = window
        self.length = length

    def transform(self, X):
        return istft(
            X,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            window=self.window,
            length=self.length,
        )


class MelSpectrogramTransformer(SignalTransformer):
    def __init__(self, sample_rate, frame_length=256, hop_length=None, n_fft=None, n_mels=26, fmin=0.0, fmax=None, window="hann", power=2.0):
        self.sample_rate = float(sample_rate)
        self.frame_length = int(frame_length)
        self.hop_length = None if hop_length is None else int(hop_length)
        self.n_fft = None if n_fft is None else int(n_fft)
        self.n_mels = int(n_mels)
        self.fmin = float(fmin)
        self.fmax = fmax
        self.window = window
        self.power = float(power)

    def transform(self, X):
        return mel_spectrogram(
            X,
            sample_rate=self.sample_rate,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            window=self.window,
            power=self.power,
        )


class MFCCTransformer(SignalTransformer):
    def __init__(self, sample_rate, n_mfcc=13, frame_length=256, hop_length=None, n_fft=None, n_mels=26, fmin=0.0, fmax=None, window="hann"):
        self.sample_rate = float(sample_rate)
        self.n_mfcc = int(n_mfcc)
        self.frame_length = int(frame_length)
        self.hop_length = None if hop_length is None else int(hop_length)
        self.n_fft = None if n_fft is None else int(n_fft)
        self.n_mels = int(n_mels)
        self.fmin = float(fmin)
        self.fmax = fmax
        self.window = window

    def transform(self, X):
        return mfcc(
            X,
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            window=self.window,
        )


__all__ = [
    "BandpassFilterTransformer",
    "ISTFTTransformer",
    "MFCCTransformer",
    "MelSpectrogramTransformer",
    "NormalizeSignalTransformer",
    "PreEmphasisTransformer",
    "SignalPipeline",
    "STFTTransformer",
    "SignalTransformer",
]
