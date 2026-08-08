import numpy as np
import pytest

from mastermlx.signal import StreamingIIRFilter, StreamingSTFT, iir_filter, stft


def _collect_spectra(results):
    spectra = [result["spectrogram"] for result in results if result["spectrogram"].size]
    return np.vstack(spectra) if spectra else np.empty((0, 0), dtype=complex)


def test_streaming_iir_matches_batch_filter_across_irregular_chunks():
    rng = np.random.default_rng(4)
    signal = rng.normal(size=127)
    b = np.array([0.2, 0.15, 0.1])
    a = np.array([1.0, -0.65, 0.12])
    stream = StreamingIIRFilter(b, a, sample_rate=1000.0)

    output = np.concatenate([stream.push(signal[:5]), stream.push(signal[5:51]), stream.push(signal[51:])])

    assert np.allclose(output, iir_filter(signal, b, a))
    assert stream.state()["samples_seen"] == signal.size
    assert stream.state()["filter_state"]["input"].shape == (2,)


def test_streaming_sos_matches_equivalent_single_iir_section():
    rng = np.random.default_rng(5)
    signal = rng.normal(size=64)
    b = np.array([0.2, 0.1, 0.0])
    a = np.array([1.0, -0.7, 0.0])
    stream = StreamingIIRFilter(sos=np.asarray([[*b, *a]]))

    output = np.concatenate([stream.push(signal[:31]), stream.push(signal[31:])])

    assert np.allclose(output, iir_filter(signal, b, a))
    assert stream.state()["uses_sos"] is True


def test_streaming_iir_resets_across_gaps_and_rate_changes():
    stream = StreamingIIRFilter([0.5], [1.0, -0.5], sample_rate=100.0)
    stream.push(np.ones(4))
    after_gap = stream.push(np.ones(3), gap_samples=5)

    assert np.allclose(after_gap, iir_filter(np.ones(3), [0.5], [1.0, -0.5]))
    assert stream.state()["gap_samples"] == 5
    assert stream.state()["last_reset_reason"] == "gap"
    with pytest.raises(ValueError, match="require reset"):
        stream.set_sample_rate(200.0, reset=False)

    after_rate_change = stream.push(np.ones(3), sample_rate=200.0)
    assert np.allclose(after_rate_change, iir_filter(np.ones(3), [0.5], [1.0, -0.5]))
    assert stream.state()["last_reset_reason"] == "sample_rate_change"


def test_streaming_stft_matches_batch_stft_across_chunk_boundaries():
    rng = np.random.default_rng(6)
    signal = rng.normal(size=93)
    stream = StreamingSTFT(frame_length=16, hop_length=8, n_fft=32, sample_rate=80.0, pad_end=True)

    results = [
        stream.push(signal[:3]),
        stream.push(signal[3:38]),
        stream.push(signal[38:70]),
        stream.push(signal[70:]),
        stream.flush(),
    ]
    output = _collect_spectra(results)
    expected = stft(signal, frame_length=16, hop_length=8, n_fft=32, pad_end=True)

    assert np.allclose(output, expected)
    assert results[1]["frame_end_times"][0] == pytest.approx(15.0 / 80.0)


def test_streaming_stft_does_not_bridge_packet_loss_or_rate_change():
    stream = StreamingSTFT(frame_length=8, hop_length=4, sample_rate=10.0)

    assert stream.push(np.arange(6, dtype=float))["spectrogram"].size == 0
    after_gap = stream.push(np.arange(8, dtype=float), gap_samples=3)
    assert np.array_equal(after_gap["frame_start_samples"], np.array([9]))
    assert after_gap["frame_end_times"][0] == pytest.approx(1.6)
    assert stream.state()["last_reset_reason"] == "gap"

    with pytest.raises(ValueError, match="require reset"):
        stream.set_sample_rate(20.0, reset=False)
    assert stream.push(np.arange(8, dtype=float), sample_rate=20.0)["frame_start_samples"][0] == 0
    assert stream.state()["last_reset_reason"] == "sample_rate_change"
