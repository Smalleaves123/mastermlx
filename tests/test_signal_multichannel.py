import numpy as np

from mastermlx.signal import MultiChannelSignalMonitor, MultiChannelStreamAligner


def test_multichannel_aligner_resamples_different_source_rates():
    aligner = MultiChannelStreamAligner(
        ["fast", "slow"],
        target_sample_rate=10.0,
        source_sample_rates={"fast": 20.0, "slow": 10.0},
    )
    fast_time = np.arange(20, dtype=float) / 20.0
    slow_time = np.arange(10, dtype=float) / 10.0

    result = aligner.push(
        {"fast": 2.0 * fast_time + 1.0, "slow": 2.0 * slow_time + 1.0},
        timestamps={"fast": fast_time, "slow": slow_time},
    )

    assert np.allclose(result["timestamps"], slow_time)
    assert result["channel_names"] == ("fast", "slow")
    assert np.allclose(result["samples"], np.column_stack([2.0 * slow_time + 1.0] * 2))


def test_multichannel_aligner_exposes_long_sampling_gaps_as_missing():
    aligner = MultiChannelStreamAligner(
        ["left", "right"],
        target_sample_rate=10.0,
        max_interpolation_gap=0.15,
    )
    times = np.array([0.0, 0.1, 0.4])

    result = aligner.push(
        {"left": times, "right": times}, timestamps={"left": times, "right": times}
    )

    assert np.allclose(result["timestamps"], np.arange(5, dtype=float) / 10.0)
    assert np.isnan(result["samples"][2:4]).all()
    assert np.allclose(result["samples"][[0, 1, 4]], np.column_stack([times[[0, 1, 2]]] * 2))


def test_multichannel_monitor_fuses_quality_drift_and_lifecycle_events():
    monitor = MultiChannelSignalMonitor(
        ["motor", "housing"],
        sample_rate=10.0,
        feature_fn=lambda frame: {"mean": np.mean(frame), "rms": np.sqrt(np.mean(frame * frame))},
        frame_length=4,
        baseline_frames=2,
        baseline_std_floor=0.1,
        adaptation_rate=0.0,
        confirmation_frames=2,
        recovery_frames=2,
    )
    healthy = {"motor": np.zeros(4), "housing": np.zeros(4)}
    abnormal = {"motor": np.full(4, 10.0), "housing": np.full(4, 10.0)}

    first = monitor.push(healthy)
    calibrated = monitor.push(healthy)
    suspected = monitor.push(abnormal)
    alerted = monitor.push(abnormal)
    recovering = monitor.push(healthy)
    recovered = monitor.push(healthy)

    assert first["statuses"] == ("initializing",)
    assert [event["type"] for event in calibrated["events"]] == ["baseline_ready"]
    assert suspected["statuses"] == ("healthy",)
    assert alerted["statuses"] == ("critical",)
    assert alerted["health_scores"][0] == 0.0
    assert [event["type"] for event in alerted["events"]] == ["alert"]
    assert recovering["statuses"] == ("critical",)
    assert recovered["statuses"] == ("healthy",)
    assert [event["type"] for event in recovered["events"]] == ["recovered"]
    assert monitor.state()["baseline_ready"] is True


def test_multichannel_monitor_penalizes_missing_channels_after_calibration():
    monitor = MultiChannelSignalMonitor(
        ["motor", "housing"],
        sample_rate=10.0,
        frame_length=4,
        baseline_frames=1,
        adaptation_rate=0.0,
        confirmation_frames=1,
        fusion_weights={"motor": 3.0, "housing": 1.0},
    )
    monitor.push({"motor": np.ones(4), "housing": np.ones(4)})

    result = monitor.push({"motor": np.full(4, np.nan), "housing": np.ones(4)})

    assert result["quality"][0][0]["valid"] is False
    assert result["quality"][0][0]["reasons"] == ("missing_samples",)
    assert result["health_scores"][0] < 50.0
    assert result["statuses"] == ("critical",)
    assert result["events"][0]["type"] == "alert"
