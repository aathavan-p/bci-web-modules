"""Data Loading module logic."""
import numpy as np

from modules.data_loading import core


def test_window_shape_and_units(raw):
    t, x = core.get_window(raw, ["C3", "C4"], tmin=10.0, duration=4.0)
    assert x.shape == (2, 640)                  # 4 s x 160 Hz
    assert np.isclose(t[0], 10.0)
    assert 1 < np.abs(x).mean() < 200           # microvolts, not volts


def test_alias_frequency_formula():
    assert core.alias_frequency(50, 60) == 10
    assert core.alias_frequency(50, 160) == 50   # below Nyquist: unchanged
    assert core.alias_frequency(70, 100) == 30


def test_naive_decimation_aliases_but_antialias_does_not():
    sf = 160.0
    t = np.arange(int(8 * sf)) / sf
    x = np.sin(2 * np.pi * 50 * t)[None]          # pure 50 Hz "mains"
    _, naive, fs_n = core.resample_window(t, x, sf, 40, anti_alias=False)
    _, clean, fs_c = core.resample_window(t, x, sf, 40, anti_alias=True)
    assert fs_n == fs_c == 40
    f, p = core.window_psd(naive, fs_n, seg_sec=2)
    assert abs(f[np.argmax(p[0])] - 10) <= 0.5    # 50 Hz folded to 10 Hz
    assert clean.std() < 0.1 * naive.std()        # filtered away before resampling


def test_channel_stats_flags_flat_channel(raw):
    _, x = core.get_window(raw, raw.ch_names, 0, 8)
    x[3] = 0.0
    df = core.channel_stats(x, raw.ch_names)
    assert df.loc[3, "flag"] == "flat"


def test_annotations_in_window(raw):
    df = core.annotations_in_window(raw, 0, 30, labels=["left_hand", "right_hand", "rest"])
    assert len(df) > 0 and set(df.label) <= {"left_hand", "right_hand", "rest"}
    assert (df.onset <= 30).all()
