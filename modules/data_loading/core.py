"""Signal logic for the Data Loading module (no Streamlit, fully testable)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch

import mne

CHANNEL_STATS = {
    "rms": "RMS amplitude (µV)",
    "ptp": "Peak-to-peak (µV)",
    "zlogvar": "Variance z-score (outlier check)",
}


def get_window(raw: mne.io.BaseRaw, picks: list[str], tmin: float, duration: float):
    """Return ``(times, data_uV)`` for ``picks`` in ``[tmin, tmin+duration)``."""
    sf = raw.info["sfreq"]
    start = int(max(0, round(tmin * sf)))
    stop = int(min(raw.n_times, start + round(duration * sf)))
    data, times = raw.get_data(picks=picks, start=start, stop=stop, return_times=True)
    return times, data * 1e6


def resample_window(times: np.ndarray, data: np.ndarray, sfreq: float,
                    new_sfreq: float, anti_alias: bool = True):
    """Change the sampling rate of a window.

    * ``anti_alias=True``  -> polyphase resampling (low-pass first), what
      ``raw.resample()`` does.
    * ``anti_alias=False`` -> naive re-sampling: read off the signal at the new
      sample times with no low-pass first (linear interpolation between the
      original samples). Energy above the new Nyquist frequency folds back into
      the spectrum (aliasing).
    """
    if new_sfreq >= sfreq:
        return times, data, sfreq
    if anti_alias:
        from scipy.signal import resample_poly
        from fractions import Fraction
        frac = Fraction(new_sfreq / sfreq).limit_denominator(200)
        out = resample_poly(data, frac.numerator, frac.denominator, axis=-1)
        eff = sfreq * frac.numerator / frac.denominator
        new_t = times[0] + np.arange(out.shape[-1]) / eff
        return new_t, out, eff
    eff = float(new_sfreq)
    new_t = times[0] + np.arange(int(np.floor((times[-1] - times[0]) * eff)) + 1) / eff
    out = np.stack([np.interp(new_t, times, row) for row in data])
    return new_t, out, eff


def alias_frequency(f: float, fs: float) -> float:
    """Frequency at which a pure tone ``f`` appears after sampling at ``fs``."""
    f_mod = f % fs
    return float(min(f_mod, fs - f_mod))


def channel_stats(data_uV: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Per-channel quality statistics for a window (data in µV)."""
    rms = np.sqrt(np.mean((data_uV - data_uV.mean(axis=1, keepdims=True)) ** 2, axis=1))
    ptp = np.ptp(data_uV, axis=1)
    logvar = np.log(np.var(data_uV, axis=1) + 1e-12)
    sd = logvar.std()
    z = (logvar - np.median(logvar)) / (sd if sd > 0 else 1.0)
    df = pd.DataFrame({"channel": names, "rms": rms, "ptp": ptp, "zlogvar": z})
    df["flag"] = np.where(np.var(data_uV, axis=1) < 1e-4, "flat",
                          np.where(np.abs(z) > 3, "outlier", ""))
    return df


def window_psd(data_uV: np.ndarray, sfreq: float, seg_sec: float = 1.0):
    """Welch PSD (µV²/Hz) of each row."""
    nperseg = int(min(data_uV.shape[-1], max(8, round(seg_sec * sfreq))))
    f, p = welch(data_uV, fs=sfreq, nperseg=nperseg, axis=-1)
    return f, p


def annotations_in_window(raw: mne.io.BaseRaw, tmin: float, tmax: float,
                          labels: list[str] | None = None) -> pd.DataFrame:
    ann = raw.annotations
    rows = []
    first = raw.first_time if ann.orig_time is not None else 0.0
    for on, du, de in zip(ann.onset - first, ann.duration, ann.description):
        if labels is not None and de not in labels:
            continue
        if on + du >= tmin and on <= tmax:
            rows.append({"onset": float(on), "duration": float(du), "label": str(de)})
    return pd.DataFrame(rows, columns=["onset", "duration", "label"])


def channel_table(raw: mne.io.BaseRaw) -> pd.DataFrame:
    rows = []
    types = raw.get_channel_types()
    for ch, ctype in zip(raw.info["chs"], types):
        loc = ch["loc"][:3]
        has_pos = bool(np.any(loc) and not np.any(np.isnan(loc)))
        rows.append({
            "name": ch["ch_name"],
            "type": ctype,
            "unit": "V" if ch["unit"] == 107 else str(ch["unit"]),
            "has_position": has_pos,
            "bad": ch["ch_name"] in raw.info["bads"],
        })
    return pd.DataFrame(rows)


def default_picks(raw: mne.io.BaseRaw, n: int = 10) -> list[str]:
    """A spread of channels that covers frontal, central and occipital sites."""
    preferred = ["Fp1", "Fz", "FC3", "C3", "Cz", "C4", "CP3", "Pz", "O1", "Oz",
                 "Fp2", "FC4", "CP4", "O2"]
    have = [c for c in preferred if c in raw.ch_names]
    if len(have) >= min(n, len(raw.ch_names)):
        return have[:n]
    eeg = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)] or raw.ch_names
    return (have + [c for c in eeg if c not in have])[:n]
