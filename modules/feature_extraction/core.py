"""Signal logic for the Feature Extraction module (no Streamlit, fully testable).

Turns epoched EEG ``(n_trials, n_channels, n_times)`` into a feature matrix
``X`` of shape ``(n_trials, n_features)``, scores how well each feature
separates the classes, and runs a leak-free cross-validated sanity check.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mne
import numpy as np
from scipy.signal import welch
from scipy.stats import binom
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

METHODS = {
    "bandpower": "Band power (Welch PSD)",
    "csp": "CSP log-variance (spatial filters)",
    "hjorth": "Hjorth parameters (time domain)",
}
HJORTH_PARAMS = ["activity", "mobility", "complexity"]
BANDS = {"theta": (4, 8), "alpha / mu": (8, 13), "beta": (13, 30), "low gamma": (30, 45)}


# ------------------------------------------------------------------ epoching
def make_epochs(raw: mne.io.BaseRaw, classes: list[str], tmin: float, tmax: float,
                picks: list[str] | None = None, l_freq: float | None = None,
                h_freq: float | None = None) -> mne.Epochs:
    """Cut trials for ``classes`` from annotations, optionally band-pass first.

    Filtering is done on the continuous data (before cutting) to avoid edge
    artefacts at the start and end of each trial.
    """
    raw = raw.copy()
    if picks:
        raw.pick(picks)
    else:
        raw.pick("eeg")
    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq, h_freq, method="iir",
                   iir_params=dict(order=4, ftype="butter"), verbose=False)
    present = sorted(set(raw.annotations.description))
    event_id = {c: i + 1 for i, c in enumerate(classes) if c in present}
    if len(event_id) < 2:
        raise ValueError("Need at least two classes that exist in the annotations.")
    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
    epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, baseline=None,
                        preload=True, reject_by_annotation=False, verbose=False)
    return epochs


# ------------------------------------------------------------------ features
def band_power(X: np.ndarray, sfreq: float, band: tuple[float, float],
               welch_sec: float = 1.0, relative: bool = False, log: bool = True,
               total_band: tuple[float, float] = (1.0, 45.0)):
    """Mean Welch PSD inside ``band`` for each trial and channel.

    Returns ``(features (n_trials, n_ch), freqs, psd (n_trials, n_ch, n_freqs))``.
    """
    nperseg = int(min(X.shape[-1], max(16, round(welch_sec * sfreq))))
    f, psd = welch(X, fs=sfreq, nperseg=nperseg, axis=-1)
    m = (f >= band[0]) & (f <= band[1])
    if not m.any():
        raise ValueError(f"Band {band} has no frequency bins at resolution "
                         f"{f[1] - f[0]:.2f} Hz - widen it or lengthen the Welch window.")
    bp = psd[..., m].mean(-1)
    if relative:
        tm = (f >= total_band[0]) & (f <= min(total_band[1], f[-1]))
        bp = bp / psd[..., tm].mean(-1)
    if log:
        bp = np.log10(bp + 1e-30)
    return bp, f, psd


def hjorth(X: np.ndarray, which: str = "activity", log: bool = True) -> np.ndarray:
    """Hjorth activity (variance), mobility and complexity per trial/channel."""
    d1 = np.diff(X, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0, v1, v2 = X.var(-1), d1.var(-1), d2.var(-1)
    mobility = np.sqrt(v1 / v0)
    if which == "activity":
        out = v0
        return np.log10(out + 1e-30) if log else out
    if which == "mobility":
        return mobility
    if which == "complexity":
        return np.sqrt(v2 / v1) / mobility
    raise ValueError(which)


class BandPowerTransformer(BaseEstimator, TransformerMixin):
    """sklearn wrapper so band power can sit inside a CV pipeline."""

    def __init__(self, sfreq=160.0, band=(8, 13), welch_sec=1.0, relative=False, log=True):
        self.sfreq, self.band, self.welch_sec = sfreq, band, welch_sec
        self.relative, self.log = relative, log

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return band_power(X, self.sfreq, self.band, self.welch_sec, self.relative, self.log)[0]


class HjorthTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, which="activity", log=True):
        self.which, self.log = which, log

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return hjorth(X, self.which, self.log)


def make_extractor(method: str, sfreq: float, band, welch_sec=1.0, relative=False,
                   log=True, n_csp=4, hjorth_param="activity"):
    """Feature extractor as an sklearn transformer (CSP is *fitted*, so it must
    be trained inside each CV fold to avoid leakage)."""
    if method == "bandpower":
        return BandPowerTransformer(sfreq, tuple(band), welch_sec, relative, log)
    if method == "csp":
        from mne.decoding import CSP
        return CSP(n_components=n_csp, log=log, transform_into="average_power",
                   reg="ledoit_wolf", norm_trace=False)
    if method == "hjorth":
        return HjorthTransformer(hjorth_param, log)
    raise ValueError(method)


@dataclass
class FeatureResult:
    X: np.ndarray                    # (n_trials, n_features)
    y: np.ndarray                    # (n_trials,)
    feature_names: list[str]
    channel_of_feature: list[str | None]
    extractor: object
    extra: dict = field(default_factory=dict)


def extract(epochs: mne.Epochs, method: str, band, welch_sec=1.0, relative=False,
            log=True, n_csp=4, hjorth_param="activity") -> FeatureResult:
    """Fit/transform on *all* trials - for visualisation only. Use
    :func:`cv_accuracy` for any performance number."""
    Xe = epochs.get_data(copy=True)
    y = epochs.events[:, 2]
    chs = epochs.ch_names
    ext = make_extractor(method, epochs.info["sfreq"], band, welch_sec, relative, log,
                         n_csp, hjorth_param)
    if method == "csp":
        if len(np.unique(y)) != 2:
            raise ValueError("This CSP demo uses exactly two classes.")
        F = ext.fit_transform(Xe, y)
        names = [f"CSP{i + 1}" for i in range(F.shape[1])]
        extra = {"patterns": ext.patterns_[:n_csp], "filters": ext.filters_[:n_csp]}
        return FeatureResult(F, y, names, [None] * len(names), ext, extra)
    F = ext.fit_transform(Xe, y)
    suffix = {"bandpower": f"{'rel' if relative else ''}BP{band[0]:g}-{band[1]:g}",
              "hjorth": hjorth_param}[method]
    names = [f"{c}·{suffix}" for c in chs]
    return FeatureResult(F, y, names, list(chs), ext)


# ------------------------------------------------------------------ scoring
def fisher_score(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Between-class / within-class variance ratio per feature (higher = more
    discriminative). Works for 2+ classes."""
    classes = np.unique(y)
    mu = X.mean(0)
    num = np.zeros(X.shape[1])
    den = np.zeros(X.shape[1])
    for c in classes:
        Xc = X[y == c]
        num += len(Xc) * (Xc.mean(0) - mu) ** 2
        den += len(Xc) * Xc.var(0)
    return num / (den + 1e-30)


def chance_threshold(n_trials: int, n_classes: int, alpha: float = 0.05) -> float:
    """Accuracy needed to beat chance at level ``alpha`` with ``n_trials``
    (binomial test, Combrisson & Jerbi 2015)."""
    return float(binom.ppf(1 - alpha, n_trials, 1.0 / n_classes) / n_trials)


def cv_accuracy(epochs: mne.Epochs, method: str, band, welch_sec=1.0, relative=False,
                log=True, n_csp=4, hjorth_param="activity", n_splits=5, seed=0):
    """Leak-free CV: the extractor (incl. CSP) and scaler are re-fitted inside
    each training fold, followed by a simple LDA. Returns per-fold accuracies."""
    Xe = epochs.get_data(copy=True)
    y = epochs.events[:, 2]
    n_min = int(np.bincount(y)[np.unique(y)].min())
    k = max(2, min(n_splits, n_min))
    pipe = make_pipeline(
        make_extractor(method, epochs.info["sfreq"], band, welch_sec, relative, log,
                       n_csp, hjorth_param),
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    cv = StratifiedKFold(k, shuffle=True, random_state=seed)
    return cross_val_score(pipe, Xe, y, cv=cv)


def class_mean_psd(psd: np.ndarray, y: np.ndarray, ch_idx: int):
    """Per-class mean and SEM of PSD at one channel, in dB re 1 µV²/Hz
    (``psd`` is in V²/Hz as returned by :func:`band_power`)."""
    out = {}
    lp = 10 * np.log10(psd[:, ch_idx, :] * 1e12 + 1e-30)
    for c in np.unique(y):
        v = lp[y == c]
        out[int(c)] = (v.mean(0), v.std(0) / np.sqrt(max(1, len(v))))
    return out


def default_classes(labels: list[str]) -> list[str]:
    for pair in (["left_hand", "right_hand"], ["both_hands", "both_feet"]):
        if all(p in labels for p in pair):
            return pair
    usable = [l for l in labels if l.lower() not in ("blink", "bad", "bad_", "edge")]
    return usable[:2]
