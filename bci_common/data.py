"""Shared EEG data sources for the teaching modules.

Pure functions only (no Streamlit), so they can be unit-tested and reused by
any module in the merged course app.

Three sources are supported:

* ``synthetic`` - a deterministic, offline motor-imagery recording generated
  from a fixed seed. It has realistic 1/f background activity, occipital alpha,
  sensorimotor mu/beta rhythms with contralateral desynchronisation (ERD)
  during left/right-hand imagery, eye blinks and 50 Hz mains noise. Because
  the ground truth is known, students can check that what they see is right.
* ``physionet`` - the PhysioNet EEG Motor Movement/Imagery dataset (EEGBCI),
  downloaded on first use through ``mne.datasets.eegbci``.
* ``upload`` - any file MNE can read in one piece (.fif, .edf, .bdf, .gdf,
  .set with embedded data) or a CSV with one column per channel.
"""
from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd

mne.set_log_level("ERROR")

# 10-20 subset that covers frontal (blinks), sensorimotor (mu/beta) and
# occipital (alpha) regions.
SYNTH_CHANNELS = [
    "Fp1", "Fp2", "F3", "Fz", "F4",
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CPz", "CP4",
    "P3", "Pz", "P4",
    "O1", "Oz", "O2",
]
SYNTH_EVENT_ID = {"rest": 1, "left_hand": 2, "right_hand": 3}

# PhysioNet EEGBCI run numbers and what happens in them.
PHYSIONET_RUNS = {
    1: "Baseline, eyes open",
    2: "Baseline, eyes closed",
    3: "Task 1: real left/right fist",
    4: "Task 2: imagined left/right fist",
    5: "Task 3: real both fists/feet",
    6: "Task 4: imagined both fists/feet",
    7: "Task 1 (repeat)",
    8: "Task 2 (repeat)",
    9: "Task 3 (repeat)",
    10: "Task 4 (repeat)",
    11: "Task 1 (repeat)",
    12: "Task 2 (repeat)",
    13: "Task 3 (repeat)",
    14: "Task 4 (repeat)",
}
# T0 = rest, T1/T2 meaning depends on the run.
PHYSIONET_LR_RUNS = {3, 4, 7, 8, 11, 12}

UPLOAD_EXTENSIONS = ["fif", "edf", "bdf", "gdf", "set", "csv"]


@dataclass
class LoadResult:
    """A loaded recording plus a human-readable note on where it came from."""

    raw: mne.io.BaseRaw
    source: str
    description: str


# --------------------------------------------------------------------------
# Synthetic motor-imagery recording
# --------------------------------------------------------------------------
def _pink_noise(rng: np.random.Generator, n_ch: int, n_times: int) -> np.ndarray:
    """1/f noise via spectral shaping (unit variance per channel)."""
    white = rng.standard_normal((n_ch, n_times))
    spec = np.fft.rfft(white, axis=1)
    freqs = np.fft.rfftfreq(n_times)
    freqs[0] = freqs[1]
    spec /= np.sqrt(freqs)
    pink = np.fft.irfft(spec, n=n_times, axis=1)
    pink -= pink.mean(axis=1, keepdims=True)
    pink /= pink.std(axis=1, keepdims=True)
    return pink


def _narrowband(rng, n_times, sfreq, f0, bw=1.5):
    """Band-limited oscillation with a slowly wandering amplitude/phase."""
    white = rng.standard_normal(n_times)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_times, 1 / sfreq)
    spec *= np.exp(-0.5 * ((freqs - f0) / bw) ** 2)
    sig = np.fft.irfft(spec, n=n_times)
    return sig / sig.std()


def _spatial_weights(ch_pos: np.ndarray, centre: np.ndarray, width: float) -> np.ndarray:
    d = np.linalg.norm(ch_pos - centre, axis=1)
    return np.exp(-0.5 * (d / width) ** 2)


def make_synthetic_mi(
    n_trials_per_class: int = 20,
    sfreq: float = 160.0,
    trial_dur: float = 4.0,
    iti: float = 2.5,
    seed: int = 42,
) -> mne.io.RawArray:
    """Generate a deterministic left/right-hand motor-imagery recording.

    Ground truth built into the signal:

    * Occipital alpha (~10 Hz) at O1/Oz/O2, always present.
    * Mu (~11 Hz) and beta (~22 Hz) over C3 and C4. During *left_hand*
      imagery the right hemisphere (C4) desynchronises, during *right_hand*
      imagery C3 does. The amplitude drop varies per trial (0-65 %), so some
      trials are hard to classify. *rest* trials keep full power.
    * Eye blinks (large, slow, frontal) at random times.
    * 50 Hz mains interference on all channels.

    Data are in volts, as MNE expects.
    """
    rng = np.random.default_rng(seed)
    montage = mne.channels.make_standard_montage("standard_1020")
    pos_dict = montage.get_positions()["ch_pos"]
    ch_pos = np.array([pos_dict[ch] for ch in SYNTH_CHANNELS])
    n_ch = len(SYNTH_CHANNELS)

    # Trial schedule (balanced, shuffled)
    labels = np.repeat(list(SYNTH_EVENT_ID.values()), n_trials_per_class)
    rng.shuffle(labels)
    lead_in = 3.0
    onsets = lead_in + np.arange(len(labels)) * (trial_dur + iti)
    total = onsets[-1] + trial_dur + lead_in
    n_times = int(round(total * sfreq))
    t = np.arange(n_times) / sfreq

    # Background: spatially smoothed pink noise, ~8 µV
    bg = _pink_noise(rng, n_ch, n_times)
    dist = np.linalg.norm(ch_pos[:, None] - ch_pos[None], axis=2)
    mix = np.exp(-0.5 * (dist / 0.035) ** 2)
    mix /= mix.sum(axis=1, keepdims=True)
    data = 8e-6 * (mix @ bg) / (mix @ bg).std(axis=1, keepdims=True)

    # Occipital alpha
    alpha = _narrowband(rng, n_times, sfreq, 10.0, 1.0)
    w_occ = _spatial_weights(ch_pos, pos_dict["Oz"], 0.04)
    data += 10e-6 * np.outer(w_occ, alpha)

    # Sensorimotor mu & beta with ERD envelopes
    # Trial-to-trial variability: some imagery trials desynchronise strongly,
    # others hardly at all (as in real subjects), so features are not perfect.
    depth = rng.uniform(0.35, 1.0, size=len(labels))

    def erd_envelope(hemi_class: int) -> np.ndarray:
        env = np.ones(n_times)
        ramp = int(0.5 * sfreq)
        for onset, lab, dep in zip(onsets, labels, depth):
            if lab != hemi_class:
                continue
            a = int((onset + 0.5) * sfreq)  # ERD develops ~0.5 s after cue
            b = int((onset + trial_dur) * sfreq)
            env[a:b] = dep
            env[a - ramp:a] = np.linspace(1, dep, ramp)
            env[b:b + ramp] = np.linspace(dep, 1, ramp)
        return env

    for site, erd_class in (("C3", SYNTH_EVENT_ID["right_hand"]),
                            ("C4", SYNTH_EVENT_ID["left_hand"])):
        env = erd_envelope(erd_class)
        w = _spatial_weights(ch_pos, pos_dict[site], 0.03)
        mu = _narrowband(rng, n_times, sfreq, 11.0, 1.2)
        beta = _narrowband(rng, n_times, sfreq, 22.0, 2.0)
        data += np.outer(w, env * (9e-6 * mu + 4e-6 * beta))

    # Eye blinks (frontal, ~150 µV, ~300 ms)
    w_front = _spatial_weights(ch_pos, (pos_dict["Fp1"] + pos_dict["Fp2"]) / 2, 0.05)
    blink_times = np.sort(rng.uniform(1, total - 1, size=int(total / 6)))
    for bt in blink_times:
        blink = 150e-6 * np.exp(-0.5 * ((t - bt) / 0.08) ** 2)
        data += np.outer(w_front, blink)

    # Mains noise
    data += 3e-6 * np.sin(2 * np.pi * 50.0 * t)[None, :]

    info = mne.create_info(SYNTH_CHANNELS, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage(montage)
    inv = {v: k for k, v in SYNTH_EVENT_ID.items()}
    raw.set_annotations(mne.Annotations(
        onset=onsets, duration=[trial_dur] * len(onsets),
        description=[inv[x] for x in labels]))
    blink_ann = mne.Annotations(blink_times - 0.15, [0.3] * len(blink_times),
                                ["blink"] * len(blink_times))
    raw.set_annotations(raw.annotations + blink_ann)
    raw.info["description"] = "Synthetic motor imagery (BCI course sample)"
    return raw


# --------------------------------------------------------------------------
# PhysioNet EEGBCI
# --------------------------------------------------------------------------
def load_physionet(subject: int = 1, run: int = 4, path: str | None = None) -> mne.io.BaseRaw:
    """Download (first time only) and load one PhysioNet EEGBCI run.

    Channel names are cleaned (``C3..`` -> ``C3``), the standard 10-05 montage
    is applied and the T0/T1/T2 annotations are renamed to readable labels.
    """
    from mne.datasets import eegbci

    files = eegbci.load_data(subject, [run], path=path, update_path=False)
    raw = mne.io.read_raw_edf(files[0], preload=True)
    eegbci.standardize(raw)
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
    if run in PHYSIONET_LR_RUNS:
        mapping = {"T0": "rest", "T1": "left_hand", "T2": "right_hand"}
    elif run in (1, 2):
        mapping = {"T0": "rest"}
    else:
        mapping = {"T0": "rest", "T1": "both_hands", "T2": "both_feet"}
    raw.annotations.rename({k: v for k, v in mapping.items()
                            if k in set(raw.annotations.description)})
    return raw


# --------------------------------------------------------------------------
# Uploaded files
# --------------------------------------------------------------------------
def read_uploaded(name: str, content: bytes, csv_sfreq: float = 250.0) -> mne.io.BaseRaw:
    """Read an uploaded file from bytes.

    For CSV: one column per channel, optional ``time`` column (ignored),
    values assumed to be in microvolts.
    """
    ext = name.rsplit(".", 1)[-1].lower()
    if ext not in UPLOAD_EXTENSIONS:
        raise ValueError(f"Unsupported file type '.{ext}'. Use one of: "
                         + ", ".join("." + e for e in UPLOAD_EXTENSIONS))
    if ext == "csv":
        return read_csv_eeg(io.BytesIO(content), sfreq=csv_sfreq)

    suffix = "_raw.fif" if ext == "fif" else f".{ext}"
    with tempfile.TemporaryDirectory() as tmp:
        fpath = os.path.join(tmp, "upload" + suffix)
        with open(fpath, "wb") as fh:
            fh.write(content)
        raw = mne.io.read_raw(fpath, preload=True)
    try_set_montage(raw)
    return raw


def read_csv_eeg(buf, sfreq: float = 250.0) -> mne.io.RawArray:
    df = pd.read_csv(buf)
    drop = [c for c in df.columns if str(c).strip().lower() in ("time", "t", "timestamp", "sample")]
    df = df.drop(columns=drop)
    df = df.select_dtypes(include=[np.number])
    if df.shape[1] == 0:
        raise ValueError("CSV has no numeric channel columns.")
    info = mne.create_info([str(c) for c in df.columns], sfreq, ch_types="eeg")
    raw = mne.io.RawArray(df.to_numpy().T * 1e-6, info, verbose=False)
    try_set_montage(raw)
    return raw


def try_set_montage(raw: mne.io.BaseRaw) -> bool:
    """Apply the standard 10-05 montage to any channels it knows. Returns True
    if at least 3 EEG channels received positions."""
    montage = mne.channels.make_standard_montage("standard_1005")
    known = {c.lower(): c for c in montage.ch_names}
    rename = {}
    for ch in raw.ch_names:
        clean = ch.strip().rstrip(".").replace("EEG ", "").replace("-REF", "")
        if clean.lower() in known and clean != ch:
            rename[ch] = known[clean.lower()]
    if rename:
        try:
            raw.rename_channels(rename)
        except Exception:
            pass
    try:
        raw.set_montage(montage, match_case=False, on_missing="ignore")
    except Exception:
        return False
    return len(channel_positions_2d(raw)) >= 3


# --------------------------------------------------------------------------
# Helpers used by several modules
# --------------------------------------------------------------------------
def channel_positions_2d(raw_or_info) -> dict[str, tuple[float, float]]:
    """Project 3-D sensor positions onto a 2-D head plane (azimuthal
    equidistant, nose up). Channels without positions are skipped."""
    info = raw_or_info.info if hasattr(raw_or_info, "info") else raw_or_info
    out = {}
    for ch in info["chs"]:
        if ch["kind"] != mne.io.constants.FIFF.FIFFV_EEG_CH:
            continue
        loc = ch["loc"][:3]
        if not np.any(loc) or np.any(np.isnan(loc)):
            continue
        x, y, z = loc
        r = np.sqrt(x * x + y * y + z * z)
        theta = np.arccos(np.clip(z / r, -1, 1))  # angle from vertex
        phi = np.arctan2(y, x)
        out[ch["ch_name"]] = (float(theta * np.cos(phi)), float(theta * np.sin(phi)))
    return out


def events_summary(raw: mne.io.BaseRaw) -> pd.DataFrame:
    """Count annotations per label with mean duration."""
    ann = raw.annotations
    if len(ann) == 0:
        return pd.DataFrame(columns=["label", "count", "mean_duration_s"])
    df = pd.DataFrame({"label": ann.description, "duration": ann.duration})
    return (df.groupby("label")
              .agg(count=("duration", "size"), mean_duration_s=("duration", "mean"))
              .reset_index()
              .round(3))


def dimension_notation(raw: mne.io.BaseRaw) -> dict:
    """The C x T description of a continuous recording."""
    eeg = mne.pick_types(raw.info, eeg=True)
    return {
        "channels_C": len(raw.ch_names),
        "eeg_channels": int(len(eeg)),
        "samples_T": int(raw.n_times),
        "sfreq_Hz": float(raw.info["sfreq"]),
        "duration_s": float(round(raw.n_times / raw.info["sfreq"], 2)),
        "n_annotations": len(raw.annotations),
        "highpass_Hz": float(raw.info["highpass"]),
        "lowpass_Hz": float(raw.info["lowpass"]),
    }
