"""Shared data layer: sample generator, uploads, helpers."""
import io

import mne
import numpy as np
import pandas as pd
import pytest

from bci_common import data as D


def test_synthetic_is_deterministic_and_well_formed(raw):
    again = D.make_synthetic_mi(seed=42)
    assert np.allclose(raw.get_data()[:, :1000], again.get_data()[:, :1000])
    assert raw.info["sfreq"] == 160.0
    assert raw.ch_names == D.SYNTH_CHANNELS
    counts = D.events_summary(raw).set_index("label")["count"]
    assert counts["left_hand"] == counts["right_hand"] == counts["rest"] == 20
    # realistic scalp-EEG amplitude: tens of microvolts, stored in volts
    assert 5e-6 < np.median(raw.get_data().std(axis=1)) < 100e-6


def test_synthetic_ground_truth_contralateral_erd(raw):
    """Left-hand imagery lowers mu power at C4 vs C3, right-hand the reverse."""
    ev, eid = mne.events_from_annotations(raw, event_id={"left_hand": 2, "right_hand": 3})
    ep = mne.Epochs(raw, ev, eid, tmin=0.5, tmax=3.5, baseline=None, preload=True)
    p = ep.compute_psd(fmin=8, fmax=13, picks=["C3", "C4"]).get_data().mean(-1)
    left, right = p[ep.events[:, 2] == 2].mean(0), p[ep.events[:, 2] == 3].mean(0)
    assert left[1] < left[0]      # C4 < C3 during left-hand imagery
    assert right[0] < right[1]    # C3 < C4 during right-hand imagery


def test_fif_roundtrip_upload(raw, tmp_path):
    f = tmp_path / "rec_raw.fif"
    raw.copy().crop(0, 20).save(f)
    up = D.read_uploaded("rec_raw.fif", f.read_bytes())
    assert up.ch_names == raw.ch_names
    assert len(D.channel_positions_2d(up)) == len(raw.ch_names)


def test_csv_upload_units_and_montage():
    t = np.arange(500) / 250
    df = pd.DataFrame({"time": t, "C3": 10 * np.sin(2 * np.pi * 10 * t),
                       "Cz": np.zeros(500), "C4": np.ones(500), "Pz": np.ones(500)})
    raw = D.read_uploaded("x.csv", df.to_csv(index=False).encode(), csv_sfreq=250)
    assert raw.ch_names == ["C3", "Cz", "C4", "Pz"]           # time column dropped
    assert np.isclose(raw.get_data()[0].max(), 10e-6, rtol=1e-2)  # µV -> V
    assert len(D.channel_positions_2d(raw)) == 4


def test_unsupported_extension_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        D.read_uploaded("notes.txt", b"hello")


def test_physionet_loader_cleans_names_and_events(raw, tmp_path, monkeypatch):
    """Offline check of load_physionet: a fake PhysioNet-style EDF (names like
    'C3..', events T0/T1/T2) is standardised and relabelled."""
    pytest.importorskip("edfio")
    fake = raw.copy().crop(0, 60).set_montage(None)
    fake.rename_channels({c: (c + "..")[:4] for c in fake.ch_names if len(c) < 4})
    fake.annotations.delete(np.where(fake.annotations.description == "blink")[0])
    fake.annotations.rename({"rest": "T0", "left_hand": "T1", "right_hand": "T2"})
    f = tmp_path / "S001R04.edf"
    mne.export.export_raw(f, fake, overwrite=True)
    monkeypatch.setattr(mne.datasets.eegbci, "load_data", lambda *a, **k: [str(f)])
    out = D.load_physionet(1, 4)
    assert "C3" in out.ch_names and "Fp1" in out.ch_names
    assert {"rest", "left_hand", "right_hand"} <= set(out.annotations.description)
    assert len(D.channel_positions_2d(out)) == len(out.ch_names)
