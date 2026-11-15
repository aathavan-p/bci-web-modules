"""Streamlit widgets shared by every module so that merged modules look and
behave the same: data-source picker, Explain panel with questions, export."""
from __future__ import annotations

import hashlib

import mne
import streamlit as st

from . import data as D


# ---------------------------------------------------------------- caching
@st.cache_resource(show_spinner="Generating the synthetic sample...")
def _cached_synthetic(seed: int) -> mne.io.BaseRaw:
    return D.make_synthetic_mi(seed=seed)


@st.cache_resource(show_spinner="Downloading/loading PhysioNet run (first time can take ~30 s)...")
def _cached_physionet(subject: int, run: int) -> mne.io.BaseRaw:
    return D.load_physionet(subject, run)


@st.cache_resource(show_spinner="Reading your file...")
def _cached_upload(digest: str, name: str, _content: bytes, csv_sfreq: float) -> mne.io.BaseRaw:
    return D.read_uploaded(name, _content, csv_sfreq=csv_sfreq)


# ---------------------------------------------------------------- picker
def data_source_picker(key: str, container=None) -> tuple[mne.io.BaseRaw | None, dict]:
    """Render the data-source controls and return ``(raw_copy, source_dict)``.

    The returned Raw is a copy, so the module may modify it freely.
    """
    c = container or st.sidebar
    c.markdown("#### Data source")
    kind_label = c.radio(
        "Load data from",
        ["Sample (synthetic motor imagery)", "PhysioNet EEGBCI (real EEG)", "Upload my own file"],
        key=f"{key}_kind",
        help="The synthetic sample works offline and has a known ground truth. "
             "PhysioNet is real 64-channel EEG, downloaded on first use.",
    )
    source: dict = {}
    raw = None
    try:
        if kind_label.startswith("Sample"):
            seed = c.number_input("Random seed", 0, 9999, 42, key=f"{key}_seed",
                                  help="Same seed -> identical recording.")
            source = {"kind": "synthetic", "seed": int(seed)}
            raw = _cached_synthetic(int(seed))
        elif kind_label.startswith("PhysioNet"):
            subject = c.number_input("Subject (1-109)", 1, 109, 1, key=f"{key}_subj")
            run = c.selectbox("Run", list(D.PHYSIONET_RUNS), index=3,
                              format_func=lambda r: f"R{r:02d} - {D.PHYSIONET_RUNS[r]}",
                              key=f"{key}_run")
            source = {"kind": "physionet", "subject": int(subject), "run": int(run)}
            raw = _cached_physionet(int(subject), int(run))
        else:
            up = c.file_uploader("EEG file", type=D.UPLOAD_EXTENSIONS, key=f"{key}_file",
                                 help=".fif .edf .bdf .gdf .set (data embedded) or .csv "
                                      "(one column per channel, µV).")
            csv_sfreq = 250.0
            if up is not None and up.name.lower().endswith(".csv"):
                csv_sfreq = c.number_input("CSV sampling rate (Hz)", 1.0, 10000.0, 250.0,
                                           key=f"{key}_csvfs")
            if up is None:
                c.info("Choose a file to continue.")
                return None, {"kind": "upload"}
            content = up.getvalue()
            digest = hashlib.sha1(content).hexdigest()
            source = {"kind": "upload", "filename": up.name, "csv_sfreq": csv_sfreq}
            raw = _cached_upload(digest, up.name, content, csv_sfreq)
    except Exception as exc:  # surface loader errors in the UI, not a traceback
        c.error(f"Could not load data: {exc}")
        return None, source
    return raw.copy(), source


# ---------------------------------------------------------------- explain
def explain_panel(markdown_text: str, questions: list[dict], key: str,
                  expanded: bool = False) -> None:
    """Explanation (may contain live values) followed by self-check questions.

    Each question: ``{"q": str, "options": [str], "answer": int, "why": str}``.
    """
    with st.expander("💡 Explain: what am I looking at?", expanded=expanded):
        st.markdown(markdown_text)
        if not questions:
            return
        st.markdown("---\n**Check your understanding**")
        for i, q in enumerate(questions):
            choice = st.radio(f"**Q{i + 1}.** {q['q']}", q["options"], index=None,
                              key=f"{key}_q{i}")
            if choice is not None:
                if q["options"].index(choice) == q["answer"]:
                    st.success(f"Correct. {q['why']}")
                else:
                    st.error(f"Not quite. {q['why']}")


# ---------------------------------------------------------------- export
def export_panel(script: str, params_json: str, key: str, extra: list[tuple] | None = None):
    """Show the generated code and offer downloads.

    ``extra`` is a list of ``(label, bytes_or_str, filename, mime)``.
    """
    with st.expander("⬇️ Export code and parameters"):
        st.caption("The script reproduces exactly what you see, using the current parameters. "
                   "Run it from the repository root.")
        st.code(script, language="python")
        cols = st.columns(2 + len(extra or []))
        cols[0].download_button("Python script", script, f"{key}_export.py",
                                "text/x-python", key=f"{key}_dl_py")
        cols[1].download_button("Parameters (JSON)", params_json, f"{key}_params.json",
                                "application/json", key=f"{key}_dl_json")
        for i, (label, payload, fname, mime) in enumerate(extra or []):
            cols[2 + i].download_button(label, payload, fname, mime, key=f"{key}_dl_x{i}")


def selected_customdata(event) -> list[str]:
    """Extract ``customdata`` values from a ``st.plotly_chart`` selection event."""
    try:
        pts = event.selection.points  # type: ignore[attr-defined]
    except AttributeError:
        try:
            pts = event["selection"]["points"]
        except Exception:
            return []
    out = []
    for p in pts or []:
        cd = p.get("customdata") if isinstance(p, dict) else None
        if cd is None:
            continue
        out.append(cd[0] if isinstance(cd, (list, tuple)) else cd)
    return out
