"""Module 1 - Data loading: what is inside an EEG recording?

Exposes a single ``render()`` function, as required for the merged course app.
"""
from __future__ import annotations

import mne
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bci_common import data as D
from bci_common import export as X
from bci_common.plotting import class_color, head_map
from bci_common.ui import data_source_picker, explain_panel, export_panel, selected_customdata

from . import core

K = "dl"  # widget-key prefix, keeps this module's state separate from others
MODULE_NAME = "Data loading"


def _consume_selection(plot_key: str, last_key: str):
    """Return the new selection of a plotly chart, or None if unchanged."""
    sel = tuple(selected_customdata(st.session_state.get(plot_key)))
    if sel and sel != st.session_state.get(last_key):
        st.session_state[last_key] = sel
        return sel
    if not sel:
        st.session_state[last_key] = ()
    return None


def render() -> None:
    st.header("Data loading · what is inside an EEG recording?")
    st.caption("Load a recording, browse it, and see how sampling rate, units, channels and "
               "annotations shape what you can do with it later in the BCI pipeline.")

    raw, source = data_source_picker(K)
    if raw is None:
        st.info("Pick or upload a recording in the sidebar to begin.")
        return

    sf = raw.info["sfreq"]
    duration = raw.n_times / sf
    eeg_names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)] or raw.ch_names
    labels_all = sorted(set(raw.annotations.description))
    pos2d = D.channel_positions_2d(raw)

    # Reset per-recording state when the source changes
    src_sig = repr(sorted(source.items()))
    if st.session_state.get(f"{K}_src") != src_sig:
        st.session_state[f"{K}_src"] = src_sig
        st.session_state[f"{K}_picks"] = core.default_picks(raw)
        st.session_state[f"{K}_focus"] = next(
            (c for c in ("C3", "C4", "Cz") if c in st.session_state[f"{K}_picks"]),
            st.session_state[f"{K}_picks"][0])
        st.session_state[f"{K}_t0"] = 0.0
        st.session_state[f"{K}_labels"] = [l for l in labels_all if l != "blink"] or labels_all

    # ---- linked selections from the previous run (must run before widgets)
    sel = _consume_selection(f"{K}_map", f"{K}_map_last")
    if sel:
        st.session_state[f"{K}_focus"] = sel[0]
        picks = list(st.session_state[f"{K}_picks"])
        for ch in sel:
            if ch not in picks:
                picks.append(ch)
        st.session_state[f"{K}_picks"] = picks
    nav = _consume_selection(f"{K}_nav", f"{K}_nav_last")
    if nav:
        st.session_state[f"{K}_t0"] = float(nav[0])

    # ---- parameters
    sb = st.sidebar
    sb.markdown("#### Parameters")
    win = sb.select_slider("Window length (s)", [1, 2, 4, 5, 8, 10, 20, 30],
                           value=8 if duration >= 8 else 1, key=f"{K}_win")
    max_t0 = max(0.0, float(duration - win))
    st.session_state[f"{K}_t0"] = float(min(st.session_state.get(f"{K}_t0", 0.0), max_t0))
    t0 = sb.slider("Window start (s)", 0.0, max_t0, step=0.5, key=f"{K}_t0",
                   help="Or click an event in the navigator plot.")
    picks = sb.multiselect("Channels to display", eeg_names, key=f"{K}_picks",
                           help="Or click/box-select sensors on the head map.")
    if not picks:
        st.warning("Select at least one channel.")
        return
    if st.session_state.get(f"{K}_focus") not in picks:
        st.session_state[f"{K}_focus"] = picks[0]
    focus = sb.selectbox("Focus channel (spectrum)", picks, key=f"{K}_focus")
    spacing = sb.slider("Amplitude spacing (µV between traces)", 10, 400, 60, 10,
                        key=f"{K}_spacing")
    rs_options = ["original"] + [f for f in (250, 200, 128, 100, 80, 64, 60, 50, 40)
                                 if f < sf]
    rs = sb.selectbox("Resample to (Hz)", rs_options, key=f"{K}_rs",
                      format_func=lambda v: f"original ({sf:g} Hz)" if v == "original"
                      else f"{v} Hz")
    anti_alias = sb.checkbox("Anti-alias filter before resampling", True, key=f"{K}_aa",
                             help="Untick to see what goes wrong without it.")
    show_labels = sb.multiselect("Annotations to show", labels_all, key=f"{K}_labels")
    stat = sb.selectbox("Head map colour", list(core.CHANNEL_STATS),
                        format_func=core.CHANNEL_STATS.get, key=f"{K}_stat")

    # ---- computation
    times, win_data = core.get_window(raw, picks, t0, win)
    new_sf = sf if rs == "original" else float(rs)
    r_times, r_data, eff_sf = core.resample_window(times, win_data, sf, new_sf, anti_alias)
    map_chs = [c for c in eeg_names if c in pos2d] or eeg_names
    _, map_data = core.get_window(raw, map_chs, t0, win)
    stats = core.channel_stats(map_data, map_chs)
    anns = core.annotations_in_window(raw, t0, t0 + win, show_labels)
    dims = D.dimension_notation(raw)

    # ---- header metrics
    m = st.columns(5)
    m[0].metric("Channels C", dims["channels_C"])
    m[1].metric("Samples T", f"{dims['samples_T']:,}")
    m[2].metric("Sampling rate", f"{sf:g} Hz")
    m[3].metric("Duration", f"{duration:.1f} s")
    m[4].metric("Annotations", dims["n_annotations"])

    tab_sig, tab_anat = st.tabs(["📈 Signals", "🗂️ File anatomy"])

    with tab_sig:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(_trace_figure(times, win_data, r_times, r_data, picks, focus,
                                          spacing, anns, labels_all, rs != "original"),
                            key=f"{K}_ts")
        with c2:
            if len(pos2d) >= 3:
                vals = dict(zip(stats["channel"], stats[stat]))
                fig = head_map({c: pos2d[c] for c in map_chs}, vals,
                               title=f"{core.CHANNEL_STATS[stat]} in window",
                               colorbar_title="", highlight=set(picks) | {focus},
                               colorscale="RdBu_r" if stat == "zlogvar" else "Viridis",
                               symmetric=stat == "zlogvar")
                st.plotly_chart(fig, key=f"{K}_map", on_select="rerun",
                                selection_mode=("points", "box", "lasso"))
                st.caption("Click a sensor (or use the box-select tool in the plot toolbar) to add it to the display; "
                           "the first one becomes the focus channel.")
            else:
                st.info("No sensor positions in this file, so no head map. "
                        "Showing a bar chart instead.")
                fig = go.Figure(go.Bar(x=stats["channel"], y=stats[stat],
                                       customdata=stats["channel"]))
                fig.update_layout(height=380, title=core.CHANNEL_STATS[stat],
                                  margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, key=f"{K}_map", on_select="rerun")
            flagged = stats[stats["flag"] != ""]
            if len(flagged):
                st.warning("Unusual channels in this window (artifact or bad contact?): "
                           + ", ".join(f"{r.channel} ({r.flag})" for r in flagged.itertuples()))

        c3, c4 = st.columns([3, 2])
        with c3:
            fi = picks.index(focus)
            st.plotly_chart(_psd_figure(win_data[fi], sf, r_data[fi], eff_sf, focus,
                                        rs != "original", anti_alias),
                            key=f"{K}_psd")
        with c4:
            st.plotly_chart(_navigator_figure(raw, t0, win, labels_all, duration),
                            key=f"{K}_nav", on_select="rerun", selection_mode="points")
            st.caption("Click an event to jump the window to it.")

    with tab_anat:
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**Dimensions** (continuous data is a C × T matrix)")
            st.dataframe(pd.DataFrame(dims.items(), columns=["property", "value"])
                         .astype(str), hide_index=True)
            st.markdown("**Annotations / events**")
            st.dataframe(D.events_summary(raw), hide_index=True)
            if len(raw.annotations):
                ev, ev_id = mne.events_from_annotations(raw)
                st.markdown(f"**`mne.events_from_annotations` → events array** "
                            f"(shape {ev.shape}: sample, previous value, event id)")
                st.code(f"event_id = {ev_id}\n" + np.array2string(ev[:8]) +
                        ("\n..." if len(ev) > 8 else ""), language="python")
        with a2:
            st.markdown("**Channels**")
            st.dataframe(core.channel_table(raw), hide_index=True, height=420)
        with st.expander("Full `raw.info`"):
            st.text(str(raw.info))

    # ---- explain
    nyq = sf / 2
    alias50 = core.alias_frequency(50.0, eff_sf)
    n_win = len(times)
    rs_line = ""
    if rs != "original":
        rs_line = (f"\n\n**Resampling.** You reduced the rate to **{eff_sf:g} Hz**, so the new "
                   f"Nyquist frequency is **{eff_sf / 2:g} Hz**. ")
        if 50 > eff_sf / 2:
            rs_line += (f"Mains noise at 50 Hz is above it. "
                        + ("The anti-alias filter removed it before resampling, so it is "
                           "simply gone from the spectrum." if anti_alias else
                           f"With no anti-alias filter it folds back and appears at "
                           f"**{alias50:g} Hz** - a fake rhythm that was never in the brain. "
                           f"Look for the spike in the red spectrum."))
    explain_md = f"""
**The recording as a matrix.** This file holds **C = {dims['channels_C']} channels × T = {dims['samples_T']:,} samples**.
At **{sf:g} Hz** each second contains {sf:g} samples, so your {win} s window is
**{n_win} samples** per channel. The highest frequency the data can represent is the
**Nyquist frequency, {nyq:g} Hz** (half the sampling rate).

**Units.** MNE stores EEG in **volts**; the plots show **microvolts (µV)** because scalp EEG
is typically 10-100 µV. A channel reading 1000 µV or a perfectly flat line is almost always a
recording problem, not brain activity. The head map's *variance z-score* flags such channels.

**Annotations** mark what happened when (cue onsets, artefacts). Each has an *onset*, a
*duration* and a *label*. Later modules convert them with `mne.events_from_annotations` into
an **events array** of `[sample, 0, id]` rows, which is what epoching needs.

**Sensor positions (montage)** tell MNE where each electrode sits. They are required for head
maps and any spatial method (CSP, source analysis). {len(pos2d)} of {len(eeg_names)} EEG
channels here have positions.{rs_line}
"""
    if source.get("kind") == "synthetic":
        explain_md += ("\n\n**About the sample.** It is synthetic with a known ground truth: "
                       "alpha at O1/Oz/O2, mu/beta over C3/C4 that weakens on the opposite side "
                       "during hand imagery, eye blinks at Fp1/Fp2 and 50 Hz mains noise.")
    explain_panel(explain_md, _questions(sf), key=K)

    # ---- export
    params = {"picks": picks, "window_start_s": t0, "window_len_s": win,
              "resample_hz": None if rs == "original" else new_sf,
              "anti_alias": anti_alias, "annotation_labels": show_labels,
              "focus_channel": focus}
    body = f"""
print(raw)
print("C x T =", len(raw.ch_names), "x", raw.n_times, "| sfreq =", raw.info["sfreq"], "Hz")
print(raw.annotations)

picks = {picks!r}
tmin, tmax = {t0}, {t0 + win}
window = raw.copy().pick(picks).crop(tmin=tmin, tmax=min(tmax, raw.times[-1]))
"""
    if rs != "original":
        if anti_alias:
            body += f"window.resample({new_sf})  # low-pass + resample (no aliasing)\n"
        else:
            body += (f"# naive resampling WITHOUT anti-alias filter (demonstrates aliasing)\n"
                     f"new_t = np.arange(0, window.times[-1], 1 / {new_sf})\n"
                     f"data = np.stack([np.interp(new_t, window.times, ch) "
                     f"for ch in window.get_data()])\n"
                     f"window = mne.io.RawArray(data, mne.create_info(picks, "
                     f"{new_sf}, 'eeg'))\n")
    body += """
data_uV = window.get_data() * 1e6
print("window shape (channels, samples):", data_uV.shape)
np.savetxt("window_uV.csv", data_uV.T, delimiter=",", header=",".join(picks), comments="")

# Interactive browser (opens a window when run locally)
window.plot(scalings=dict(eeg=%g), block=True)
""" % (spacing * 1e-6 / 2)
    script = X.build_script(MODULE_NAME, source, body, params)
    csv = pd.DataFrame(r_data.T, columns=picks)
    csv.insert(0, "time_s", r_times)
    export_panel(script, X.params_json(MODULE_NAME, source, params), key=K,
                 extra=[("Window as CSV (µV)", csv.to_csv(index=False), "window_uV.csv",
                         "text/csv")])


# ---------------------------------------------------------------- figures
def _trace_figure(times, data, r_times, r_data, picks, focus, spacing, anns, labels_all,
                  resampled):
    fig = go.Figure()
    n = len(picks)
    offsets = (n - 1 - np.arange(n)) * spacing
    for i, ch in enumerate(picks):
        is_focus = ch == focus
        if resampled:
            fig.add_trace(go.Scatter(x=times, y=data[i] + offsets[i], mode="lines",
                                     line=dict(color="#cbd5e1", width=1),
                                     hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=r_times, y=r_data[i] + offsets[i], mode="lines", name=ch,
            line=dict(color="#f59e0b" if is_focus else "#1e293b", width=2 if is_focus else 1),
            customdata=r_data[i], hovertemplate=f"{ch}<br>%{{x:.3f}} s<br>%{{customdata:.1f}} µV"
                                                "<extra></extra>",
            showlegend=False))
    for row in anns.itertuples():
        col = class_color(labels_all.index(row.label)) if row.label in labels_all else "#64748b"
        fig.add_vrect(x0=row.onset, x1=row.onset + max(row.duration, 0.02), fillcolor=col,
                      opacity=0.12, line_width=0)
        fig.add_annotation(x=row.onset, y=1.0, yref="paper", text=row.label, showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(size=10, color=col))
    # scale bar
    fig.add_shape(type="line", x0=times[0], x1=times[0], y0=offsets[0] + spacing * 0.2,
                  y1=offsets[0] + spacing * 0.2 + spacing / 2, line=dict(width=3, color="#0f172a"))
    fig.add_annotation(x=times[0], y=offsets[0] + spacing * 0.45, text=f"{spacing / 2:g} µV",
                       showarrow=False, xanchor="left", xshift=4, font=dict(size=10))
    fig.update_layout(
        height=max(380, 34 * n + 80), margin=dict(l=10, r=10, t=30, b=40),
        xaxis_title="Time (s)",
        yaxis=dict(tickmode="array", tickvals=offsets, ticktext=picks, showgrid=False,
                   zeroline=False),
        hovermode="closest", title=dict(text="Raw traces", x=0, font=dict(size=14)))
    return fig


def _psd_figure(x, sf, xr, eff_sf, ch, resampled, anti_alias):
    f, p = core.window_psd(x[None], sf)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=10 * np.log10(p[0] + 1e-20), name=f"original ({sf:g} Hz)",
                             line=dict(color="#1e293b")))
    if resampled:
        fr, pr = core.window_psd(xr[None], eff_sf)
        fig.add_trace(go.Scatter(
            x=fr, y=10 * np.log10(pr[0] + 1e-20),
            name=f"resampled {eff_sf:g} Hz " + ("(anti-aliased)" if anti_alias else "(naive)"),
            line=dict(color="#059669" if anti_alias else "#dc2626", width=2)))
        fig.add_vline(x=eff_sf / 2, line_dash="dash", line_color="#64748b",
                      annotation_text=f"new Nyquist {eff_sf / 2:g} Hz")
    fig.add_vline(x=sf / 2, line_dash="dot", line_color="#94a3b8",
                  annotation_text="Nyquist", annotation_position="bottom right")
    for name, lo, hi, col in (("alpha/mu", 8, 13, "#bfdbfe"), ("beta", 13, 30, "#e2e8f0")):
        fig.add_vrect(x0=lo, x1=hi, fillcolor=col, opacity=0.45, line_width=0,
                      annotation_text=name, annotation_position="top",
                      annotation_font_size=10)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=40),
                      title=dict(text=f"Spectrum of {ch} in this window", x=0,
                                 font=dict(size=14)),
                      xaxis_title="Frequency (Hz)", yaxis_title="Power (dB µV²/Hz)",
                      legend=dict(orientation="h", y=-0.3))
    return fig


def _navigator_figure(raw, t0, win, labels_all, duration):
    ann = raw.annotations
    first = raw.first_time if ann.orig_time is not None else 0.0
    fig = go.Figure()
    for i, lab in enumerate(labels_all):
        mask = ann.description == lab
        on = ann.onset[mask] - first
        fig.add_trace(go.Scatter(x=on, y=[lab] * len(on), mode="markers", name=lab,
                                 marker=dict(color=class_color(i), size=8, symbol="line-ns-open",
                                             line=dict(width=2, color=class_color(i))),
                                 customdata=on, hovertemplate="%{y} @ %{x:.1f} s<extra></extra>"))
    fig.add_vrect(x0=t0, x1=t0 + win, fillcolor="#f59e0b", opacity=0.25, line_width=0)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=40), showlegend=False,
                      title=dict(text="Navigator: whole recording", x=0, font=dict(size=14)),
                      xaxis=dict(title="Time (s)", range=[0, duration]),
                      clickmode="event+select")
    if not labels_all:
        fig.add_annotation(text="No annotations in this file", x=duration / 2, y=0,
                           showarrow=False)
    return fig


def _questions(sf):
    return [
        {"q": f"This recording is sampled at {sf:g} Hz. How many samples does one "
              f"channel contain in a 4-second window?",
         "options": [f"{sf / 4:g}", f"{sf * 4:g}", f"{sf + 4:g}", "Depends on the number of channels"],
         "answer": 1,
         "why": f"Samples = duration × sampling rate = 4 × {sf:g} = {sf * 4:g}. "
                "The channel count changes C, not T."},
        {"q": "You decimate data with 50 Hz mains noise to 60 Hz by keeping every k-th sample, "
              "with no filtering. Where does the 50 Hz noise appear?",
         "options": ["It disappears", "At 50 Hz", "At 10 Hz", "At 30 Hz"],
         "answer": 2,
         "why": "The new Nyquist is 30 Hz. 50 Hz folds back to |60 - 50| = 10 Hz, right in the "
                "alpha band, where it could be mistaken for a brain rhythm. Try it with the "
                "resample controls and untick the anti-alias filter."},
        {"q": "Two large (~150 µV), slow (~0.3 s) deflections appear only on Fp1 and Fp2. "
              "What are they most likely to be?",
         "options": ["P300 responses", "Eye blinks", "Alpha bursts", "Mains interference"],
         "answer": 1,
         "why": "Blinks are large, slow and strongest at the most frontal electrodes, close to "
                "the eyes. Artifact handling (a later module) deals with them."},
        {"q": "Which information do you need from the file before you can cut it into "
              "trials (epochs)?",
         "options": ["The sensor positions", "The event onsets and their labels",
                     "The patient's name", "The amplifier's filter settings"],
         "answer": 1,
         "why": "Epoching cuts windows around event onsets and groups them by label. "
                "Positions matter for spatial methods, not for epoching."},
        {"q": "Why is it a red flag if a channel's variance z-score is far below -3?",
         "options": ["It is the reference channel or flat/disconnected",
                     "It carries the strongest brain signal",
                     "It was sampled at a lower rate", "It contains eye blinks"],
         "answer": 0,
         "why": "Near-zero variance means the electrode recorded almost nothing: it is flat, "
                "bridged, disconnected or is the reference itself."},
    ]
