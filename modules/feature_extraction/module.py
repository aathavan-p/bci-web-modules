"""Module 7 - Feature extraction: from EEG trials to a feature matrix.

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

K = "fe"
MODULE_NAME = "Feature extraction"
SENSORIMOTOR = ["FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                "CP3", "CP1", "CPz", "CP2", "CP4"]


@st.cache_resource(show_spinner="Cutting trials...", max_entries=8)
def _epochs(src_sig: str, classes: tuple, tmin: float, tmax: float, picks: tuple,
            l_freq, h_freq, _raw) -> mne.Epochs:
    return core.make_epochs(_raw, list(classes), tmin, tmax, list(picks), l_freq, h_freq)


def _consume(plot_key: str, last_key: str):
    sel = tuple(selected_customdata(st.session_state.get(plot_key)))
    if sel and sel != st.session_state.get(last_key):
        st.session_state[last_key] = sel
        return sel
    if not sel:
        st.session_state[last_key] = ()
    return None


def render() -> None:
    st.header("Feature extraction · turning trials into numbers a classifier can use")
    st.caption("Each trial is a channels × samples block. Feature extraction compresses it into a "
               "short vector that keeps the information that differs between classes.")

    raw, source = data_source_picker(K)
    if raw is None:
        st.info("Pick or upload a recording in the sidebar to begin.")
        return
    labels = sorted(set(raw.annotations.description))
    if len(labels) < 2:
        st.error("Feature extraction needs a recording with at least two labelled event types "
                 "(annotations). This file has: " + (", ".join(labels) or "none") + ". "
                 "Try the sample, or PhysioNet run 4 (imagined left/right fist).")
        return

    sf = raw.info["sfreq"]
    eeg = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
    pos2d = D.channel_positions_2d(raw)
    task_mask = np.isin(raw.annotations.description, core.default_classes(labels))
    trial_len = (float(np.median(raw.annotations.duration[task_mask]))
                 if task_mask.any() else 4.0)
    trial_len = trial_len if trial_len >= 1.0 else 4.0

    src_sig = repr(sorted(source.items()))
    if st.session_state.get(f"{K}_src") != src_sig:
        st.session_state[f"{K}_src"] = src_sig
        st.session_state[f"{K}_classes"] = core.default_classes(labels)
        st.session_state[f"{K}_focus"] = next((c for c in ("C3", "C4", "Cz") if c in eeg), eeg[0])
        st.session_state[f"{K}_trial"] = None
        st.session_state[f"{K}_xfeat"] = None
        st.session_state[f"{K}_yfeat"] = None

    # linked selections from the previous run
    s = _consume(f"{K}_map", f"{K}_map_last")
    if s:
        st.session_state[f"{K}_focus"] = s[0]
        st.session_state[f"{K}_pending_x_channel"] = s[0]
    s = _consume(f"{K}_scatter", f"{K}_scatter_last")
    if s:
        st.session_state[f"{K}_trial"] = int(s[0])
    s = _consume(f"{K}_bars", f"{K}_bars_last")
    if s:
        st.session_state[f"{K}_pending_x_feature"] = s[0]

    # ---------------------------------------------------------- parameters
    sb = st.sidebar
    sb.markdown("#### Trials")
    classes = sb.multiselect("Classes to compare", labels, key=f"{K}_classes")
    t_hi = round(max(trial_len, 1.0) + 1.0, 1)
    tmin, tmax = sb.slider("Epoch window relative to cue (s)", -2.0, t_hi,
                           (0.5, min(3.5, t_hi)), 0.1, key=f"{K}_twin",
                           help="Try a window that ends before 0 s - there is no task "
                                "information there.")
    ch_mode = sb.radio("Channels", ["All EEG", "Sensorimotor strip", "Custom"],
                       key=f"{K}_chmode", horizontal=True)
    if ch_mode == "All EEG":
        picks = eeg
    elif ch_mode == "Sensorimotor strip":
        picks = [c for c in SENSORIMOTOR if c in eeg] or eeg
    else:
        picks = sb.multiselect("Pick channels", eeg, default=[c for c in SENSORIMOTOR if c in eeg][:6]
                               or eeg[:6], key=f"{K}_custom")
    sb.markdown("#### Feature method")
    method = sb.radio("Method", list(core.METHODS), format_func=core.METHODS.get,
                      key=f"{K}_method")
    preset = sb.selectbox("Band preset", ["custom"] + list(core.BANDS), index=2,
                          key=f"{K}_preset")
    if preset != "custom" and st.session_state.get(f"{K}_last_preset") != preset:
        st.session_state[f"{K}_band"] = tuple(float(b) for b in core.BANDS[preset])
    st.session_state[f"{K}_last_preset"] = preset
    st.session_state.setdefault(f"{K}_band", (8.0, 13.0))
    band = sb.slider("Frequency band (Hz)", 1.0, float(min(45.0, sf / 2 - 1)), step=0.5,
                     key=f"{K}_band")
    welch_sec, relative, n_csp, hj = 1.0, False, 4, "activity"
    if method == "bandpower":
        welch_sec = sb.select_slider("Welch segment length (s)", [0.25, 0.5, 1.0, 2.0],
                                     value=1.0, key=f"{K}_welch",
                                     help="Longer segments -> finer frequency resolution "
                                          "(1/length Hz) but noisier estimates.")
        relative = sb.checkbox("Relative power (÷ total 1-45 Hz)", False, key=f"{K}_rel")
    elif method == "csp":
        n_csp = sb.slider("CSP components", 2, min(10, max(2, len(picks))), 4, 2,
                          key=f"{K}_ncsp")
    else:
        hj = sb.selectbox("Hjorth parameter", core.HJORTH_PARAMS, key=f"{K}_hj")
    log = sb.checkbox("Log-transform features", True, key=f"{K}_log",
                      help="Power is skewed; the log makes it closer to Gaussian, which "
                           "linear classifiers like LDA assume.")

    # ---------------------------------------------------------- checks
    if len(classes) < 2:
        st.warning("Choose at least two classes.")
        return
    if method == "csp" and len(classes) != 2:
        st.warning("CSP here is the classic two-class version - choose exactly two classes.")
        return
    if len(picks) < (2 if method == "csp" else 1):
        st.warning("Choose more channels.")
        return
    if tmax - tmin < 0.5:
        st.warning("Use an epoch window of at least 0.5 s.")
        return

    # ---------------------------------------------------------- compute
    try:
        ep_broad = _epochs(src_sig, tuple(classes), tmin, tmax, tuple(picks), None, None, raw)
        if method == "bandpower":
            ep_feat = ep_broad
        else:
            ep_feat = _epochs(src_sig, tuple(classes), tmin, tmax, tuple(picks),
                              band[0], band[1], raw)
        kw = dict(band=band, welch_sec=welch_sec, relative=relative, log=log, n_csp=n_csp,
                  hjorth_param=hj)
        res = core.extract(ep_feat, method, **kw)
        _, freqs, psd = core.band_power(ep_broad.get_data(copy=True), sf, band, welch_sec)
        accs = core.cv_accuracy(ep_feat, method, **kw)
    except ValueError as exc:
        st.error(str(exc))
        return

    y = res.y
    inv_id = {v: k for k, v in ep_feat.event_id.items()}
    cls_codes = sorted(np.unique(y))
    n_per = {inv_id[c]: int((y == c).sum()) for c in cls_codes}
    fisher = core.fisher_score(res.X, y)
    order = np.argsort(fisher)[::-1]

    # resolve scatter axes (respect clicks on map / bars)
    names = res.feature_names
    pend_ch = st.session_state.pop(f"{K}_pending_x_channel", None)
    pend_f = st.session_state.pop(f"{K}_pending_x_feature", None)
    if pend_ch and pend_ch in res.channel_of_feature:
        st.session_state[f"{K}_xfeat"] = names[res.channel_of_feature.index(pend_ch)]
    if pend_f in names:
        st.session_state[f"{K}_xfeat"] = pend_f
    if st.session_state.get(f"{K}_xfeat") not in names:
        st.session_state[f"{K}_xfeat"] = names[order[0]]
    if st.session_state.get(f"{K}_yfeat") not in names:
        st.session_state[f"{K}_yfeat"] = names[order[1] if len(order) > 1 else order[0]]
    if st.session_state.get(f"{K}_focus") not in picks:
        st.session_state[f"{K}_focus"] = picks[0]

    sb.markdown("#### Views")
    focus = sb.selectbox("Focus channel (spectrum)", picks, key=f"{K}_focus")
    xf = sb.selectbox("Scatter x-axis feature", names, key=f"{K}_xfeat")
    yf = sb.selectbox("Scatter y-axis feature", names, key=f"{K}_yfeat")
    trial = st.session_state.get(f"{K}_trial")
    if trial is not None and trial >= len(y):
        trial = None

    # ---------------------------------------------------------- metrics
    n_tr = len(y)
    thr = core.chance_threshold(n_tr, len(cls_codes))
    acc = float(accs.mean())
    m = st.columns(5)
    m[0].metric("Trials", n_tr, help=", ".join(f"{k}: {v}" for k, v in n_per.items()))
    m[1].metric("Trial size (C × T)", f"{len(picks)} × {ep_feat.get_data(copy=False).shape[-1]}")
    m[2].metric("Features per trial", res.X.shape[1])
    m[3].metric(f"CV accuracy ({len(accs)}-fold LDA)", f"{acc:.0%} ± {accs.std():.0%}",
                help="Mean ± SD over folds. Extractor, scaler and LDA are re-fitted in "
                     "every training fold.")
    m[4].metric("Chance / significant at p<.05", f"{1 / len(cls_codes):.0%} / {thr:.0%}")
    if acc < thr:
        st.info(f"Accuracy {acc:.0%} is **not** above the {thr:.0%} needed to beat chance with "
                f"{n_tr} trials. These features carry little class information.")

    # ---------------------------------------------------------- plots
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(_psd_fig(freqs, psd, y, picks.index(focus), focus, band, inv_id, trial),
                        key=f"{K}_psd")
    with c2:
        if len(pos2d) >= 3 and all(p in pos2d for p in picks):
            if method == "csp":
                comp = st.selectbox("CSP component shown", range(n_csp),
                                    format_func=lambda i: f"CSP{i + 1} (Fisher "
                                                          f"{fisher[i]:.2f})",
                                    key=f"{K}_comp")
                vals = dict(zip(picks, res.extra["patterns"][comp]))
                fig = head_map({c: pos2d[c] for c in picks}, vals,
                               title=f"Spatial pattern of CSP{comp + 1}", symmetric=True,
                               highlight={focus})
            else:
                per_ch = dict(zip(res.channel_of_feature, fisher))
                fig = head_map({c: pos2d[c] for c in picks}, per_ch,
                               title="Class separability (Fisher score) per channel",
                               colorscale="Viridis", highlight={focus})
            st.plotly_chart(fig, key=f"{K}_map", on_select="rerun",
                            selection_mode=("points",))
            st.caption("Click a sensor to see its spectrum"
                       + ("" if method == "csp" else " and plot its feature on the x-axis") + ".")
        else:
            st.info("No sensor positions for the selected channels, so no head map.")

    c3, c4 = st.columns([3, 2])
    with c3:
        st.plotly_chart(_scatter_fig(res.X, y, names, xf, yf, inv_id, trial),
                        key=f"{K}_scatter", on_select="rerun", selection_mode=("points",))
        st.caption("Each dot is one trial. Click one to overlay its spectrum above.")
    with c4:
        st.plotly_chart(_bars_fig(names, fisher, order, xf), key=f"{K}_bars",
                        on_select="rerun", selection_mode=("points",))
        st.caption("Click a bar to put that feature on the scatter x-axis.")

    with st.expander("🔎 Feature matrix X (what the classifier receives)"):
        df = pd.DataFrame(res.X, columns=names)
        df.insert(0, "label", [inv_id[v] for v in y])
        st.dataframe(df.round(4), height=260)
        st.caption(f"Shape {res.X.shape} = (trials, features). Displayed features were computed "
                   "on all trials for visualisation; the accuracy above re-fits everything "
                   "inside each fold.")

    with st.expander("📊 Band sweep: which frequencies carry the information?"):
        st.caption("Runs the same leak-free CV for a sliding 4 Hz band (takes a few seconds).")
        if st.button("Run band sweep", key=f"{K}_sweep_btn"):
            st.session_state[f"{K}_sweep"] = _band_sweep(src_sig, classes, tmin, tmax, picks,
                                                         method, kw, raw, ep_broad)
        sw = st.session_state.get(f"{K}_sweep")
        if sw is not None:
            fig = go.Figure(go.Scatter(x=sw["centre"], y=sw["acc"], mode="lines+markers"))
            fig.add_hline(y=thr, line_dash="dash", annotation_text="p<.05 chance threshold")
            fig.add_vrect(x0=band[0], x1=band[1], fillcolor="#f59e0b", opacity=0.15,
                          line_width=0)
            fig.update_layout(height=300, xaxis_title="Band centre (Hz)",
                              yaxis_title="CV accuracy", yaxis_range=[0, 1.05],
                              margin=dict(l=10, r=10, t=20, b=40))
            st.plotly_chart(fig, key=f"{K}_sweep_fig")

    # ---------------------------------------------------------- explain
    explain_panel(_explain_md(method, sf, band, welch_sec, freqs, picks, res, tmin, tmax,
                              acc, thr, n_tr, relative, log, source, classes, ep_feat),
                  _questions(n_tr), key=K)

    # ---------------------------------------------------------- export
    params = {"classes": classes, "tmin": tmin, "tmax": tmax, "picks": picks,
              "method": method, "band_hz": list(band), "welch_sec": welch_sec,
              "relative": relative, "log": log, "n_csp": n_csp, "hjorth_param": hj}
    lf, hf = (None, None) if method == "bandpower" else (band[0], band[1])
    body = f"""
from modules.feature_extraction import core

epochs = core.make_epochs(raw, classes={classes!r}, tmin={tmin}, tmax={tmax},
                          picks={picks!r}, l_freq={lf}, h_freq={hf})
print(epochs)

kw = dict(band={tuple(band)}, welch_sec={welch_sec}, relative={relative}, log={log},
          n_csp={n_csp}, hjorth_param={hj!r})
res = core.extract(epochs, {method!r}, **kw)   # for inspection / plotting only
print("Feature matrix X:", res.X.shape)

# Leak-free estimate: extractor + scaler + LDA re-fitted inside every fold
accs = core.cv_accuracy(epochs, {method!r}, **kw)
thr = core.chance_threshold(len(res.y), {len(cls_codes)})
print(f"CV accuracy {{accs.mean():.3f}} +/- {{accs.std():.3f}}  (chance threshold p<.05: {{thr:.3f}})")

fisher = core.fisher_score(res.X, res.y)
for i in np.argsort(fisher)[::-1][:5]:
    print(f"  {{res.feature_names[i]:<20s}} Fisher = {{fisher[i]:.3f}}")

import pandas as pd
inv = {{v: k for k, v in epochs.event_id.items()}}
df = pd.DataFrame(res.X, columns=res.feature_names)
df.insert(0, "label", [inv[v] for v in res.y])
df.to_csv("features.csv", index=False)
print("saved features.csv")
"""
    script = X.build_script(MODULE_NAME, source, body, params)
    feat_df = pd.DataFrame(res.X, columns=names)
    feat_df.insert(0, "label", [inv_id[v] for v in y])
    export_panel(script, X.params_json(MODULE_NAME, source, params), key=K,
                 extra=[("Features (CSV)", feat_df.to_csv(index=False), "features.csv",
                         "text/csv")])


# ---------------------------------------------------------------- helpers
def _band_sweep(src_sig, classes, tmin, tmax, picks, method, kw, raw, ep_broad):
    nyq = raw.info["sfreq"] / 2
    centres = np.arange(4, min(42, nyq - 3) + 0.1, 2.0)
    accs = []
    prog = st.progress(0.0)
    for i, c in enumerate(centres):
        b = (float(c - 2), float(c + 2))
        kw2 = dict(kw, band=b)
        ep = ep_broad if method == "bandpower" else _epochs(
            src_sig, tuple(classes), tmin, tmax, tuple(picks), b[0], b[1], raw)
        try:
            accs.append(float(core.cv_accuracy(ep, method, **kw2).mean()))
        except ValueError:
            accs.append(np.nan)
        prog.progress((i + 1) / len(centres))
    prog.empty()
    return {"centre": centres, "acc": accs}


def _psd_fig(freqs, psd, y, ci, ch, band, inv_id, trial):
    fig = go.Figure()
    stats = core.class_mean_psd(psd, y, ci)
    for i, (code, (mu, sem)) in enumerate(stats.items()):
        col = class_color(i)
        fig.add_trace(go.Scatter(x=np.r_[freqs, freqs[::-1]], y=np.r_[mu + sem, (mu - sem)[::-1]],
                                 fill="toself", fillcolor=col, opacity=0.15, line_width=0,
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=freqs, y=mu, name=f"{inv_id[code]} (mean ± SEM)",
                                 line=dict(color=col, width=2)))
    if trial is not None:
        fig.add_trace(go.Scatter(x=freqs, y=10 * np.log10(psd[trial, ci] * 1e12 + 1e-30),
                                 name=f"trial #{trial} ({inv_id[y[trial]]})",
                                 line=dict(color="#0f172a", width=1, dash="dot")))
    fig.add_vrect(x0=band[0], x1=band[1], fillcolor="#f59e0b", opacity=0.18, line_width=0,
                  annotation_text="feature band", annotation_position="top left")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=40),
                      title=dict(text=f"Spectrum at {ch}, by class", x=0, font=dict(size=14)),
                      xaxis=dict(title="Frequency (Hz)", range=[1, min(45, freqs[-1])]),
                      yaxis_title="Power (dB µV²/Hz)", legend=dict(orientation="h", y=-0.25))
    return fig


def _scatter_fig(F, y, names, xf, yf, inv_id, trial):
    xi, yi = names.index(xf), names.index(yf)
    fig = go.Figure()
    for i, code in enumerate(sorted(np.unique(y))):
        idx = np.where(y == code)[0]
        fig.add_trace(go.Scatter(
            x=F[idx, xi], y=F[idx, yi], mode="markers", name=inv_id[code],
            marker=dict(color=class_color(i), size=9, opacity=0.8,
                        line=dict(color="white", width=1)),
            customdata=idx, hovertemplate="trial #%{customdata}<br>x=%{x:.3g}<br>y=%{y:.3g}"
                                          "<extra>" + inv_id[code] + "</extra>"))
    if trial is not None:
        fig.add_trace(go.Scatter(x=[F[trial, xi]], y=[F[trial, yi]], mode="markers",
                                 marker=dict(size=18, color="rgba(0,0,0,0)",
                                             line=dict(color="#0f172a", width=2)),
                                 hoverinfo="skip", showlegend=False))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=40),
                      title=dict(text="Feature space (one dot = one trial)", x=0,
                                 font=dict(size=14)),
                      xaxis_title=xf, yaxis_title=yf, clickmode="event+select",
                      legend=dict(orientation="h", y=-0.25))
    return fig


def _bars_fig(names, fisher, order, xf):
    top = order[:20]
    fig = go.Figure(go.Bar(
        x=[fisher[i] for i in top][::-1], y=[names[i] for i in top][::-1], orientation="h",
        customdata=[names[i] for i in top][::-1],
        marker_color=["#f59e0b" if names[i] == xf else "#2563eb" for i in top][::-1],
        hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=40),
                      title=dict(text="Most discriminative features (Fisher score)", x=0,
                                 font=dict(size=14)),
                      xaxis_title="Fisher score", clickmode="event+select",
                      yaxis=dict(tickmode="linear", dtick=1, tickfont=dict(size=10)))
    return fig


def _explain_md(method, sf, band, welch_sec, freqs, picks, res, tmin, tmax, acc, thr,
                n_tr, relative, log, source, classes, ep):
    n_t = ep.get_data(copy=False).shape[-1]
    res_hz = freqs[1] - freqs[0]
    n_bins = int(((freqs >= band[0]) & (freqs <= band[1])).sum())
    txt = (f"**From trial to vector.** Each trial is **{len(picks)} channels × {n_t} samples** "
           f"= {len(picks) * n_t:,} numbers. The method below reduces it to "
           f"**{res.X.shape[1]} features**, giving a feature matrix **X** of shape "
           f"{res.X.shape} (trials × features).\n\n")
    if method == "bandpower":
        txt += (f"**Band power.** For each channel the power spectrum is estimated with Welch's "
                f"method: the trial is cut into {welch_sec:g} s segments, each is Fourier-"
                f"transformed and the results are averaged. Frequency resolution is "
                f"1 / {welch_sec:g} s = **{res_hz:.2g} Hz**, so your {band[0]:g}-{band[1]:g} Hz "
                f"band covers **{n_bins} bins**; their mean is the feature"
                + (" (divided by total 1-45 Hz power, which cancels differences in overall "
                   "signal level between trials)" if relative else "") + ".\n\n")
    elif method == "csp":
        txt += ("**Common Spatial Patterns.** The data are first band-pass filtered to "
                f"{band[0]:g}-{band[1]:g} Hz. CSP then learns spatial filters (weighted sums of "
                "channels) whose output variance is as large as possible for one class and as "
                "small as possible for the other. The log of that variance is the feature. "
                "Because CSP is **trained with the labels**, it must be fitted on training "
                "trials only - the accuracy here re-fits CSP inside every fold. The head map "
                "shows the *pattern* (where the source projects on the scalp), which is "
                "interpretable; the filter itself is not.\n\n")
    else:
        txt += ("**Hjorth parameters** describe a signal in the time domain: *activity* is the "
                "variance (= total power), *mobility* ≈ the mean frequency, *complexity* ≈ how "
                "much the waveform deviates from a pure sine. Computed here on data band-passed "
                f"to {band[0]:g}-{band[1]:g} Hz. They are cheap and need no Fourier transform, "
                "which suits real-time BCIs.\n\n")
    if log:
        txt += ("**Log transform.** Power values are positive and right-skewed; the log makes "
                "class distributions closer to Gaussian, which LDA assumes.\n\n")
    txt += (f"**Is this any good?** The Fisher score ranks each feature by between-class vs "
            f"within-class spread. The CV accuracy (**{acc:.0%}**) comes from a simple LDA with "
            f"everything re-fitted inside each fold. With {n_tr} trials you need **{thr:.0%}** to "
            f"beat chance at p<.05 - not just {1 / len(classes):.0%}.")
    if tmax <= 0:
        txt += ("\n\n⚠️ **Your window ends before the cue.** The subject had not been told what "
                "to imagine yet, so any accuracy above chance would be a bug or leakage.")
    if source.get("kind") == "synthetic" and set(classes) == {"left_hand", "right_hand"}:
        txt += ("\n\n**Ground truth for the sample.** Left-hand imagery weakens mu (~11 Hz) and "
                "beta (~22 Hz) over the *right* hemisphere (C4) and vice versa, starting ~0.5 s "
                "after the cue. Good features should therefore point at C3/C4 in 8-30 Hz. Try "
                "30-45 Hz or a pre-cue window and watch the accuracy fall to chance.")
    return txt


def _questions(n_tr):
    thr = core.chance_threshold(n_tr, 2)
    return [
        {"q": "You set the Welch segment length to 0.5 s. What is the frequency resolution "
              "of the spectrum?",
         "options": ["0.5 Hz", "1 Hz", "2 Hz", "It depends on the sampling rate"],
         "answer": 2,
         "why": "Resolution = 1 / segment length = 1 / 0.5 s = 2 Hz. The sampling rate sets the "
                "highest frequency, not the spacing between bins."},
        {"q": "During imagined LEFT-hand movement, where do you expect mu-band power to drop?",
         "options": ["Over C3 (left hemisphere)", "Over C4 (right hemisphere)",
                     "Over Oz", "Equally everywhere"],
         "answer": 1,
         "why": "Motor cortex is organised contralaterally: the right hemisphere (C4) controls "
                "the left hand, so that is where event-related desynchronisation appears."},
        {"q": "Why must CSP be fitted inside each cross-validation training fold?",
         "options": ["It is too slow to fit once",
                     "CSP uses class labels, so fitting on all trials leaks test information",
                     "MNE requires it", "Otherwise the features are not log-transformed"],
         "answer": 1,
         "why": "CSP is supervised. Fitting it on test trials lets their labels shape the "
                "features, which inflates accuracy. Band power has no fitted parameters, so it "
                "cannot leak that way (but the scaler still can)."},
        {"q": f"With {n_tr} trials in two balanced classes, you get 60 % accuracy. "
              f"Is that significantly above chance (p<.05)?",
         "options": ["Yes, anything above 50 % is", f"No, you need about {thr:.0%}",
                     "Only if the classes are unbalanced", "Cannot be decided"],
         "answer": 1 if thr > 0.6 else 0,
         "why": f"With few trials, chance performance varies a lot. A binomial test gives a "
                f"threshold of {thr:.0%} for {n_tr} trials (Combrisson & Jerbi, 2015)."},
        {"q": "You use 26 channels and band power in one band. How many features per trial?",
         "options": ["1", "26", "26 × number of samples", "4"],
         "answer": 1,
         "why": "One band-power value per channel per band: 26 × 1 = 26. Adding a second band "
                "would double it."},
    ]
