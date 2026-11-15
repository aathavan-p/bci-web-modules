"""Plotly figures shared across modules (no Streamlit imports)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import RBFInterpolator

CLASS_COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]


def class_color(i: int) -> str:
    return CLASS_COLORS[i % len(CLASS_COLORS)]


def head_map(
    pos2d: dict[str, tuple[float, float]],
    values: dict[str, float] | None = None,
    title: str = "",
    colorbar_title: str = "",
    highlight: set[str] | None = None,
    colorscale: str = "RdBu_r",
    symmetric: bool = False,
    interpolate: bool = True,
    height: int = 380,
) -> go.Figure:
    """Top-down sensor map (nose up). With ``values`` the scalp is coloured by
    interpolation between sensors. Sensors are a clickable scatter trace whose
    ``customdata`` is the channel name, so Streamlit selections can link back
    to other plots."""
    names = list(pos2d)
    xy = np.array([pos2d[n] for n in names]) if names else np.zeros((0, 2))
    radius = float(np.max(np.linalg.norm(xy, axis=1)) * 1.08) if len(xy) else 1.0
    fig = go.Figure()

    vals = None
    if values:
        vals = np.array([values.get(n, np.nan) for n in names], dtype=float)
        ok = ~np.isnan(vals)
        if symmetric:
            m = float(np.nanmax(np.abs(vals))) or 1.0
            zmin, zmax = -m, m
        else:
            zmin, zmax = float(np.nanmin(vals)), float(np.nanmax(vals))
            if zmin == zmax:
                zmax = zmin + 1e-12
        if interpolate and ok.sum() >= 4:
            g = np.linspace(-radius, radius, 80)
            gx, gy = np.meshgrid(g, g)
            interp = RBFInterpolator(xy[ok], vals[ok], kernel="thin_plate_spline",
                                     smoothing=1e-3)
            gz = interp(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
            gz[np.hypot(gx, gy) > radius] = np.nan
            fig.add_trace(go.Heatmap(
                x=g, y=g, z=gz, zmin=zmin, zmax=zmax, colorscale=colorscale,
                colorbar=dict(title=colorbar_title, thickness=12),
                hoverinfo="skip", zsmooth="best"))

    # Head outline, nose, ears
    th = np.linspace(0, 2 * np.pi, 200)
    line = dict(color="#334155", width=2)
    fig.add_trace(go.Scatter(x=radius * np.cos(th), y=radius * np.sin(th), mode="lines",
                             line=line, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[-0.12 * radius, 0, 0.12 * radius],
                             y=[0.99 * radius, 1.12 * radius, 0.99 * radius],
                             mode="lines", line=line, hoverinfo="skip", showlegend=False))
    for s in (-1, 1):
        ey = np.linspace(-0.15, 0.15, 20) * radius
        fig.add_trace(go.Scatter(x=s * (radius + 0.04 * radius * np.cos(ey / radius * 10)),
                                 y=ey, mode="lines", line=line, hoverinfo="skip",
                                 showlegend=False))

    highlight = highlight or set()
    if len(names):
        hover = [f"{n}" + (f"<br>{values[n]:.3g}" if values and n in values else "")
                 for n in names]
        fig.add_trace(go.Scatter(
            x=xy[:, 0], y=xy[:, 1], mode="markers+text", text=names,
            textposition="top center", textfont=dict(size=9, color="#0f172a"),
            customdata=names, hovertext=hover, hoverinfo="text",
            marker=dict(size=[13 if n in highlight else 8 for n in names],
                        color=["#f59e0b" if n in highlight else "#0f172a" for n in names],
                        line=dict(color="white", width=1)),
            showlegend=False, name="sensors"))

    lim = radius * 1.2
    fig.update_layout(
        title=title, height=height, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False, range=[-lim, lim], scaleanchor="y"),
        yaxis=dict(visible=False, range=[-lim, lim]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        clickmode="event+select", dragmode=False,
    )
    return fig
