"""
Plotly filtration plotting helpers shared by forest classes.

The public entry points are ``plot_at_filtration_plotly`` and
``plot_filtration_interactive``. They expect a forest-like object exposing the
same small interface used by the Matplotlib helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import numpy as np


def _require_plotly():
    """Import Plotly graph objects or raise an installation hint."""
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise RuntimeError(
            "Plotly support requires the optional dependency 'plotly'. "
            "Install it with `pip install \".[plotly]\"` (local repo) or "
            "`pip install persforest[plotly]`."
        ) from e
    return go

def _plotly_traces_for_snapshot_2d(
    forest,
    snapshot: Dict[str, Any],
    color_map: Dict[object, str],
    show_cycles: bool = True,
    fill_triangles: bool = True,
    signed: bool = False,
    min_bar_length: float = 0.0,
    vertex_size: float = 3.0,
    linewidth_filt: float = 1.0,
    linewidth_cycle: float = 3.0,
    max_cycle_slots: Optional[int] = None,
) -> List[Any]:
    """
    Build Plotly traces for a 2D filtration snapshot.

    Parameters
    ----------
    forest : object
        Forest-like object exposing active bar/cycle queries and point-cloud
        coordinates.
    snapshot : dict
        Output of _complex_snapshot_at_filtration.
    color_map : dict[PFBar, str]
        Bar-to-color mapping.
    show_cycles : bool
        If True, include active cycle traces.
    fill_triangles : bool
        If True, fill present triangles.
    signed : bool
        If False, cancel opposite-oriented duplicate edges before
        rendering cycles.
    min_bar_length : float
        Only include active cycles for bars with lifespan at least this
        value.
    vertex_size : float
        Plotly marker size for point-cloud vertices.
    linewidth_filt : float
        Line width for complex edges.
    linewidth_cycle : float
        Line width for cycle edges.
    max_cycle_slots : int | None
        Fixed number of cycle trace slots. When provided, empty traces are
        added so animation frames keep a stable trace layout.

    Returns
    -------
    list
        List of Plotly traces.
    """
    go = _require_plotly()
    pts = snapshot["points"]
    filt_val = snapshot["filt_val"]
    edges = snapshot["edges"]
    triangles = snapshot["triangles"]

    traces: List[Any] = []

    # Filled triangles
    x_tri = []
    y_tri = []
    if fill_triangles and triangles:
        for tri in triangles:
            tri_pts = pts[list(tri)]
            x_tri.extend([tri_pts[0, 0], tri_pts[1, 0], tri_pts[2, 0], tri_pts[0, 0], None])
            y_tri.extend([tri_pts[0, 1], tri_pts[1, 1], tri_pts[2, 1], tri_pts[0, 1], None])

    traces.append(
        go.Scatter(
            x=x_tri,
            y=y_tri,
            mode="lines",
            fill="toself",
            line=dict(width=0.5, color="rgba(120,120,120,0.25)"),
            fillcolor="rgba(31,119,180,0.18)",
            hoverinfo="skip",
            name="triangles",
            showlegend=True,
        )
    )

    # Complex edges
    x_edge = []
    y_edge = []
    if edges:
        for e in edges:
            p = pts[list(e)]
            x_edge.extend([p[0, 0], p[1, 0], None])
            y_edge.extend([p[0, 1], p[1, 1], None])

    traces.append(
        go.Scatter(
            x=x_edge,
            y=y_edge,
            mode="lines",
            line=dict(width=linewidth_filt, color="rgba(60,60,60,0.9)"),
            hoverinfo="skip",
            name="edges",
            showlegend=True,
        )
    )

    # Points
    traces.append(
        go.Scatter(
            x=pts[:, 0],
            y=pts[:, 1],
            mode="markers",
            marker=dict(size=vertex_size, color="black"),
            name="points",
            showlegend=True,
            hovertemplate="(%{x:.3f}, %{y:.3f})<extra></extra>",
        )
    )

    # Active cycles
    active = []
    if show_cycles:
        active = forest._active_bars_with_cycles_at(
            filt_val=filt_val,
            min_bar_length=min_bar_length,
        )
        active = sorted(active, key=lambda bc: bc[0].lifespan(), reverse=True)


    if max_cycle_slots is None:
        max_cycle_slots = len(active)

    for i in range(max_cycle_slots):
        if i < len(active):
            bar, cycle = active[i]
            if cycle.dim() != 1:
                raise ValueError(
                    f"plot_at_filtration_plotly expected a 1-chain, got dim={cycle.dim()}"
                )
            segments = (
                cycle.segments(point_cloud=forest.point_cloud)
                if signed
                else cycle.unsigned().segments(point_cloud=forest.point_cloud)
            )

            x_c, y_c = [], []
            for seg in segments:
                x_c.extend([seg[0, 0], seg[1, 0], None])
                y_c.extend([seg[0, 1], seg[1, 1], None])

            traces.append(
                go.Scatter(
                    x=x_c,
                    y=y_c,
                    mode="lines",
                    line=dict(width=linewidth_cycle, color=color_map[bar]),
                    name=f"cycle {i + 1}",
                    legendgroup=f"bar-{id(bar)}",
                    showlegend=(i < 12),
                    visible=True,
                    hovertemplate=(
                        f"birth={bar.birth:.4g}<br>"
                        f"death={bar.death:.4g}<br>"
                        f"lifespan={bar.lifespan():.4g}"
                        "<extra></extra>"
                    ),
                )
            )
        else:
            traces.append(
                go.Scatter(
                    x=[np.nan],
                    y=[np.nan],
                    mode="lines",
                    line=dict(width=linewidth_cycle, color="rgba(0,0,0,0)"),
                    name=f"cycle {i + 1}",
                    showlegend=False,
                    visible=True,
                    hoverinfo="skip",
                )
            )

    return traces

def _plotly_traces_for_snapshot_3d(
    forest,
    snapshot: Dict[str, Any],
    color_map: Dict["PFBar", str],
    signed: bool = False,
    show_cycles: bool = True,
    show_complex: bool = True,
    min_bar_length: float = 0.0,
    complex_opacity: float = 0.20,
    cycle_opacity: float = 0.55,
    vertex_size: float = 3.0,
    max_cycle_slots: Optional[int] = None,
) -> List[Any]:
    """
    Build Plotly traces for a 3D filtration snapshot.

    This version is animation-safe: it always returns the same trace layout
    when max_cycle_slots is fixed.

    Trace order:
        0. complex boundary mesh
        1. points
        2+. cycle mesh slots

    Parameters
    ----------
    forest : object
        Forest-like object exposing active bar/cycle queries.
    snapshot : dict
        Output of ``_complex_snapshot_at_filtration``.
    color_map : dict[PFBar, str]
        Bar-to-color mapping.
    signed : bool
        If False, cancel opposite-oriented duplicate triangles before
        rendering cycles.
    show_cycles : bool
        If True, include active cycle meshes.
    show_complex : bool
        If True, include the complex boundary mesh.
    min_bar_length : float
        Only include active cycles for bars with lifespan at least this
        value.
    complex_opacity : float
        Opacity for the complex boundary mesh.
    cycle_opacity : float
        Opacity for cycle meshes.
    vertex_size : float
        Plotly marker size for point-cloud vertices.
    max_cycle_slots : int | None
        Fixed number of cycle trace slots. When provided, empty traces are
        added so animation frames keep a stable trace layout.

    Returns
    -------
    list
        List of Plotly traces.
    """
    go = _require_plotly()
    pts = snapshot["points"]
    filt_val = snapshot["filt_val"]
    boundary_triangles = snapshot["triangles"]

    traces: List[Any] = []

    # --- complex boundary mesh: always present as trace slot 0
    if show_complex and boundary_triangles:
        i_idx = [tri[0] for tri in boundary_triangles]
        j_idx = [tri[1] for tri in boundary_triangles]
        k_idx = [tri[2] for tri in boundary_triangles]

        traces.append(
            go.Mesh3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                i=i_idx,
                j=j_idx,
                k=k_idx,
                opacity=complex_opacity,
                color="lightblue",
                name="complex boundary",
                hoverinfo="skip",
                showlegend=True,
            )
        )
    else:
        traces.append(
            go.Mesh3d(
                x=[np.nan],
                y=[np.nan],
                z=[np.nan],
                i=[],
                j=[],
                k=[],
                opacity=complex_opacity,
                color="lightblue",
                name="complex boundary",
                hoverinfo="skip",
                showlegend=True,
                visible=True,
            )
        )

    # --- points: always present as trace slot 1
    traces.append(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(size=vertex_size, color="black"),
            name="points",
            showlegend=True,
            hovertemplate="(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
        )
    )

    # --- active cycles
    active = []
    if show_cycles:
        active = forest._active_bars_with_cycles_at(
            filt_val=filt_val,
            min_bar_length=min_bar_length,
        )
        active = sorted(active, key=lambda bc: bc[0].lifespan(), reverse=True)

    if max_cycle_slots is None:
        max_cycle_slots = len(active)

    for idx in range(max_cycle_slots):
        if idx < len(active):
            bar, cycle = active[idx]
            if cycle.dim() != 2:
                raise ValueError(
                    f"plot_at_filtration_plotly expected a 2-chain, got dim={cycle.dim()}"
                )
            tri_faces = cycle.simplices(signed=signed)

            if tri_faces:
                i_idx = [tri[0] for tri in tri_faces]
                j_idx = [tri[1] for tri in tri_faces]
                k_idx = [tri[2] for tri in tri_faces]

                traces.append(
                    go.Mesh3d(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        z=pts[:, 2],
                        i=i_idx,
                        j=j_idx,
                        k=k_idx,
                        opacity=cycle_opacity,
                        color=color_map[bar],
                        name=f"cycle {idx + 1}",
                        legendgroup=f"bar-{id(bar)}",
                        showlegend=(idx < 12),
                        visible=True,
                        hovertemplate=(
                            f"birth={bar.birth:.4g}<br>"
                            f"death={bar.death:.4g}<br>"
                            f"lifespan={bar.lifespan():.4g}"
                            "<extra></extra>"
                        ),
                    )
                )
            else:
                traces.append(
                    go.Mesh3d(
                        x=[np.nan],
                        y=[np.nan],
                        z=[np.nan],
                        i=[],
                        j=[],
                        k=[],
                        opacity=cycle_opacity,
                        color=color_map[bar],
                        name=f"cycle {idx + 1}",
                        legendgroup=f"bar-{id(bar)}",
                        showlegend=False,
                        visible=True,
                        hoverinfo="skip",
                    )
                )
        else:
            traces.append(
                go.Mesh3d(
                    x=[np.nan],
                    y=[np.nan],
                    z=[np.nan],
                    i=[],
                    j=[],
                    k=[],
                    opacity=cycle_opacity,
                    color="rgba(0,0,0,0)",
                    name=f"cycle {idx + 1}",
                    showlegend=False,
                    visible=True,
                    hoverinfo="skip",
                )
            )

    return traces
# ------ Plotly public API ---------

def plot_at_filtration_plotly(
    forest,
    filt_val: float,
    coloring: Literal["forest", "bars"] = "forest",
    show_cycles: bool = True,
    fill_triangles: bool = True,
    signed: bool = False,
    min_bar_length: float = 0.0,
    show_complex: bool = True,
    complex_opacity: float = 0.20,
    cycle_opacity: float = 0.55,
    show: bool = True,
    renderer: Optional[str] = None,
    vertex_size: float = 3.0,
    width: Optional[int] = None,   
    height: Optional[int] = None,
):
    """
    Plot the complex and active cycles at a single filtration value using Plotly.

    Supports ambient dimensions 2 and 3.

    Parameters
    ----------
    forest : object
        Forest-like object exposing ``dim``, ``barcode``, color maps and
        filtration snapshots.
    filt_val : float
        Filtration threshold.
    coloring : {"forest", "bars"}
        Which color map to use for active bars.
    show_cycles : bool
        If True, overlay cycle representatives.
    fill_triangles : bool
        For 2D only: if True, fill present triangles.
    signed : bool
        If False, cancel opposite-oriented duplicate edges before rendering
        2D cycle representatives or duplicate triangular faces before
        rendering 3D cycle representatives.
    min_bar_length : float
        Only show bars with lifespan >= min_bar_length.
    show_complex : bool
        For 3D only: whether to show the complex boundary mesh.
    complex_opacity : float
        Opacity of the 3D complex mesh.
    cycle_opacity : float
        Opacity of 3D cycle meshes.
    show : bool
        If True, call fig.show().
    renderer : str | None
        Plotly renderer used by ``fig.show()``.
    vertex_size : float
        Plotly marker size for point-cloud vertices.
    width, height : int | None
        Figure dimensions in pixels, forwarded to Plotly layout.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _require_plotly()
    if forest.dim not in (2, 3):
        raise ValueError("plot_at_filtration_plotly is only implemented for dimensions 2 and 3.")

    color_map = forest._get_color_map(coloring=coloring)
    snapshot = forest._complex_snapshot_at_filtration(filt_val=filt_val)

    if forest.dim == 2:
        traces = _plotly_traces_for_snapshot_2d(
            forest=forest,
            snapshot=snapshot,
            color_map=color_map,
            show_cycles=show_cycles,
            fill_triangles=fill_triangles,
            signed=signed,
            min_bar_length=min_bar_length,
            vertex_size=vertex_size,
        )

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f"Filtration value r = {filt_val:.4g}",
            xaxis_title="x",
            yaxis_title="y",
            template="plotly_white",
            hovermode="closest",
            width=width,
            height=height,
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

    else:
        traces = _plotly_traces_for_snapshot_3d(
            forest=forest,
            snapshot=snapshot,
            color_map=color_map,
            show_cycles=show_cycles,
            show_complex=show_complex,
            min_bar_length=min_bar_length,
            complex_opacity=complex_opacity,
            cycle_opacity=cycle_opacity,
            signed=signed,
            vertex_size=vertex_size,
        )

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f"Filtration value r= {filt_val:.4g}",
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="z",
                aspectmode="data",
            ),
            template="plotly_white",
            width = width,   
            height = height,
        )

    if show:
        fig.show(renderer=renderer)

    return fig

def plot_filtration_interactive(
    forest,
    coloring: Literal["forest", "bars"] = "forest",
    show_cycles: bool = True,
    signed: bool = False,
    filt_max: Optional[float] = None,
    min_bar_length: float = 0.0,
    show_complex: bool = True,
    complex_opacity: float = 0.20,
    cycle_opacity: float = 0.55,
    resolution: int =100,
    show: bool = True,
    renderer: Optional[str] = None,
    fill_triangles: bool = True,
    vertex_size: float = 3.0,
    width: Optional[int] = None,   
    height: Optional[int] = None,
):
    """
    Create a Plotly slider that moves through filtration values.

    Supports ambient dimensions 2 and 3.

    Parameters
    ----------
    forest : object
        Forest-like object exposing ``dim``, ``barcode``, color maps and
        filtration snapshots.
    coloring : {"forest", "bars"}
        Which bar color map to use.
    show_cycles : bool
        If True, overlay active cycle representatives.
    signed : bool
        If False, removes simplices which appear with both orientations.
    filt_max : float | None
        Maximum filtration value for the slider. If None, uses
        ``1.03 * max(bar.death for bar in forest.barcode)``.
    min_bar_length : float
        Only show bars with lifespan >= min_bar_length.
    show_complex : bool
        For 3D only: whether to show the complex boundary mesh.
    complex_opacity : float
        Opacity of 3D complex mesh.
    cycle_opacity : float
        Opacity of 3D cycle meshes.
    resolution : int
        Number of slider intervals. The figure contains
        ``resolution + 1`` equally spaced frames from 0 to ``filt_max``.
    show : bool
        If True, call fig.show().
    renderer : str | None
        Plotly renderer used by ``fig.show()``.
    fill_triangles : bool
        For 2D only: whether to fill present triangles.
    vertex_size : float
        Plotly marker size for point-cloud vertices.
    width, height : int | None
        Figure dimensions in pixels, forwarded to Plotly layout.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    go = _require_plotly()
    if forest.dim not in (2, 3):
        raise ValueError("plot_filtration_interactive is only implemented for dimensions 2 and 3.")

    if filt_max is None:
        filt_max = max( [bar.death for bar in forest.barcode] )*1.03

    filtration_values = np.linspace(0,filt_max,resolution+1)

    if len(filtration_values) == 0:
        raise ValueError("No filtration values available for interactive plot.")

    color_map = forest._get_color_map(coloring=coloring)

    if show_cycles:
        max_cycle_slots = max(
            len(forest._active_bars_with_cycles_at(v, min_bar_length=min_bar_length))
            for v in filtration_values
        )
    else:
        max_cycle_slots = 0

    frames = []
    for v in filtration_values:
        snapshot = forest._complex_snapshot_at_filtration(filt_val=v)

        if forest.dim == 2:
            frame_traces = _plotly_traces_for_snapshot_2d(
                forest=forest,
                snapshot=snapshot,
                color_map=color_map,
                show_cycles=show_cycles,
                fill_triangles=fill_triangles,
                signed=signed,
                min_bar_length=min_bar_length,
                max_cycle_slots=max_cycle_slots,
                vertex_size=vertex_size
            )

        elif forest.dim == 3:
            frame_traces = _plotly_traces_for_snapshot_3d(
                forest=forest,
                snapshot=snapshot,
                color_map=color_map,
                show_cycles=show_cycles,
                show_complex=show_complex,
                min_bar_length=min_bar_length,
                complex_opacity=complex_opacity,
                cycle_opacity=cycle_opacity,
                signed=signed,
                max_cycle_slots=max_cycle_slots,
                vertex_size=vertex_size
            )
        else:
            raise ValueError("plot_filtration_interactive is only implemented for dimensions 2 and 3.")

        frames.append(go.Frame(name=f"{v:.12g}", data=frame_traces, traces=list(range(len(frame_traces))),))

    fig = go.Figure(data=frames[0].data, frames=frames)

    slider_steps = []
    for v in filtration_values:
        slider_steps.append(
            dict(
                method="animate",
                label=f"{v:.4g}",
                args=[
                    [f"{v:.12g}"],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
            )
        )

    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "r = "},
            pad={"t": 40},
            steps=slider_steps,
        )
    ]

    if forest.dim == 2:
        fig.update_layout(
            title="Interactive filtration plot",
            xaxis_title="x",
            yaxis_title="y",
            template="plotly_white",
            hovermode="closest",
            sliders=sliders,
            width=width,
            height=height,
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    else:
        fig.update_layout(
            title="Interactive filtration plot",
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="z",
                aspectmode="data",
            ),
            template="plotly_white",
            sliders=sliders,
            width=width,
            height=height,
        )

    fig.update_layout(
        legend=dict(
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
        ),
        margin=dict(r=180),
    )

    if show:
        fig.show(renderer=renderer)

    return fig
