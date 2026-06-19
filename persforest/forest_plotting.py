"""
Shared plotting utilities for forest-like persistence objects.

This module is designed to be *forest-agnostic*: it only assumes that a
"forest" object provides

    - forest.barcode        : iterable of bar objects
    - each bar has .birth, .death, and preferably .lifespan()
    - (optionally) forest._build_color_map_forest()
                    forest._build_color_map_bars()
                    forest.color_map_forest
                    forest.color_map_bars
    - (for animations) forest.filtration : iterable of (simplex, filt_val)
      and a method
          forest.plot_at_filtration(filt_val: float, ax=None, **kwargs)

`PersistenceForest` satisfies these assumptions. Future forest classes can
reuse these utilities by exposing the same small interface.

Typical use inside a class:

    from forest_plotting import plot_barcode as _plot_barcode_generic
    from forest_plotting import animate_filtration as _animate_filtration_generic

    class PersistenceForest:
        ...
        def plot_barcode(self, *args, **kwargs):
            return _plot_barcode_generic(self, *args, **kwargs)

        def animate_filtration(self, *args, **kwargs):
            return _animate_filtration_generic(self, *args, **kwargs)

You are free to adapt the wrappers (defaults, docstrings, etc.) per class.
"""

from typing import Any, Literal, Optional, Tuple
from numbers import Real
from pathlib import Path
import shutil
import subprocess
import tempfile
import warnings
import numpy as np
import matplotlib.pyplot as plt


def _plot_barcode_generic(
        forest,
        *,
        ax=None,
        sort: str | None = "birth",   # "length" | "birth" | "death" | None
        title: str = "Barcode",
        xlabel: str = "filtration value",
        coloring: Literal["forest", "bars","none","grey"] = "forest",
        max_bars: int = 0,
        min_bar_length: float = 0.0,
        bar_width: float = 2.0,
        descending: bool = False,
        tight_layout: bool = True,
    ):
    """
    Plot a 1D barcode from ``forest.barcode``.

    Each Bar contributes a horizontal segment from birth to death.
    If death is +inf, an arrow is drawn to the right.

    Parameters
    ----------
    ax : matplotlib.axes.Axes | None
        If given, draw on this axes. Otherwise a new figure/axes is created.
    sort : {"length","birth","death",None}
        Sort bars before plotting (None preserves current order).
        Default is "birth".
    title : str
        Plot title.
    xlabel : str
        Label for the x-axis.
    coloring : {"forest","bars","none","grey"}
        Which color scheme to use:
        - "forest": use forest.color_map_forest (tree-structured colors).
        - "bars":   use forest.color_map_bars (ignores tree structure).
        - "none":   all bars share matplotlib defaults.
        - "grey":   draw all bars in black.
        If the chosen color map does not exist yet, it is built as in
        `plot_at_filtration`.
    max_bars : int
        If > 0, display at most this many bars, keeping the longest ones
        (by lifespan). 0 means show all bars.
    min_bar_length : float
        Filter out bars with lifespan < min_bar_length before plotting.
    bar_width : float
        Line width used for each barcode interval.
    descending : bool
        If True, reverse the selected sort order.
    tight_layout : bool
        If True, call ``fig.tight_layout()`` after drawing.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes the barcode was drawn on.
    """
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    if not getattr(forest, "barcode", None):
        raise ValueError("No bars to plot: `forest.barcode` is empty.")

    # ---- Prepare color map (same logic as plot_at_filtration) ----
    if coloring == "forest":
        if not hasattr(forest, "color_map_forest"):
            forest._build_color_map_forest()
        color_map = forest.color_map_forest
    elif coloring == "bars":
        if not hasattr(forest, "color_map_bars"):
            forest._build_color_map_bars()
        color_map = forest.color_map_bars
    else:
        color_map = {}

    # ---- Work on a copy so we don't mutate original order ----
    bars = list(forest.barcode)

    # Filter by minimum length (Gudhi-like)
    if min_bar_length > 0.0:
        bars = [b for b in bars if b.lifespan() >= min_bar_length]

    if not bars:
        raise ValueError(
            "No bars to plot after applying min_bar_length filter "
            f"(min_bar_length = {min_bar_length})."
        )

    # Limit to longest `max_bars` bars if requested (Gudhi-like)
    if max_bars and max_bars > 0 and len(bars) > max_bars:
        bars = sorted(bars, key=lambda b: b.lifespan(), reverse=True)[:max_bars]

    # Optional sorting for display
    if sort == "birth":
        bars.sort(key=lambda b: (b.birth, b.death), reverse=descending)
    elif sort == "death":
        def dkey(b):
            d = b.death
            return (math.inf if not math.isfinite(d) else d, b.birth)
        bars.sort(key=dkey, reverse=descending)
    elif sort == "length":
        def length(b):
            d = b.death
            d_val = math.inf if not math.isfinite(d) else d
            return d_val - b.birth
        bars.sort(key=length, reverse=descending)
    elif sort is None:
        # Keep whatever order came out of filtering
        pass
    else:
        raise ValueError(
            f"Unknown sort option {sort!r}. "
            "Expected one of 'birth', 'death', 'length', or None."
        )

    n_bars = len(bars)

    # ---- Create axes if needed, with controlled figure height ----
    created_ax = False
    if ax is None:
        # Height grows sublinearly and is capped to avoid gigantic figures
        base_height = 2.5
        extra_height = 0.12 * min(n_bars, 80)   # at most ~9.1 total
        fig, ax = plt.subplots(figsize=(7, base_height + extra_height))
        created_ax = True
    else:
        fig = ax.figure

    # ---- Determine x-limits with a bit of padding ----
    deaths = np.array([b.death for b in bars], dtype=float)
    finite_deaths = deaths[np.isfinite(deaths)]

    xmin = 0
    xmax = float(np.nanmax(finite_deaths))

    if not np.isfinite(xmax):  # extreme corner case
        raise ValueError("max of deaths is not finie")

    pad = (xmax - xmin) * 0.05 if xmax > xmin else 1.0
    ax.set_xlim(xmin, xmax + pad)

    # ---- Draw segments ----
    for i, b in enumerate(bars):
        x0, x1 = float(b.birth), float(b.death)

        # Guard against inverted bars due to numerical issues
        if math.isfinite(x1) and x1 < x0:
            x0, x1 = x1, x0

        color = color_map.get(b, None)
        if coloring == "grey":
            color = "black"

        line_kwargs = {
            "y": i,
            "xmin": x0,
            "xmax": x1 if math.isfinite(x1) else ax.get_xlim()[1] - 0.25 * pad,
            "linewidth": bar_width,  # thicker bars
        }
        if color is not None:
            line_kwargs["color"] = color

        if math.isfinite(x1):
            # Finite bar: simple thick line, NO endpoint markers
            ax.hlines(**line_kwargs)
        else:
            # Infinite bar: truncated line + arrow
            right = ax.get_xlim()[1]
            line_kwargs["xmax"] = right - 0.25 * pad
            ax.hlines(**line_kwargs)

            # Draw arrow for infinity
            arrow_kwargs = {
                "xy":   (right - 0.15 * pad, i),
                "xytext": (x0, i),
                "arrowprops": dict(arrowstyle="->", lw=2),
                "va": "center",
            }
            if color is not None:
                arrow_kwargs["arrowprops"]["color"] = color

            ax.annotate("", **arrow_kwargs)

    # ---- Cosmetics ----
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax.set_ylim(-1, n_bars)  # keep bars nicely framed
    if tight_layout:
        fig.tight_layout()

    # If we created the axes, show it immediately (so this works in scripts)
    if created_ax:
        import matplotlib.pyplot as plt
        plt.show()

    return ax

def _plot_dendrogram_generic(
        forest,
        ax=None,
        show: bool = True,
        annotate_ids: bool = False,
        leaf_spacing: float = 1.0,
        tree_gap_leaves: int = 1,
        check_reduced: bool = True,
        small_on_top: bool = False,
        threshold: float = 0.0,
    ):
    """
    Plot a dendrogram-style view of a forest.

    The y-coordinate is each node's ``filt_val``. This generic helper is used
    by ``PersistenceForest.plot_dendrogram``.

    Parameters
    ----------
    forest : Forest
        Forest containing nodes with `filt_val`, `parent`, `children`, and `id`.
    ax : matplotlib.axes.Axes | None
        Axes to draw on. If None, a new figure/axes is created.
    show : bool, optional
        If True (default) call ``plt.show`` after plotting.
    annotate_ids : bool, optional
        If True, annotate each node with its integer id.
    leaf_spacing : float, optional
        Horizontal spacing between consecutive leaves.
    tree_gap_leaves : int, optional
        Additional empty leaf slots inserted between separate trees.
    check_reduced : bool, optional
        If True, warn if any parent and child share the same filtration value.
    small_on_top : bool, optional
        If True, invert the y-axis so smaller filtration values appear on top.
    threshold : float, optional
        Minimum vertical span, measured from root to leaves, required to keep a
        tree. Trees with ``max_leaf_delta <= threshold`` are omitted.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes on which the dendrogram was drawn.
    """
    import warnings

    if not forest.nodes:
        raise ValueError("Forest has no nodes to plot.")

    all_nodes = forest.nodes
    all_ids = set(all_nodes.keys())

    # Recompute roots robustly from the current structure (these are Node objects)
    all_roots = [n for n in all_nodes.values() if n.parent is None or n.parent not in all_ids]

    if check_reduced:
        equal_pairs = [
            (p.id, c)
            for p in all_nodes.values()
            for c in p.children
            if c in all_nodes and all_nodes[c].filt_val == p.filt_val
        ]
        if equal_pairs:
            warnings.warn(
                f"Forest does not appear reduced: found {len(equal_pairs)} parent–child pairs "
                f"with equal filt_val. Plotting anyway."
            )

    bad_direction = [
        (p.id, c)
        for p in all_nodes.values()
        for c in p.children
        if c in all_nodes and all_nodes[c].filt_val < p.filt_val
    ]
    if bad_direction:
        warnings.warn(
            f"{len(bad_direction)} edges have child.filt_val > parent.filt_val."
        )

    # ---------- threshold filtering (key addition) ----------
    def _subtree_ids(root_id: int) -> set[int]:
        """All node ids reachable from (and including) root_id that are present in all_nodes."""
        stack = [root_id]
        seen: set[int] = set()
        while stack:
            nid = stack.pop()
            if nid in seen or nid not in all_nodes:
                continue
            seen.add(nid)
            stack.extend([cid for cid in all_nodes[nid].children if cid in all_nodes])
        return seen

    def _is_leaf_in(sub_ids: set[int], nid: int) -> bool:
        """Leaf = no children inside this same subgraph."""
        return not any((cid in sub_ids) for cid in all_nodes[nid].children)

    included_root_ids: list[int] = []
    included_ids: set[int] = set()

    for r in all_roots:
        sub_ids = _subtree_ids(r.id)
        # Identify leaves within this subgraph
        leaves = [nid for nid in sub_ids if _is_leaf_in(sub_ids, nid)]
        root_val = float(all_nodes[r.id].filt_val)
        max_leaf_delta = max((abs(float(all_nodes[l].filt_val) - root_val) for l in leaves), default=0.0)

        if max_leaf_delta > float(threshold):
            included_root_ids.append(r.id)
            included_ids.update(sub_ids)

    if threshold <= 0.0:
        # No filtering requested: include everything
        nodes = all_nodes
        roots = sorted(all_roots, key=lambda n: (n.filt_val, n.id))
    else:
        if not included_ids:
            # Nothing to plot under this threshold — return an empty/annotated axes.
            if ax is None:
                _, ax = plt.subplots(figsize=(8, 6))
            ax.set_title(f"Forest dendrogram (y = filt_val) — no trees exceed threshold {threshold}")
            ax.set_axis_off()
            if show:
                plt.show()
            return ax
        nodes = {nid: all_nodes[nid] for nid in included_ids}
        roots = [all_nodes[rid] for rid in included_root_ids]
        roots.sort(key=lambda n: (n.filt_val, n.id))

    node_ids = set(nodes.keys())

    # ---------- positions ----------
    x: dict[int, float] = {}
    y: dict[int, float] = {n.id: float(n.filt_val) for n in nodes.values()}
    visited: set[int] = set()
    leaf_counter = 0

    def _assign_x(nid: int):
        nonlocal leaf_counter
        if nid in visited:
            return
        visited.add(nid)

        child_ids = [cid for cid in nodes[nid].children if cid in nodes]
        child_ids.sort(key=lambda cid: (nodes[cid].filt_val, nodes[cid].id))

        if len(child_ids) == 0:
            x[nid] = leaf_counter * leaf_spacing
            leaf_counter += 1
        else:
            for cid in child_ids:
                _assign_x(cid)
            xs = [x[cid] for cid in child_ids]
            x[nid] = sum(xs) / len(xs)

    # Lay out each tree; insert spacing between trees
    for i, r in enumerate(roots):
        start_before = leaf_counter
        _assign_x(r.id)
        if i != len(roots) - 1 and leaf_counter > start_before:
            leaf_counter += tree_gap_leaves

    # Place any stray components (shouldn't happen, but be safe) — within the filtered set only
    for nid in list(node_ids):
        if nid not in x:
            _assign_x(nid)
            leaf_counter += tree_gap_leaves

    # ---------- draw ----------
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    line_color = "0.25"
    merge_color = "0.1"
    node_edge = "white"

    for p in nodes.values():
        pid = p.id
        px, py = x[pid], y[pid]
        child_ids = [cid for cid in p.children if cid in nodes]
        if not child_ids:
            continue

        child_ids.sort(key=lambda cid: x[cid])
        child_xs = [x[cid] for cid in child_ids]
        child_ys = [y[cid] for cid in child_ids]

        for cx, cy in zip(child_xs, child_ys):
            ax.plot([cx, cx], [cy, py], linewidth=1.2, color=line_color, zorder=1)

        if len(child_ids) >= 2:
            ax.plot([min(child_xs), max(child_xs)], [py, py], linewidth=1.5, color=merge_color, zorder=2)

    for n in nodes.values():
        ax.scatter(x[n.id], y[n.id], s=24, zorder=3, color="C0", edgecolors=node_edge, linewidths=0.6)
        if annotate_ids:
            ax.annotate(
                str(n.id),
                (x[n.id], y[n.id]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                color="0.2",
            )

    ax.set_xlabel("leaf order")
    ax.set_ylabel("filt_val (y)")
    title = "Forest dendrogram (y = filt_val)"
    if threshold > 0.0:
        title += f" — threshold > {threshold}"
    ax.set_title(title)
    ax.margins(x=0.05, y=0.05)
    ax.grid(False)

    if x:
        xs = list(x.values())
        ax.set_xlim(min(xs) - leaf_spacing, max(xs) + leaf_spacing)

    if small_on_top:
        ax.invert_yaxis()

    if show:
        plt.show()
    return ax

def _animate_filtration_generic(
        forest,
        filename: Optional[str] = None,
        *,
        fps: int = 20,
        frames: int = 200,
        coloring: Literal["forest", "bars"] = "forest",
        with_barcode: bool = False,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        dpi: int = 300,
        figsize: Optional[tuple[float, float]] = None,
        pixel_size: Optional[tuple[int, int]] = None,
        panel_width_ratios: tuple[float, float] = (3.5, 2.0),
        panel_spacing: float = 0.08,
        figure_margins: Optional[dict[str, float]] = None,
        filtration_kwargs: Optional[dict] = None,
        barcode_kwargs: Optional[dict] = None,
        # Deprecated aliases, kept for backward compatibility
        cloud_figsize: Optional[tuple[float, float]] = None,
        total_figsize: Optional[tuple[float, float]] = None,
        plot_kwargs: Optional[dict] = None,
        alpha_digits: Optional[int] = None,
    ):
        """
        Create a Matplotlib animation across filtration values.

        This is the generic target for ``PersistenceForest.animate_filtration``
        when using Matplotlib-backed animation paths.

        Parameters
        ----------
        forest : Forest
            Forest exposing ``filtration``, ``barcode``, and ``plot_at_filtration``.
        filename : str | None, optional
            If given, the animation is written to this path.
        fps : int, optional
            Frames per second for the saved animation.
        frames : int, optional
            Number of time steps (frames) sampled between ``t_min`` and ``t_max``.
        coloring : {"forest","bars"}, optional
            Which color scheme to apply consistently to both the cloud panel and
            barcode panel.
        with_barcode : bool, optional
            If True, show a second panel with the barcode and a moving vertical
            line indicating the current filtration value.
        t_min, t_max : float | None, optional
            Optional lower/upper bounds on the filtration values to animate.
        dpi : int, optional
            DPI for saving the animation.
        figsize : (float, float) | None, optional
            Figure size in inches. If None, defaults to (6,6) without barcode
            and (10,5) with barcode.
        pixel_size : (int, int) | None, optional
            Figure size in pixels. If provided, takes precedence over
            ``figsize`` and is converted using ``dpi``.
        panel_width_ratios : (float, float), optional
            Width ratios for cloud and barcode panels when
            ``with_barcode=True``.
        panel_spacing : float, optional
            Horizontal spacing between cloud and barcode panels (GridSpec
            ``wspace``) when ``with_barcode=True``.
        figure_margins : dict[str, float] | None, optional
            Outer margins applied via ``fig.subplots_adjust`` with keys
            ``{"left","right","bottom","top"}``.
        filtration_kwargs : dict | None, optional
            Extra keyword arguments forwarded to ``plot_at_filtration``.
            The keys ``ax``, ``show`` and ``filt_val`` are reserved and are
            managed by this helper. If ``coloring`` is supplied here and
            conflicts with the top-level ``coloring``, the top-level value
            wins.
            Example::
                filtration_kwargs=dict(
                    show_complex=True,
                    vertex_size=3,
                    coloring="forest",
                )
        barcode_kwargs : dict | None, optional
            Extra keyword arguments forwarded to ``_plot_barcode`` **except**
            ``ax`` and ``coloring``, which are managed by this method.
            Example::
                barcode_kwargs=dict(
                    max_bars=150,
                    min_bar_length=1e-3,
                    sort="length",
                    title="Barcode",
                )
        cloud_figsize, total_figsize : tuple[float, float] | None, optional
            Deprecated aliases used only when ``figsize`` is omitted.
        plot_kwargs : dict | None, optional
            Deprecated alias for ``filtration_kwargs``.
        alpha_digits : int | None, optional
            Number of decimal places in the filtration-value overlay. If None,
            uses compact formatting.

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            The created animation. If ``filename`` is not None, the animation
            is also saved to disk.
        fig : matplotlib.figure.Figure
            The figure on which the animation is drawn.
        """
        from matplotlib.animation import FuncAnimation, FFMpegWriter

        if not hasattr(forest, "filtration") or not forest.filtration:
            raise ValueError("Forest has no filtration data to animate.")
        is_3d = (getattr(forest, "dim", None) == 3)
        if int(dpi) <= 0:
            raise ValueError("dpi must be a positive integer.")

        if len(panel_width_ratios) != 2:
            raise ValueError("panel_width_ratios must have exactly two entries.")
        if float(panel_width_ratios[0]) <= 0 or float(panel_width_ratios[1]) <= 0:
            raise ValueError("panel_width_ratios entries must be positive.")
        if not isinstance(panel_spacing, Real):
            raise ValueError("panel_spacing must be a numeric value.")
        panel_spacing = float(panel_spacing)
        if panel_spacing < 0.0:
            raise ValueError("panel_spacing must be non-negative.")

        if cloud_figsize is not None or total_figsize is not None:
            warnings.warn(
                "`cloud_figsize` and `total_figsize` are deprecated in animate_filtration; "
                "use `figsize=(w, h)`.",
                DeprecationWarning,
                stacklevel=2,
            )
            if figsize is None:
                if with_barcode and total_figsize is not None:
                    figsize = total_figsize
                elif (not with_barcode) and cloud_figsize is not None:
                    figsize = cloud_figsize

        if plot_kwargs is not None:
            warnings.warn(
                "`plot_kwargs` is deprecated in animate_filtration; "
                "use `filtration_kwargs`.",
                DeprecationWarning,
                stacklevel=2,
            )
            if filtration_kwargs is None:
                filtration_kwargs = dict(plot_kwargs)
            else:
                merged = dict(plot_kwargs)
                merged.update(filtration_kwargs)
                filtration_kwargs = merged

        # Optional restriction to a time window
        if t_min is None:
            t_min = 0.0
        if t_max is None:
            finite_deaths = [
                float(bar.death)
                for bar in getattr(forest, "barcode", [])
                if np.isfinite(float(bar.death))
            ]
            if finite_deaths:
                t_max = max(finite_deaths)
            else:
                finite_filtration = [float(f) for _, f in forest.filtration if np.isfinite(float(f))]
                if not finite_filtration:
                    raise ValueError("Could not infer finite t_max from barcode or filtration.")
                t_max = max(finite_filtration)

        # Uniformly spaced in filtration value → uniform speed
        frame_times = np.linspace(t_min, t_max, frames).tolist()  # pyright: ignore[reportCallIssue, reportArgumentType]

        # ---- Common kwargs for plot_at_filtration (cloud panel) ----
        if filtration_kwargs is None:
            filtration_kwargs = {}
        else:
            filtration_kwargs = dict(filtration_kwargs)

        banned_filtration_keys = {"ax", "show", "filt_val"}
        bad_keys = [k for k in banned_filtration_keys if k in filtration_kwargs]
        if bad_keys:
            raise ValueError(
                "filtration_kwargs contains reserved keys that are managed by animate_filtration: "
                f"{sorted(bad_keys)}"
            )
        if "coloring" in filtration_kwargs and filtration_kwargs["coloring"] != coloring:
            warnings.warn(
                "Ignoring `filtration_kwargs['coloring']` because top-level `coloring` controls both panels.",
                UserWarning,
                stacklevel=2,
            )
            filtration_kwargs.pop("coloring", None)

        # Reasonable defaults (only used if not explicitly overridden)
        if is_3d:
            filtration_kwargs = {
                "show_complex": True,
                "vertex_size": 3,
                "coloring": coloring,
                "show": False,
                **filtration_kwargs,
            }
        else:
            filtration_kwargs = {
                "vertex_size": 3,
                "coloring": coloring,
                "show": False,
                **filtration_kwargs,
            }
        filtration_kwargs["show"] = False

        # ---- Barcode kwargs & shared color dict ----
        rendered_figsize, _ = _resolve_matplotlib_figsize(
            with_barcode=with_barcode,
            dpi=int(dpi),
            figsize=figsize,
            pixel_size=pixel_size,
        )
        margins = _resolve_figure_margins(
            with_barcode=with_barcode,
            figure_margins=figure_margins,
        )
        if with_barcode:
            fig = plt.figure(figsize=rendered_figsize, constrained_layout=False)
            if is_3d:
                gs = fig.add_gridspec(
                    1,
                    2,
                    width_ratios=panel_width_ratios,
                    wspace=panel_spacing,
                )
                ax_cloud = fig.add_subplot(gs[0, 0], projection="3d")
                ax_bar = fig.add_subplot(gs[0, 1])
            else:
                gs = fig.add_gridspec(
                    1,
                    2,
                    width_ratios=panel_width_ratios,
                    wspace=panel_spacing,
                )
                ax_cloud = fig.add_subplot(gs[0, 0])
                ax_bar = fig.add_subplot(gs[0, 1])
            fig.subplots_adjust(**margins)

            # Draw the (static) barcode once
            if not getattr(forest, "barcode", None):
                raise ValueError("`with_barcode=True` but `forest.barcode` is empty.")

            if barcode_kwargs is None:
                barcode_kwargs = {}
            else:
                barcode_kwargs = dict(barcode_kwargs)
            if "ax" in barcode_kwargs:
                raise ValueError("barcode_kwargs cannot contain 'ax'; axis is managed internally.")
            if "coloring" in barcode_kwargs and barcode_kwargs["coloring"] != coloring:
                warnings.warn(
                    "Ignoring `barcode_kwargs['coloring']` because top-level `coloring` controls both panels.",
                    UserWarning,
                    stacklevel=2,
                )
                barcode_kwargs.pop("coloring", None)
            # Do not let the caller override ax here
            # Defaults for the barcode panel – user can override sort/title/xlabel
            barcode_kwargs = {
                "sort": "length",
                "title": "Barcode",
                "xlabel": "filtration value",
                "tight_layout": False,
                "coloring": coloring,
                **barcode_kwargs,
            }


            forest.plot_barcode(
                ax=ax_bar,
                **barcode_kwargs,
            )

            # Vertical line that will move with the filtration
            current_t0 = frame_times[0]
            barcode_line = ax_bar.axvline(current_t0, color="k", linewidth=2)
        else:
            if is_3d:
                fig = plt.figure(figsize=rendered_figsize, constrained_layout=False)
                ax_cloud = fig.add_subplot(111, projection="3d")
            else:
                fig = plt.figure(figsize=rendered_figsize, constrained_layout=False)
                ax_cloud = fig.add_subplot(111)
            fig.subplots_adjust(**margins)
            ax_bar = None
            barcode_line = None

        # ---- Helper to draw a single frame on the cloud panel ----
        def _draw_frame_at_time(t: float, frame_idx: int = 0):
            """Helper: clear the cloud axis and redraw for filtration value t."""
            ax_cloud.clear()
            local_plot_kwargs = filtration_kwargs
            if is_3d:
                local_plot_kwargs = dict(filtration_kwargs)
                camera_mode = local_plot_kwargs.pop("camera_mode", "fixed")
                style_3d = dict(local_plot_kwargs.get("style_3d", {}) or {})
                camera_eye = style_3d.get("camera_eye", None)
                if camera_mode not in {"fixed", "orbit"}:
                    raise ValueError("camera_mode must be 'fixed' or 'orbit'.")
                if camera_mode == "orbit":
                    base_elev = 22.0
                    base_azim = -55.0
                    if isinstance(camera_eye, dict):
                        base_elev = float(camera_eye.get("elev", base_elev))
                        base_azim = float(camera_eye.get("azim", base_azim))
                    elif isinstance(camera_eye, (tuple, list)) and len(camera_eye) >= 2:
                        base_elev = float(camera_eye[0])
                        base_azim = float(camera_eye[1])
                    denom = max(1, len(frame_times) - 1)
                    style_3d["camera_eye"] = (
                        base_elev,
                        base_azim + 360.0 * float(frame_idx) / float(denom),
                    )
                    local_plot_kwargs["style_3d"] = style_3d
            # Delegate the heavy lifting to the existing helper
            forest.plot_at_filtration(filt_val=t, ax=ax_cloud, **local_plot_kwargs)

            # Optional: overlay a small text box with the current filtration value.
            # Comment this out if you prefer only the built-in title.
            if alpha_digits is None:
                radius_text = rf"$\alpha = {t:.3g}$"
            else:
                radius_text = rf"$\alpha = {t:.{alpha_digits}f}$"
            ax_cloud.annotate(
                radius_text,
                xy=(0.02, 0.98),
                xycoords="axes fraction",
                va="top",
                ha="left",
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

        # ---- Animation callbacks ----
        def init():
            _draw_frame_at_time(frame_times[0], frame_idx=0)
            if with_barcode and barcode_line is not None:
                t0 = frame_times[0]
                barcode_line.set_xdata([t0, t0])
            return []

        def update(frame_idx: int):
            t = frame_times[frame_idx]
            _draw_frame_at_time(t, frame_idx=frame_idx)
            if with_barcode and barcode_line is not None:
                barcode_line.set_xdata([t, t])
            return []

        anim = FuncAnimation(
            fig,
            update,
            frames=len(frame_times),
            init_func=init,
            blit=False,
        )

        # ---- Optionally write to disk ----
        if filename is not None:
            # Force a draw so text/ticks/layout are finalized
            fig.canvas.draw()


            fname = str(filename)
            ext = fname.lower().rsplit(".", 1)[-1] if "." in fname else ""

            savefig_kwargs = {
                "pad_inches": 0.0,  
                "facecolor": fig.get_facecolor(),
            }

            if ext == "mp4":
                writer = FFMpegWriter(fps=fps, bitrate=2000)
                anim.save(fname, writer=writer, dpi=dpi, savefig_kwargs=savefig_kwargs)
            elif ext in {"gif", "gifv"}:
                try:
                    from matplotlib.animation import PillowWriter
                except ImportError as e:
                    raise RuntimeError(
                        "Saving as GIF requires Pillow. Install it with `pip install pillow` "
                        "or `pip install \".[animation]\"`."
                    ) from e
                writer = PillowWriter(fps=fps)
                anim.save(fname, writer=writer, dpi=dpi, savefig_kwargs=savefig_kwargs)
            else:
                anim.save(fname, dpi=dpi, fps=fps, savefig_kwargs=savefig_kwargs)
        return anim, fig


def _compute_frame_times(
    forest,
    *,
    frames: int,
    t_min: Optional[float],
    t_max: Optional[float],
) -> np.ndarray:
    """
    Return a monotone time grid used for filtration animations.

    Parameters
    ----------
    forest : object
        Forest-like object exposing ``filtration`` and ``barcode``.
    frames : int
        Number of frame times to generate.
    t_min, t_max : float | None
        Optional lower and upper bounds.

    Returns
    -------
    np.ndarray
        Array of shape (frames,) with filtration values.
    """
    if frames <= 0:
        raise ValueError("frames must be a positive integer.")

    if t_min is None:
        t0 = 0.0
    else:
        t0 = float(t_min)

    if t_max is None:
        finite_deaths = [
            float(bar.death)
            for bar in getattr(forest, "barcode", [])
            if np.isfinite(float(bar.death))
        ]
        if finite_deaths:
            t1 = max(finite_deaths)
        else:
            finite_filtration = [float(f) for _, f in forest.filtration if np.isfinite(float(f))]
            if not finite_filtration:
                raise ValueError("Could not infer finite t_max from barcode or filtration.")
            t1 = max(finite_filtration)
    else:
        t1 = float(t_max)

    if t1 < t0:
        raise ValueError(f"Invalid time window: t_max ({t1}) is smaller than t_min ({t0}).")

    return np.linspace(t0, t1, int(frames))


def _resolve_matplotlib_figsize(
    *,
    with_barcode: bool,
    dpi: int,
    figsize: Optional[tuple[float, float]] = None,
    pixel_size: Optional[tuple[int, int]] = None,
    # Deprecated aliases
    width: Optional[int] = None,
    height: Optional[int] = None,
    cloud_figsize: Optional[tuple[float, float]] = None,
    total_figsize: Optional[tuple[float, float]] = None,
) -> tuple[tuple[float, float], tuple[int, int]]:
    """
    Resolve a deterministic even-pixel canvas and matching figure size.

    Parameters
    ----------
    with_barcode : bool
        Whether the figure contains a barcode panel.
    dpi : int
        Dots per inch used to convert between inches and pixels.
    figsize : tuple[float, float] | None
        Preferred figure size in inches.
    pixel_size : tuple[int, int] | None
        Preferred canvas size in pixels. Takes precedence over ``figsize``.
    width, height : int | None
        Deprecated pixel-size aliases; both must be supplied together.
    cloud_figsize, total_figsize : tuple[float, float] | None
        Deprecated figure-size aliases used when ``figsize`` is omitted.

    Returns
    -------
    tuple
        ``(figsize_inches, pixel_size_even)``.
    """
    if int(dpi) <= 0:
        raise ValueError("dpi must be a positive integer.")

    def _even(v: int) -> int:
        return v if (v % 2 == 0) else (v + 1)

    if width is not None or height is not None:
        if width is None or height is None:
            raise ValueError("width and height must be provided together.")
        if pixel_size is None:
            pixel_size = (int(width), int(height))

    if pixel_size is not None:
        if len(pixel_size) != 2:
            raise ValueError("pixel_size must be a 2-tuple (width, height).")
        px_w = _even(max(2, int(round(float(pixel_size[0])))))
        px_h = _even(max(2, int(round(float(pixel_size[1])))))
        return (px_w / float(dpi), px_h / float(dpi)), (px_w, px_h)

    if figsize is not None:
        base_w, base_h = float(figsize[0]), float(figsize[1])
    else:
        if with_barcode:
            if total_figsize is None:
                base_w, base_h = 10.0, 5.0
            else:
                base_w, base_h = float(total_figsize[0]), float(total_figsize[1])
        else:
            if cloud_figsize is None:
                base_w, base_h = 6.0, 6.0
            else:
                base_w, base_h = float(cloud_figsize[0]), float(cloud_figsize[1])

    px_w = _even(max(2, int(round(base_w * float(dpi)))))
    px_h = _even(max(2, int(round(base_h * float(dpi)))))
    return (px_w / float(dpi), px_h / float(dpi)), (px_w, px_h)


def _resolve_figure_margins(
    *,
    with_barcode: bool,
    figure_margins: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """
    Resolve validated Matplotlib subplot margins.

    Parameters
    ----------
    with_barcode : bool
        Selects defaults for one-panel or two-panel layouts.
    figure_margins : dict[str, float] | None
        Optional dict with exactly ``left``, ``right``, ``bottom`` and ``top``.

    Returns
    -------
    dict[str, float]
        Margins suitable for ``fig.subplots_adjust``.
    """
    if figure_margins is None:
        if with_barcode:
            return {"left": 0.02, "right": 0.98, "bottom": 0.10, "top": 0.90}
        return {"left": 0.02, "right": 0.98, "bottom": 0.10, "top": 0.92}

    if not isinstance(figure_margins, dict):
        raise ValueError(
            "figure_margins must be a dict with keys {'left','right','bottom','top'}."
        )
    required_keys = {"left", "right", "bottom", "top"}
    provided_keys = set(figure_margins.keys())
    if provided_keys != required_keys:
        raise ValueError(
            "figure_margins must contain exactly the keys "
            "{'left','right','bottom','top'}."
        )

    margins: dict[str, float] = {}
    for key in ("left", "right", "bottom", "top"):
        value = figure_margins[key]
        if not isinstance(value, Real):
            raise ValueError(f"figure_margins['{key}'] must be numeric.")
        margins[key] = float(value)

    if not (0.0 <= margins["left"] < margins["right"] <= 1.0):
        raise ValueError("figure_margins must satisfy 0 <= left < right <= 1.")
    if not (0.0 <= margins["bottom"] < margins["top"] <= 1.0):
        raise ValueError("figure_margins must satisfy 0 <= bottom < top <= 1.")

    return margins


def _parse_camera_eye(camera_eye: Optional[Any]) -> tuple[float, float]:
    """
    Convert a camera specification to (elev, azim) for matplotlib 3D.
    """
    elev, azim = 22.0, -55.0
    if camera_eye is None:
        return elev, azim
    if isinstance(camera_eye, dict):
        elev = float(camera_eye.get("elev", elev))
        azim = float(camera_eye.get("azim", azim))
        return elev, azim
    if isinstance(camera_eye, (tuple, list)) and len(camera_eye) >= 2:
        return float(camera_eye[0]), float(camera_eye[1])
    raise ValueError(
        "camera_eye must be None, a dict with keys {'elev','azim'}, "
        "or a tuple/list (elev, azim)."
    )


def _ffmpeg_path_or_raise() -> str:
    """
    Return ffmpeg executable path, or raise with a clear install hint.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg was not found in PATH. Install ffmpeg to export MP4 "
            "(for example: `brew install ffmpeg`)."
        )
    return ffmpeg_path


def _assemble_mp4_with_ffmpeg(
    *,
    frame_dir: Path,
    fps: int,
    output_file: Path,
) -> None:
    """
    Assemble PNG frames into an MP4 via ffmpeg.
    """
    ffmpeg = _ffmpeg_path_or_raise()
    input_pattern = str(frame_dir / "frame_%04d.png")
    cmd = [
        ffmpeg,
        "-y",
        "-r",
        str(int(fps)),
        "-i",
        input_pattern,
        "-vcodec",
        "libx264",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        str(output_file),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed while assembling MP4:\n{err}") from e


def _animate_filtration_generic_3d_matplotlib(
    forest,
    filename: str,
    *,
    with_barcode: bool = False,
    fps: int = 20,
    frames: int = 200,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    coloring: Literal["forest", "bars"] = "forest",
    show_cycles: bool = True,
    signed: bool = False,
    min_bar_length: float = 0.0,
    show_complex: bool = True,
    complex_opacity: float = 0.20,
    cycle_opacity: float = 0.55,
    vertex_size: float = 3.0,
    cloud_figsize: tuple[float, float] = (6.0, 6.0),
    total_figsize: Optional[tuple[float, float]] = None,
    barcode_kwargs: Optional[dict] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    camera_mode: Literal["fixed", "orbit"] = "fixed",
    camera_eye: Optional[Any] = None,
    dpi: int = 200,
    alpha_digits: Optional[int] = None,
) -> None:
    """
    Animate a 3D filtration and export as MP4 using matplotlib.

    This implementation avoids browser-based rendering and instead
    generates frames directly via matplotlib, which are then assembled
    into a video using ffmpeg.

    Parameters
    ----------
    forest : PersistenceForest
        Forest instance with 3D point cloud data.
    filename : str
        Output MP4 path.
    with_barcode : bool
        Whether to render a second panel with barcode and moving time marker.
    fps : int
        Frames per second for the resulting video.
    frames : int
        Number of animation frames.
    t_min, t_max : float | None
        Filtration interval. If omitted, inferred from forest data.
    coloring : {"forest", "bars"}
        Bar color map strategy used for cycle surfaces.
    show_cycles : bool
        If True, show active cycle representatives.
    signed : bool
        If False, cancel opposite-oriented duplicate simplices in cycle chains.
    min_bar_length : float
        Exclude bars shorter than this threshold.
    show_complex : bool
        If True, draw complex edges and boundary triangles.
    complex_opacity : float
        Opacity used for complex boundary surfaces.
    cycle_opacity : float
        Opacity used for cycle surfaces.
    vertex_size : float
        Marker size for vertices.
    cloud_figsize : tuple[float, float]
        Figure size used when ``with_barcode=False``.
    total_figsize : tuple[float, float] | None
        Total figure size used when ``with_barcode=True``.
    barcode_kwargs : dict | None
        Extra kwargs forwarded to ``forest.plot_barcode`` (except ``ax``).
    width, height : int | None
        Optional figure size in pixels.
    camera_mode : {"fixed", "orbit"}
        Camera behavior. Orbit rotates azimuth uniformly across frames.
    camera_eye : Any
        Camera specification as ``(elev, azim)`` or ``{"elev": ..., "azim": ...}``.
    dpi : int
        PNG rendering DPI before ffmpeg assembly.
    alpha_digits : int | None
        Number of digits shown in the filtration value overlay.

    Returns
    -------
    None
        Writes the MP4 to ``filename``.
    """
    from matplotlib import colors as mcolors
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    if int(fps) <= 0:
        raise ValueError("fps must be a positive integer.")
    if forest.dim != 3:
        raise ValueError("_animate_filtration_generic_3d_matplotlib requires ambient dimension 3.")
    if camera_mode not in {"fixed", "orbit"}:
        raise ValueError("camera_mode must be 'fixed' or 'orbit'.")

    frame_times = _compute_frame_times(
        forest=forest,
        frames=frames,
        t_min=t_min,
        t_max=t_max,
    )
    color_map = forest._get_color_map(coloring=coloring)

    pts = np.asarray(forest.point_cloud, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("point_cloud must be an (n_points, 3) array-like.")

    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    spans = np.maximum(maxs - mins, 1e-9)
    pad = 0.05 * np.max(spans)
    xlim = (mins[0] - pad, maxs[0] + pad)
    ylim = (mins[1] - pad, maxs[1] + pad)
    zlim = (mins[2] - pad, maxs[2] + pad)
    box_aspect = (xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])

    elev0, azim0 = _parse_camera_eye(camera_eye)
    figsize, _frame_px = _resolve_matplotlib_figsize(
        with_barcode=with_barcode,
        width=width,
        height=height,
        dpi=int(dpi),
        cloud_figsize=cloud_figsize,
        total_figsize=total_figsize,
    )
    out_path = Path(filename).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if barcode_kwargs is None:
        barcode_kwargs = {}
    else:
        barcode_kwargs = dict(barcode_kwargs)
    barcode_kwargs.pop("ax", None)
    barcode_kwargs = {
        "sort": "length",
        "title": "Barcode",
        "xlabel": "filtration value",
        "tight_layout": False,
        "coloring": coloring,
        "min_bar_length": min_bar_length,
        **barcode_kwargs,
    }

    def _draw_scene(ax_scene, t: float, frame_idx: int) -> None:
        ax_scene.cla()
        snapshot = forest._complex_snapshot_at_filtration(float(t))
        frame_pts = snapshot["points"]

        if show_complex:
            edges = snapshot.get("edges", [])
            if edges:
                segments = [frame_pts[list(edge)] for edge in edges]
                edge_coll = Line3DCollection(
                    segments,
                    colors="0.35",
                    linewidths=0.7,
                    alpha=max(0.2, float(complex_opacity)),
                )
                ax_scene.add_collection3d(edge_coll)

            triangles = snapshot.get("triangles", [])
            if triangles:
                tri_polys = [frame_pts[list(tri)] for tri in triangles]
                tri_coll = Poly3DCollection(
                    tri_polys,
                    facecolors=mcolors.to_rgba("lightblue", alpha=complex_opacity),
                    edgecolors="none",
                )
                ax_scene.add_collection3d(tri_coll)

        ax_scene.scatter(
            frame_pts[:, 0],
            frame_pts[:, 1],
            frame_pts[:, 2],
            s=vertex_size,
            c="black",
            depthshade=False,
        )

        if show_cycles:
            active = forest._active_bars_with_cycles_at(
                filt_val=float(t),
                min_bar_length=min_bar_length,
            )
            active = sorted(active, key=lambda bc: bc[0].lifespan(), reverse=True)

            for bar, cycle in active:
                tri_faces = forest._chain_triangles_3d(cycle, signed=signed)
                if not tri_faces:
                    continue
                cycle_polys = [frame_pts[list(face)] for face in tri_faces]
                color = color_map.get(bar, "#d62728")
                cycle_coll = Poly3DCollection(
                    cycle_polys,
                    facecolors=mcolors.to_rgba(color, alpha=cycle_opacity),
                    edgecolors=mcolors.to_rgba(color, alpha=min(1.0, cycle_opacity + 0.25)),
                    linewidths=0.2,
                )
                ax_scene.add_collection3d(cycle_coll)

        ax_scene.set_xlim(*xlim)
        ax_scene.set_ylim(*ylim)
        ax_scene.set_zlim(*zlim)
        ax_scene.set_box_aspect(box_aspect)
        ax_scene.set_xlabel("x")
        ax_scene.set_ylabel("y")
        ax_scene.set_zlabel("z")
        ax_scene.set_title(f"Filtration value r = {float(t):.4g}")
        if alpha_digits is None:
            radius_text = rf"$\alpha = {float(t):.3g}$"
        else:
            radius_text = rf"$\alpha = {float(t):.{alpha_digits}f}$"
        ax_scene.text2D(
            0.02,
            0.98,
            radius_text,
            transform=ax_scene.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

        if camera_mode == "orbit":
            denom = max(1, len(frame_times) - 1)
            azim = azim0 + 360.0 * float(frame_idx) / float(denom)
        else:
            azim = azim0
        ax_scene.view_init(elev=elev0, azim=azim)

    with tempfile.TemporaryDirectory(prefix="persforest_3d_frames_") as tmp_dir:
        frame_dir = Path(tmp_dir)
        for idx, t in enumerate(frame_times):
            fig = plt.figure(figsize=figsize)
            if with_barcode:
                gs = fig.add_gridspec(1, 2, width_ratios=[3.5, 2.0])
                ax_scene = fig.add_subplot(gs[0, 0], projection="3d")
                ax_bar = fig.add_subplot(gs[0, 1])

                forest.plot_barcode(
                    ax=ax_bar,
                    **barcode_kwargs,
                )
                ax_bar.set_xlim(float(frame_times[0]), float(frame_times[-1]))
                ax_bar.axvline(float(t), color="k", linewidth=2.0)
            else:
                ax_scene = fig.add_subplot(111, projection="3d")

            _draw_scene(ax_scene=ax_scene, t=float(t), frame_idx=idx)

            frame_file = frame_dir / f"frame_{idx:04d}.png"
            fig.savefig(frame_file, dpi=dpi, facecolor=fig.get_facecolor())
            plt.close(fig)

        _assemble_mp4_with_ffmpeg(
            frame_dir=frame_dir,
            fps=int(fps),
            output_file=out_path,
        )

def animate_filtration_pair(
    forest1,
    forest2,
    filename: Optional[str] = None,
    *,
    fps: int = 20,
    frames: int = 200,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    dpi: int = 200,
    total_figsize: Optional[Tuple[float, float]] = None,
    plot_kwargs_forest1: Optional[dict] = None,
    plot_kwargs_forest2: Optional[dict] = None,
    barcode_kwargs_forest1: Optional[dict] = None,
    barcode_kwargs_forest2: Optional[dict] = None,
):
    """
    Animate two Forests side-by-side: for each forest, show the evolving
    cycle representatives in the point cloud together with its barcode.

    The filtration panels and barcodes are styled via `plot_at_filtration` and
    `_plot_barcode`, so they match what `animate_filtration` produces.

    Parameters
    ----------
    forest1, forest2 : Forest
        The two forests to animate.
    filename : str or None, optional
        If not None, the animation is also written to this file.
        The extension ('.mp4', '.gif', etc.) determines the writer.
    fps : int, optional
        Frames per second for the saved animation.
    frames : int, optional
        Number of frames in the animation (shared time grid).
    t_min, t_max : float or None, optional
        Filtration time window. If None, t_min = 0 and t_max is the max
        filtration value across both forests.
    dpi : int, optional
        DPI for saving.
    total_figsize : (float, float) or None, optional
        Overall figure size. If None, a reasonable default is used.
    plot_kwargs_forest1, plot_kwargs_forest2 : dict or None, optional
        Extra kwargs forwarded to `plot_at_filtration` for each forest.
        Used on the *cloud/loop* panels.
    barcode_kwargs_forest1, barcode_kwargs_forest2 : dict or None, optional
        Extra kwargs forwarded to `_plot_barcode` for each forest.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    fig : matplotlib.figure.Figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    # --- 1) Sanity checks -----------------------------------------------------
    for forest, name in ((forest1, "forest1"), (forest2, "forest2")):
        if not hasattr(forest, "filtration") or not forest.filtration:
            raise ValueError(f"{name} has no filtration data to animate.")
        if not getattr(forest, "barcode", None):
            raise ValueError(f"{name} has an empty barcode.")

    # --- 2) Time grid ---------------------------------------------------------
    if t_min is None:
        t_start: float = 0.0
    else:
        t_start = float(t_min)

    if t_max is None:
        max1 = max(node.filt_val for node in forest1.nodes.values())
        max2 = max(node.filt_val for node in forest2.nodes.values())
        t_end: float = float(max(max1, max2))*1.05
    else:
        t_end = float(t_max)*1.05

    n_frames: int = int(frames)

    frame_times = np.linspace(start = t_start, stop = t_end, num = n_frames).tolist()

    # --- 3) Figure layout: 2x2 (clouds on top, barcodes below) ----------------
    if total_figsize is None:
        total_figsize = (12.0, 8.0)

    fig, ((ax_cloud_1, ax_cloud_2),
          (ax_bar_1,   ax_bar_2)) = plt.subplots(
        2,
        2,
        figsize=total_figsize,
        gridspec_kw={"height_ratios": [5, 2]},
    )

    # --- 4) Barcode plotting --------------------------------------------------
    # Base kwargs to resemble your existing barcode style
    base_barcode_kwargs = {
        "sort": "length",
        "xlabel": "filtration value",
        "tight_layout": False,
    }

    if barcode_kwargs_forest1 is None:
        barcode_kwargs_forest1 = {}
    if barcode_kwargs_forest2 is None:
        barcode_kwargs_forest2 = {}

    kwargs_bar_1 = {**base_barcode_kwargs, **barcode_kwargs_forest1}
    kwargs_bar_2 = {**base_barcode_kwargs, **barcode_kwargs_forest2}

    if "title" not in kwargs_bar_1:
        kwargs_bar_1["title"] = "Barcode"
    if "title" not in kwargs_bar_2:
        kwargs_bar_2["title"] = "Barcode"

    forest1.plot_barcode(ax=ax_bar_1, **kwargs_bar_1)
    forest2.plot_barcode(ax=ax_bar_2, **kwargs_bar_2)

    # Force both barcodes to share the same x-range as the animation time
    ax_bar_1.set_xlim(t_start, t_end)
    ax_bar_2.set_xlim(t_start, t_end)

    # Vertical lines tracking the current filtration value
    t0 = frame_times[0]
    barcode_line_1 = ax_bar_1.axvline(t0, color="k", linewidth=2)
    barcode_line_2 = ax_bar_2.axvline(t0, color="k", linewidth=2)

    # --- 5) Filtration plot kwargs (match animate_filtration defaults) --------
    base_plot_kwargs = {
        "fill_triangles": True,
        "loop_vertex_markers": False,
        "vertex_size": 3,
        "coloring": "forest",  # uses the forest's color dict, shared with barcode
        "show": False,         # we manage figure/axes ourselves
    }

    if plot_kwargs_forest1 is None:
        plot_kwargs_forest1 = {}
    if plot_kwargs_forest2 is None:
        plot_kwargs_forest2 = {}

    kwargs_cloud_1 = {**base_plot_kwargs, **plot_kwargs_forest1}
    kwargs_cloud_2 = {**base_plot_kwargs, **plot_kwargs_forest2}

    if "title" not in plot_kwargs_forest1:
        plot_kwargs_forest1["title"] = "Acitve Loops"
    if "title" not in plot_kwargs_forest2:
        plot_kwargs_forest2["title"] = "Acitve Loops"

    # --- 6) Helpers to draw a single frame -----------------------------------
    def _draw_frame_at_time(t: float):
        # First forest
        ax_cloud_1.clear()
        forest1.plot_at_filtration(filt_val=t, ax=ax_cloud_1, **kwargs_cloud_1)
        ax_cloud_1.text(
            0.02, 0.98, rf"$\alpha = {t:.3g}$",
            transform=ax_cloud_1.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

        # Second forest
        ax_cloud_2.clear()
        forest2.plot_at_filtration(filt_val=t, ax=ax_cloud_2, **kwargs_cloud_2)
        ax_cloud_2.text(
            0.02, 0.98, rf"$\alpha = {t:.3g}$",
            transform=ax_cloud_2.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    def _align_barcode_axes():
        # Make barcode axes have same left/right as their corresponding cloud axes
        for cloud_ax, bar_ax in ((ax_cloud_1, ax_bar_1), (ax_cloud_2, ax_bar_2)):
            cloud_pos = cloud_ax.get_position()
            bar_pos = bar_ax.get_position()
            # Keep bar's vertical position/height, but match horizontal start and width
            bar_ax.set_position([
                cloud_pos.x0,    # left
                bar_pos.y0,      # bottom
                cloud_pos.width, # width
                bar_pos.height,  # height
            ])

    # --- 7) Animation callbacks -----------------------------------------------
    def init():
        _draw_frame_at_time(frame_times[0])
        _align_barcode_axes()
        t_init = frame_times[0]
        barcode_line_1.set_xdata([t_init, t_init])
        barcode_line_2.set_xdata([t_init, t_init])
        return []

    def update(frame_idx: int):
        t = frame_times[frame_idx]
        _draw_frame_at_time(t)
        _align_barcode_axes()
        barcode_line_1.set_xdata([t, t])
        barcode_line_2.set_xdata([t, t])
        return []

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_times),
        init_func=init,
        blit=False,
    )

    # --- 8) Optional save to disk ---------------------------------------------
    if filename is not None:
        fname = str(filename)
        ext = fname.lower().rsplit(".", 1)[-1] if "." in fname else ""
        if ext == "mp4":
            writer = FFMpegWriter(
                fps=fps, 
                #bitrate=2000,
                codec="libx264",
                extra_args=[
                    "-pix_fmt", "yuv420p",
                    "-vf", "scale=1920:-2",     
                    "-profile:v", "baseline",
                    "-level", "3.1",
                    "-movflags", "+faststart",
                ],)
            anim.save(fname, writer=writer, dpi=dpi)
        elif ext in {"gif", "gifv"}:
            try:
                from matplotlib.animation import PillowWriter
            except ImportError as e:
                raise RuntimeError(
                    "Saving as GIF requires Pillow. Install it with `pip install pillow` "
                    "or `pip install \".[animation]\"`."
                ) from e
            writer = PillowWriter(fps=fps)
            anim.save(fname, writer=writer, dpi=dpi)
        else:
            # Fallback to default writer chosen by matplotlib
            anim.save(fname, dpi=dpi, fps=fps)

    return anim, fig
