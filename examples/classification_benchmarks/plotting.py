"""Plotting utilities for embedded planar graph benchmark examples."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt


def _as_nodes(nodes: Any) -> np.ndarray:
    """Return validated node coordinates as a float array."""
    nodes_array = np.asarray(nodes, dtype=np.float64)
    if nodes_array.ndim != 2 or nodes_array.shape[1] != 2:
        raise ValueError("graph.nodes must have shape (n_nodes, 2)")
    return nodes_array


def _as_edges(edges: Any, n_nodes: int) -> np.ndarray:
    """Return validated edge endpoint indices as an int array."""
    edges_array = np.asarray(edges, dtype=np.int64)
    if edges_array.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if edges_array.ndim != 2 or edges_array.shape[1] != 2:
        raise ValueError("graph.edges must have shape (n_edges, 2)")
    if np.any(edges_array < 0) or np.any(edges_array >= n_nodes):
        raise ValueError("graph.edges contains node indices outside graph.nodes")
    return edges_array


def _edge_segments(nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return line segments for a set of graph edges."""
    if edges.size == 0:
        return np.empty((0, 2, 2), dtype=np.float64)
    return nodes[edges]


def _final_edge_indices_from_original(labels: dict[str, object], original_indices: Sequence[int]) -> list[int]:
    """Map original edge indices to final edge indices after removals."""
    removed = sorted(int(index) for index in labels.get("removed_edges_original", []))
    mapped: list[int] = []
    for index in original_indices:
        original = int(index)
        if original in removed:
            continue
        final = original - sum(removed_index < original for removed_index in removed)
        mapped.append(final)
    return mapped


def _plot_edge_subset(
    ax,
    nodes: np.ndarray,
    edges: np.ndarray,
    edge_indices: Sequence[int],
    *,
    color: str,
    linewidth: float,
    zorder: int,
) -> None:
    """Overlay a highlighted subset of existing graph edges."""
    valid_indices = [int(index) for index in edge_indices if 0 <= int(index) < edges.shape[0]]
    if not valid_indices:
        return
    segments = _edge_segments(nodes, edges[np.asarray(valid_indices, dtype=np.int64)])
    ax.add_collection(LineCollection(segments, colors=color, linewidths=linewidth, zorder=zorder))


def _plot_removed_edges(
    ax,
    removed_edge_coordinate_pairs: object,
    *,
    color: str,
    linewidth: float,
) -> None:
    """Overlay removed edges encoded directly as coordinate pairs."""
    pairs = np.asarray(removed_edge_coordinate_pairs, dtype=np.float64)
    if pairs.size == 0:
        return
    if pairs.ndim != 3 or pairs.shape[1:] != (2, 2):
        return
    collection = LineCollection(
        pairs,
        colors=color,
        linewidths=linewidth,
        linestyles="dashed",
        alpha=0.8,
        zorder=3,
    )
    ax.add_collection(collection)


def plot_planar_graph(
    graph,
    ax=None,
    *,
    node_size: float = 8,
    edge_width: float = 1.0,
    edge_color: str = "0.25",
    node_color: str = "black",
    title: str | None = None,
    show_nodes: bool = True,
    equal_aspect: bool = True,
    highlight_metadata: bool = True,
):
    """Plot an embedded planar graph.

    Parameters
    ----------
    graph:
        Any object with ``nodes`` and ``edges`` attributes. ``nodes`` must be a
        numeric ``(n_nodes, 2)`` array and ``edges`` must be an integer
        ``(n_edges, 2)`` array of node indices.
    ax:
        Optional Matplotlib axes. A new figure and axes are created when this is
        omitted.
    highlight_metadata:
        If true, highlight benchmark-specific metadata such as tendril edges,
        branch edges, thin transport edges, and removed transport edges.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the graph plot.
    """
    nodes = _as_nodes(graph.nodes)
    edges = _as_edges(graph.edges, nodes.shape[0])
    labels = getattr(graph, "labels", {}) or {}

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    segments = _edge_segments(nodes, edges)
    if segments.size > 0:
        ax.add_collection(
            LineCollection(
                segments,
                colors=edge_color,
                linewidths=edge_width,
                capstyle="round",
                joinstyle="round",
                zorder=1,
            )
        )

    if highlight_metadata and isinstance(labels, dict):
        family = labels.get("family")
        if family == "loop_tendrils":
            _plot_edge_subset(
                ax,
                nodes,
                edges,
                labels.get("tendril_edge_indices", []),
                color="#d95f02",
                linewidth=max(1.8 * edge_width, edge_width + 0.6),
                zorder=2,
            )
            _plot_edge_subset(
                ax,
                nodes,
                edges,
                labels.get("branch_edge_indices", []),
                color="#7570b3",
                linewidth=max(1.8 * edge_width, edge_width + 0.6),
                zorder=3,
            )
        elif isinstance(labels.get("protrusion_edge_indices"), list):
            _plot_edge_subset(
                ax,
                nodes,
                edges,
                labels.get("protrusion_edge_indices", []),
                color="#d95f02",
                linewidth=max(1.8 * edge_width, edge_width + 0.6),
                zorder=2,
            )
        elif family == "damaged_transport_network":
            thin_indices = _final_edge_indices_from_original(labels, labels.get("thin_edges", []))
            _plot_edge_subset(
                ax,
                nodes,
                edges,
                thin_indices,
                color="#d95f02",
                linewidth=max(2.2 * edge_width, edge_width + 0.8),
                zorder=2,
            )
            _plot_removed_edges(
                ax,
                labels.get("removed_edge_coordinate_pairs", []),
                color="#b2182b",
                linewidth=max(1.7 * edge_width, edge_width + 0.5),
            )

    if show_nodes and nodes.size > 0:
        ax.scatter(
            nodes[:, 0],
            nodes[:, 1],
            s=node_size,
            color=node_color,
            linewidths=0,
            zorder=4,
        )

    if title is not None:
        ax.set_title(title)

    ax.autoscale()
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    ax.margins(0.08)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    return ax


def plot_sampled_planar_graph(
    graph,
    sample_points,
    ax=None,
    *,
    sample_size: float = 5,
    vertex_size: float = 13,
    sample_color: str = "#2c7fb8",
    vertex_color: str = "#d95f02",
    edge_width: float = 0.8,
    edge_color: str = "0.72",
    title: str | None = None,
    equal_aspect: bool = True,
    highlight_metadata: bool = False,
):
    """Plot a planar graph with sampled edge points overlaid."""
    nodes = _as_nodes(graph.nodes)
    samples = np.asarray(sample_points, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("sample_points must have shape (n_samples, 2)")

    ax = plot_planar_graph(
        graph,
        ax=ax,
        edge_width=edge_width,
        edge_color=edge_color,
        show_nodes=False,
        equal_aspect=equal_aspect,
        highlight_metadata=highlight_metadata,
    )

    if samples.size > 0:
        ax.scatter(
            samples[:, 0],
            samples[:, 1],
            s=sample_size,
            color=sample_color,
            linewidths=0,
            alpha=0.85,
            zorder=4,
        )

    if nodes.size > 0:
        ax.scatter(
            nodes[:, 0],
            nodes[:, 1],
            s=vertex_size,
            color=vertex_color,
            edgecolors="white",
            linewidths=0.35,
            zorder=5,
        )

    if title is not None:
        ax.set_title(title)

    return ax
