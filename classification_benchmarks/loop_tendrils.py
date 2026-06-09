"""Generate loop-with-inward-tendrils benchmark graphs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_CLASSES = (
    "clean",
    "short_sparse",
    "long_sparse",
    "short_dense",
    "long_dense",
    "branching_sparse",
    "branching_dense",
    "polarized",
    "paired_spikes",
)
SUPPORTED_TENDRIL_MODES = ("single", "paired", "branching")
SUPPORTED_ROOT_MODES = ("uniform", "polarized")


@dataclass(frozen=True)
class EmbeddedPlanarGraph:
    """Embedded planar graph stored as nodes, edges, and metadata labels."""

    nodes: np.ndarray
    edges: np.ndarray
    labels: dict[str, object]


def _normalize_nodes(nodes: np.ndarray) -> np.ndarray:
    """Fit nodes into [0, 1]^2 while preserving aspect ratio."""
    normalized = np.asarray(nodes, dtype=np.float64).copy()
    mins = normalized.min(axis=0)
    maxs = normalized.max(axis=0)
    spans = maxs - mins
    max_span = float(spans.max())

    if max_span == 0.0:
        return np.full_like(normalized, 0.5, dtype=np.float64)

    normalized = (normalized - mins) / max_span
    normalized += (1.0 - spans / max_span) / 2.0
    return normalized.astype(np.float64, copy=False)


def _edge_lengths(nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return Euclidean lengths for graph edges."""
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        return np.empty(0, dtype=np.float64)
    vectors = nodes[edges[:, 1]] - nodes[edges[:, 0]]
    return np.linalg.norm(vectors, axis=1).astype(np.float64)


def _loop_nodes(
    n_loop_vertices: int,
    loop_radius: float,
    loop_noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate polygonal loop nodes around a noisy circle."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_loop_vertices, endpoint=False)
    if loop_noise > 0.0:
        radii = loop_radius * (1.0 + loop_noise * rng.normal(size=n_loop_vertices))
        radii = np.maximum(radii, 0.1 * loop_radius)
    else:
        radii = np.full(n_loop_vertices, loop_radius, dtype=np.float64)

    return np.column_stack((radii * np.cos(theta), radii * np.sin(theta))).astype(np.float64)


def _loop_edges(n_loop_vertices: int) -> np.ndarray:
    """Return cycle edges around the outer loop."""
    edges = [(i, (i + 1) % n_loop_vertices) for i in range(n_loop_vertices)]
    return np.asarray(edges, dtype=np.int64)


def _circular_distance(i: int, j: int, n: int) -> int:
    """Return circular index distance between two loop indices."""
    delta = abs(int(i) - int(j))
    return min(delta, n - delta)


def _select_roots(
    n_loop_vertices: int,
    n_roots: int,
    min_separation: int,
    rng: np.random.Generator,
    allowed_indices: np.ndarray | None = None,
) -> list[int]:
    """Select loop roots with circular separation when feasible."""
    if n_roots <= 0:
        return []

    allowed = (
        np.arange(n_loop_vertices, dtype=np.int64)
        if allowed_indices is None
        else np.asarray(allowed_indices, dtype=np.int64)
    )
    allowed = np.unique(allowed)
    if allowed.size == 0:
        allowed = np.arange(n_loop_vertices, dtype=np.int64)

    for separation in range(min_separation, -1, -1):
        candidates = allowed.copy()
        rng.shuffle(candidates)
        roots: list[int] = []
        for candidate in candidates:
            root = int(candidate)
            if all(_circular_distance(root, existing, n_loop_vertices) >= separation for existing in roots):
                roots.append(root)
                if len(roots) == n_roots:
                    return sorted(roots)

    sampled = rng.choice(allowed, size=min(n_roots, allowed.size), replace=False)
    if sampled.size < n_roots:
        extra = rng.choice(np.arange(n_loop_vertices), size=n_roots - sampled.size, replace=False)
        sampled = np.concatenate((sampled, extra))
    return sorted(int(index) for index in sampled[:n_roots])


def _angle_in_sector(theta: np.ndarray, center: float, width: float) -> np.ndarray:
    """Return mask for angles lying inside a circular sector."""
    wrapped = (theta - center + np.pi) % (2.0 * np.pi) - np.pi
    return np.abs(wrapped) <= width / 2.0


def _unit_inward_and_tangent(point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return inward radial unit vector and tangent for a loop point."""
    norm = float(np.linalg.norm(point))
    if norm == 0.0:
        inward = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        inward = -point / norm
    tangent = np.asarray([-inward[1], inward[0]], dtype=np.float64)
    return inward, tangent


def _add_single_tendril(
    nodes_list: list[list[float]],
    edges_list: list[tuple[int, int]],
    root_idx: int,
    length: float,
    segments_per_tendril: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """Add one curved inward tendril path from a loop root."""
    root = np.asarray(nodes_list[root_idx], dtype=np.float64)
    inward, tangent = _unit_inward_and_tangent(root)
    curvature = float(rng.uniform(-0.04, 0.04))

    new_nodes: list[int] = []
    new_edges: list[int] = []
    previous = root_idx
    for segment in range(1, segments_per_tendril + 1):
        s = segment / segments_per_tendril
        point = root + s * length * inward + curvature * np.sin(np.pi * s) * tangent
        current = len(nodes_list)
        nodes_list.append([float(point[0]), float(point[1])])
        edges_list.append((previous, current))
        new_nodes.append(current)
        new_edges.append(len(edges_list) - 1)
        previous = current

    return new_nodes, new_edges


def _add_branching_tendril(
    nodes_list: list[list[float]],
    edges_list: list[tuple[int, int]],
    root_idx: int,
    length: float,
    segments_per_tendril: int,
    branching_probability: float,
    rng: np.random.Generator,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Add one inward tendril, optionally with two small side branches."""
    new_nodes, tendril_edges = _add_single_tendril(
        nodes_list,
        edges_list,
        root_idx,
        length,
        segments_per_tendril,
        rng,
    )
    branch_edges: list[int] = []
    branch_points: list[int] = []

    if new_nodes and rng.random() < branching_probability:
        branch_point = new_nodes[-2] if len(new_nodes) >= 2 else new_nodes[-1]
        point = np.asarray(nodes_list[branch_point], dtype=np.float64)
        inward, tangent = _unit_inward_and_tangent(point)
        branch_length = 0.35 * length

        for sign in (-1.0, 1.0):
            direction = inward + sign * 0.85 * tangent
            direction /= np.linalg.norm(direction)
            tip = point + branch_length * direction
            tip_idx = len(nodes_list)
            nodes_list.append([float(tip[0]), float(tip[1])])
            edges_list.append((branch_point, tip_idx))
            new_nodes.append(tip_idx)
            branch_edges.append(len(edges_list) - 1)
            branch_points.append(branch_point)

    return new_nodes, tendril_edges, branch_edges, branch_points


def _add_paired_spike(
    nodes_list: list[list[float]],
    edges_list: list[tuple[int, int]],
    root_i: int,
    root_j: int,
    length: float,
) -> tuple[list[int], list[int]]:
    """Add one inward V-shaped paired spike from two nearby loop roots."""
    point_i = np.asarray(nodes_list[root_i], dtype=np.float64)
    point_j = np.asarray(nodes_list[root_j], dtype=np.float64)
    midpoint = 0.5 * (point_i + point_j)
    inward, _ = _unit_inward_and_tangent(midpoint)
    tip = midpoint + length * inward
    tip_idx = len(nodes_list)
    nodes_list.append([float(tip[0]), float(tip[1])])

    edge_indices: list[int] = []
    edges_list.append((root_i, tip_idx))
    edge_indices.append(len(edges_list) - 1)
    edges_list.append((root_j, tip_idx))
    edge_indices.append(len(edges_list) - 1)
    return [tip_idx], edge_indices


def _deduplicate_edges(
    edges: list[tuple[int, int]],
) -> tuple[np.ndarray, dict[int, int | None]]:
    """Deduplicate edges and map original edge indices to final indices."""
    seen: dict[tuple[int, int], int] = {}
    unique_edges: list[tuple[int, int]] = []
    old_to_new: dict[int, int | None] = {}

    for old_idx, (u, v) in enumerate(edges):
        if u == v:
            old_to_new[old_idx] = None
            continue
        edge = (u, v) if u < v else (v, u)
        if edge in seen:
            old_to_new[old_idx] = seen[edge]
            continue
        old_to_new[old_idx] = len(unique_edges)
        seen[edge] = len(unique_edges)
        unique_edges.append(edge)

    return np.asarray(unique_edges, dtype=np.int64), old_to_new


def _remap_edge_indices(indices: list[int], old_to_new: dict[int, int | None]) -> list[int]:
    """Remap metadata edge indices after edge deduplication."""
    remapped: list[int] = []
    for index in indices:
        new_index = old_to_new.get(index)
        if new_index is not None and int(new_index) not in remapped:
            remapped.append(int(new_index))
    return remapped


def _class_defaults(
    class_name: str,
    n_tendrils: int | None,
    tendril_length: float | None,
    tendril_mode: str,
    branching_probability: float | None,
) -> tuple[int, float, str, float]:
    """Resolve class-specific defaults."""
    defaults: dict[str, tuple[int, float, str, float]] = {
        "clean": (0, 0.0, "single", 0.0),
        "short_sparse": (6, 0.18, "single", 0.0),
        "long_sparse": (6, 0.38, "single", 0.0),
        "short_dense": (18, 0.18, "single", 0.0),
        "long_dense": (18, 0.38, "single", 0.0),
        "branching_dense": (14, 0.25, "branching", 0.75),
        "polarized": (14, 0.22, "single", 0.0),
        "paired_spikes": (8, 0.25, "paired", 0.0),
        "branching_sparse": (6, 0.32, "branching", 0.9),
    }
    default_count, default_length, default_mode, default_branching = defaults[class_name]
    count = default_count if n_tendrils is None else int(n_tendrils)
    length = default_length if tendril_length is None else float(tendril_length)
    mode = default_mode if class_name in {"branching_dense", "paired_spikes", "branching_sparse"} else tendril_mode
    branch_prob = default_branching if branching_probability is None else float(branching_probability)
    return count, length, mode, branch_prob


def _validate_inputs(
    class_name: str,
    n_loop_vertices: int,
    loop_radius: float,
    loop_noise: float,
    n_tendrils: int,
    tendril_length: float,
    tendril_length_variation: float,
    segments_per_tendril: int,
    tendril_mode: str,
    branching_probability: float,
    root_mode: str,
    polarized_sector_width: float,
    min_root_separation: int,
    *,
    parametric: bool = False,
) -> None:
    """Validate generator inputs with clear errors."""
    supported_class_names = SUPPORTED_CLASSES + (("parametric",) if parametric else ())
    if class_name not in supported_class_names:
        raise ValueError(
            f"Unknown class_name {class_name!r}. "
            f"Supported values are: {', '.join(supported_class_names)}"
        )
    if tendril_mode not in SUPPORTED_TENDRIL_MODES:
        raise ValueError(
            f"Unknown tendril_mode {tendril_mode!r}. "
            f"Supported values are: {', '.join(SUPPORTED_TENDRIL_MODES)}"
        )
    if root_mode not in SUPPORTED_ROOT_MODES:
        raise ValueError(
            f"Unknown root_mode {root_mode!r}. "
            f"Supported values are: {', '.join(SUPPORTED_ROOT_MODES)}"
        )
    if not isinstance(n_loop_vertices, int):
        raise TypeError("n_loop_vertices must be an int")
    if n_loop_vertices < 12:
        raise ValueError("n_loop_vertices must be at least 12")
    if loop_radius <= 0.0:
        raise ValueError("loop_radius must be positive")
    if loop_noise < 0.0:
        raise ValueError("loop_noise must be non-negative")
    if not isinstance(n_tendrils, int):
        raise TypeError("n_tendrils must be an int")
    if n_tendrils < 0:
        raise ValueError("n_tendrils must be non-negative")
    if not 0.0 <= tendril_length_variation < 1.0:
        raise ValueError("tendril_length_variation must be in [0, 1)")
    if segments_per_tendril < 1:
        raise ValueError("segments_per_tendril must be at least 1")
    if parametric:
        if n_tendrils > 0 and tendril_length <= 0.0:
            raise ValueError("tendril_length must be positive when n_tendrils is positive")
        if n_tendrils == 0 and tendril_length != 0.0:
            raise ValueError("tendril_length must be 0.0 when n_tendrils is zero")
    elif class_name != "clean" and tendril_length <= 0.0:
        raise ValueError("tendril_length must be positive for non-clean classes")
    if not 0.0 <= branching_probability <= 1.0:
        raise ValueError("branching_probability must be between 0 and 1")
    if polarized_sector_width <= 0.0 or polarized_sector_width > 2.0 * np.pi:
        raise ValueError("polarized_sector_width must be in (0, 2*pi]")
    if min_root_separation < 0:
        raise ValueError("min_root_separation must be nonnegative")


def _generate_loop_tendrils_graph(
    class_name: str,
    n_loop_vertices: int = 80,
    loop_radius: float = 1.0,
    loop_noise: float = 0.0,
    n_tendrils: int = 0,
    tendril_length: float = 0.0,
    tendril_length_variation: float = 0.0,
    segments_per_tendril: int = 3,
    tendril_mode: str = "single",
    branching_probability: float = 0.0,
    root_mode: str = "uniform",
    polarized_sector_width: float = 1.5707963267948966,
    min_root_separation: int = 3,
    normalize: bool = True,
    seed: int | None = None,
    parametric: bool = False,
) -> EmbeddedPlanarGraph:
    """Generate one loop-with-inward-tendrils graph from resolved parameters."""
    _validate_inputs(
        class_name,
        n_loop_vertices,
        loop_radius,
        loop_noise,
        n_tendrils,
        tendril_length,
        tendril_length_variation,
        segments_per_tendril,
        tendril_mode,
        branching_probability,
        root_mode,
        polarized_sector_width,
        min_root_separation,
        parametric=parametric,
    )

    rng = np.random.default_rng(seed)
    loop_points = _loop_nodes(n_loop_vertices, loop_radius, loop_noise, rng)
    nodes_list: list[list[float]] = [[float(x), float(y)] for x, y in loop_points]
    edges_list: list[tuple[int, int]] = [tuple(map(int, edge)) for edge in _loop_edges(n_loop_vertices)]

    loop_edge_indices = list(range(n_loop_vertices))
    tendril_root_indices: list[int] = []
    tendril_tip_indices: list[int] = []
    tendril_edge_indices: list[int] = []
    branch_edge_indices: list[int] = []
    branch_point_indices: list[int] = []
    paired_root_pairs: list[list[int]] = []
    sector_center_angle: float | None = None
    notes: list[str] = ["intended_branch_count counts generated side-branch edges"]

    allowed_indices: np.ndarray | None = None
    if root_mode == "polarized":
        theta = np.linspace(0.0, 2.0 * np.pi, n_loop_vertices, endpoint=False)
        sector_center_angle = float(rng.uniform(0.0, 2.0 * np.pi))
        allowed_indices = np.flatnonzero(_angle_in_sector(theta, sector_center_angle, polarized_sector_width))
        if allowed_indices.size < n_tendrils:
            notes.append("polarized sector had fewer roots than requested; root selection was relaxed")

    if n_tendrils * max(1, min_root_separation) > n_loop_vertices:
        notes.append("requested root separation was too strict; selection may have been relaxed")

    if n_tendrils > 0:
        roots = _select_roots(
            n_loop_vertices,
            n_tendrils,
            min_root_separation,
            rng,
            allowed_indices=allowed_indices,
        )
    else:
        roots = []

    if tendril_length_variation > 0.0 and roots:
        low = 1.0 - tendril_length_variation
        high = 1.0 + tendril_length_variation
        root_lengths = {
            int(root): float(tendril_length * factor)
            for root, factor in zip(roots, rng.uniform(low, high, size=len(roots)))
        }
    else:
        root_lengths = {int(root): float(tendril_length) for root in roots}

    generated_tendril_lengths: list[float] = []
    if tendril_mode == "paired":
        used_roots: set[int] = set()
        for root in roots:
            if root in used_roots:
                continue
            root_j = (root + 2) % n_loop_vertices
            if root_j in used_roots:
                root_j = (root + 1) % n_loop_vertices
            used_roots.add(root)
            used_roots.add(root_j)
            current_length = root_lengths[int(root)]
            new_nodes, new_edges = _add_paired_spike(
                nodes_list,
                edges_list,
                root,
                root_j,
                current_length,
            )
            tendril_root_indices.extend([root, root_j])
            paired_root_pairs.append([int(root), int(root_j)])
            tendril_tip_indices.extend(new_nodes)
            tendril_edge_indices.extend(new_edges)
            generated_tendril_lengths.append(current_length)
    else:
        for root in roots:
            tendril_root_indices.append(int(root))
            current_length = root_lengths[int(root)]
            if tendril_mode == "branching":
                new_nodes, new_edges, branch_edges, branch_points = _add_branching_tendril(
                    nodes_list,
                    edges_list,
                    root,
                    current_length,
                    segments_per_tendril,
                    branching_probability,
                    rng,
                )
                tendril_edge_indices.extend(new_edges)
                branch_edge_indices.extend(branch_edges)
                branch_point_indices.extend(branch_points)
            else:
                new_nodes, new_edges = _add_single_tendril(
                    nodes_list,
                    edges_list,
                    root,
                    current_length,
                    segments_per_tendril,
                    rng,
                )
                tendril_edge_indices.extend(new_edges)
            if new_nodes:
                tendril_tip_indices.append(int(new_nodes[-1]))
            generated_tendril_lengths.append(current_length)

    edges, old_to_new = _deduplicate_edges(edges_list)
    loop_edge_indices = _remap_edge_indices(loop_edge_indices, old_to_new)
    tendril_edge_indices = _remap_edge_indices(tendril_edge_indices, old_to_new)
    branch_edge_indices = _remap_edge_indices(branch_edge_indices, old_to_new)

    nodes = np.asarray(nodes_list, dtype=np.float64)
    if normalize:
        nodes = _normalize_nodes(nodes)

    total_edge_indices = tendril_edge_indices + branch_edge_indices
    intended_total_tendril_length = 0.0
    if total_edge_indices:
        lengths = _edge_lengths(nodes, edges[np.asarray(total_edge_indices, dtype=np.int64)])
        intended_total_tendril_length = float(np.sum(lengths))

    labels: dict[str, object] = {
        "family": "loop_tendrils",
        "class_name": class_name,
        "seed": seed,
        "n_loop_vertices": n_loop_vertices,
        "loop_radius": float(loop_radius),
        "loop_noise": float(loop_noise),
        "n_tendrils": int(n_tendrils),
        "tendril_length": float(tendril_length),
        "tendril_length_variation": float(tendril_length_variation),
        "tendril_lengths": [float(length) for length in generated_tendril_lengths],
        "segments_per_tendril": segments_per_tendril,
        "tendril_mode": tendril_mode,
        "branching_probability": float(branching_probability),
        "root_mode": root_mode,
        "polarized_sector_width": float(polarized_sector_width),
        "min_root_separation": int(min_root_separation),
        "normalize": bool(normalize),
        "loop_node_indices": [int(index) for index in range(n_loop_vertices)],
        "loop_edge_indices": [int(index) for index in loop_edge_indices],
        "tendril_root_indices": [int(index) for index in tendril_root_indices],
        "tendril_tip_indices": [int(index) for index in tendril_tip_indices],
        "tendril_edge_indices": [int(index) for index in tendril_edge_indices],
        "branch_edge_indices": [int(index) for index in branch_edge_indices],
        "branch_point_indices": [int(index) for index in branch_point_indices],
        "paired_root_pairs": paired_root_pairs,
        "intended_tendril_count": int(n_tendrils),
        "intended_single_tendril_length": float(tendril_length),
        "intended_single_tendril_lengths": [float(length) for length in generated_tendril_lengths],
        "intended_total_tendril_length": intended_total_tendril_length,
        "intended_branch_count": int(len(branch_edge_indices)),
        "sector_center_angle": sector_center_angle,
        "notes": "; ".join(notes),
    }

    return EmbeddedPlanarGraph(nodes=nodes, edges=edges, labels=labels)


def generate_loop_tendrils(
    class_name: str = "clean",
    n_loop_vertices: int = 80,
    loop_radius: float = 1.0,
    loop_noise: float = 0.0,
    n_tendrils: int | None = None,
    tendril_length: float | None = None,
    segments_per_tendril: int = 3,
    tendril_mode: str = "single",
    branching_probability: float | None = None,
    polarized_sector_width: float = 1.5707963267948966,
    min_root_separation: int = 3,
    normalize: bool = True,
    seed: int | None = None,
) -> EmbeddedPlanarGraph:
    """Generate one class-preset loop-with-inward-tendrils graph."""
    if class_name not in SUPPORTED_CLASSES:
        raise ValueError(
            f"Unknown class_name {class_name!r}. "
            f"Supported values are: {', '.join(SUPPORTED_CLASSES)}"
        )

    effective_count, effective_length, effective_mode, effective_branching = _class_defaults(
        class_name,
        n_tendrils,
        tendril_length,
        tendril_mode,
        branching_probability,
    )
    if class_name == "clean":
        effective_count = 0
        effective_length = 0.0
        effective_mode = "single"
        effective_branching = 0.0

    return _generate_loop_tendrils_graph(
        class_name=class_name,
        n_loop_vertices=n_loop_vertices,
        loop_radius=loop_radius,
        loop_noise=loop_noise,
        n_tendrils=effective_count,
        tendril_length=effective_length,
        tendril_length_variation=0.0,
        segments_per_tendril=segments_per_tendril,
        tendril_mode=effective_mode,
        branching_probability=effective_branching,
        root_mode="polarized" if class_name == "polarized" else "uniform",
        polarized_sector_width=polarized_sector_width,
        min_root_separation=min_root_separation,
        normalize=normalize,
        seed=seed,
    )


def generate_parametric_loop_tendrils(
    n_tendrils: int,
    tendril_length: float,
    n_loop_vertices: int = 80,
    loop_radius: float = 1.0,
    loop_noise: float = 0.0,
    tendril_length_variation: float = 0.0,
    segments_per_tendril: int = 3,
    tendril_mode: str = "single",
    branching_probability: float = 0.0,
    root_mode: str = "uniform",
    polarized_sector_width: float = np.pi / 2,
    min_root_separation: int = 3,
    normalize: bool = True,
    seed: int | None = None,
) -> EmbeddedPlanarGraph:
    """Generate one parameterized loop-tendril graph for regression tasks."""
    return _generate_loop_tendrils_graph(
        class_name="parametric",
        n_loop_vertices=n_loop_vertices,
        loop_radius=loop_radius,
        loop_noise=loop_noise,
        n_tendrils=n_tendrils,
        tendril_length=tendril_length,
        tendril_length_variation=tendril_length_variation,
        segments_per_tendril=segments_per_tendril,
        tendril_mode=tendril_mode,
        branching_probability=branching_probability,
        root_mode=root_mode,
        polarized_sector_width=polarized_sector_width,
        min_root_separation=min_root_separation,
        normalize=normalize,
        seed=seed,
        parametric=True,
    )


def save_graph(graph: EmbeddedPlanarGraph, path: str | Path) -> None:
    """Save an embedded planar graph to a compressed NumPy archive."""
    labels_json = json.dumps(graph.labels, sort_keys=True)
    np.savez_compressed(
        Path(path),
        nodes=np.asarray(graph.nodes, dtype=np.float64),
        edges=np.asarray(graph.edges, dtype=np.int64),
        labels_json=np.asarray(labels_json),
    )


def load_graph(path: str | Path) -> EmbeddedPlanarGraph:
    """Load an embedded planar graph from ``save_graph`` format."""
    with np.load(Path(path), allow_pickle=False) as data:
        nodes = np.asarray(data["nodes"], dtype=np.float64)
        edges = np.asarray(data["edges"], dtype=np.int64)
        labels_raw: Any = data["labels_json"].item()

    labels = json.loads(str(labels_raw))
    return EmbeddedPlanarGraph(nodes=nodes, edges=edges, labels=labels)


def generate_dataset(
    n_per_class: int,
    classes: tuple[str, ...] = SUPPORTED_CLASSES,
    seed: int | None = None,
    **kwargs: object,
) -> list[EmbeddedPlanarGraph]:
    """Generate a class-major list of loop-tendril graphs."""
    if not isinstance(n_per_class, int):
        raise TypeError("n_per_class must be an int")
    if n_per_class < 1:
        raise ValueError("n_per_class must be at least 1")

    rng = np.random.default_rng(seed)
    graphs: list[EmbeddedPlanarGraph] = []

    for class_name in classes:
        if class_name not in SUPPORTED_CLASSES:
            raise ValueError(
                f"Unknown class_name {class_name!r}. "
                f"Supported values are: {', '.join(SUPPORTED_CLASSES)}"
            )
        for _ in range(n_per_class):
            graph_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            graphs.append(generate_loop_tendrils(class_name=class_name, seed=graph_seed, **kwargs))

    return graphs


if __name__ == "__main__":
    for name in SUPPORTED_CLASSES:
        graph = generate_loop_tendrils(class_name=name, seed=0)
        print(
            f"{name}: "
            f"nodes={graph.nodes.shape[0]}, "
            f"edges={graph.edges.shape[0]}, "
            f"intended_tendrils={graph.labels['intended_tendril_count']}, "
            f"intended_total_tendril_length={graph.labels['intended_total_tendril_length']:.6f}, "
            f"intended_branches={graph.labels['intended_branch_count']}"
        )
