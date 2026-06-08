"""Additional embedded graph families for loop-geometry benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import Callable, Optional

import numpy as np


SUPPORTED_CLASSES = (
    "hairs_uniform",
    "hairs_polarized",
    "hairs_random",
    "blebs_uniform",
    "blebs_random",
    "blebs_polarized",
    "branch_single",
    "branch_Y",
    "fractal_koch",
    "fractal_midpoint",
    "tortuous",
    "serrated",
    "mixed_half",
    "mixed_random",
    "mixed_alternating",
    "mild_grid_regular",
    "mild_grid_sinusoidal",
    "mild_grid_wavy",
    "mild_grid_jittered",
)


@dataclass(frozen=True)
class EmbeddedPlanarGraph:
    """
    nodes : float64 array of shape (n_nodes, 2) - coordinates in the plane
    edges : int64 array of shape (n_edges, 2) - undirected edge index pairs
    labels: dict[str, object] - JSON-serializable metadata
    """

    nodes: np.ndarray
    edges: np.ndarray
    labels: dict[str, object]


def _normalize_nodes(nodes: np.ndarray) -> np.ndarray:
    """Fit nodes into [0, 1]^2 while preserving aspect ratio."""
    normalized = np.asarray(nodes, dtype=np.float64).copy()
    if normalized.ndim != 2 or normalized.shape[1] != 2:
        raise ValueError("nodes must have shape (n_nodes, 2)")

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
    """Return Euclidean lengths for all graph edges."""
    nodes = np.asarray(nodes, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        return np.empty(0, dtype=np.float64)
    vectors = nodes[edges[:, 1]] - nodes[edges[:, 0]]
    return np.linalg.norm(vectors, axis=1).astype(np.float64)


def _make_loop(
    n_loop_vertices: int,
    loop_radius: float,
    loop_noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create noisy circular loop vertices in counterclockwise order."""
    _validate_int("n_loop_vertices", n_loop_vertices, minimum=3)
    _validate_nonnegative("loop_noise", loop_noise)
    _validate_positive("loop_radius", loop_radius)

    theta = np.linspace(0.0, 2.0 * np.pi, n_loop_vertices, endpoint=False)
    radii = np.full(n_loop_vertices, loop_radius, dtype=np.float64)
    if loop_noise > 0.0:
        radii *= 1.0 + loop_noise * rng.normal(size=n_loop_vertices)
        radii = np.maximum(radii, 0.1 * loop_radius)
    return np.column_stack((radii * np.cos(theta), radii * np.sin(theta))).astype(np.float64)


def _select_roots(
    n_loop_vertices: int,
    n_roots: int,
    min_separation: int,
    rng: np.random.Generator,
    allowed_indices: Optional[np.ndarray] = None,
) -> list[int]:
    """Select loop-root indices, preserving requested separation when feasible."""
    _validate_int("n_loop_vertices", n_loop_vertices, minimum=1)
    _validate_int("n_roots", n_roots, minimum=0)
    _validate_int("min_separation", min_separation, minimum=0)
    if n_roots == 0:
        return []

    allowed = np.arange(n_loop_vertices, dtype=np.int64)
    if allowed_indices is not None:
        allowed = np.asarray(allowed_indices, dtype=np.int64)
        allowed = allowed[(0 <= allowed) & (allowed < n_loop_vertices)]
        allowed = np.unique(allowed)
        if allowed.size == 0:
            allowed = np.arange(n_loop_vertices, dtype=np.int64)

    if n_roots > allowed.size:
        raise ValueError("n_roots cannot exceed the number of selectable loop vertices")

    for separation in range(min_separation, -1, -1):
        for _ in range(max(200, 20 * int(allowed.size))):
            candidates = allowed.copy()
            rng.shuffle(candidates)
            roots: list[int] = []
            for candidate in candidates:
                root = int(candidate)
                if all(_circular_distance(root, existing, n_loop_vertices) >= separation for existing in roots):
                    roots.append(root)
                    if len(roots) == n_roots:
                        return sorted(roots)

    raise AssertionError("root selection fallback was not reached")


def _add_polyline(
    nodes_list: list[list[float]],
    edges_list: list[tuple[int, int]],
    start_idx: int,
    direction: np.ndarray,
    length: float,
    segments: int,
) -> tuple[list[int], list[int]]:
    """Append a straight polyline and return its new node and edge indices."""
    _validate_positive("length", length)
    _validate_int("segments", segments, minimum=1)

    start = np.asarray(nodes_list[start_idx], dtype=np.float64)
    unit = _unit(direction)
    node_indices: list[int] = []
    edge_indices: list[int] = []
    previous = int(start_idx)

    for segment in range(1, segments + 1):
        point = start + (segment / segments) * length * unit
        current = len(nodes_list)
        nodes_list.append(_point_to_list(point))
        edges_list.append((previous, current))
        node_indices.append(current)
        edge_indices.append(len(edges_list) - 1)
        previous = current

    return node_indices, edge_indices


def _insert_arc_bleb(
    nodes_list: list[list[float]],
    edges_list: list[tuple[int, int]],
    root_idx: int,
    next_idx: int,
    bleb_radius: float,
    bleb_points_per_arc: int,
) -> tuple[list[int], list[int]]:
    """Replace one loop edge by an outward circular arc anchored at its endpoints."""
    _validate_positive("bleb_radius", bleb_radius)
    _validate_int("bleb_points_per_arc", bleb_points_per_arc, minimum=1)

    p0 = np.asarray(nodes_list[root_idx], dtype=np.float64)
    p1 = np.asarray(nodes_list[next_idx], dtype=np.float64)
    midpoint = 0.5 * (p0 + p1)
    tangent = _unit(p1 - p0)
    outward = _outward_direction(midpoint)
    half_chord = 0.5 * float(np.linalg.norm(p1 - p0))
    sagitta = float(bleb_radius)
    circle_radius = (half_chord * half_chord + sagitta * sagitta) / (2.0 * sagitta)
    center_offset = sagitta - circle_radius

    node_indices: list[int] = []
    edge_indices: list[int] = []
    previous = int(root_idx)
    x_positions = np.linspace(-half_chord, half_chord, bleb_points_per_arc + 2)[1:-1]

    for x_pos in x_positions:
        y_pos = center_offset + np.sqrt(max(0.0, circle_radius * circle_radius - x_pos * x_pos))
        point = midpoint + x_pos * tangent + y_pos * outward
        current = len(nodes_list)
        nodes_list.append(_point_to_list(point))
        edges_list.append((previous, current))
        node_indices.append(current)
        edge_indices.append(len(edges_list) - 1)
        previous = current

    edges_list.append((previous, int(next_idx)))
    edge_indices.append(len(edges_list) - 1)
    return node_indices, edge_indices


def generate_hair_loop(
    n_loop_vertices: int = 80,
    n_hairs: int = 8,
    hair_length: float = 0.18,
    hair_mode: str = "uniform",
    hair_segment_count: int = 1,
    polar_sector_width: float = np.pi / 2,
    min_root_separation: int = 3,
    loop_radius: float = 1.0,
    loop_noise: float = 0.01,
    normalize: bool = True,
    seed: Optional[int] = None,
) -> EmbeddedPlanarGraph:
    """Generate a one-cycle loop with straight outward hair attachments."""
    _validate_mode("hair_mode", hair_mode, ("uniform", "polarized", "random"))
    _validate_int("n_hairs", n_hairs, minimum=0)
    _validate_positive("hair_length", hair_length)
    _validate_int("hair_segment_count", hair_segment_count, minimum=1)
    _validate_positive("polar_sector_width", polar_sector_width)
    rng = np.random.default_rng(seed)

    loop = _make_loop(n_loop_vertices, loop_radius, loop_noise, rng)
    nodes_list = [_point_to_list(point) for point in loop]
    edges_list = _loop_edges_list(n_loop_vertices)
    roots = _roots_by_mode(n_loop_vertices, n_hairs, hair_mode, polar_sector_width, min_root_separation, rng)

    hair_edges: list[int] = []
    hair_nodes: list[int] = []
    for root in roots:
        new_nodes, new_edges = _add_polyline(
            nodes_list,
            edges_list,
            root,
            _outward_direction(loop[root]),
            hair_length,
            hair_segment_count,
        )
        hair_nodes.extend(new_nodes)
        hair_edges.extend(new_edges)

    return _build_graph(
        nodes_list,
        edges_list,
        normalize,
        {
            "family": "hair_loop",
            "class_name": f"hairs_{hair_mode}",
            "seed": seed,
            "n_loop_vertices": n_loop_vertices,
            "n_hairs": n_hairs,
            "hair_length": float(hair_length),
            "hair_mode": hair_mode,
            "hair_segment_count": hair_segment_count,
            "polar_sector_width": float(polar_sector_width),
            "min_root_separation": min_root_separation,
            "actual_min_root_separation": _actual_min_root_separation(roots, n_loop_vertices),
            "loop_radius": float(loop_radius),
            "loop_noise": float(loop_noise),
            "normalize": bool(normalize),
            "loop_node_indices": _int_list(range(n_loop_vertices)),
            "hair_root_indices": _int_list(roots),
            "hair_node_indices": _int_list(hair_nodes),
            "hair_edge_indices": _int_list(hair_edges),
            "protrusion_edge_indices": _int_list(hair_edges),
        },
    )


def generate_bleb_loop(
    n_loop_vertices: int = 80,
    n_blebs: int = 4,
    bleb_radius: float = 0.12,
    bleb_mode: str = "uniform",
    bleb_points_per_arc: int = 10,
    min_root_separation: int = 6,
    loop_radius: float = 1.0,
    loop_noise: float = 0.01,
    normalize: bool = True,
    seed: Optional[int] = None,
) -> EmbeddedPlanarGraph:
    """Generate a one-cycle loop with outward semicircular bleb arcs."""
    _validate_mode("bleb_mode", bleb_mode, ("uniform", "random", "polarized"))
    _validate_int("n_blebs", n_blebs, minimum=0)
    _validate_positive("bleb_radius", bleb_radius)
    _validate_int("bleb_points_per_arc", bleb_points_per_arc, minimum=1)
    rng = np.random.default_rng(seed)

    loop = _make_loop(n_loop_vertices, loop_radius, loop_noise, rng)
    roots = _roots_by_mode(n_loop_vertices, n_blebs, bleb_mode, np.pi / 2, min_root_separation, rng)
    removed = {(root, (root + 1) % n_loop_vertices) for root in roots}
    nodes_list = [_point_to_list(point) for point in loop]
    edges_list = [
        edge for edge in _loop_edges_list(n_loop_vertices) if edge not in removed
    ]

    bleb_edges: list[int] = []
    bleb_nodes: list[int] = []
    for root in roots:
        new_nodes, new_edges = _insert_arc_bleb(
            nodes_list,
            edges_list,
            root,
            (root + 1) % n_loop_vertices,
            bleb_radius,
            bleb_points_per_arc,
        )
        bleb_nodes.extend(new_nodes)
        bleb_edges.extend(new_edges)

    return _build_graph(
        nodes_list,
        edges_list,
        normalize,
        {
            "family": "bleb_loop",
            "class_name": f"blebs_{bleb_mode}",
            "seed": seed,
            "n_loop_vertices": n_loop_vertices,
            "n_blebs": n_blebs,
            "bleb_radius": float(bleb_radius),
            "bleb_mode": bleb_mode,
            "bleb_points_per_arc": bleb_points_per_arc,
            "min_root_separation": min_root_separation,
            "actual_min_root_separation": _actual_min_root_separation(roots, n_loop_vertices),
            "loop_radius": float(loop_radius),
            "loop_noise": float(loop_noise),
            "normalize": bool(normalize),
            "loop_node_indices": _int_list(range(n_loop_vertices)),
            "bleb_root_indices": _int_list(roots),
            "bleb_node_indices": _int_list(bleb_nodes),
            "bleb_edge_indices": _int_list(bleb_edges),
            "protrusion_edge_indices": _int_list(bleb_edges),
        },
    )


def generate_branching_loop(
    n_loop_vertices: int = 80,
    n_branches: int = 6,
    branch_length: float = 0.25,
    branch_mode: str = "single",
    secondary_probability: float = 0.8,
    min_root_separation: int = 4,
    loop_radius: float = 1.0,
    loop_noise: float = 0.01,
    normalize: bool = True,
    seed: Optional[int] = None,
) -> EmbeddedPlanarGraph:
    """Generate a one-cycle loop with inward single or Y-shaped protrusions."""
    _validate_mode("branch_mode", branch_mode, ("single", "Y"))
    _validate_int("n_branches", n_branches, minimum=0)
    _validate_positive("branch_length", branch_length)
    _validate_probability("secondary_probability", secondary_probability)
    rng = np.random.default_rng(seed)

    loop = _make_loop(n_loop_vertices, loop_radius, loop_noise, rng)
    nodes_list = [_point_to_list(point) for point in loop]
    edges_list = _loop_edges_list(n_loop_vertices)
    roots = _select_roots(n_loop_vertices, n_branches, min_root_separation, rng)

    branch_edges: list[int] = []
    branch_nodes: list[int] = []
    branch_points: list[int] = []
    for root in roots:
        new_nodes, new_edges, split = _add_inward_branch(
            nodes_list,
            edges_list,
            root,
            branch_length,
            branch_mode,
            secondary_probability,
            rng,
        )
        branch_nodes.extend(new_nodes)
        branch_edges.extend(new_edges)
        if split is not None:
            branch_points.append(split)

    return _build_graph(
        nodes_list,
        edges_list,
        normalize,
        {
            "family": "branching_loop",
            "class_name": "branch_Y" if branch_mode == "Y" else "branch_single",
            "seed": seed,
            "n_loop_vertices": n_loop_vertices,
            "n_branches": n_branches,
            "branch_length": float(branch_length),
            "branch_mode": branch_mode,
            "secondary_probability": float(secondary_probability),
            "min_root_separation": min_root_separation,
            "actual_min_root_separation": _actual_min_root_separation(roots, n_loop_vertices),
            "loop_radius": float(loop_radius),
            "loop_noise": float(loop_noise),
            "normalize": bool(normalize),
            "loop_node_indices": _int_list(range(n_loop_vertices)),
            "branch_root_indices": _int_list(roots),
            "branch_node_indices": _int_list(branch_nodes),
            "branch_edge_indices": _int_list(branch_edges),
            "branch_point_indices": _int_list(branch_points),
            "protrusion_edge_indices": _int_list(branch_edges),
        },
    )


def generate_fractal_loop(
    n_sides: int = 6,
    fractal_iterations: int = 2,
    perturbation_scale: float = 0.1,
    fractal_type: str = "koch",
    loop_radius: float = 1.0,
    loop_noise: float = 0.0,
    normalize: bool = True,
    seed: Optional[int] = None,
) -> EmbeddedPlanarGraph:
    """Generate a single rough boundary cycle using Koch or midpoint rules."""
    _validate_int("n_sides", n_sides, minimum=3)
    _validate_int("fractal_iterations", fractal_iterations, minimum=0)
    _validate_mode("fractal_type", fractal_type, ("koch", "midpoint"))
    _validate_nonnegative("perturbation_scale", perturbation_scale)
    rng = np.random.default_rng(seed)

    points = _make_loop(n_sides, loop_radius, loop_noise, rng)
    for _ in range(fractal_iterations):
        if fractal_type == "koch":
            points = _koch_iteration(points)
        else:
            points = _midpoint_iteration(points, perturbation_scale, rng)

    nodes_list = [_point_to_list(point) for point in points]
    edges_list = _loop_edges_list(len(nodes_list))
    return _build_graph(
        nodes_list,
        edges_list,
        normalize,
        {
            "family": "fractal_loop",
            "class_name": f"fractal_{fractal_type}",
            "seed": seed,
            "n_sides": n_sides,
            "fractal_iterations": fractal_iterations,
            "fractal_iteration": fractal_iterations,
            "perturbation_scale": float(perturbation_scale),
            "fractal_type": fractal_type,
            "loop_radius": float(loop_radius),
            "loop_noise": float(loop_noise),
            "normalize": bool(normalize),
            "loop_node_indices": _int_list(range(len(nodes_list))),
            "protrusion_edge_indices": [],
        },
    )


def generate_tortuous_loop(
    n_loop_vertices: int = 80,
    amplitude: float = 0.05,
    frequency: int = 5,
    loop_radius: float = 1.0,
    loop_noise: float = 0.01,
    normalize: bool = True,
    seed: Optional[int] = None,
) -> EmbeddedPlanarGraph:
    """Generate a one-cycle loop with sinusoidal microvascular tortuosity."""
    _validate_nonnegative("amplitude", amplitude)
    _validate_int("frequency", frequency, minimum=1)
    rng = np.random.default_rng(seed)

    loop = _make_loop(n_loop_vertices, loop_radius, loop_noise, rng)
    subdivisions = max(4, int(2 * frequency))
    nodes_list: list[list[float]] = []

    for index in range(n_loop_vertices):
        p0 = loop[index]
        p1 = loop[(index + 1) % n_loop_vertices]
        for sub in range(subdivisions):
            t = sub / subdivisions
            global_t = (index + t) / n_loop_vertices
            base = (1.0 - t) * p0 + t * p1
            normal = _outward_direction(base)
            wave = np.sin(2.0 * np.pi * frequency * global_t)
            nodes_list.append(_point_to_list(base + amplitude * wave * normal))

    edges_list = _loop_edges_list(len(nodes_list))
    return _build_graph(
        nodes_list,
        edges_list,
        normalize,
        {
            "family": "tortuous_loop",
            "class_name": "tortuous",
            "seed": seed,
            "n_loop_vertices": n_loop_vertices,
            "amplitude": float(amplitude),
            "frequency": frequency,
            "subdivisions_per_edge": subdivisions,
            "loop_radius": float(loop_radius),
            "loop_noise": float(loop_noise),
            "normalize": bool(normalize),
            "loop_node_indices": _int_list(range(len(nodes_list))),
            "protrusion_edge_indices": [],
        },
    )


def generate_serrated_loop(
    n_loop_vertices: int = 20,
    serration_count: int = 10,
    serration_depth: float = 0.1,
    serration_width: float = 0.04,
    loop_radius: float = 1.0,
    loop_noise: float = 0.01,
    normalize: bool = True,
    seed: Optional[int] = None,
) -> EmbeddedPlanarGraph:
    """Generate a one-cycle loop with alternating inward and outward serrations."""
    _validate_int("serration_count", serration_count, minimum=1)
    _validate_nonnegative("serration_depth", serration_depth)
    _validate_nonnegative("serration_width", serration_width)
    rng = np.random.default_rng(seed)

    base = _make_loop(n_loop_vertices, loop_radius, loop_noise, rng)
    nodes_list: list[list[float]] = []
    serration_tip_indices: list[int] = []
    for index in range(n_loop_vertices):
        p0 = base[index]
        p1 = base[(index + 1) % n_loop_vertices]
        nodes_list.append(_point_to_list(p0))
        midpoint = 0.5 * (p0 + p1)
        outward = _outward_direction(midpoint)
        tangent = _unit(p1 - p0)
        phase = int(np.floor(index * serration_count / n_loop_vertices))
        direction = -1.0 if phase % 2 == 0 else 1.0
        lateral = serration_width * np.sin(2.0 * np.pi * index / max(1, serration_count))
        tip = midpoint + direction * serration_depth * outward + lateral * tangent
        tip_idx = len(nodes_list)
        nodes_list.append(_point_to_list(tip))
        serration_tip_indices.append(tip_idx)

    edges_list = _loop_edges_list(len(nodes_list))
    return _build_graph(
        nodes_list,
        edges_list,
        normalize,
        {
            "family": "serrated_loop",
            "class_name": "serrated",
            "seed": seed,
            "n_loop_vertices": n_loop_vertices,
            "serration_count": serration_count,
            "serration_depth": float(serration_depth),
            "serration_width": float(serration_width),
            "loop_radius": float(loop_radius),
            "loop_noise": float(loop_noise),
            "normalize": bool(normalize),
            "loop_node_indices": _int_list(range(len(nodes_list))),
            "serration_tip_indices": _int_list(serration_tip_indices),
            "protrusion_edge_indices": [],
        },
    )


def generate_mixed_loop(
    n_loop_vertices: int = 80,
    n_hairs: int = 6,
    hair_length: float = 0.18,
    n_blebs: int = 4,
    bleb_radius: float = 0.12,
    min_root_separation: int = 3,
    mix_mode: str = "half_half",
    loop_radius: float = 1.0,
    loop_noise: float = 0.01,
    normalize: bool = True,
    seed: Optional[int] = None,
) -> EmbeddedPlanarGraph:
    """Generate a one-cycle loop with both hair attachments and bleb arcs."""
    _validate_mode("mix_mode", mix_mode, ("half_half", "random", "alternating"))
    _validate_int("n_hairs", n_hairs, minimum=0)
    _validate_int("n_blebs", n_blebs, minimum=0)
    _validate_positive("hair_length", hair_length)
    _validate_positive("bleb_radius", bleb_radius)
    rng = np.random.default_rng(seed)

    n_attachments = n_hairs + n_blebs
    loop = _make_loop(n_loop_vertices, loop_radius, loop_noise, rng)
    roots = _select_roots(n_loop_vertices, n_attachments, min_root_separation, rng)
    root_types = _mixed_root_types(n_attachments, n_hairs, n_blebs, mix_mode, rng)
    hair_roots = [root for root, root_type in zip(roots, root_types) if root_type == "hair"]
    bleb_roots = [root for root, root_type in zip(roots, root_types) if root_type == "bleb"]

    removed = {(root, (root + 1) % n_loop_vertices) for root in bleb_roots}
    nodes_list = [_point_to_list(point) for point in loop]
    edges_list = [
        edge for edge in _loop_edges_list(n_loop_vertices) if edge not in removed
    ]

    bleb_edges: list[int] = []
    bleb_nodes: list[int] = []
    for root in bleb_roots:
        new_nodes, new_edges = _insert_arc_bleb(
            nodes_list,
            edges_list,
            root,
            (root + 1) % n_loop_vertices,
            bleb_radius,
            8,
        )
        bleb_nodes.extend(new_nodes)
        bleb_edges.extend(new_edges)

    hair_edges: list[int] = []
    hair_nodes: list[int] = []
    for root in hair_roots:
        new_nodes, new_edges = _add_polyline(
            nodes_list,
            edges_list,
            root,
            _outward_direction(loop[root]),
            hair_length,
            1,
        )
        hair_nodes.extend(new_nodes)
        hair_edges.extend(new_edges)

    return _build_graph(
        nodes_list,
        edges_list,
        normalize,
        {
            "family": "mixed_loop",
            "class_name": "mixed_half" if mix_mode == "half_half" else f"mixed_{mix_mode}",
            "seed": seed,
            "n_loop_vertices": n_loop_vertices,
            "n_hairs": len(hair_roots),
            "requested_n_hairs": n_hairs,
            "hair_length": float(hair_length),
            "n_blebs": len(bleb_roots),
            "requested_n_blebs": n_blebs,
            "bleb_radius": float(bleb_radius),
            "min_root_separation": min_root_separation,
            "actual_min_root_separation": _actual_min_root_separation(roots, n_loop_vertices),
            "mix_mode": mix_mode,
            "loop_radius": float(loop_radius),
            "loop_noise": float(loop_noise),
            "normalize": bool(normalize),
            "loop_node_indices": _int_list(range(n_loop_vertices)),
            "hair_root_indices": _int_list(hair_roots),
            "hair_node_indices": _int_list(hair_nodes),
            "hair_edge_indices": _int_list(hair_edges),
            "bleb_root_indices": _int_list(bleb_roots),
            "bleb_node_indices": _int_list(bleb_nodes),
            "bleb_edge_indices": _int_list(bleb_edges),
            "protrusion_edge_indices": _int_list(hair_edges + bleb_edges),
        },
    )


def generate_mild_iso_cycle_grid(
    n_rows: int = 6,
    n_cols: int = 6,
    distortion_type: str = "sinusoidal",
    distortion_amplitude: float = 0.08,
    jitter: float = 0.01,
    normalize: bool = True,
    seed: Optional[int] = None,
) -> EmbeddedPlanarGraph:
    """Generate a fixed rectangular grid with mild geometric distortion."""
    _validate_int("n_rows", n_rows, minimum=2)
    _validate_int("n_cols", n_cols, minimum=2)
    _validate_mode("distortion_type", distortion_type, ("regular", "sinusoidal", "wavy", "jittered"))
    _validate_nonnegative("distortion_amplitude", distortion_amplitude)
    _validate_nonnegative("jitter", jitter)
    rng = np.random.default_rng(seed)

    nodes = _grid_nodes(n_rows, n_cols)
    x = nodes[:, 0].copy()
    y = nodes[:, 1].copy()

    if distortion_type == "sinusoidal":
        nodes[:, 1] = y + distortion_amplitude * np.sin(2.0 * np.pi * x)
    elif distortion_type == "wavy":
        nodes[:, 0] = x + 0.5 * distortion_amplitude * np.sin(2.0 * np.pi * y)
        nodes[:, 1] = y + distortion_amplitude * np.sin(2.0 * np.pi * x)
    elif distortion_type == "jittered":
        weights = _grid_boundary_weights(n_rows, n_cols)
        nodes += rng.normal(0.0, jitter, size=nodes.shape) * weights[:, None]
    elif distortion_type == "regular":
        pass

    if normalize:
        nodes = _normalize_nodes(nodes)

    boundary = _grid_boundary_indices(n_rows, n_cols)
    return EmbeddedPlanarGraph(
        nodes=np.asarray(nodes, dtype=np.float64),
        edges=_grid_edges(n_rows, n_cols),
        labels={
            "family": "mild_iso_cycle_grid",
            "class_name": f"mild_grid_{distortion_type}",
            "seed": seed,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "distortion_type": distortion_type,
            "distortion_amplitude": float(distortion_amplitude),
            "jitter": float(jitter),
            "normalize": bool(normalize),
            "loop_node_indices": _int_list(boundary),
            "boundary_node_indices": _int_list(boundary),
            "protrusion_edge_indices": [],
        },
    )


def generate_dataset(
    classes: tuple[str, ...],
    n_per_class: int = 50,
    seed: Optional[int] = None,
    **kwargs: object,
) -> list[EmbeddedPlanarGraph]:
    """Generate a class-major dataset across the supported loop families."""
    _validate_int("n_per_class", n_per_class, minimum=1)
    rng = np.random.default_rng(seed)
    graphs: list[EmbeddedPlanarGraph] = []
    _validate_dataset_kwargs(classes, kwargs)

    for class_name in classes:
        if class_name not in SUPPORTED_CLASSES:
            raise ValueError(
                f"Unknown class {class_name!r}. Supported classes are: {', '.join(SUPPORTED_CLASSES)}"
            )
        generator, defaults = _DATASET_GENERATORS[class_name]
        for _ in range(n_per_class):
            graph_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            params = {**defaults, **_accepted_kwargs(generator, kwargs), "seed": graph_seed}
            graphs.append(generator(**params))

    return graphs


def _build_graph(
    nodes_list: list[list[float]],
    edges_list: list[tuple[int, int]],
    normalize: bool,
    labels: dict[str, object],
) -> EmbeddedPlanarGraph:
    """Convert mutable node and edge lists into a validated graph object."""
    nodes = np.asarray(nodes_list, dtype=np.float64)
    edges = _deduplicate_edges(edges_list)
    if normalize:
        nodes = _normalize_nodes(nodes)
    return EmbeddedPlanarGraph(nodes=nodes, edges=edges, labels=_json_labels(labels))


def _loop_edges_list(n_loop_vertices: int) -> list[tuple[int, int]]:
    """Return edge pairs for a simple cycle over consecutive node indices."""
    return [(index, (index + 1) % n_loop_vertices) for index in range(n_loop_vertices)]


def _deduplicate_edges(edges: list[tuple[int, int]]) -> np.ndarray:
    """Deduplicate undirected edges while preserving first occurrence order."""
    seen: set[tuple[int, int]] = set()
    unique: list[tuple[int, int]] = []
    for u, v in edges:
        if u == v:
            continue
        key = (min(int(u), int(v)), max(int(u), int(v)))
        if key not in seen:
            seen.add(key)
            unique.append((int(u), int(v)))
    return np.asarray(unique, dtype=np.int64)


def _roots_by_mode(
    n_loop_vertices: int,
    n_roots: int,
    mode: str,
    sector_width: float,
    min_separation: int,
    rng: np.random.Generator,
) -> list[int]:
    """Select roots using uniform, polarized, or random angular placement."""
    if n_roots == 0:
        return []
    if n_roots > n_loop_vertices:
        raise ValueError("n_roots cannot exceed n_loop_vertices")
    if mode == "uniform":
        roots = np.linspace(0, n_loop_vertices, n_roots, endpoint=False, dtype=np.int64)
        offset = int(rng.integers(0, max(1, n_loop_vertices)))
        selected = sorted(int((root + offset) % n_loop_vertices) for root in roots)
        _require_root_separation(selected, n_loop_vertices, min_separation)
        return selected
    if mode == "polarized":
        theta = np.linspace(0.0, 2.0 * np.pi, n_loop_vertices, endpoint=False)
        center = float(rng.uniform(0.0, 2.0 * np.pi))
        allowed = np.flatnonzero(_angle_in_sector(theta, center, sector_width))
        return _select_roots(n_loop_vertices, n_roots, min_separation, rng, allowed)
    if mode == "random":
        return _select_roots(n_loop_vertices, n_roots, min_separation, rng)
    raise AssertionError("validated root mode was not handled")


def _add_inward_branch(
    nodes_list: list[list[float]],
    edges_list: list[tuple[int, int]],
    root_idx: int,
    length: float,
    branch_mode: str,
    secondary_probability: float,
    rng: np.random.Generator,
) -> tuple[list[int], list[int], Optional[int]]:
    """Append one curved inward protrusion, optionally splitting into a Y."""
    root = np.asarray(nodes_list[root_idx], dtype=np.float64)
    inward = -_outward_direction(root)
    tangent = np.asarray([-inward[1], inward[0]], dtype=np.float64)
    curvature = float(rng.uniform(-0.04, 0.04))
    segments = int(rng.integers(2, 4))
    stem_length = 0.5 * length if branch_mode == "Y" else length

    node_indices: list[int] = []
    edge_indices: list[int] = []
    previous = int(root_idx)
    for segment in range(1, segments + 1):
        s = segment / segments
        point = root + s * stem_length * inward + curvature * np.sin(np.pi * s) * tangent
        current = len(nodes_list)
        nodes_list.append(_point_to_list(point))
        edges_list.append((previous, current))
        node_indices.append(current)
        edge_indices.append(len(edges_list) - 1)
        previous = current

    split_idx: Optional[int] = None
    if branch_mode == "Y" and rng.random() < secondary_probability:
        split_idx = previous
        split = np.asarray(nodes_list[split_idx], dtype=np.float64)
        angle = float(rng.uniform(np.deg2rad(20.0), np.deg2rad(40.0)))
        for sign in (-1.0, 1.0):
            direction = _rotate(inward, sign * angle)
            tip = split + 0.5 * length * direction
            tip_idx = len(nodes_list)
            nodes_list.append(_point_to_list(tip))
            edges_list.append((split_idx, tip_idx))
            node_indices.append(tip_idx)
            edge_indices.append(len(edges_list) - 1)

    return node_indices, edge_indices, split_idx


def _koch_iteration(points: np.ndarray) -> np.ndarray:
    """Apply one outward Koch replacement to every edge of a closed polygon."""
    new_points: list[np.ndarray] = []
    n_points = points.shape[0]
    for index in range(n_points):
        p0 = points[index]
        p1 = points[(index + 1) % n_points]
        vector = p1 - p0
        outward = _right_normal(vector)
        new_points.append(p0)
        new_points.append(p0 + vector / 3.0)
        new_points.append(p0 + 0.5 * vector + (np.sqrt(3.0) / 6.0) * np.linalg.norm(vector) * outward)
        new_points.append(p0 + 2.0 * vector / 3.0)
    return np.asarray(new_points, dtype=np.float64)


def _midpoint_iteration(
    points: np.ndarray,
    perturbation_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Insert outward-displaced midpoints along a closed polygon."""
    new_points: list[np.ndarray] = []
    n_points = points.shape[0]
    for index in range(n_points):
        p0 = points[index]
        p1 = points[(index + 1) % n_points]
        midpoint = 0.5 * (p0 + p1)
        length = float(np.linalg.norm(p1 - p0))
        displacement = abs(float(rng.normal(0.0, perturbation_scale * length)))
        new_points.append(p0)
        new_points.append(midpoint + displacement * _outward_direction(midpoint))
    return np.asarray(new_points, dtype=np.float64)


def _mixed_root_types(
    n_attachments: int,
    n_hairs: int,
    n_blebs: int,
    mix_mode: str,
    rng: np.random.Generator,
) -> list[str]:
    """Assign hair or bleb type to each selected mixed-loop root."""
    if n_attachments == 0:
        return []
    if n_hairs + n_blebs != n_attachments:
        raise ValueError("n_attachments must equal n_hairs + n_blebs")
    if mix_mode == "half_half":
        n_hair_actual = n_attachments // 2
        n_bleb_actual = n_attachments - n_hair_actual
        return ["hair"] * n_hair_actual + ["bleb"] * n_bleb_actual
    if mix_mode == "random":
        hair_probability = n_hairs / n_attachments
        return [
            "hair" if rng.random() < hair_probability else "bleb"
            for _ in range(n_attachments)
        ]
    if mix_mode == "alternating":
        return ["hair" if index % 2 == 0 else "bleb" for index in range(n_attachments)]
    raise AssertionError("validated mix mode was not handled")


def _grid_nodes(n_rows: int, n_cols: int) -> np.ndarray:
    """Return row-major rectangular grid nodes in [0, 1]^2."""
    y = np.linspace(0.0, 1.0, n_rows, dtype=np.float64)
    x = np.linspace(0.0, 1.0, n_cols, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64, copy=False)


def _grid_edges(n_rows: int, n_cols: int) -> np.ndarray:
    """Return horizontal and vertical edges for a rectangular grid."""
    edges: list[tuple[int, int]] = []
    for row in range(n_rows):
        for col in range(n_cols):
            node = row * n_cols + col
            if col + 1 < n_cols:
                edges.append((node, node + 1))
            if row + 1 < n_rows:
                edges.append((node, node + n_cols))
    return np.asarray(edges, dtype=np.int64)


def _grid_boundary_weights(n_rows: int, n_cols: int) -> np.ndarray:
    """Return jitter weights that keep boundary nodes nearly fixed."""
    weights = np.ones(n_rows * n_cols, dtype=np.float64)
    for row in range(n_rows):
        for col in range(n_cols):
            if row in (0, n_rows - 1) or col in (0, n_cols - 1):
                weights[row * n_cols + col] = 0.15
    return weights


def _grid_boundary_indices(n_rows: int, n_cols: int) -> list[int]:
    """Return boundary grid node indices in clockwise order."""
    top = [col for col in range(n_cols)]
    right = [row * n_cols + n_cols - 1 for row in range(1, n_rows)]
    bottom = [(n_rows - 1) * n_cols + col for col in range(n_cols - 2, -1, -1)]
    left = [row * n_cols for row in range(n_rows - 2, 0, -1)]
    return top + right + bottom + left


def _accepted_kwargs(
    generator: Callable[..., EmbeddedPlanarGraph],
    kwargs: dict[str, object],
) -> dict[str, object]:
    """Filter dataset overrides to parameters accepted by one generator."""
    accepted = set(signature(generator).parameters)
    return {key: value for key, value in kwargs.items() if key in accepted}


def _validate_dataset_kwargs(classes: tuple[str, ...], kwargs: dict[str, object]) -> None:
    """Raise on dataset overrides accepted by none of the selected classes."""
    accepted: set[str] = set()
    for class_name in classes:
        if class_name in _DATASET_GENERATORS:
            generator, _ = _DATASET_GENERATORS[class_name]
            accepted.update(signature(generator).parameters)
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        raise ValueError(f"unsupported dataset override(s): {', '.join(unknown)}")


def _angle_in_sector(theta: np.ndarray, center: float, width: float) -> np.ndarray:
    """Return a mask for angles lying inside a circular sector."""
    wrapped = (theta - center + np.pi) % (2.0 * np.pi) - np.pi
    return np.abs(wrapped) <= width / 2.0


def _outward_direction(point: np.ndarray) -> np.ndarray:
    """Return the radial outward unit vector for a point near the origin."""
    norm = float(np.linalg.norm(point))
    if norm == 0.0:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    return np.asarray(point, dtype=np.float64) / norm


def _right_normal(vector: np.ndarray) -> np.ndarray:
    """Return the right-hand unit normal of a nonzero 2D vector."""
    unit = _unit(vector)
    return np.asarray([unit[1], -unit[0]], dtype=np.float64)


def _rotate(vector: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 2D vector by angle radians."""
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]], dtype=np.float64)


def _unit(vector: np.ndarray) -> np.ndarray:
    """Return a normalized vector, raising on zero length."""
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero-length vector")
    return vector / norm


def _circular_distance(i: int, j: int, n: int) -> int:
    """Return circular index distance between two vertices on an n-cycle."""
    delta = abs(int(i) - int(j))
    return min(delta, n - delta)


def _require_root_separation(roots: list[int], n_loop_vertices: int, min_separation: int) -> None:
    """Validate circular spacing for already selected roots."""
    for left_index, left in enumerate(roots):
        for right in roots[left_index + 1 :]:
            if _circular_distance(left, right, n_loop_vertices) < min_separation:
                raise ValueError(
                    "could not select uniformly spaced roots satisfying min_separation; "
                    "reduce n_roots or min_root_separation"
                )


def _actual_min_root_separation(roots: list[int], n_loop_vertices: int) -> Optional[int]:
    """Return the achieved circular minimum root separation, or None for fewer than two roots."""
    if len(roots) < 2:
        return None
    return min(
        _circular_distance(left, right, n_loop_vertices)
        for left_index, left in enumerate(roots)
        for right in roots[left_index + 1 :]
    )


def _point_to_list(point: np.ndarray) -> list[float]:
    """Convert one point to JSON-friendly Python floats."""
    return [float(point[0]), float(point[1])]


def _int_list(values: object) -> list[int]:
    """Convert iterable integer-like values to Python ints."""
    return [int(value) for value in values]


def _json_labels(labels: dict[str, object]) -> dict[str, object]:
    """Convert NumPy scalar metadata values into JSON-friendly Python values."""
    clean: dict[str, object] = {}
    for key, value in labels.items():
        if isinstance(value, np.integer):
            clean[key] = int(value)
        elif isinstance(value, np.floating):
            clean[key] = float(value)
        elif isinstance(value, np.ndarray):
            clean[key] = value.tolist()
        else:
            clean[key] = value
    return clean


def _validate_mode(name: str, value: str, supported: tuple[str, ...]) -> None:
    """Validate a string mode value."""
    if value not in supported:
        raise ValueError(f"{name} must be one of {supported}; got {value!r}")


def _validate_int(name: str, value: int, minimum: int) -> None:
    """Validate an integer parameter with a lower bound."""
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _validate_positive(name: str, value: float) -> None:
    """Validate a strictly positive numeric parameter."""
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_nonnegative(name: str, value: float) -> None:
    """Validate a non-negative numeric parameter."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_probability(name: str, value: float) -> None:
    """Validate a probability parameter."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


_DATASET_GENERATORS: dict[str, tuple[Callable[..., EmbeddedPlanarGraph], dict[str, object]]] = {
    "hairs_uniform": (generate_hair_loop, {"hair_mode": "uniform"}),
    "hairs_polarized": (generate_hair_loop, {"hair_mode": "polarized"}),
    "hairs_random": (generate_hair_loop, {"hair_mode": "random"}),
    "blebs_uniform": (generate_bleb_loop, {"bleb_mode": "uniform"}),
    "blebs_random": (generate_bleb_loop, {"bleb_mode": "random"}),
    "blebs_polarized": (generate_bleb_loop, {"bleb_mode": "polarized"}),
    "branch_single": (generate_branching_loop, {"branch_mode": "single", "secondary_probability": 0.0}),
    "branch_Y": (generate_branching_loop, {"branch_mode": "Y"}),
    "fractal_koch": (generate_fractal_loop, {"fractal_type": "koch"}),
    "fractal_midpoint": (generate_fractal_loop, {"fractal_type": "midpoint"}),
    "tortuous": (generate_tortuous_loop, {}),
    "serrated": (generate_serrated_loop, {}),
    "mixed_half": (generate_mixed_loop, {"mix_mode": "half_half"}),
    "mixed_random": (generate_mixed_loop, {"mix_mode": "random"}),
    "mixed_alternating": (generate_mixed_loop, {"mix_mode": "alternating"}),
    "mild_grid_regular": (generate_mild_iso_cycle_grid, {"distortion_type": "regular"}),
    "mild_grid_sinusoidal": (generate_mild_iso_cycle_grid, {"distortion_type": "sinusoidal"}),
    "mild_grid_wavy": (generate_mild_iso_cycle_grid, {"distortion_type": "wavy"}),
    "mild_grid_jittered": (generate_mild_iso_cycle_grid, {"distortion_type": "jittered"}),
}


if __name__ == "__main__":
    examples = generate_dataset(SUPPORTED_CLASSES, n_per_class=1, seed=0)
    for graph in examples:
        print(
            f"{graph.labels['class_name']}: "
            f"nodes={graph.nodes.shape[0]}, "
            f"edges={graph.edges.shape[0]}, "
            f"label_keys={sorted(graph.labels.keys())}"
        )
