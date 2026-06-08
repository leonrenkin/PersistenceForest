import math
from typing import Iterable, Tuple
from numpy.typing import NDArray
import numpy as np

_EPS = 1e-12

def _to_xy(points: Iterable[Tuple[float, float]]) -> np.ndarray:
    """
    Convert input points to an ``(N, 2)`` float64 array, dropping a duplicated
    closing point when the first and last coordinates coincide.

    Parameters
    ----------
    points : iterable of (float, float)
        Polygon vertices in order; a trailing copy of the first vertex is
        tolerated and removed.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 2)`` with dtype float64.

    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("points must be an (N,2) array-like of x,y pairs")
    if np.linalg.norm(P[0] - P[-1]) < _EPS:  # drop repeated closure point
        P = P[:-1].copy()
    return P

def polygon_length(points: Iterable[Tuple[float, float]]) -> float:
    """
    Perimeter (sum of edge lengths) of the closed polygonal loop.

    Parameters
    ----------
    points : iterable of (float, float)
        Vertices of the polygon; a trailing duplicate of the first point
        is allowed.

    Returns
    -------
    float
        Total Euclidean length of all edges.
    """
    P = _to_xy(points)
    diffs = np.roll(P, -1, axis=0) - P
    return float(np.linalg.norm(diffs, axis=1).sum())

def polygon_area(points: Iterable[Tuple[float, float]], signed: bool = False) -> float:
    """
    Shoelace area of the polygon; optionally return the signed value.

    Parameters
    ----------
    points : iterable of (float, float)
        Vertices of the polygon; a trailing duplicate of the first point
        is allowed.
    signed : bool, optional
        If True, return the signed area (CCW positive, CW negative).

    Returns
    -------
    float
        Polygon area (signed or absolute depending on ``signed``).
    """
    P = _to_xy(points)
    x, y = P[:, 0], P[:, 1]
    xs, ys = np.roll(x, -1), np.roll(y, -1)
    a = 0.5 * float(np.dot(x, ys) - np.dot(y, xs))
    return a if signed else abs(a)

def polygon_area_length_ratio(points: Iterable[Tuple[float, float]]) -> float:
    """
    Area divided by squared perimeter (isoperimetric ratio).

    Parameters
    ----------
    points : iterable of (float, float)
        Polygon vertices.

    Returns
    -------
    float
        ``area / perimeter**2``.
    """
    return polygon_area(points=points)/(polygon_length(points=points)**2)

def polygon_area_length_squared_ratio(points: Iterable[Tuple[float, float]]) -> float:
    """
    Area divided by perimeter.

    Parameters
    ----------
    points : iterable of (float, float)
        Polygon vertices.

    Returns
    -------
    float
        ``area / perimeter``.
    """
    return polygon_area(points=points)/polygon_length(points=points)

def polygon_length_area_ratio(points: Iterable[Tuple[float, float]], tol = 1e-8) -> float:
    """
    Perimeter divided by area; returns 0 when area is below ``tol``.

    Parameters
    ----------
    points : iterable of (float, float)
        Polygon vertices.
    tol : float, optional
        If area < ``tol``, return 0 to avoid division by tiny values.

    Returns
    -------
    float
        ``perimeter / area`` or 0 when the area threshold is not met.
    """
    if polygon_area(points=points) < tol:
        return 0
    else:
        return polygon_length(points=points)/polygon_area(points=points)

def polygon_length_squared_area_ratio_normalized(points: Iterable[Tuple[float, float]], tol = 1e-8) -> float:
    """
    Isoperimetric deficit: (perimeter^2 / area) - 4π.

    Returns 0 when area is below ``tol``.

    Parameters
    ----------
    points : iterable of (float, float)
        Polygon vertices.
    tol : float, optional
        If area < ``tol``, return 0 to avoid division by tiny values.

    Returns
    -------
    float
        ``perimeter**2 / area - 4π`` or 0 when the area threshold is not met.
    """
    if polygon_area(points=points) < tol:
        return 0
    else:
        return polygon_length(points=points)**2/polygon_area(points=points) - 4*np.pi

def polygon_length_squared_area_ratio(points: Iterable[Tuple[float, float]]) -> float:
    """
    Perimeter squared divided by area (isoperimetric quotient).

    Parameters
    ----------
    points : iterable of (float, float)
        Polygon vertices.

    Returns
    -------
    float
        ``perimeter**2 / area``.
    """
    return (polygon_length(points=points)**2)/polygon_area(points=points)

def total_curvature(points: Iterable[Tuple[float, float]]) -> float:
    """
    Total curvature of a simple closed polygonal loop:
        K_total = sum_i |kappa_i|
    where kappa_i is the signed exterior turning angle at vertex i.

    Definition at vertex i:
        a = P[i]   - P[i-1]   (incoming edge)
        b = P[i+1] - P[i]     (outgoing edge)
        kappa_i = atan2( cross(a,b), dot(a,b) ) ∈ (-π, π]

    Properties
    ----------
    - For a simple CCW convex polygon (e.g., regular n-gon), sum(kappa_i) ≈ +2π and
      total_curvature ≈ 2π.
    - For non-convex/star-shaped polygons (some kappa_i < 0), total_curvature > 2π.
    - Invariant to translation and rotation; scale-invariant (angles only).

    Parameters
    ----------
    points : sequence of (x,y) defining a simple loop (last connects to first).

    Returns
    -------
    K : float
        Total curvature (radians).
    """
    P = _to_xy(points)

    prevP = np.roll(P, 1, axis=0)
    nextP = np.roll(P, -1, axis=0)
    a = P - prevP
    b = nextP - P

    a_len = np.linalg.norm(a, axis=1)
    b_len = np.linalg.norm(b, axis=1)
    bad = (a_len < _EPS) | (b_len < _EPS)
    if np.any(bad):
        raise ValueError("Zero-length edge detected near vertices: "
                         f"{np.where(bad)[0].tolist()}")

    cross = a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]     # z-component
    dot   = a[:, 0]*b[:, 0] + a[:, 1]*b[:, 1]
    kappa = np.arctan2(cross, dot)                # signed exterior angle ∈ (-π, π]
    abs_kappa = np.abs(kappa)

    K = float(abs_kappa.sum())
    return K

def curvature_excess(points: Iterable[Tuple[float, float]], normalize: bool = True) -> float:
    """
    Excess curvature: total_curvature - 2π

    Returns 0 for a convex loop (total curvature ≈ 2π) and positive values
    for star-shaped or non-convex loops.

    Parameters
    ----------
    points : iterable of (float, float)
        Polygon vertices.
    normalize : bool, optional
        If True, return excess curvature normalized by 2π.    

    Returns
    -------
    float
        Excess curvature normalized by 2π.
    """
    K = total_curvature(points) - 2.0*math.pi
    if normalize:
        K /= (2.0*math.pi)
    return K 

def signed_chain_edge_length(signed_chain, point_cloud: NDArray[np.float64]) -> float:
    """
    Sum of Euclidean lengths of all 1-simplices in a signed chain.

    Parameters
    ----------
    signed_chain
        Object exposing ``signed_simplices`` that yields (simplex, sign) pairs.
        Only simplices with two vertices (edges) contribute.
    point_cloud : (n_points, dim) np.ndarray
        Coordinates for the ambient point cloud.

    Returns
    -------
    float: Total edge length (orientation ignored).

    Notes
    -----
    Signs/orientations are ignored; the chain is treated as unsigned for length.
    """
    if signed_chain.dim() != 1:
        raise ValueError("Function only defined for 1-dimensional chains")
    
    total = 0.0
    for simplex, sign in signed_chain.signed_simplices:
        # Make sure we have exactly two vertices: an edge
        verts_idx = list(simplex)
        if len(verts_idx) != 2:
            continue

        p0 = point_cloud[verts_idx[0]]
        p1 = point_cloud[verts_idx[1]]
        length = float(np.linalg.norm(p1 - p0))

        # Orientation/sign is ignored; we just accumulate edge lengths.
        total += length

    return total

def constant_one_functional(signed_chain, point_cloud: NDArray[np.float64]) -> float:
    """
    Return the constant value 1; useful as a trivial chain functional.

    Parameters
    ----------
    signed_chain
        Unused; present for signature compatibility.
    point_cloud : np.ndarray
        Unused; present for signature compatibility.

    Returns
    -------
    int
        Always 1.
    """
    return 1

def signed_chain_connected_components(signed_chain, point_cloud: NDArray[np.float64]) -> float:
    """
    Number of disjoint polyhedral paths in the signed chain.

    Parameters
    ----------
    signed_chain
        Object exposing ``polyhedral_paths(point_cloud)``.
    point_cloud : (n_points, dim) np.ndarray
        Coordinates for the ambient point cloud.

    Returns
    -------
    int
        Count of connected components.
    """
    if signed_chain.dim() != 1:
        raise ValueError("Function only defined for 1-dimensional chains")
    return len( signed_chain.polyhedral_paths(point_cloud) )

def signed_chain_excess_connected_components(signed_chain, point_cloud: NDArray[np.float64]) -> float:
    """
    Number of components minus one; zero when the chain is connected.

    Parameters
    ----------
    signed_chain
        Object exposing ``polyhedral_paths(point_cloud)``.
    point_cloud : (n_points, dim) np.ndarray
        Coordinates for the ambient point cloud.

    Returns
    -------
    int
        Components count minus one.
    """
    if signed_chain.dim() != 1:
        raise ValueError("Function only defined for 1-dimensional chains")
    return len( signed_chain.polyhedral_paths(point_cloud) ) - 1

def signed_chain_connected_components_only_signed_simplices(signed_chain, point_cloud: NDArray[np.float64]):
    return signed_chain_connected_components( signed_chain=signed_chain.only_double_simplices(), point_cloud=point_cloud)

def signed_chain_area(signed_chain, point_cloud:  NDArray[np.float64]) -> float:
    """
    Area enclosed by a chain with possible holes.

    Assumes exactly one outer boundary and any number of inner boundaries.
    The path whose x-coordinate attains the global maximum is treated as the
    outer boundary; its area is added while all others are subtracted.

    Parameters
    ----------
    signed_chain
        Object exposing ``polyhedral_paths(point_cloud)`` that yield index
        sequences for polygon boundaries.
    point_cloud : (n_points, dim) np.ndarray
        Coordinates for the ambient point cloud.

    Returns
    -------
    float
        Signed area of the chain (outer minus inner regions).
    """
    if signed_chain.dim() != 1:
        raise ValueError("Function only defined for 1-dimensional chains")

    paths = list( signed_chain.polyhedral_paths(point_cloud) )
    x_max_list = np.array([point_cloud[path, 0].max() for path in paths])
    index_max = np.argmax(x_max_list)

    total_area = 0

    for index, path in enumerate(paths):
        if len(path) < 2:
            print(paths)
            raise ValueError("Paths too short")
        if index == index_max:
            total_area += polygon_area(point_cloud[path])
        else:
            total_area -= polygon_area(point_cloud[path])
    return total_area

def signed_chain_excess_curvature(signed_chain, point_cloud: NDArray[np.float64]) -> float:
    """
    Sum of ``curvature_excess`` over every polyhedral path in the chain.

    Parameters
    ----------
    signed_chain
        Object containing signed simplices of the form (simplex, sign), 
        where simplex is of the form tuple[int] corresponing to indices in point cloud
        and sign is +-1.
    point_cloud : (n_points, dim) np.ndarray
        Coordinates for the ambient point cloud.

    Returns
    -------
    float
        Total excess curvature across all paths.
    """
    if signed_chain.dim() != 1:
        raise ValueError("Function only defined for 1-dimensional chains")

    paths = list( signed_chain.polyhedral_paths(point_cloud) )
    total = 0
    for path in paths:
        if len(path) < 2:
            print(paths)
            raise ValueError("Paths too short")
        total += curvature_excess(point_cloud[path], normalize=False)

    return total

def signed_chain_excess_curvature_diff_to_unsigned(signed_chain, point_cloud: NDArray[np.float64]):
    diff = signed_chain_excess_curvature(signed_chain=signed_chain,point_cloud=point_cloud) - signed_chain_excess_curvature(signed_chain=signed_chain.unsigned(),point_cloud=point_cloud)
    return diff

def signed_chain_excess_curvature_normalized(signed_chain, point_cloud: NDArray[np.float64]) -> float:
    """
    Sum of ``curvature_excess`` with normalized=True over every polyhedral path in the chain.


    Parameters
    ----------
    signed_chain
        Object containing signed simplices of the form (simplex, sign), 
        where simplex is of the form tuple[int] corresponing to indices in point cloud
        and sign is +-1.
    point_cloud : (n_points, dim) np.ndarray
        Coordinates for the ambient point cloud.

    Returns
    -------
    float
        Total normalized excess curvature across all paths.
    """
    return signed_chain_excess_curvature(signed_chain, point_cloud) / (2.0*math.pi)

def signed_chain_circularity(signed_chain, point_cloud: NDArray[np.float64]) -> float:
    """
    Circularity functional: 4pi * area / perimeter^2  for the chain's paths.
    This is 1 for a perfect circle and between 0 and 1 for less circular shapes.

    Parameters
    ----------
    signed_chain
        Object containing signed simplices of the form (simplex, sign), 
        where simplex is of the form tuple[int] corresponing to indices in point cloud
        and sign is +-1
    point_cloud : (n_points, dim) np.ndarray
        Coordinates for the ambient point cloud.

    Returns
    -------
    float
        Circularity measure; higher values indicate more circular shapes.
    """
    if signed_chain.dim() != 1:
        raise ValueError("Function only defined for 1-dimensional chains")
    
    length = signed_chain_edge_length(signed_chain=signed_chain, point_cloud=point_cloud)
    area = signed_chain_area(signed_chain=signed_chain,point_cloud=point_cloud)
    circularity = (4.0*math.pi*area)/length**2

    return circularity

def signed_chain_circularity_complement(signed_chain, point_cloud: NDArray[np.float64]) -> float:
    """
    Return 1-circularity
    Circularity functional: 4pi * area / perimeter^2  for the chain's paths.
    This is 0 for a perfect circle and between 0 and 1 for less circular shapes.

    Parameters
    ----------
    signed_chain
        Object containing signed simplices of the form (simplex, sign), 
        where simplex is of the form tuple[int] corresponing to indices in point cloud
        and sign is +-1.
    point_cloud : (n_points, dim) np.ndarray
        Coordinates for the ambient point cloud.

    Returns
    -------
    float
        Non-Circularity measure; higher values indicate more circular shapes.
    """
    return 1- signed_chain_circularity(signed_chain=signed_chain, point_cloud=point_cloud)

def signed_chain_non_circularity(signed_chain, point_cloud: NDArray[np.float64]) -> float:
    """
    Circularity functional: perimeter^2 / (4pi*area) - 1 for the chain's paths.
    This is zero for a perfect circle and positive for less circular shapes.

    Parameters
    ----------
    signed_chain
        Object containing signed simplices of the form (simplex, sign), 
        where simplex is of the form tuple[int] corresponing to indices in point cloud
        and sign is +-1.
    point_cloud : (n_points, dim) np.ndarray
        Coordinates for the ambient point cloud.

    Returns
    -------
    float
        Non-Circularity measure; higher values indicate less circular shapes.
    """
    if signed_chain.dim() != 1:
        raise ValueError("Function only defined for 1-dimensional chains")
    length = signed_chain_edge_length(signed_chain=signed_chain, point_cloud=point_cloud)
    area = signed_chain_area(signed_chain=signed_chain,point_cloud=point_cloud)
    non_circularity = length**2/(4.0*math.pi*area) - 1

    return non_circularity

def signed_chain_volume(signed_chain, point_cloud: NDArray[np.float64]) -> float:
    """
    Returns volume of a signed chain. 
    For a 2d point cloud, it corresponds to length (and not contained area), 
    For a 3d point cloud, it corresponds to area (and not contained volume).

    Parameters
    ----------
    signed_chain
        Object containing signed simplices of the form (simplex, sign), 
        where simplex is of the form tuple[int] corresponing to indices in point cloud
        and sign is +-1.
    point_cloud : (n_points, dim) np.ndarray
        Coordinates for the ambient point cloud.

    Returns
    -------
    float
    """

    simplices = np.asarray([simplex for simplex, sign in signed_chain.signed_simplices])

    # Gather all matrices at once: shape (m, d, d)
    mats = point_cloud[simplices]

    # Batched determinant: shape (m,)
    dets = np.linalg.det(mats)

    return float(np.abs(dets).sum())

