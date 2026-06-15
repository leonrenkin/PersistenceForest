import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Literal, Iterable, Callable, Union, Sequence, Set
from collections import defaultdict
from numpy.typing import NDArray
import itertools
import gudhi as gd
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from matplotlib.collections import LineCollection, PolyCollection
import time
import seaborn as sns
from bisect import bisect_right
import warnings

# ------- helper function -----------

def key(simplex):
    """
    Return a canonical, orientation-free key for a simplex.

    Parameters
    ----------
    simplex : iterable[int]
        Vertex ids of the simplex.

    Returns
    -------
    tuple
        Sorted tuple of vertex ids, suitable for hashing regardless of
        orientation.
    """
    return tuple(sorted(simplex))

def sign_of_determinant(vectors):
    """
    Computes the sign of the determinant of d vectors in R^d.

    Parameters
    ----------
    vectors : Iterable[Iterable[float]]
        Collection of d vectors of length d.

    Returns
    -------
    int
        +1 if det > 0, -1 if det < 0, 0 if det = 0.
    """
    A = np.array(vectors, dtype=float)
    d = A.shape[0]

    sign = 1

    for i in range(d):
        # Find pivot
        pivot = i + np.argmax(abs(A[i:, i]))
        if abs(A[pivot, i]) < 1e-12:
            return 0  # determinant is zero

        # Row swap changes sign
        if pivot != i:
            A[[i, pivot]] = A[[pivot, i]]
            sign *= -1

        # Eliminate below pivot
        for j in range(i + 1, d):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]

    # Sign of determinant is the product of the signs of diagonal entries
    diag_sign = np.sign(np.prod(np.sign(np.diag(A))))
    return int(sign * diag_sign)

def are_dict_keys_sorted(d):
    """
    Return True if dict keys are in ascending order (linear time).

    Parameters
    ----------
    d : dict
        Dictionary whose key insertion order is inspected.

    Returns
    -------
    bool
        True if keys appear in ascending order, False otherwise.
    """
    it = iter(d)  # iterates over keys in insertion order
    try:
        prev = next(it)
    except StopIteration:
        return True  # empty dict is trivially sorted

    for k in it:
        if k < prev:
            return False
        prev = k
    return True

def union_optional_sets(set1: Optional[Set], set2: Optional[Set]) -> Optional[Set]:
    """
    Return the union of two sets, where either set may be None.

    Parameters
    ----------
    set1 : set or None
        First set to union.
    set2 : set or None
        Second set to union.

    Returns
    -------
    set or None
        Union of the two sets, or None if both inputs are None.
    """
    if set1 is None and set2 is None:
        return None
    elif set1 is None:
        return set2
    elif set2 is None:
        return set1
    else:
        return set1 | set2

def simplex_orientation(simplex, point_cloud):
    """
    Compute the orientation of a simplex with respect to the ambient point cloud.

    Parameters
    ----------
    simplex : sequence[int]
        Vertex ids of the simplex.
    point_cloud : ndarray
        Coordinates of all vertices.

    Returns
    -------
    int
        +1 for positive orientation, -1 for negative orientation, 0 for
        degenerate simplices.
    """
    vectors = [point_cloud[i]-point_cloud[simplex[0]] for i in simplex[1:]]
    return sign_of_determinant(vectors=vectors)

def signed_boundary(simplex, orientation: int): 
    """
    Compute the oriented boundary of a simplex.

    Parameters
    ----------
    simplex : list[int]
        Vertices of the simplex whose boundary is computed.
    orientation : int
        Orientation (+1 or -1) of the simplex.

    Returns
    -------
    set[tuple]
        Set of oriented faces stored as (simplex_tuple, orientation).
    """
    return {(tuple(simplex[:i] + simplex[i+1:]), orientation* (-1)**i) for i in range(len(simplex))}

def merge_at_simplex(cycle1: "SignedChain", cycle2: "SignedChain", simplex: list[int]) -> "SignedChain":
    """
    Union two chains and remove the specified simplex if it
    appears with opposite orientation in the two chains.

    Parameters
    ----------
    cycle1 : SignedChain
        Chain to merge.
    cycle2 : SignedChain
        Chain to merge.
    simplex : list[int]
        Simplex that is being collapsed when the chains merge.

    Returns
    -------
    SignedChain
        New chain representing the merged cycles.
    """
    union = cycle1.signed_simplices | cycle2.signed_simplices
    return SignedChain(signed_simplices= union.difference({(tuple(simplex),1),(tuple(simplex),-1)}) )

def update_chain_with_diff(signed_simplices: Set[tuple], interior_diff: Set[tuple] | None, codim1_simplex_diff: Set[tuple] | None):
    """
    Apply stored simplex differences to a mutable set of signed simplices.

    Parameters
    ----------
    signed_simplices : set[tuple]
        Current oriented codimension-one simplices, stored as
        ``(simplex_tuple, orientation)``.
    interior_diff : set[tuple] | None
        Oriented full-dimensional simplices whose signed boundaries should be
        added to ``signed_simplices``.
    codim1_simplex_diff : set[tuple] | None
        Oriented codimension-one simplices to remove from ``signed_simplices``.

    Returns
    -------
    set[tuple]
        The same set object after applying the additions and removals.
    """

    if interior_diff is not None:
        for oriented_simplex in interior_diff:
            signed_simplices.update(signed_boundary(simplex =oriented_simplex[0], orientation=oriented_simplex[1]))

    if codim1_simplex_diff is not None:
        for oriented_simplex in codim1_simplex_diff:
            signed_simplices.remove(oriented_simplex)
    
    return signed_simplices

# ------------ Classes ---------------

@dataclass(slots=True)
class SignedChain: 
    """
    Oriented simplicial chain used as a cycle representative.

    Attributes
    ----------
    signed_simplices : set[tuple]
        Oriented codimension-one simplices, stored as
        ``(simplex_tuple, orientation)``.
    active_start, active_end : float
        Half-open filtration interval ``[active_start, active_end)`` on which
        this representative is active.
    interior_available : bool
        True when ``interior`` has been computed.
    interior : set[tuple] | None
        Oriented full-dimensional simplices whose boundary gives the chain's
        tracked interior.
    """
    signed_simplices: Set[tuple]        # oriented codimension-one simplices, stored as (simplex, orientation)
    active_start: float = float("-inf")
    active_end: float   = float("-inf")
    interior_available: bool = False
    interior: Set[tuple] | None = None        # oriented (d+1)-simplices, stored as (simplex, orientation)
    
    def cancel_simplex(self, simplex: list[int]) -> "SignedChain":
        """
        Remove a simplex that appears with both orientations in this chain.

        Parameters
        ----------
        simplex : list[int]
            Simplex to delete from the chain.

        Returns
        -------
        SignedChain
            Chain with the simplex removed in both orientations.
        """
        return SignedChain(signed_simplices= self.signed_simplices.difference({(tuple(simplex),1),(tuple(simplex),-1)}) )
    
    def unsigned(self) -> "SignedChain":
        """
        Return a new SignedChain where simplices that appear with opposite
        orientations cancel out.

        If an underlying simplex appears only with one orientation, it is kept.

        Returns
        -------
        SignedChain
            Chain without simplices that appear with both orientations.
        """
        # Aggregate orientations per underlying simplex.
        coeffs: Dict[tuple, int] = {}
        for simplex, orientation in self.signed_simplices:
            coeffs[simplex] = coeffs.get(simplex, 0) + int(orientation)

        cleaned: Set[tuple] = set()
        for simplex, c in coeffs.items():
            if c > 0:
                cleaned.add((simplex, 1))
            elif c < 0:
                cleaned.add((simplex, -1))
            # If c == 0, the +1 and -1 copies cancel.
        
        return SignedChain(
            signed_simplices=cleaned,
            active_start=self.active_start,
            active_end=self.active_end,
        )

    def segments(self, point_cloud: NDArray):
        """
        Convert oriented edges in the chain to geometric segments.

        Parameters
        ----------
        point_cloud : ndarray
            Ambient coordinates for simplex vertices.

        Returns
        -------
        list[np.ndarray]
            List of 2xD arrays representing oriented edge segments.
        """
        segments = []
        for signed_simplex in self.signed_simplices:
            if signed_simplex[1]==1:
                segments.append(np.array(point_cloud[list(signed_simplex[0])]))
            else:
                segments.append(np.array(point_cloud[list(reversed(signed_simplex[0]))]))
        return segments
    
    def dim(self):
        """Return the topological dimension of the simplices in this chain."""
        for signed_simplex in self.signed_simplices:
            return len(signed_simplex[0])-1

    def polyhedral_paths(self, point_cloud: NDArray) -> List[NDArray[np.int32]]:
        """
        Decompose a 1-dimensional SignedChain (collection of oriented edges in R^2)
        into polyhedral paths, choosing at each branching point the leftmost
        outgoing edge (smallest counterclockwise angle).

        Parameters
        ----------
        point_cloud : ndarray, shape (n_points, dim>=2)
            Ambient point cloud. Vertex indices in the simplices refer into this
            array. Only the first two coordinates (x,y) are used.

        Returns
        -------
        paths : list of 1D int ndarrays
            Each array `v = paths[k]` is a cyclic list of vertex indices,
            analogous to `Loop.vertex_list` in LoopForest:
                edges are (v[i], v[i+1]) and the final edge (v[-1], v[0]).
            The first vertex is repeated at the end if the greedy walk closes
            up naturally.
        """
        if not self.signed_simplices:
            return []

        if self.dim() !=1:
            raise ValueError(f"Polyhedral path methods only works for Signed 1-chains. Dimemsion of chain: {self.dim}")

        from collections import defaultdict

        def _start_end(simplex: Tuple[int, ...], orientation: int) -> Tuple[int, int]:
            """Return (start, end) vertex of an oriented 1-simplex."""
            if len(simplex) != 2:
                raise ValueError(
                    "polyhedral_paths is only implemented for 1-chains (edges)."
                )
            a, b = simplex
            if orientation == 1:
                return a, b
            elif orientation == -1:
                return b, a
            else:
                raise ValueError("Orientation must be ±1.")

        def _angle_ccw(prev_vec: np.ndarray, next_vec: np.ndarray) -> float:
            """
            Signed angle from prev_vec to next_vec in [0, 2π),
            measured counterclockwise.
            """
            # Work in 2D; take first two coordinates
            pv = prev_vec[:2]
            nv = next_vec[:2]

            # Skip zero-length directions at caller level
            cross = pv[0] * nv[1] - pv[1] * nv[0]
            dot   = pv[0] * nv[0] + pv[1] * nv[1]
            angle = math.atan2(cross, dot)  # ∈ (-π, π]
            if np.all(pv == -nv):
                angle = -math.pi
            return angle

        # Precompute adjacency: start vertex -> list of (signed_simplex, end_vertex)
        edges_by_start: Dict[int, List[Tuple[tuple, int]]] = defaultdict(list)
        for signed_simplex in self.signed_simplices:
            simplex, orientation = signed_simplex
            start, end = _start_end(simplex, orientation)
            edges_by_start[start].append((signed_simplex, end))

        visited: Set[tuple] = set()
        paths: List[NDArray[np.int32]] = []

        for signed_simplex in self.signed_simplices:
            if signed_simplex in visited:
                continue

            simplex, orientation = signed_simplex
            start, end = _start_end(simplex, orientation)

            # Start a new path with this edge
            path_vertices: List[int] = [int(start), int(end)]
            visited.add(signed_simplex)

            prev_vertex = start
            cur_vertex = end

            p_prev = np.asarray(point_cloud[prev_vertex], dtype=float)[:2]
            p_cur  = np.asarray(point_cloud[cur_vertex],  dtype=float)[:2]
            prev_vec = p_cur - p_prev

            while True:
                candidates = edges_by_start.get(cur_vertex, [])
                if not candidates:
                    # No outgoing edges from this vertex
                    raise ValueError("No outgoing edges, this should not happen")

                best_edge: Optional[tuple] = None
                best_next_vertex: Optional[int] = None
                best_angle: Optional[float] = None

                for edge_key, next_vertex in candidates:
                    p_next = np.asarray(point_cloud[next_vertex], dtype=float)[:2]
                    next_vec = p_next - p_cur

                    # Ignore degenerate directions
                    if np.allclose(next_vec, 0.0):
                        continue

                    angle = _angle_ccw(prev_vec, next_vec)

                    if best_angle is None or angle > best_angle:
                        best_angle = angle
                        best_edge = edge_key
                        best_next_vertex = int(next_vertex)

                if best_edge is None or best_next_vertex is None:
                    # Only degenerate candidates
                    raise ValueError("Only degenerate candidates, this should not happen")

                # Stop if we would traverse an already covered signed simplex
                if best_edge in visited:
                    break

                # Advance along the chosen leftmost edge
                path_vertices.append(best_next_vertex)
                visited.add(best_edge)

                prev_vertex, cur_vertex = cur_vertex, best_next_vertex
                p_prev, p_cur = p_cur, np.asarray(point_cloud[cur_vertex], dtype=float)[:2]
                prev_vec = p_cur - p_prev

            paths.append(np.array(path_vertices[:-1], dtype=np.int32)) #last vertex is repeated, remove it

        for path in paths:
            if len(path)<2:
                print(paths)
                raise ValueError("Path too short in SignedChain.polyhedral_path() method")

        return paths
    
    def vertex_coordinates(self, point_cloud: NDArray, signed = True) -> NDArray[np.float64]:
        """
        Extract the coordinates of the vertices in this SignedChain.

        Parameters
        ----------
        point_cloud : ndarray, shape (n_points, dim)
            Ambient point cloud.
        signed : bool
            If True, use the stored signed simplices. If False, first cancel
            opposite-oriented copies via ``unsigned()``.

        Returns
        -------
        ndarray, shape (n_vertices, dim)
            Coordinates of the vertices in the chain.
        """
        vertex_indices: Set[int] = set()
        if signed: 
            signed_simplices = self.signed_simplices
        else:
            signed_simplices = self.unsigned().signed_simplices

        for signed_simplex in signed_simplices:
            simplex, _ = signed_simplex
            vertex_indices.update(simplex)

        coords = np.array([point_cloud[i] for i in sorted(vertex_indices)], dtype=float)
        return coords

@dataclass(slots=True)
class PFNode:
    """
    Node in the PersistenceForest graph.

    Each node stores a cycle representative and links to its parent and
    children in the forest. Optional diff fields store how to reconstruct cycle
    representatives and interiors when ``keep_simplex_diff=True``.
    """
    id: int # unique node identifier
    filt_val: float
    cycle: SignedChain 
    children: set[int]                                    #ids of children
    parent: Optional[int] = None
    # is_root would be True for tree roots; parent is the current source of truth.
    _barcode_covered: int = 0

    _simplex_diff_available: bool = False       # True when simplex diffs were stored for this node.
    _interior_diff: set[tuple] | None = None     # oriented full-dimensional simplices, stored as (simplex, orientation)
    _codim1_simplex_diff: set[tuple] | None = None  # oriented codimension-one simplices to remove, stored as (simplex, orientation)
    # Cycle at node: union of child cycles plus boundaries from _interior_diff,
    # minus _codim1_simplex_diff.
    _barcode_interior_diff: set[tuple] | None = None  
    _barcode_codim1_simplex_diff: set[tuple] | None = None  

    def __repr__(self) -> str:
        return f"Node(id={self.id}, f={self.filt_val})"

class PFBar:
    """ 
    Persistence bar together with a progression of cycle representatives.

    For an ambient ``dim``-dimensional point cloud, these bars represent
    codimension-one homology classes. Each cycle representative has
    ``active_start`` and ``active_end`` attributes giving the interval on which
    this representative is active.
    The cycle reps are a strictly decreasing chain w.r.t. inclusion.
    """

    def __init__(self, 
                 birth: float, 
                 death: float, 
                 _node_progression: tuple[int,...], 
                 cycle_reps: list[SignedChain], 
                 is_max_tree_bar: Optional[bool]=None, 
                 root_id: Optional[int]=None):
        """
        Initialize a persistence bar together with its representative cycles.

        Parameters
        ----------
        birth : float
            Filtration value where the class is born.
        death : float
            Filtration value where the class dies.
        _node_progression : tuple[int, ...]
            Node ids (from leaves to root) describing where the representative changes.
        cycle_reps : list[SignedChain]
            Cycle representatives active on consecutive subintervals.
        is_max_tree_bar : bool | None
            True if this bar is the longest in its tree, False if truncated by
            another bar, None if unknown.
        root_id : int | None
            Id of the tree root that contains this bar.
        """
        self.birth = birth
        self.death = death
        self._node_progression = _node_progression #nodes saved as node_ids
        self.cycle_reps = cycle_reps
        self.is_max_tree_bar = is_max_tree_bar
        self.root_id = root_id


    def cycle_at_filtration_value(self, filt_val)->SignedChain:
        """
        Return the active representative cycle at a given filtration value.

        Parameters
        ----------
        filt_val : float
            Filtration value inside ``[birth, death)`` of the bar.

        Returns
        -------
        SignedChain
            Cycle representative active at ``filt_val``.

        Raises
        ------
        ValueError
            If ``filt_val`` lies outside the lifespan or no representative is
            active (should not happen in a valid bar).
        """

        if filt_val < self.birth:
            raise ValueError(f"Filtration value {filt_val} is too small and not in lifespan of the bar")
        if filt_val >= self.death:
            raise ValueError(f"Filtration value {filt_val} is too large and not in lifespan of the bar")

        if len(self._node_progression)==1:
            return self.cycle_reps[0]

        first = 0
        last = len(self.cycle_reps)-1
        best = None

        while first<=last:
            midpoint = (first+last) // 2

            if self.cycle_reps[midpoint].active_start <= filt_val:
                best = self.cycle_reps[midpoint]
                first = midpoint+1
            else:
                last = midpoint-1

        if best == None:
            raise ValueError(f"Binary search in barcode returned {None}, check if bar is empty list")
        elif best.active_start>filt_val or best.active_end<= filt_val:
            raise ValueError("Output of binary search incorrect, loop not active at filtration value. Check correctness of loop_at_filtration method.")
        
        return best
    
    def lifespan(self):
        """Returns lifespan of bar"""
        return self.death - self.birth
    
class PersistenceForest:
    """
    Compute and store alpha-complex cycle progressions in forest format.

    For an ambient ``dim``-dimensional point cloud, the forest tracks
    codimension-one cycle representatives and their barcode.
    """

    def __init__(self, 
                 point_cloud,
                 reduce: bool = True,
                 compute_barcode: bool = True,
                 print_info: bool = False,
                 keep_simplex_diff: bool = False,
                 compute_interior: bool = False,
                 low_memory_mode: bool =False,
                 filtration_tol: float = 1e-12,
                 ):
        """
        Build a PersistenceForest from a point cloud using the alpha complex.

        Parameters
        ----------
        point_cloud : array-like, shape (n_points, dim)
            Coordinates of the input point set.
        reduce : bool
            If True, collapse parent-child pairs whose filtration values differ
            by at most ``filtration_tol``.
        compute_barcode : bool
            If True, compute and store the barcode after building the forest.
        print_info : bool
            If True, print timing information during construction.
        keep_simplex_diff : bool
            If True, store simplex additions/removals on nodes so cycle
            representatives and interiors can be reconstructed from diffs.
        compute_interior : bool
            If True, reconstruct cycle representatives with their interior
            simplices. Requires ``keep_simplex_diff=True``.
        low_memory_mode : bool
            Reserved for a future diff-only representation. Currently raises
            ``ValueError`` when enabled.
        filtration_tol : float
            Absolute tolerance used when reducing parent-child pairs at the
            same filtration value.
        """
        self.point_cloud = np.array(point_cloud) #point cloud is list of n-dim arrays
        self.filtration_tol = float(filtration_tol)
        if self.filtration_tol < 0 or not math.isfinite(self.filtration_tol):
            raise ValueError("filtration_tol must be a finite non-negative number")

        #check if point cloud has correct shape
        self.dim = self.point_cloud.shape[1]
        if print_info:
            print(f'dimension = {self.dim}')

        self._node_id = itertools.count(1)         #used to assign unique id to each node in the forest
        self._cycle_id = itertools.count(1)
        self.nodes: Dict[int, PFNode] = {}
        self.cycles: Dict[int, SignedChain] = {}

        self._active_node_ids: set[int] = set()               #used for bookkeeping of active nodes in forest computation algorithm
        self.roots: set[int]  = set()                 #List of roots of trees in the forest

        self.levels: list[float] = []               #Critical filtration values, might give duplicates in current implementation

        start = time.perf_counter()
        self._alpha_complex = gd.AlphaComplex(points=point_cloud) # pyright: ignore[reportAttributeAccessIssue]
        self.simplex_tree = self._alpha_complex.create_simplex_tree()
        alpha_complex_time = time.perf_counter()-start
        if print_info:
            print(f"Alpha complex generated in {alpha_complex_time}")

        start = time.perf_counter()
        # Take square root of filtration value since Gudhi alpha-complex filtration values are squared.
        for simplex, filtration in self.simplex_tree.get_filtration():
            self.simplex_tree.assign_filtration(simplex, (filtration**0.5)*2)
        
        # Extract s filtration up to order d
        self.filtration =  [(simplex,filtration) for simplex, filtration in self.simplex_tree.get_filtration() if len(simplex) >= self.dim] #keep simplices up to codim 1
        filtration_time = time.perf_counter()-start
        if print_info:
            print(f"Filtration processed in {filtration_time}")

        self.barcode: set[PFBar] = set()

        self.landscape_families: Dict[str, Any] = {}
        self.barcode_functionals: Dict[str, Any] = {}

        if low_memory_mode:
            raise ValueError("This feature is a work in progress. We only story diff insteaf of full cycle reps which should drastically memory usage.")
        if low_memory_mode and not keep_simplex_diff:
            raise ValueError("low_memory_mode=True requires keep_simplex_diff=True")
        self.keep_simplex_diff = keep_simplex_diff
        self.low_memory_mode = low_memory_mode
        self.compute_interior = compute_interior

        self._compute_forest(print_info = print_info)

        self.reduced = reduce
        
        # Compute where each cycle is active.
        self._compute_loop_activity()

        if reduce:
            self._reduce_forest(print_info = print_info)

        if compute_barcode:
            if self.keep_simplex_diff:
                self.compute_barcode_diff(print_info=print_info)
            else:
                self.compute_barcode(print_info = print_info)

        if compute_interior and not keep_simplex_diff:
            raise ValueError("compute_interior=True requires keep_simplex_diff=True")

        if compute_interior:
            self._compute_interior_of_cycle_reps(print_info = print_info)


        return

    # ---------- builders ---------


    def add_leaf(self, simplex: List[int], filt_val: float, orientation: int):
        """
        Create a new leaf node for a full-dimensional simplex.

        In the reverse-filtration algorithm this corresponds to the death of a
        codimension-one homology class; the new cycle is the signed boundary of
        ``simplex``.

        Parameters
        ----------
        simplex : list[int]
            Vertices of the full-dimensional simplex being added. It has
            length ``self.dim + 1``.
        filt_val : float
            Filtration value where the simplex enters.
        orientation : int
            Orientation (+/-1) of the simplex in the complex.

        Returns
        -------
        PFNode
            Newly created leaf node.
        """

        nid = next(self._node_id)
        
        new_cycle = SignedChain(signed_simplices=signed_boundary(simplex=simplex,orientation=orientation))

        if self.keep_simplex_diff:
            new_node = PFNode(id=nid,
                              filt_val=filt_val, 
                              children=set(), 
                              cycle=new_cycle, 
                              _simplex_diff_available = True,
                              _interior_diff = {(key(simplex=simplex), orientation)},
                              _codim1_simplex_diff = None)
        else: 
            new_node = PFNode(id=nid,filt_val=filt_val, children=set(), cycle=new_cycle, _simplex_diff_available = False)

        self.nodes[nid]=new_node
        self._active_node_ids.add(nid)  # roots are active nodes; at termination they are the roots of the forest

        self.levels.append(filt_val)

        return new_node
    
    def make_root(self, node: PFNode, filt_val: float):
        """
        End a tree by creating a root parent for an active node.

        In the reverse-filtration algorithm this corresponds to the birth of a
        codimension-one homology class.

        Parameters
        ----------
        node : PFNode
            Child node that becomes the child of the new root.
        filt_val : float
            Filtration value at which the root is created.
        """
        nid = next(self._node_id)

        self._active_node_ids.remove(node.id)
        node.parent = nid #new root node is parent of input node

        if self.keep_simplex_diff:
            root_node = PFNode(id=nid, 
                               filt_val=filt_val, 
                               cycle=node.cycle, 
                               children={node.id},
                               _simplex_diff_available = True,
                               _interior_diff = None,
                               _codim1_simplex_diff = None)
        else:
            root_node = PFNode(id=nid, filt_val=filt_val, cycle=node.cycle, children={node.id}, _simplex_diff_available = False)


        self.nodes[nid]=root_node
        self.roots.add(root_node.id)

        self.levels.append(filt_val)

        return

    def merge_nodes(self, node1: PFNode, node2: PFNode, simplex: List[int], filt_val: float):
        """ 
        Create a parent node by merging two active child nodes.

        The parent cycle is obtained by unioning the child cycles and removing
        both orientations of ``simplex``. In the reverse-filtration algorithm,
        this records a split event and creates a new barcode branch.

        Parameters
        ----------
        node1, node2 : PFNode
            Child nodes that are merged.
        simplex : list[int]
            Codimension-one simplex at which the two children cycles merge.
        filt_val : float
            Filtration value of the merge event.
        """

        nid = next(self._node_id)
        node1.parent = nid
        node2.parent = nid

        parent_cycle = merge_at_simplex(cycle1 = node1.cycle, cycle2=node2.cycle,  simplex=simplex) 

        if self.keep_simplex_diff:
            parent_node = PFNode(id=nid, 
                                 filt_val=filt_val, 
                                 children={node1.id,node2.id}, 
                                 cycle=parent_cycle,
                                 _simplex_diff_available = True,
                                 _interior_diff = None,
                                 _codim1_simplex_diff = {(key(simplex),1),(key(simplex),-1)})
        else:
            parent_node = PFNode(id=nid, filt_val=filt_val, children={node1.id,node2.id}, cycle=parent_cycle, _simplex_diff_available = False)
        self.nodes[nid]=parent_node
        
        self._active_node_ids.add(nid)
        self._active_node_ids.remove(node1.id)
        self._active_node_ids.remove(node2.id)

        self.levels.append(filt_val)

        return
    
    def update_node(self, node: PFNode, simplex: List[int], filt_val:float):
        """
        Create a parent node for an active node whose cycle changes.

        The updated cycle is obtained by removing both orientations of
        ``simplex`` from ``node.cycle``.

        Parameters
        ----------
        node : PFNode
            Node whose representative is being updated.
        simplex : list[int]
            Codimension-one simplex to remove from the representative.
        filt_val : float
            Filtration value of the update event.
        """

        updated_cycle = node.cycle.cancel_simplex(simplex=simplex)

        nid = next(self._node_id)
        node.parent=nid

        if self.keep_simplex_diff:
            update_node = PFNode(id=nid, 
                                 filt_val=filt_val, 
                                 children={node.id},
                                 cycle = updated_cycle,
                                 _simplex_diff_available = True,
                                 _interior_diff = None,
                                 _codim1_simplex_diff = {(key(simplex),1),(key(simplex),-1)})
        else:
            update_node = PFNode(id=nid, filt_val=filt_val, children={node.id},cycle = updated_cycle, _simplex_diff_available = False)

        self.nodes[nid]=update_node

        self._active_node_ids.add(nid)
        self._active_node_ids.remove(node.id)

        self.levels.append(filt_val)

        return
    
    
    # ----- compute the forest ----------

    def _compute_forest(self, print_info: bool = False):
        """ 
        Compute the persistence forest from the alpha-complex filtration.

        Parameters
        ----------
        print_info : bool
            If True, print timing information.
        """

        loop_forest_start = time.perf_counter()

        face_cycle_dict = {}

        #simplices is already ordered in ascending order by number of simplices 
        for simplex, filt_val in reversed(self.filtration):
            
            if len(simplex) == self.dim+1:
                orientation = simplex_orientation(simplex=simplex, point_cloud=self.point_cloud)

                new_node = self.add_leaf( simplex=simplex, filt_val=filt_val, orientation = orientation)

                faces = list(itertools.combinations(simplex, len(simplex)-1 ) )
                for face in faces:
                    if key(face) in face_cycle_dict:
                       face_cycle_dict[key(face)].append(new_node.id)
                    else:
                        face_cycle_dict[key(face)] = [new_node.id]

            elif len(simplex) == self.dim:
                #L is nodes containing simplex, can be of the form [],[l1], [l1,l2], [l1,l1]
                #L is active nodes over L_tmp

                #If key exists, get its value and remove it
                #if key does not exists, get []
                L_tmp_ids = face_cycle_dict.pop(key(simplex), [])

                L = self._update_node_list(L_tmp_ids)


                #if no cycle contains simplex, nothing happens
                if len(L) == 0:
                    continue

                #if cycle is only contained in a single loop and appears only once in that loop once, remove that loop from the active loops 
                elif len(L) == 1:
                    # Update the face dictionary for all faces contained in the cycle we just removed.
                    for signed_simplex in L[0].cycle.signed_simplices:

                            simplex = signed_simplex[0]

                            L_simplex_tmp = face_cycle_dict.pop(key(simplex), None)
                            if L_simplex_tmp is None:
                                continue

                            L_simplex = self._update_node_list(L_simplex_tmp)

                            if len(L_simplex)> 2:
                                raise ValueError("L_edge too long in loop removal process")

                            if len(L_simplex)==1:
                                continue
                            elif L_simplex[0] != L[0]:
                                face_cycle_dict[key(simplex)] = [L_simplex[0].id]
                            elif L_simplex[1] != L[0]: 
                                face_cycle_dict[key(simplex)] = [L_simplex[1].id]
                            else:
                                continue
                                
                    self.make_root(node=L[0],filt_val=filt_val)

                    continue

                elif len(L) == 2 and L[0]!=L[1]: 
                    self.merge_nodes( node1=L[0], node2=L[1], simplex =simplex, filt_val=filt_val)
                    """if not loop_in_filtration_check(parent_loop.vertex_list, simplex_tree=self.simplex_tree, filt_value=filt_val):
                            print('edge', simplex)
                            print('edge dict entry')
                            print('filtration value', filt_val)
                            print(f'first loop', L[0].cycle)
                            print(f'second loop', L[1].cycle)
                            raise ValueError("Loop not in simplex, Loop concat Case")"""


                elif len(L) == 2 and L[0]==L[1]:
                    #Same simplex is contained in a cycle in both orientations -> we remove it from the the cycle
                    self.update_node(node=L[0], simplex=simplex, filt_val=filt_val)
                    """if not loop_in_filtration_check(vertex_loop, simplex_tree=self.simplex_tree, filt_value=filt_val):
                                print('edge', simplex)
                                print(f'starting loop', L[0].loop)
                                print(f'outer loop', vertex_loop)
                                raise ValueError("Loop not in simplex, Tiebreak Case")"""
                    
                else:
                    print(L)
                    print(simplex)
                    raise ValueError("Error, L is of the wrong form")

        loop_forest_time = time.perf_counter() - loop_forest_start
        if print_info:
            print(f"Forest succesfully computed in {loop_forest_time} sec")

        return


    # ----- methods to work with the forest

    def active_nodes_at(self, filt_val: float) -> List[PFNode]:
        """
        Return nodes active at a given filtration value.

        A non-root node is active at ``r`` if ``node.filt_val >= r`` and its
        parent has filtration value ``< r``.

        Parameters
        ----------
        filt_val : float
            Filtration value to query.

        Returns
        -------
        list[PFNode]
            Active nodes sorted by decreasing node filtration value and then id.
        """
        nodes = self.nodes

        active: List[PFNode] = []
        for n in nodes.values():
            if n.filt_val < filt_val:
                continue
            if n.parent == None:
            # all children must exist and be strictly above alpha
                continue
            else:
                parent = nodes[n.parent]
                if parent.filt_val >= filt_val:
                    continue

            active.append(n)
        
        # deterministic order: higher filt_val first, then id
        active.sort(key=lambda n: (-n.filt_val, n.id))
        return active

    def active_cycles_at(self, filt_val: float) -> List[SignedChain]:
        """Return cycle representatives of nodes active at ``filt_val``."""
        active_nodes = self.active_nodes_at(filt_val=filt_val)
        return [node.cycle for node in active_nodes]

    def leaves_below_node(self, node: PFNode) -> set[int]:
        """Return ids of all descendant leaves below ``node``."""
        leaf_ids: set[int] = set()

        if len(node.children) == 0:
            leaf_ids.add(node.id)
            return leaf_ids
        
        for cid in node.children:
            child = self.nodes[cid]
            leaf_ids.update(self.leaves_below_node(child))


        return leaf_ids

    def leaf_to_node_path(self, leaf: PFNode, node: PFNode) -> List[int]:
        """Return node ids on the path from ``leaf`` up to ancestor ``node``."""
        path = [leaf.id]

        active_node = leaf
        while active_node.parent != None:
            active_node = self.nodes[active_node.parent]
            path.append(active_node.id)
            if active_node.id == node.id:
                return path

        if active_node.id != node.id:
            raise ValueError(f"Node {node} is not above leaf {leaf}")

        return path

    def node_to_leaf_path(self, leaf: PFNode, node: PFNode) -> List[int]:
        """Return node ids on the path from ancestor ``node`` down to ``leaf``."""
        return list(reversed(self.leaf_to_node_path(leaf=leaf, node=node)))

    def get_root(self, node: PFNode) -> PFNode:
        """Return the highest ancestor of ``node`` with no parent."""
        while node.parent != None:
                    pid = node.parent
                    node = self.nodes[pid]

        return node

    def _update_node_list(self, node_id_list: List[int]) -> List[PFNode]:
        """Return current root ancestors for ``node_id_list``."""
        L = [ self.get_root( self.nodes[id] ) for id in node_id_list ]
        return L

    # ----- reduce forest (collapses trivial edges which happen at the same filtration value) -------------

    def _collapse_parent_child(self, parent: PFNode, child: PFNode):
        """
        Collapse a parent-child pair into the parent node.

        Intended for pairs whose filtration values agree up to
        ``self.filtration_tol``. The parent node is kept; the child is removed
        and its children are re-parented to ``parent``.
        """

        parent.children.remove(child.id)
        parent.children.update(child.children)

        #re-parent grandchildren
        for gcid in child.children:
            self.nodes[gcid].parent = parent.id

        #remove child node from forest
        del self.nodes[child.id]

        #update node simplex diffs
        if self.keep_simplex_diff:

            parent._interior_diff = union_optional_sets(parent._interior_diff, child._interior_diff)
            parent._codim1_simplex_diff = union_optional_sets(parent._codim1_simplex_diff, child._codim1_simplex_diff)

        #if parent is now isolated point in forest, delete it from forest completely
        if len(parent.children)==0 and parent.parent == None:
            del self.nodes[parent.id]
            self.roots.remove(parent.id)

        return

    def _reduce_forest(self, print_info: bool = False):
        """
        Reduce the forest by collapsing every parent-child pair whose
        filtration values differ by at most ``self.filtration_tol``.

        Collapse rule for an edge (parent p, child c) with
        p.filt_val <= c.filt_val <= p.filt_val + self.filtration_tol:
        - Keep the *parent* node p (its cycle stays as-is).
        - Remove the child node c from the forest.
        - The parent of the resulting (collapsed) node remains p.parent (i.e., the
            parent of the original parent), if any.
        - Children of the resulting node are the union of p.children and c.children,
            minus the removed child c itself.
        - For every grandchild g in c.children, set g.parent = p.id.
        Repeats until no collapsible edges remain.

        Parameters
        ----------
        print_info : bool
            If True, print timing information and node counts.
        """
        if print_info:
            print("Reducing the forest")
        reduction_start = time.perf_counter()

        collapses = 0

        if print_info:
            print(f"Number of nodes before reduction: {len(self.nodes.keys())}")

        #iterate over snapshot of the nodes in the tree, the nodes dict might be changed each iteration
        node_list = list(self.nodes.values())

        for p in node_list:

            # p might have been deleted as a child in a previous iteration of this outer loop
            if p.id not in self.nodes.keys():
                print("continue case")
                continue

            for cid in p.children.copy():
                child = self.nodes[cid]
                if child.filt_val < p.filt_val:
                    raise ValueError(
                        f"Invalid parent-child filtration order: "
                        f"parent {p.id} has filt_val={p.filt_val}, "
                        f"child {child.id} has filt_val={child.filt_val}"
                    )

                if child.filt_val <= p.filt_val + self.filtration_tol:
                    self._collapse_parent_child(parent=p, child=child)
                    collapses += 1

        reduction_time = time.perf_counter() - reduction_start
        if print_info:
            print(f"Reduction complete in {reduction_time} sec")
            print(f"Number of nodes after reduction: {len(self.nodes.keys())}")
        
        return

    # If we have multiple edges appearing at the same filtration value, we might get a root node which is also a merge in the reduction process
    # This will lead to a node which appears in the root list but has type merge
    # Not a mistake in the code, simply the way the edge case is currently handled
    # -> use node.parent == None to check if a node is a root
    # If a merge is also a root, then the merge should be split into 2 separate roots as the merge only lives for 0 time
    # This is not implemented yet and should not occur for points in general position

    # ------ Add active period of each cycle ----------

    def _node_activity(self, node):
        """Return ``(active_start, active_end)`` for a node's cycle."""
        if node.parent == None:
            return node.filt_val, node.filt_val

        parent = self.nodes[node.parent]
        active_start = parent.filt_val
        active_end = node.filt_val
        return active_start, active_end

    def _compute_loop_activity(self):
        """Annotate each non-root node cycle with its active interval."""

        for node in self.nodes.values():

            if node.parent == None:
                continue
            
            parent = self.nodes[node.parent]
            node.cycle.active_end = node.filt_val
            node.cycle.active_start = parent.filt_val

        return

    # ----- Compute barcode sequence ---------

    def compute_barcode(self, print_info: bool = False):
        """
        Compute the barcode from the forest structure.

        Parameters
        ----------
        print_info : bool
            If True, print timing information.

        Notes
        -----
        Iterates over leaves, walks to the root (or first covered node), and
        records the sequence of representatives. Bars are stored in
        ``self.barcode``.
        """
        
        if print_info:
            print("Computing Barcode")
        barcode_start = time.perf_counter()

        #dict should be ordered with filtration values decreasing since nodes are added in that order
        if not are_dict_keys_sorted(self.nodes):
            raise ValueError("Node dict keys are not sorted. This should not happen. Easy fix: sort keys in compute_barcode function (currently not implemented)")
    

        for id, node in self.nodes.items():
            #every barcode starts at leaf
            if len(node.children)>0:
                continue

            death = node.filt_val
            node_id_progession = [id]
            cycle_progression = [node.cycle]
            node._barcode_covered += 1
            is_max_tree_bar = True
            root_id = self.get_root(node).id

            if node.parent == None:
                raise ValueError("Leaf has no Parent, this should not happen")
            else:
                parent = self.nodes[node.parent]

            #walk up forest until a root or an already _barcode_covered node is discovered
            while parent.parent is not None:
                #check if parent node has already been covered by leaf with larger filtration value
                if parent._barcode_covered != 0:
                    is_max_tree_bar = False
                    break

                node_id_progession.append(parent.id)
                cycle_progression.append(parent.cycle)
                parent._barcode_covered += 1

                #move to parent of parent
                parent = self.nodes[parent.parent]

            birth = parent.filt_val


            #reverse lists to get progression which is ascending with respect to filtration value
            bar = PFBar(birth=birth,
                      death=death, 
                      _node_progression = tuple(reversed(node_id_progession)), 
                      cycle_reps=list(reversed(cycle_progression)), 
                      is_max_tree_bar=is_max_tree_bar,
                      root_id=root_id)
            self.barcode.add(bar)
 

        barcode_time = time.perf_counter() - barcode_start
        if print_info:
            print(f"Barcode computation completed in {barcode_time} sec")
    
        return
         
    def compute_barcode_diff(self, print_info: bool = False):
        """
        Compute the barcode from the forest structure using stored node diffs.

        Parameters
        ----------
        print_info : bool
            If True, print timing information.

        Notes
        -----
        Iterates over leaves in reverse node order, walks to the root or to a
        merge node with an undiscovered branch, and records the sequence of
        representatives. When ``keep_simplex_diff=True``, accumulated barcode
        diffs are stored on merge nodes for later reconstruction of cycle
        representatives with interiors. Bars are stored in ``self.barcode``.
        """
        
        if print_info:
            print("Computing Barcode")
        barcode_start = time.perf_counter()

        #dict should be ordered with filtration values decreasing since nodes are added in that order
        if not are_dict_keys_sorted(self.nodes):
            raise ValueError("Node dict keys are not sorted. This should not happen. Easy fix: sort keys in compute_barcode function (currently not implemented)")
    

        for id, node in reversed(self.nodes.items()):
            #every barcode starts at leaf
            if len(node.children)>0:
                continue

            death = node.filt_val
            node_id_progession = [id]
            cycle_progression = [node.cycle]
            node._barcode_covered += 1
            is_max_tree_bar = True
            root_id = self.get_root(node).id

            barcode_interior_diff = node._interior_diff
            barcode_codim1_simplex_diff = node._codim1_simplex_diff

            if node.parent == None:
                raise ValueError("Leaf has no Parent, this should not happen")
            else:
                parent = self.nodes[node.parent]

            #walk up forest until a root or a merge node with node._barcode_covered < number of children
            while parent.parent is not None:
                #check if parent node still has undiscovered leaves
                if parent._barcode_covered < len(parent.children) - 1:
                    parent._barcode_covered += 1
                    is_max_tree_bar = False

                    #Write down total bar diff in merge node
                    if self.keep_simplex_diff:
                        parent._barcode_interior_diff = union_optional_sets(parent._barcode_interior_diff,barcode_interior_diff)
                        parent._barcode_codim1_simplex_diff =  union_optional_sets(parent._barcode_codim1_simplex_diff, barcode_codim1_simplex_diff)
        
                    break

                if self.keep_simplex_diff:
                    barcode_interior_diff = union_optional_sets(barcode_interior_diff, parent._interior_diff)
                    barcode_codim1_simplex_diff = union_optional_sets(barcode_codim1_simplex_diff, parent._codim1_simplex_diff)

                    if parent._barcode_covered != 0:
                        barcode_interior_diff = union_optional_sets(barcode_interior_diff, parent._barcode_interior_diff)
                        barcode_codim1_simplex_diff = union_optional_sets(barcode_codim1_simplex_diff, parent._barcode_codim1_simplex_diff)

                node_id_progession.append(parent.id)
                cycle_progression.append(parent.cycle)
                parent._barcode_covered += 1

                #move to parent of parent
                parent = self.nodes[parent.parent]

            birth = parent.filt_val


            #reverse lists to get progression which is ascending with respect to filtration value
            bar = PFBar(birth=birth,
                      death=death, 
                      _node_progression = tuple(reversed(node_id_progession)), 
                      cycle_reps=list(reversed(cycle_progression)), 
                      is_max_tree_bar=is_max_tree_bar,
                      root_id=root_id)
            self.barcode.add(bar)
 

        barcode_time = time.perf_counter() - barcode_start
        if print_info:
            print(f"Barcode computation completed in {barcode_time} sec")
    
        return

    def _cycle_reps_from_node_diff(self, bar: PFBar) -> list[SignedChain]:
        """
        Reconstruct cycle representatives for a bar from node diffs.

        Parameters
        ----------
        bar : PFBar
            Bar whose node progression should be reconstructed.

        Returns
        -------
        list[SignedChain]
            Cycle representatives ordered by increasing filtration value. Each
            returned chain has ``interior_available=True`` and an ``interior``
            set containing accumulated full-dimensional simplices.

        Raises
        ------
        ValueError
            If any node in the progression does not have simplex diffs.
        """

        interior = set()
        simplices = set()
        cycle_reps = []

        for nid in reversed(bar._node_progression):
            node = self.nodes[nid]

            active_start, active_end = self._node_activity(node)

            if not node._simplex_diff_available:
                raise ValueError("set PersistenceForest(..., keep_simplex_diff=True)")
            simplices = update_chain_with_diff(signed_simplices=simplices, interior_diff=node._barcode_interior_diff, codim1_simplex_diff=node._barcode_codim1_simplex_diff)
            simplices = update_chain_with_diff(signed_simplices=simplices, interior_diff=node._interior_diff, codim1_simplex_diff=node._codim1_simplex_diff)

            if node._interior_diff is not None:
                interior.update(node._interior_diff)
            if node._barcode_interior_diff is not None:
                interior.update(node._barcode_interior_diff)

            signed_chain = SignedChain(
                signed_simplices=simplices.copy(),
                active_start=active_start,
                active_end=active_end,
                interior_available=True,
                interior=interior.copy()
            )
            cycle_reps.append(signed_chain)


        return list(reversed(cycle_reps))

    def _compute_interior_of_cycle_reps(self,print_info: bool =False):
        """
        Replace barcode cycle representatives by diff-reconstructed chains.

        Parameters
        ----------
        print_info : bool
            If True, print timing information.

        Notes
        -----
        Requires the forest to have been built with ``keep_simplex_diff=True``.
        The reconstructed representatives carry accumulated interior simplex
        data used by ``interior_simplex_activity``.
        """
        if print_info:
            print("Computing interior of cycle reps")
        interior_start = time.perf_counter()

        for bar in self.barcode:
            bar.cycle_reps = self._cycle_reps_from_node_diff(bar)
        
        interior_end = time.perf_counter()-interior_start
        if print_info:
            print(f"interior of cycle reps computed in {interior_end}")

        return

    def interior_simplex_activity(self) -> dict[tuple[int, ...], list[tuple[PFBar, float, float]]]:
        """
        Return active interior intervals for full-dimensional simplices.

        The return value maps each unsigned full-dimensional simplex to a list
        of ``(bar, active_start, active_end)`` tuples. The interval is the
        half-open filtration range where the simplex is in the interior of the
        active cycle representative for that bar.

        Returns
        -------
        dict[tuple[int, ...], list[tuple[PFBar, float, float]]]
            Mapping from simplex keys to bars and active intervals.

        Notes
        -----
        Call this only after interiors have been computed, for example by
        constructing with ``compute_interior=True`` and
        ``keep_simplex_diff=True``.
        """
        if not self.keep_simplex_diff and self.compute_interior:
            raise ValueError("Set PersistenceForest(..., keep_simplex_diff = True, compute_interior = True)")

        activity = defaultdict(list)

        for bar in self.barcode:
            first_rep = bar.cycle_reps[0]
            first_simplex_keys = {key(simplex) for simplex, _orientation in first_rep.interior}
            last_active_end_by_simplex = {}

            for cycle_rep in bar.cycle_reps:
                cycle_rep_simplex_keys = {key(simplex) for simplex, _orientation in cycle_rep.interior}

                for simplex_key in first_simplex_keys:
                    if simplex_key in cycle_rep_simplex_keys:
                        last_active_end_by_simplex[simplex_key] = cycle_rep.active_end

            for simplex_key in first_simplex_keys:
                activity[simplex_key].append(
                    (bar, first_rep.active_start, last_active_end_by_simplex[simplex_key])
                )

        return dict(activity)

    # ----- Useful barcode functions --------

    def max_bar(self)-> PFBar:
        """Return the bar with the longest lifespan."""
        return max(self.barcode, key=lambda bar: bar.lifespan())
    
    def longest_bars(self,k:int) -> List[PFBar]:
        """Return the ``k`` longest bars in descending lifespan order."""
        return sorted(self.barcode, key=lambda bar: bar.lifespan(), reverse=True)[0:k]
    
    def active_bars_at(self, filt_val:float):
        """Return bars whose lifespan contains ``filt_val``."""
        return [bar for bar in self.barcode if (bar.birth<=filt_val and bar.death>filt_val)]

    # ------ extract cycle representatives ---------

    def cycle_reps_at(self, filt_val: float, min_bar_length:float = 0) -> List[SignedChain]:
        """Return cycle representatives active at ``filt_val``."""
        active_bars = self.active_bars_at(filt_val=filt_val)
        cycles = [bar.cycle_at_filtration_value(filt_val=filt_val) for bar in active_bars if bar.lifespan()>=min_bar_length]
        return cycles

    def _active_bars_with_cycles_at(
        self,
        filt_val: float,
        min_bar_length: float = 0.0,
    ) -> List[Tuple["PFBar", SignedChain]]:
        """Return active bars paired with their cycles at ``filt_val``."""
        active = []
        for bar in self.barcode:
            if bar.lifespan() < min_bar_length:
                continue
            if bar.birth <= filt_val < bar.death:
                active.append((bar, bar.cycle_at_filtration_value(filt_val=filt_val)))
        return active

    def barcode_cycle_reps(self, relative_position=0.1, min_bar_length: float = 0) -> List[SignedChain]:
        """
        Return the list of cycle representatives for all bars in the barcode.

        Parameters
        ----------
        relative_position : float
            Relative position in the barcode (between 0 and 1).
        min_bar_length : float
            Minimum lifespan of bars to consider.

        Returns
        -------
        list[SignedChain]
            Cycle representatives for all bars in the barcode.
        """
        if relative_position < 0 or relative_position > 1:
            raise ValueError("relative_position must be in [0,1]")

        # Get all bars in the barcode
        all_bars = sorted(list(self.barcode), key=lambda bar: bar.lifespan(), reverse=True)
        
        # Select bars based on relative position and minimum bar length
        selected_bars = [bar for bar in all_bars if bar.lifespan() >= min_bar_length]
        
        # Compute cycle representatives for each selected bar
        cycles = [bar.cycle_at_filtration_value(filt_val=bar.birth + (bar.lifespan() * relative_position)) for bar in selected_bars]
        
        return cycles

    # ------ generate color scheme  ---------

    def _build_color_map_forest(self, seed: Optional[int] = 39, start_color: Optional[str] = "#ff7f0e",):
        """
        Computes a color map which assign a color to each bar in the barcode. 
        Bars in same tree will have similiar colors. 
        Saved as a dictionary {bar: "#RRGGBB"} in self.color_map_forest
        Based on json

        Parameters
        ----------
        seed : int | None
            Random seed to make the palette repeatable.
        start_color : str | None
            Preferred color to start the palette with (hex).
        """

        from .color_scheme import color_map_for_bars

        ordered_bars = sorted(list(self.barcode), key= lambda bar: bar.lifespan(), reverse=True)

        self.color_map_forest = color_map_for_bars(
            ordered_bars,
            seed =seed,
            by_id=False,
            prefer_start=start_color
        )

        return

    def _build_color_map_bars(self):
        """
        Computes a color map which assign a color to each bar in the barcode. 
        Ignores tree structure and cycles through 20 colors from longest to shortest bar 
        Saved as a dictionary {bar: "#RRGGBB"} in self.color_map_bars
        """
        bars_sorted = sorted(list(self.barcode), key = lambda bar:bar.lifespan(), reverse=True )
        colors = sns.color_palette("tab20", len(bars_sorted))

        self.color_map_bars = {bars_sorted[i]: colors[i] for i in range(len(bars_sorted))}
        self.set_longest_bar_colors(coloring = "bars", colors = [   "#1f78b4",   "#e31a1c",  "#33a02c",  "#ff7f00",  "#6a3d9a",  "#e7298a",  "#17becf",   "#bcbd22",   "#b15928",  "#fdb462",  "#7570b3", "#66a61e"]
)
        return

    def _get_color_map(self, coloring: Literal["forest", "bars"] = "forest"):
        """Return the requested bar color map, building it on first use."""
        if coloring == "forest":
            if not hasattr(self, "color_map_forest"):
                self._build_color_map_forest()
            return self.color_map_forest
        elif coloring == "bars":
            if not hasattr(self, "color_map_bars"):
                self._build_color_map_bars()
            return self.color_map_bars
        else:
            raise ValueError("coloring must be 'forest' or 'bars'")

    def set_longest_bar_colors(
        self,
        colors: Sequence[Any],
        coloring: Literal["forest", "bars"] = "forest",
    ) -> None:
        """Overwrite colors for the longest bars in the selected color map."""
        from .color_scheme import _to_hex

        color_map = self._get_color_map(coloring=coloring)
        bars_sorted = sorted(list(self.barcode), key=lambda bar: bar.lifespan(), reverse=True)

        for bar, color in zip(bars_sorted, colors):
            color_map[bar] = _to_hex(color)

        return

    # ----- filtration plotting helpers -------

    def _simplices_present_at_filtration(self, filt_val: float) -> Dict[str, Any]:
        """
        Return simplices present at a filtration value.

        For ambient dimension 2:
            returns edges, triangles
        For ambient dimension 3:
            returns edges, triangles, tetrahedra

        Parameters
        ----------
        filt_val : float
            Filtration threshold.

        Returns
        -------
        dict
            Dictionary containing present simplices.
        """
        if self.dim not in (2, 3):
            raise ValueError("Plotting is only implemented for ambient dimensions 2 and 3.")

        out: Dict[str, Any] = {
            "edges": [],
            "triangles": [],
            "tetrahedra": [],
        }

        for simplex, f in self.filtration:
            if f > filt_val:
                break

            simplex_t = tuple(simplex)

            if self.dim == 2:
                if len(simplex_t) == 2:
                    out["edges"].append(simplex_t)
                elif len(simplex_t) == 3:
                    out["triangles"].append(simplex_t)

            elif self.dim == 3:
                if len(simplex_t) == 2:
                    out["edges"].append(simplex_t)
                elif len(simplex_t) == 3:
                    out["triangles"].append(simplex_t)
                elif len(simplex_t) == 4:
                    out["tetrahedra"].append(simplex_t)

        return out
    
    def _boundary_triangles_from_tetrahedra(
        self,
        tetrahedra: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int, int]]:
        """
        Compute the boundary triangles of a tetrahedral complex.

        A triangular face is on the boundary iff it appears in exactly one
        tetrahedron.

        Parameters
        ----------
        tetrahedra : list[tuple[int, int, int, int]]
            Present tetrahedra.

        Returns
        -------
        list[tuple[int, int, int]]
            Boundary triangular faces with canonical vertex order.
        """
        face_count: Dict[Tuple[int, int, int], int] = defaultdict(int)

        for tet in tetrahedra:
            for face in itertools.combinations(tet, 3):
                face_count[key(face)] += 1 # type: ignore

        boundary_faces = [face for face, c in face_count.items() if c == 1]
        return boundary_faces

    def _complex_snapshot_at_filtration(self, filt_val: float) -> Dict[str, Any]:
        """
        Build a backend-agnostic snapshot of the complex and active cycles at
        a given filtration value.

        Parameters
        ----------
        filt_val : float
            Filtration threshold.

        Returns
        -------
        dict
            Snapshot containing points and present simplices.
        """
        pts = np.asarray(self.point_cloud, dtype=float)
        simplices = self._simplices_present_at_filtration(filt_val)

        snapshot: Dict[str, Any] = {
            "dimension": self.dim,
            "filt_val": float(filt_val),
            "points": pts,
        }

        if self.dim == 2:
            snapshot["edges"] = simplices["edges"]
            snapshot["triangles"] = simplices["triangles"]

        elif self.dim == 3:
            tetrahedra = simplices["tetrahedra"]
            standalone_triangles = simplices["triangles"]
            edges = simplices["edges"]

            boundary_from_tets = self._boundary_triangles_from_tetrahedra(tetrahedra)

            tet_face_keys = {tuple(sorted(face)) for face in boundary_from_tets}
            standalone_keys = {tuple(sorted(face)) for face in standalone_triangles}

            # Include all boundary faces from tetrahedra, plus triangles not covered by them.
            boundary_faces = list(tet_face_keys | standalone_keys)

            snapshot["triangles"] = boundary_faces
            snapshot["edges"] = edges
            snapshot["tetrahedra"] = tetrahedra
            snapshot["standalone_triangles"] = standalone_triangles

        return snapshot

    def _chain_segments_2d(
        self,
        chain: SignedChain,
        signed: bool = False,
    ) -> List[np.ndarray]:
        """
        Convert a 1-chain to oriented segments in R^2.

        Parameters
        ----------
        chain : SignedChain
            Cycle representative consisting of oriented edges.
        signed : bool
            If False, cancel opposite-oriented duplicate edges first.

        Returns
        -------
        list[np.ndarray]
            List of 2x2 segment arrays.
        """
        if chain.dim() != 1:
            raise ValueError(f"_chain_segments_2d expected a 1-chain, got dim={chain.dim()}")

        if not signed:
            chain = chain.unsigned()

        return chain.segments(point_cloud=self.point_cloud)

    def _chain_triangles_3d(
        self,
        chain: SignedChain,
        signed: bool = False,
    ) -> List[Tuple[int, int, int]]:
        """
        Convert a 2-chain to a list of triangular faces in R^3.

        Orientation is currently ignored for rendering; the triangles are
        returned in their stored vertex order.

        Parameters
        ----------
        chain : SignedChain
            Cycle representative consisting of oriented triangles.
        signed : bool
            If False, cancel opposite-oriented duplicate triangles first.

        Returns
        -------
        list[tuple[int, int, int]]
            Triangle faces.
        """
        if chain.dim() != 2:
            raise ValueError(f"_chain_triangles_3d expected a 2-chain, got dim={chain.dim()}")
        
        if not signed:
            chain = chain.unsigned()

        return [tuple(simplex) for simplex, _orientation in chain.signed_simplices]

    @staticmethod
    def _require_plotly():
        """Import Plotly graph objects or raise an installation hint."""
        try:
            import plotly.graph_objects as go
        except ImportError as e:
            raise RuntimeError(
                "Plotly support requires the optional dependency 'plotly'. "
                "Install it with `pip install \".[plotly]\"` (local repo) or "
                "`pip install loopforest[plotly]`."
            ) from e
        return go

    def _empty_scatter_2d(self, name: str, showlegend: bool = False):
        """Return an empty 2D Plotly scatter trace."""
        go = self._require_plotly()
        return go.Scatter(
            x=[np.nan],
            y=[np.nan],
            mode="lines",
            name=name,
            showlegend=showlegend,
            hoverinfo="skip",
        )

    def _empty_cycle_trace_2d(self, name: str = "cycle", showlegend: bool = False):
        """Return an empty 2D Plotly cycle trace."""
        go = self._require_plotly()
        return go.Scatter(
            x=[np.nan],
            y=[np.nan],
            mode="lines",
            line=dict(width=3),
            name=name,
            showlegend=showlegend,
            hoverinfo="skip",
            visible=True,
        )

    # ----- filtration plotting tools -------

    def plot_at_filtration(
        self,
        filt_val: float,
        ax=None,
        signed: bool = False,
        show_cycles: bool = True,
        show: bool = True,
        show_complex: bool = True,
        figsize: tuple[float, float] = (5, 5), 
        vertex_size: float = 7,
        coloring: Literal['forest','bars'] = "forest",
        title: Optional[str] = None,
        min_bar_length: float = 0,
        point_zorder: float = 6,
        cycle_zorder: float = 5,
        dpi: int = 300,
        style_2d: Optional[dict[str, Any]] = None,
        style_3d: Optional[dict[str, Any]] = None,
    ):
        """
        Plot the simplicial filtration at a fixed filtration value.

        Parameters
        ----------
        filt_val : float
            Filtration threshold.
        ax : matplotlib.axes.Axes or None
            Axes to draw on; if None, a new figure+axes are created.
        signed : bool
            Orientation policy for cycle chains in both 2D and 3D:
            - False: cancel opposite-oriented duplicates.
            - True: preserve orientation duplicates.
        show_cycles : bool
            If True, overlay the active cycles at the filtration value.
        show : bool
            If True, calls plt.show() when done.
        show_complex : bool
            Whether to render complex geometry (edges/surfaces).
        figsize : tuple[float, float]
            Figure size used when ``ax`` is None.
        vertex_size : float
            Marker size for point cloud.
        point_zorder : float
            Matplotlib z-order for point markers in 2D plots.
            For example, use ``point_zorder=4, cycle_zorder=6`` to draw
            cycles above points.
        cycle_zorder : float
            Matplotlib z-order for cycle edges in 2D plots.
        coloring : {"forest","bars"}
            Color scheme; builds the map on first use.
        title : str | None
            Title for the axes. Defaults to a filtration summary.
        min_bar_length : float
            Only draw active cycles for bars with lifespan at least this value.
        style_2d : dict | None
            Optional 2D style overrides. Supported keys include:
            ``show_orientation_arrows``,
            ``point_color``, ``point_alpha``, ``complex_face_color``,
            ``complex_face_alpha``, ``complex_edge_color``,
            ``complex_edge_width``, ``cycle_edge_width``,
            ``arrow_linewidth``, ``arrow_scale``.
        style_3d : dict | None
            Optional 3D style overrides. Supported keys include:
            ``camera_eye``, ``remove_axes``,
            ``point_color``, ``point_alpha``, ``depthshade_points``,
            ``complex_color``, ``complex_face_alpha``, ``cycle_face_alpha``,
            ``complex_edge_color``, ``cycle_edge_color``,
            ``complex_edge_width``, ``cycle_edge_width``,
            ``complex_edge_alpha``, ``cycle_edge_alpha``,
            ``antialiased``, ``zsort``, ``desaturate_complex``.
        dpi : int
            DPI used when a new Matplotlib figure is created.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from .simplicial_filtration_plotting import _plot_at_filtration_generic
        return _plot_at_filtration_generic(
            self,
            filt_val=filt_val,
            ax=ax,
            show=show,
            show_complex=show_complex,
            figsize=figsize,
            vertex_size=vertex_size,
            coloring=coloring,
            title=title,
            show_cycles=show_cycles,
            signed=signed,
            min_bar_length=min_bar_length,
            point_zorder=point_zorder,
            cycle_zorder=cycle_zorder,
            dpi=dpi,
            style_2d=style_2d,
            style_3d=style_3d,
        )

    def plot_at_filtration_with_dual( 
        self,
        filt_val: float,
        ax=None,
        fill_triangles: bool = True,
        figsize: tuple[float, float] = (7, 7),
        vertex_size: float = 12,
        coloring: Literal['forest','bars'] = "forest",
        dual_vertex_size: float = 12,
        show_cycles: bool = True,
        linewidth_filt: float = 0.6,
        linewidth_cycle: float = 1.8,
        linewidth_dual_edge: float = 0.4,
        show_dual: bool = True,
        show: bool = True,
    ):
        """
        Plot primal and dual edges of the alpha complex at a filtration value.

        Parameters
        ----------
        filt_val : float
            Filtration threshold for displaying simplices.
        ax : matplotlib.axes.Axes or None
            Axes to draw on; a new figure is created if None.
        fill_triangles : bool
            If True, fill triangles that are present at or before filt_val.
        figsize : tuple[float, float]
            Size of the figure if ax is None.
        vertex_size : float
            Marker size for input points.
        coloring : {"forest","bars"}
            Color scheme to use for active cycles.
        dual_vertex_size : float
            Marker size for dual vertices.
        show_cycles : bool
            If True, overlay the active cycles at the filtration value.
        linewidth_filt : float
            Line width for primal edges.
        linewidth_cycle : float
            Line width for cycle edges.
        linewidth_dual_edge : float
            Line width for dual edges.
        show_dual: bool
            If True, overlay the dual edges and dual vertices.  
        show : bool
            If True, calls plt.show() when done.

        Returns
        -------
        matplotlib.axes.Axes
        """
        
        color_map = self._get_color_map(coloring=coloring)

        pts = np.asarray(self.point_cloud, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("point_cloud must be an (n_points, 2) array-like.")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        edge_info: dict[Tuple[int, int], tuple[np.ndarray, float]] = {}
        triangle_info: dict[Tuple[int, int, int], tuple[np.ndarray, np.ndarray, float]] = {}
        edge_to_triangles: dict[Tuple[int, int], list[Tuple[int, int, int]]] = defaultdict(list)
        vertex_to_dual_all: dict[int, list[tuple[np.ndarray, float]]] = {i: [] for i in range(len(pts))}

        for simplex, filtration in self.filtration:
            if len(simplex) == 2:
                i, j = simplex
                edge_key = tuple(sorted((i, j)))
                segment = np.array([pts[i], pts[j]])
                edge_info[edge_key] = (segment, filtration)
            elif len(simplex) == 3:
                i, j, k = simplex
                tri_key = tuple(sorted((i, j, k)))
                coords = np.array([pts[i], pts[j], pts[k]])
                barycenter = coords.mean(axis=0)
                triangle_info[tri_key] = (coords, barycenter, filtration)
                for v in tri_key:
                    vertex_to_dual_all[v].append((barycenter, filtration))
                for edge in itertools.combinations(tri_key, 2):
                    edge_key = tuple(sorted(edge))
                    edge_to_triangles[edge_key].append(tri_key)

        edges_present: list[np.ndarray] = []
        edges_future: list[np.ndarray] = []
        for edge_key, (segment, filtration) in edge_info.items():
            if filt_val >= filtration:
                edges_present.append(segment)
            else:
                edges_future.append(segment)

        tris_present: list[np.ndarray] = []
        tris_pending: list[np.ndarray] = []
        dual_vertices_present: list[np.ndarray] = []
        dual_vertices_future: list[np.ndarray] = []
        for coords, barycenter, filtration in triangle_info.values():
            if filtration <= filt_val:
                tris_present.append(coords)
                dual_vertices_future.append(barycenter)
            else:
                tris_pending.append(coords)
                dual_vertices_present.append(barycenter)

        dual_edges_present: list[np.ndarray] = []
        dual_edges_future: list[np.ndarray] = []
        for edge_key, tri_keys in edge_to_triangles.items():
            if len(tri_keys) != 2:
                continue  # boundary edge → no dual edge
            edge_data = edge_info.get(edge_key)
            if edge_data is None:
                continue
            edge_filtration = edge_data[1]
            tri1 = triangle_info.get(tri_keys[0])
            tri2 = triangle_info.get(tri_keys[1])
            if tri1 is None or tri2 is None:
                continue
            bary1 = tri1[1]
            bary2 = tri2[1]
            segment = np.array([bary1, bary2])
            if filt_val < edge_filtration:
                dual_edges_present.append(segment)
            else:
                dual_edges_future.append(segment)

        ax.scatter(
            pts[:, 0], 
            pts[:, 1],
            s=vertex_size, 
            color="k", 
            label="points",
            marker="o",
            edgecolors="none",
            zorder=2.8
        )

        if fill_triangles and tris_present:
            tri_coll = PolyCollection(
                tris_present,
                closed=True,
                edgecolors="none",
                facecolors="C0",
                alpha=0.2,
                zorder=1,
            )
            ax.add_collection(tri_coll)

        # if dual_vertices_pending:
        #     pending_arr = np.array(dual_vertices_pending)
        #     ax.scatter(
        #         pending_arr[:, 0],
        #         pending_arr[:, 1],
        #         s=dual_vertex_size,
        #         c="C3",
        #         alpha=0.6,
        #         marker="^",
        #         edgecolors="white",
        #         linewidths=0.4,
        #         zorder=3.2,
        #         label="dual vertices (pending)",
        #     )

        if dual_vertices_present and show_dual:
            present_arr = np.array(dual_vertices_present)
            ax.scatter(
                present_arr[:, 0],
                present_arr[:, 1],
                s=dual_vertex_size,
                c="C3",
                marker="o",
                edgecolors="none",
                zorder=2.8,
            )

        if edges_future:
            future_edge_coll = LineCollection(
                edges_future,
                linewidths=linewidth_filt,
                colors="0.45",
                alpha=0.5,
                zorder=1.6,
            )
            ax.add_collection(future_edge_coll)

        if edges_present:
            edge_coll = LineCollection(
                edges_present,
                linewidths=linewidth_filt,
                colors="0.3",
                zorder=2,
                label="edges",
            )
            ax.add_collection(edge_coll)

        if show_dual:
            if dual_edges_future:
                dual_thin_coll = LineCollection(
                    dual_edges_future,
                    colors="C3",
                    linewidths=linewidth_dual_edge,
                    alpha=0.5,
                    linestyle="dotted",
                    zorder=3.6,
                )
                ax.add_collection(dual_thin_coll)

            if dual_edges_present:
                dual_thick_coll = LineCollection(
                    dual_edges_present,
                    colors="C3",
                    linewidths=linewidth_dual_edge,
                    alpha=1,
                    zorder=4,
                    label="dual edges",
                )
                ax.add_collection(dual_thick_coll)

        if show_cycles:
            for bar in self.barcode:
                if filt_val >= bar.birth and filt_val < bar.death:
                    cycle = bar.cycle_at_filtration_value(filt_val=filt_val)
                    segments = [np.array(pts[list(signed_simplex[0])]) for signed_simplex in cycle.signed_simplices]    
                    loop_coll = LineCollection(segments, linewidths=linewidth_cycle, colors=[color_map[bar]], zorder=5)
                    ax.add_collection(loop_coll)

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(fr"$\alpha$ = {filt_val:.4g}")
        # handles, labels = ax.get_legend_handles_labels()
        # if handles:
        #     ax.legend(loc="lower right", frameon=True)

        # ax.autoscale()

        if show:
            plt.show()

        return ax

    def plot_barcode_cycle_reps(
        self,
        min_bar_length: float = 0,
        relative_position: float = 0.1,
        ax=None,
        show: bool = True,
        figsize: tuple[float, float] = (7, 7), 
        vertex_size: float = 3,
        coloring: Literal['forest','bars'] = "forest",
        title: Optional[str] = None,
        show_orientation_arrows: bool = False,
        remove_double_edges: bool = False,
        linewidth_cycle: float = 0.8,
    ):
        """
        Plot one representative cycle for each sufficiently long barcode bar.

        The representative for each selected bar is sampled at
        ``birth + lifespan * relative_position``. This method is implemented
        only for 2D point clouds.

        Parameters
        ----------
        min_bar_length : float
            Minimum lifespan of bars to consider.
        relative_position : float
            Relative position in each bar, between 0 and 1.
        ax : matplotlib.axes.Axes or None
            Axes to draw on; if None, a new figure+axes are created.
        show : bool
            If True, calls plt.show() when done.
        figsize : tuple[float, float]
            Figure size used when ``ax`` is None.
        vertex_size : float
            Marker size for point cloud.
        coloring : {"forest","bars"}
            Color scheme; builds the map on first use.
        title : str | None
            Title for the axes. Defaults to a relative-position summary.
        show_orientation_arrows : bool
            If True, draw small arrows along each cycle edge to indicate
            the orientation of the cycle representatives.
        remove_double_edges : bool
            If True, cancel edges appearing with opposite orientations before
            plotting.
        linewidth_cycle : float
            Line width for cycle edges.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if relative_position < 0 or relative_position > 1:
            raise ValueError("relative_position must be in [0,1]")

        if self.dim != 2:
            raise ValueError("plot_barcode_cycle_reps only implemented for dimension 2")
        
        color_map = self._get_color_map(coloring=coloring)

        # --- Prep
        pts = np.asarray(self.point_cloud, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("point_cloud must be an (n_points, 2) array-like.")

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)


        # --- Base scatter
        ax.scatter(pts[:, 0], pts[:, 1], s=vertex_size, color="k", zorder=3, label="points")

        # --- Overlay cycles from barcode representatives
        for bar in self.barcode:
            if bar.lifespan()>=min_bar_length:

                cycle = bar.cycle_at_filtration_value(filt_val=bar.birth + bar.lifespan()*relative_position)    

                # >>> make a Sequence[ArrayLike] (list of 2x2 arrays) for Pylance
                segments = self._chain_segments_2d(chain=cycle,signed=(not remove_double_edges))

                # Thicker colored edges along the cycle
                loop_coll = LineCollection(segments, linewidths=linewidth_cycle, colors=[color_map[bar]], zorder=5)
                ax.add_collection(loop_coll)

                # Optional arrows to show cycle edge orientation
                if show_orientation_arrows:
                    for seg in segments:
                        # seg is a 2x2 array: [start, end]
                        (x0, y0), (x1, y1) = np.asarray(seg, dtype=float)

                        dx = x1 - x0
                        dy = y1 - y0
                        length = float(np.hypot(dx, dy))
                        if length == 0.0:
                            continue  # skip degenerate segments

                        # Place arrow around the middle of the segment, slightly shortened
                        frac = 0.5  # fraction of the segment length used for the arrow body
                        mx = 0.5 * (x0 + x1)
                        my = 0.5 * (y0 + y1)

                        # Direction unit vector
                        ux = dx / length
                        uy = dy / length

                        half = 0.5 * frac * length
                        x_start = mx - ux * half
                        y_start = my - uy * half
                        x_end   = mx + ux * half
                        y_end   = my + uy * half

                        ax.annotate(
                            "",
                            xy=(x_end, y_end),
                            xytext=(x_start, y_start),
                            arrowprops=dict(
                                arrowstyle="-|>",
                                linewidth=.2,
                                color=color_map[bar],
                                mutation_scale=6
                            ),
                            zorder=6,
                    )

        # --- Aesthetics
        ax.set_aspect("equal", adjustable="box")
        if title is None:
            ax.set_title(fr"Barcode cycle representatives at relative position {relative_position:.2f}")
        else:
            ax.set_title(title)
        #ax.set_xlabel("x")
        #ax.set_ylabel("y")
        # A simple legend (points + edges); cycle colors are self-explanatory on top
        #handles, labels = ax.get_legend_handles_labels()
        #if handles:
        #    ax.legend(loc="lower right", frameon=True)

        ax.autoscale()  # fit collections
        if show:
            plt.show()
        return ax

    # ----- plotly snapshots --------

    def _plotly_traces_for_snapshot_2d(
        self,
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
        go = self._require_plotly()
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
            active = self._active_bars_with_cycles_at(
                filt_val=filt_val,
                min_bar_length=min_bar_length,
            )
            active = sorted(active, key=lambda bc: bc[0].lifespan(), reverse=True)


        if max_cycle_slots is None:
            max_cycle_slots = len(active)

        for i in range(max_cycle_slots):
            if i < len(active):
                bar, cycle = active[i]
                segments = self._chain_segments_2d(
                    cycle,
                    signed=signed,
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
        self,
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
        go = self._require_plotly()
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
            active = self._active_bars_with_cycles_at(
                filt_val=filt_val,
                min_bar_length=min_bar_length,
            )
            active = sorted(active, key=lambda bc: bc[0].lifespan(), reverse=True)

        if max_cycle_slots is None:
            max_cycle_slots = len(active)

        for idx in range(max_cycle_slots):
            if idx < len(active):
                bar, cycle = active[idx]
                tri_faces = self._chain_triangles_3d(cycle, signed=signed)

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
        self,
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
            2D cycle representatives. In the current 3D Plotly single-frame
            path, this argument is not forwarded and cycles are rendered with
            the 3D helper default.
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
        go = self._require_plotly()
        if self.dim not in (2, 3):
            raise ValueError("plot_at_filtration_plotly is only implemented for dimensions 2 and 3.")

        color_map = self._get_color_map(coloring=coloring)
        snapshot = self._complex_snapshot_at_filtration(filt_val=filt_val)

        if self.dim == 2:
            traces = self._plotly_traces_for_snapshot_2d(
                snapshot=snapshot,
                color_map=color_map,
                show_cycles=show_cycles,
                fill_triangles=fill_triangles,
                signed=signed,
                min_bar_length=min_bar_length,
                vertex_size = vertex_size
            )

            fig = go.Figure(data=traces)
            fig.update_layout(
                title=f"Filtration value r = {filt_val:.4g}",
                xaxis_title="x",
                yaxis_title="y",
                template="plotly_white",
                hovermode="closest",
            )
            fig.update_yaxes(scaleanchor="x", scaleratio=1)

        else:
            traces = self._plotly_traces_for_snapshot_3d(
                snapshot=snapshot,
                color_map=color_map,
                show_cycles=show_cycles,
                show_complex=show_complex,
                min_bar_length=min_bar_length,
                complex_opacity=complex_opacity,
                cycle_opacity=cycle_opacity,
                vertex_size = vertex_size
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
        self,
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
        coloring : {"forest", "bars"}
            Which bar color map to use.
        show_cycles : bool
            If True, overlay active cycle representatives.
        signed : bool
            If False, removes simplices which appear with both orientations.
        filt_max : float | None
            Maximum filtration value for the slider. If None, uses
            ``1.03 * max(bar.death for bar in self.barcode)``.
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
        go = self._require_plotly()
        if self.dim not in (2, 3):
            raise ValueError("plot_filtration_interactive is only implemented for dimensions 2 and 3.")

        if filt_max is None:
            filt_max = max( [bar.death for bar in self.barcode] )*1.03

        filtration_values = np.linspace(0,filt_max,resolution+1)

        if len(filtration_values) == 0:
            raise ValueError("No filtration values available for interactive plot.")

        color_map = self._get_color_map(coloring=coloring)

        if show_cycles:
            max_cycle_slots = max(
                len(self._active_bars_with_cycles_at(v, min_bar_length=min_bar_length))
                for v in filtration_values
            )
        else:
            max_cycle_slots = 0

        frames = []
        for v in filtration_values:
            snapshot = self._complex_snapshot_at_filtration(filt_val=v)

            if self.dim == 2:
                frame_traces = self._plotly_traces_for_snapshot_2d(
                    snapshot=snapshot,
                    color_map=color_map,
                    show_cycles=show_cycles,
                    fill_triangles=fill_triangles,
                    signed=signed,
                    min_bar_length=min_bar_length,
                    max_cycle_slots=max_cycle_slots,
                    vertex_size=vertex_size
                )

            elif self.dim == 3:
                frame_traces = self._plotly_traces_for_snapshot_3d(
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

        if self.dim == 2:
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

    def interactive_plot_filtration(self, *args, **kwargs):
        """
        Deprecated alias for ``plot_filtration_interactive``.

        For backward compatibility, forwards all arguments to
        ``plot_filtration_interactive``.
        """
        warnings.warn(
            "`interactive_plot_filtration` is deprecated; use `plot_filtration_interactive`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.plot_filtration_interactive(*args, **kwargs)

    # ------ forest plotting tools ------------

    def plot_dendrogram(
        self,
        *args,
        **kwargs
    ):
        """
        Plot the forest as a dendrogram.

        Parameters
        ----------
        *args, **kwargs :
            Forwarded to ``forest_plotting._plot_dendrogram_generic``. Common
            options include ``ax``, ``show``, ``annotate_ids``,
            ``leaf_spacing``, ``tree_gap_leaves``, ``check_reduced``,
            ``small_on_top`` and ``threshold``.
        """
        from .forest_plotting import _plot_dendrogram_generic
        return _plot_dendrogram_generic(self, *args, **kwargs)

    def plot_barcode(self, *args, **kwargs):
        """
        Plot a 1D barcode from self.barcode (a set[Bar]).

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
            - "forest": use self.color_map_forest (tree-structured colors).
            - "bars":   use self.color_map_bars (ignores tree structure).
            - "none":   use Matplotlib defaults.
            - "grey":   draw all bars in black.
            If the chosen color map does not exist yet, it is built as in
            `plot_at_filtration`.
        max_bars : int
            If > 0, display at most this many bars, keeping the longest ones
            (by lifespan). 0 means show all bars.
        min_bar_length : float
            Filter out bars with lifespan < min_bar_length before plotting.
        bar_width : float
            Line width used to draw each barcode interval.
        descending : bool
            If True, reverse the selected sort order.
        tight_layout : bool
            If True, call ``fig.tight_layout()`` after drawing.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes the barcode was drawn on.
        """
        from .forest_plotting import _plot_barcode_generic

        return _plot_barcode_generic(self, *args,**kwargs)

    # --------- animation -------------

    def animate_filtration(
        self,
        filename: Optional[str] = None,
        format: Optional[Literal["mp4", "html"]] = None,
        with_barcode: bool = False,
        fps: int = 20,
        frames: int = 200,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        coloring: Literal["forest", "bars"] = "forest",
        signed: bool = False,
        show_cycles: bool = True,
        show_complex: bool = True,
        dpi: int = 200,
        figsize: Optional[tuple[float, float]] = None,
        pixel_size: Optional[tuple[int, int]] = None,
        panel_width_ratios: tuple[float, float] = (1.0, 1.0),
        panel_spacing: float = 0.08,
        figure_margins: Optional[dict[str, float]] = None,
        filtration_kwargs: Optional[dict] = None,
        barcode_kwargs: Optional[dict] = None,
        camera_mode: Literal["fixed", "orbit"] = "fixed",
        # Deprecated aliases (kept for backward compatibility)
        cloud_figsize: Optional[tuple[float, float]] = None,
        total_figsize: Optional[tuple[float, float]] = None,
        plot_kwargs: Optional[dict] = None,
        alpha_digits: Optional[int] = None,
    ):
        """
        Animate the filtration with automatic 2D/3D backend dispatch.

        Dispatch rules:
        - 2D: uses the existing matplotlib animation pipeline.
        - 3D + MP4: uses a matplotlib frame renderer and ffmpeg assembly.
        - 3D + HTML: uses the existing Plotly interactive pipeline.

        Parameters
        ----------
        filename : str | None
            Output path. If omitted, a default filename is chosen for exports.
        format : {"mp4", "html"} | None
            Output format. If None, inferred from ``filename``; defaults to
            ``"mp4"`` when not inferable.
        with_barcode : bool
            If True, show a barcode panel with a moving filtration marker.
        fps : int
            Frames per second for MP4 output.
        frames : int
            Number of sampled filtration values.
        t_min, t_max : float | None
            Optional filtration interval.
        coloring : {"forest", "bars"}
            Color map used for cycle overlays and barcode.
        signed : bool
            If False, cancel opposite-oriented duplicate simplices first.
        show_cycles : bool
            Whether to display active cycle representatives.
        show_complex : bool
            Whether to render complex geometry.
        dpi : int
            DPI used for matplotlib frame rendering and video export.
        figsize : tuple[float, float] | None
            Matplotlib figure size in inches.
            Defaults to (6, 6) without barcode and (10, 5) with barcode.
        pixel_size : tuple[int, int] | None
            Figure size in pixels. If provided, overrides ``figsize``.
        panel_width_ratios : tuple[float, float]
            Relative widths of cloud/barcode panels when ``with_barcode=True``.
        panel_spacing : float
            Horizontal spacing between cloud and barcode panels.
        figure_margins : dict[str, float] | None
            Outer figure margins with keys ``left``, ``right``, ``bottom``,
            and ``top`` passed to ``subplots_adjust``.
        filtration_kwargs : dict | None
            Extra kwargs forwarded to ``plot_at_filtration`` (except ``ax``,
            ``show`` and ``filt_val``). For rendering controls, use top-level
            ``show_cycles``/``show_complex``; keep this dict for values like
            ``vertex_size`` and 3D style values such as
            ``style_3d={'camera_eye': ..., 'complex_face_alpha': ..., 'cycle_face_alpha': ...}``.
            Backward-compatible keys are translated before forwarding:
            ``fill_triangles`` to ``show_complex``, ``remove_double_edges`` to
            ``signed``, ``linewidth_filt`` and ``linewidth_cycle`` to 2D style
            widths, ``show_orientation_arrows`` to 2D style, and
            ``camera_eye``/``remove_axes`` to 3D style.
        barcode_kwargs : dict | None
            Extra kwargs forwarded to barcode plotting for matplotlib paths.
            For animations with a barcode panel, the generic helper manages
            ``ax`` and uses the top-level ``coloring`` for both panels.
        camera_mode : {"fixed", "orbit"}
            3D camera mode.
        cloud_figsize, total_figsize : tuple[float, float] | None
            Deprecated aliases for ``figsize``. ``total_figsize`` applies when
            ``with_barcode=True``; ``cloud_figsize`` applies otherwise.
        plot_kwargs : dict | None
            Deprecated alias for ``filtration_kwargs``.
        alpha_digits : int | None
            Number of digits shown in the filtration value overlay (matplotlib paths).

        Returns
        -------
        object
            - 2D path: ``(anim, fig)`` from matplotlib.
            - 3D MP4 path: output filename string.
            - 3D HTML path: Plotly figure.
        """
        from pathlib import Path
        from .forest_plotting import _animate_filtration_generic

        out_format = format
        if out_format is None and filename is not None:
            suffix = Path(filename).suffix.lower()
            if suffix in {".mp4", ".html"}:
                out_format = suffix[1:]
        if out_format is None:
            out_format = "mp4"
        if out_format not in {"mp4", "html"}:
            raise ValueError(f"Unsupported animation format: {out_format!r}. Use 'mp4' or 'html'.")

        if plot_kwargs is not None:
            warnings.warn(
                "`plot_kwargs` is deprecated; use `filtration_kwargs`.",
                DeprecationWarning,
                stacklevel=2,
            )
            if filtration_kwargs is None:
                filtration_kwargs = dict(plot_kwargs)
            else:
                merged = dict(plot_kwargs)
                merged.update(filtration_kwargs)
                filtration_kwargs = merged

        if cloud_figsize is not None or total_figsize is not None:
            warnings.warn(
                "`cloud_figsize` and `total_figsize` are deprecated; use `figsize=(w, h)`.",
                DeprecationWarning,
                stacklevel=2,
            )
            if figsize is None:
                if with_barcode and total_figsize is not None:
                    figsize = total_figsize
                elif (not with_barcode) and cloud_figsize is not None:
                    figsize = cloud_figsize

        base_filtration_kwargs: dict[str, Any] = {}
        if filtration_kwargs is not None:
            base_filtration_kwargs.update(filtration_kwargs)
    
        style_2d = dict(base_filtration_kwargs.pop("style_2d", {}) or {})
        style_3d = dict(base_filtration_kwargs.pop("style_3d", {}) or {})

        if "fill_triangles" in base_filtration_kwargs:
            base_filtration_kwargs.setdefault(
                "show_complex",
                bool(base_filtration_kwargs.pop("fill_triangles")),
            )

        if "remove_double_edges" in base_filtration_kwargs:
            base_filtration_kwargs.setdefault(
                "signed",
                not bool(base_filtration_kwargs.pop("remove_double_edges")),
            )

        if "linewidth_filt" in base_filtration_kwargs:
            style_2d.setdefault(
                "complex_edge_width",
                float(base_filtration_kwargs.pop("linewidth_filt")),
            )
        if "linewidth_cycle" in base_filtration_kwargs:
            style_2d.setdefault(
                "cycle_edge_width",
                float(base_filtration_kwargs.pop("linewidth_cycle")),
            )

        if "show_orientation_arrows" in base_filtration_kwargs:
            style_2d.setdefault(
                "show_orientation_arrows",
                bool(base_filtration_kwargs.pop("show_orientation_arrows")),
            )

        if "camera_eye" in base_filtration_kwargs:
            style_3d.setdefault("camera_eye", base_filtration_kwargs.pop("camera_eye"))
        if "remove_axes" in base_filtration_kwargs:
            style_3d.setdefault(
                "remove_axes",
                bool(base_filtration_kwargs.pop("remove_axes")),
            )

        base_filtration_kwargs["signed"] = bool(signed)
        base_filtration_kwargs["show_cycles"] = bool(show_cycles)
        base_filtration_kwargs["show_complex"] = bool(show_complex)

        if style_2d:
            base_filtration_kwargs["style_2d"] = style_2d
        if style_3d:
            base_filtration_kwargs["style_3d"] = style_3d

        if self.dim == 2:
            if out_format == "html":
                raise ValueError("HTML animation export is only implemented for ambient dimension 3.")

            resolved_filename = filename
            if resolved_filename is not None and Path(resolved_filename).suffix.lower() == "":
                resolved_filename = f"{resolved_filename}.mp4"

            filtration_panel_kwargs = dict(base_filtration_kwargs)
            barcode_panel_kwargs = {}
            if barcode_kwargs is not None:
                barcode_panel_kwargs.update(barcode_kwargs)
            t_max_2d = t_max
            if t_max_2d is None:
                finite_deaths = [float(bar.death) for bar in self.barcode if np.isfinite(float(bar.death))]
                if finite_deaths:
                    t_max_2d = max(finite_deaths)
                else:
                    finite_filtration = [float(f) for _, f in self.filtration if np.isfinite(float(f))]
                    if finite_filtration:
                        t_max_2d = max(finite_filtration)

            return _animate_filtration_generic(
                self,
                filename=resolved_filename,
                fps=fps,
                frames=frames,
                coloring=coloring,
                with_barcode=with_barcode,
                t_min=t_min,
                t_max=t_max_2d,
                dpi=dpi,
                figsize=figsize,
                pixel_size=pixel_size,
                panel_width_ratios=panel_width_ratios,
                panel_spacing=panel_spacing,
                figure_margins=figure_margins,
                filtration_kwargs=filtration_panel_kwargs,
                barcode_kwargs=barcode_panel_kwargs,
                alpha_digits=alpha_digits,
            )

        if self.dim == 3:
            if out_format == "html":
                html_width = None
                html_height = None
                if pixel_size is not None:
                    if len(pixel_size) != 2:
                        raise ValueError("pixel_size must be a 2-tuple (width, height).")
                    html_width = int(pixel_size[0])
                    html_height = int(pixel_size[1])
                elif figsize is not None:
                    html_width = int(round(float(figsize[0]) * float(dpi)))
                    html_height = int(round(float(figsize[1]) * float(dpi)))

                filtration_panel_kwargs = dict(base_filtration_kwargs)
                style_3d = dict(filtration_panel_kwargs.get("style_3d", {}) or {})

                fig = self.plot_filtration_interactive(
                    coloring=coloring,
                    show_cycles=filtration_panel_kwargs.get("show_cycles", True),
                    signed=filtration_panel_kwargs["signed"],
                    filt_max=t_max,
                    show_complex=filtration_panel_kwargs.get("show_complex", True),
                    complex_opacity=float(style_3d.get("complex_face_alpha", 0.20)),
                    cycle_opacity=float(style_3d.get("cycle_face_alpha", 0.55)),
                    resolution=frames,
                    show=False,
                    vertex_size=float(filtration_panel_kwargs.get("vertex_size", 3.0)),
                    width=html_width,
                    height=html_height,
                )
                resolved_filename = filename if filename is not None else "filtration_animation_3d.html"
                if Path(resolved_filename).suffix.lower() == "":
                    resolved_filename = f"{resolved_filename}.html"
                fig.write_html(str(resolved_filename))
                return fig

            resolved_filename = filename if filename is not None else "filtration_animation_3d.mp4"
            if Path(resolved_filename).suffix.lower() == "":
                resolved_filename = f"{resolved_filename}.mp4"

            filtration_panel_kwargs = {"camera_mode": camera_mode, **base_filtration_kwargs}

            barcode_panel_kwargs = {}
            if barcode_kwargs is not None:
                barcode_panel_kwargs.update(barcode_kwargs)

            t_max_3d = t_max
            if t_max_3d is None:
                finite_deaths = [float(bar.death) for bar in self.barcode if np.isfinite(float(bar.death))]
                if finite_deaths:
                    t_max_3d = max(finite_deaths)
                else:
                    finite_filtration = [float(f) for _, f in self.filtration if np.isfinite(float(f))]
                    if finite_filtration:
                        t_max_3d = max(finite_filtration)

            return _animate_filtration_generic(
                self,
                filename=resolved_filename,
                with_barcode=with_barcode,
                fps=fps,
                frames=frames,
                t_min=t_min,
                t_max=t_max_3d,
                coloring=coloring,
                dpi=dpi,
                figsize=figsize,
                pixel_size=pixel_size,
                panel_width_ratios=panel_width_ratios,
                panel_spacing=panel_spacing,
                figure_margins=figure_margins,
                filtration_kwargs=filtration_panel_kwargs,
                barcode_kwargs=barcode_panel_kwargs,
                alpha_digits=alpha_digits,
            )

        raise ValueError("animate_filtration is only implemented for ambient dimensions 2 and 3.")

    def animate_barcode_measurement(
                self,
                cycle_func,
                signed: bool = True,
                bar = None,
                *args,
                **kwargs,
        ):
        """
        Animate the filtration together with a barcode measurement.

        This is the animated analogue of :meth:`plot_barcode_measurement`.
        The left panel shows :meth:`plot_at_filtration` at a moving
        filtration value, while the right panel shows the associated
        barcode measurement with a vertical line indicating the current
        filtration value.

        Parameters
        ----------
        cycle_func :
            Callable ``(chain, point_cloud) -> float`` that assigns a scalar
            to each cycle representative associated to the chosen bar.
            See :meth:`plot_barcode_measurement` for details.
        signed : bool, optional
            If ``True`` (default), ``cycle_func`` is called on the signed
            cycle representatives. If ``False``, the chains are first
            passed through ``chain.unsigned()``.
        bar :
            Bar to use for the measurement. If ``None``, the method uses
            ``self.max_bar()`` inside the generic helper.
        *args, **kwargs :
            Forwarded to
            :func:`forest_landscapes.animate_barcode_measurement_generic`.
            Typical keyword arguments include ``filename``, ``fps``,
            ``frames``, ``t_min``, ``t_max``, ``dpi``, ``total_figsize``,
            ``filtration_kwargs`` and ``measurement_kwargs``.
            ``filtration_kwargs`` is copied and normalized before forwarding:
            ``fill_triangles``, ``remove_double_edges``, ``linewidth_filt``,
            ``linewidth_cycle``, ``show_orientation_arrows``, ``camera_eye``
            and ``remove_axes`` are translated to the current
            ``plot_at_filtration`` style keys. ``plot_kwargs`` is no longer
            accepted.

        Returns
        -------
        anim, fig :
            A pair ``(anim, fig)`` where ``anim`` is a
            :class:`matplotlib.animation.FuncAnimation` and ``fig`` is the
            underlying figure.
        """
        from .forest_landscapes import animate_barcode_measurement_generic
        from copy import deepcopy

        kwargs = dict(kwargs)
        if "plot_kwargs" in kwargs:
            raise TypeError(
                "`plot_kwargs` is no longer supported; use `filtration_kwargs`."
            )
        raw_filtration_kwargs = kwargs.get("filtration_kwargs", None)
        if raw_filtration_kwargs is not None:
            if not isinstance(raw_filtration_kwargs, dict):
                raise TypeError("filtration_kwargs must be a dict when provided.")
            filtration_kwargs = deepcopy(raw_filtration_kwargs)
        else:
            filtration_kwargs = {}

        style_2d = dict(filtration_kwargs.pop("style_2d", {}) or {})
        style_3d = dict(filtration_kwargs.pop("style_3d", {}) or {})

        if "fill_triangles" in filtration_kwargs:
            filtration_kwargs["show_complex"] = bool(filtration_kwargs.pop("fill_triangles"))

        if "remove_double_edges" in filtration_kwargs:
            filtration_kwargs["signed"] = not bool(filtration_kwargs.pop("remove_double_edges"))

        if "linewidth_filt" in filtration_kwargs:
            style_2d["complex_edge_width"] = float(filtration_kwargs.pop("linewidth_filt"))
        if "linewidth_cycle" in filtration_kwargs:
            style_2d["cycle_edge_width"] = float(filtration_kwargs.pop("linewidth_cycle"))

        if "show_orientation_arrows" in filtration_kwargs:
            style_2d["show_orientation_arrows"] = bool(
                filtration_kwargs.pop("show_orientation_arrows")
            )

        if "camera_eye" in filtration_kwargs:
            style_3d["camera_eye"] = filtration_kwargs.pop("camera_eye")
        if "remove_axes" in filtration_kwargs:
            style_3d["remove_axes"] = bool(filtration_kwargs.pop("remove_axes"))

        # `alpha_digits` is not a `plot_at_filtration` kwarg in this animation path.
        filtration_kwargs.pop("alpha_digits", None)

        if style_2d:
            filtration_kwargs["style_2d"] = style_2d
        if style_3d:
            filtration_kwargs["style_3d"] = style_3d

        kwargs["filtration_kwargs"] = filtration_kwargs

        if signed:
            def _cycle_value(chain, point_cloud):
                # `chain` is a SignedChain
                return float(cycle_func(chain, point_cloud))
        else:
            def _cycle_value(chain, point_cloud):
                # `chain` is a SignedChain
                return float(cycle_func(chain.unsigned(), point_cloud))

        return animate_barcode_measurement_generic(
            forest=self,
            cycle_func=_cycle_value,
            bar=bar,
            *args,
            **kwargs,
        )

    #------- generalized landscape ----------------

    def plot_barcode_measurement(
        self,
        cycle_func,
        signed: bool = False,
        bar = None,
        ax=None,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        label: Optional[str] = None,
        show_baseline: bool = True,
        show: bool = False,
        *args,
        **kwargs,
    ):
        """
        Plot a scalar measurement of cycles along each bar.

        Parameters
        ----------
        cycle_func : callable
            Function returning a scalar for a SignedChain and point cloud.
        signed : bool
            If True, evaluate ``cycle_func`` on the stored signed chain. If
            False, evaluate it after cancelling opposing simplices.
        bar : PFBar | None
            If provided, restrict the plot to a single bar.
        ax : matplotlib.axes.Axes | None
            Axes to draw on; if None, a new figure is created.
        x_range, y_range : (float, float) | None
            Manual axis limits for the step plot.
        title : str | None
            Custom plot title.
        label : str | None
            Legend label for the curve.
        show_baseline : bool
            If True, draw the baseline of the step function.
        show : bool
            If True, call ``plt.show()`` after plotting.
        *args :
            Forwarded to ``forest_landscapes.plot_barcode_measurement_generic``.
        **kwargs :
            Forwarded to ``StepFunctionData.plot`` (e.g., ``baseline_kwargs``,
            line color/linewidth, etc.).

        Returns
        -------
        tuple[matplotlib.axes.Axes, StepFunctionData]
            The axes and the piecewise-constant measurement data.
        """
        from .forest_landscapes import plot_barcode_measurement_generic

        if signed:
            def _cycle_value(chain, point_cloud):
                # `chain` is a SignedChain
                return float(cycle_func(chain, point_cloud))
        else:
            def _cycle_value(chain, point_cloud):
                # `chain` is a SignedChain
                return float(cycle_func(chain.unsigned(), point_cloud))

        return plot_barcode_measurement_generic(
            forest=self,
            cycle_func=_cycle_value,
            bar=bar,
            ax=ax,
            x_range=x_range,
            y_range=y_range,
            title=title,
            label=label,
            show_baseline=show_baseline,
            show=show,
            *args,
            **kwargs,
        )

    def compute_generalized_landscape_family(
        self,
        cycle_func,
        label: str,
        *,
        max_k: int = 5,
        num_grid_points: int = 512,
        mode: Literal["raw", "pyramid"] = "pyramid",
        min_bar_length: float = 0.0,
        x_grid: Optional[NDArray[np.float64]] = None,
        cache: bool = True,
        signed: bool = False,
        cache_functionals: bool = False,
        functionals_label: Optional[str] = None,
        compute_functionals: bool = True,
    ):
        """
        Compute a generalized landscape family for this PersistenceForest.

        Parameters
        ----------
        cycle_func : callable
            A function

                cycle_func(chain, point_cloud) -> float

            where ``chain`` is a SignedChain. This lets you define arbitrary
            functionals on cycles (e.g. total length, mass, etc.).
        label : str
            Key used to store the family in ``self.landscape_families`` when
            ``cache=True``.
        max_k : int
            Number of landscape levels λ_1..λ_max_k.
        num_grid_points : int, optional
            Number of x-grid samples (if x_grid is None).
        mode : {"raw", "pyramid"}, optional
            Kernel mode; "pyramid" matches the LoopForest convention.
        min_bar_length : float, optional
            Ignore bars with lifespan < min_bar_length.
        x_grid : np.ndarray | None, optional
            Optional common grid to evaluate all landscapes on.
        cache : bool
            If True, store the resulting family in ``self.landscape_families``
            under ``label``.
        signed : bool
            If True, evaluate ``cycle_func`` on stored signed chains. If False,
            evaluate it after cancelling opposing simplices.
        cache_functionals : bool
            If True, store the per-bar ``BarcodeFunctionals`` in
            ``self.barcode_functionals``.
        functionals_label : str | None
            Label used when caching or retrieving per-bar functionals. If None,
            the generic helper uses ``label``.
        compute_functionals : bool
            If True, compute per-bar functionals before building landscapes. If
            False, reuse cached functionals with ``functionals_label``.

        Returns
        -------
        GeneralizedLandscapeFamily
            Family of landscapes evaluated for each bar.
        """

        if signed:
            def _cycle_value(chain, point_cloud):
                # `chain` is a SignedChain
                return float(cycle_func(chain, point_cloud))
        else:
            def _cycle_value(chain, point_cloud):
                # `chain` is a SignedChain
                return float(cycle_func(chain.unsigned(), point_cloud))
    
        from .forest_landscapes import compute_generalized_landscape_family

        return compute_generalized_landscape_family(
            self,
            _cycle_value,
            max_k=max_k,
            num_grid_points=num_grid_points,
            mode=mode,
            label=label,
            min_bar_length=min_bar_length,
            x_grid=x_grid,
            cache = cache,
            cache_functionals=cache_functionals,
            functionals_label=functionals_label,
            compute_functionals=compute_functionals,
        )

    def plot_landscape_family(
        self,
        label: str,
        ks: Optional[list[int]] = None,
        ax=None,
        title: Optional[str] = None,
        *args,
        show_legend: Optional[bool] = None,
        linewidth: Optional[float] = None,
        **kwargs,
    ):
        """
        Plot a previously computed generalized landscape family.

        Parameters
        ----------
        label : str
            Identifier used when the family was computed.
        ks : list[int] | None
            Which landscape levels to plot. Defaults to all available.
        ax : matplotlib.axes.Axes | None
            Axes to draw on; if None, a new figure is created.
        title : str | None
            Custom plot title.
        show : bool
            Forwarded through ``**kwargs`` to the generic plotting helper; if
            True, call ``plt.show()`` after plotting.
        show_legend : bool | None
            Whether to show the legend. Defaults to showing it for fewer than
            10 plotted landscapes and hiding it otherwise.
        linewidth : float | None
            Line width for the landscape plots. If omitted, Matplotlib's
            default line width is used.

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the landscape plot.
        """
        from .forest_landscapes import plot_landscape_family
        return plot_landscape_family(
            self,
            label,
            ks,
            ax,
            title,
            *args,
            show_legend=show_legend,
            linewidth=linewidth,
            **kwargs,
        )

    def plot_landscape_comparison_between_functionals(
        self,
        labels: list[str],
        k: int = 1,
        ax=None,
        title: Optional[str] = None,
        *args, 
        **kwargs
    ):
        """
        Compare multiple generalized landscape families on the same axes.

        Parameters
        ----------
        labels : list[str]
            Labels of the landscape families to compare.
        k : int
            Which landscape level to compare.
        ax : matplotlib.axes.Axes | None
            Axes to draw on; if None, a new figure is created.
        title : str | None
            Custom plot title.

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the comparison plot.
        """
        from .forest_landscapes import plot_landscape_comparison_between_functionals
        return plot_landscape_comparison_between_functionals(self, labels=labels, k=k, ax=ax, title=title, *args, **kwargs)
