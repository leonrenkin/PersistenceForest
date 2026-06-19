from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal, Iterable, Callable, Union, Sequence
from collections import defaultdict
from numpy.typing import NDArray
import itertools
import numpy as np
import gudhi as gd
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Polygon
import time
import seaborn as sns
import bisect
from bisect import bisect_right


# ------- helper functions --------------

def loop_in_filtration_check(loop, simplex_tree, filt_value):

    for i in range(len(loop)):
        birth = edge_birth_time(loop[i-1],loop[i],simplex_tree=simplex_tree)

        if math.isinf(birth) and birth > 0:
            print(f"Edge {[loop[i-1],loop[i]]} never appears in the filtration (birth = +∞).")
            return False
        
        elif birth > filt_value:
            print(f"Edge {[loop[i-1],loop[i]]} is in simplicial complex but appears later in the filtration. \n Input filt_val={filt_value}, Birth of edge {birth}")
            return False

    return True

def triangle_loop_counterclockwise(simplex, point_cloud):
    """
    Given `simplex`: a length-3 iterable of point indices into point_cloud,
    return those same indices reordered so that:
      1. The first point has the smallest (x, then y) coordinates.
      2. The remaining two follow in counter-clockwise order around that first point.
    """
    if len(simplex) != 3:
        raise ValueError("Error: given simplex is not a triangle.")

    # 1) Grab the actual coordinates as an (3,2) array
    simplex_arr     = np.array(simplex, dtype=int)
    triangle_pts    = np.asarray(point_cloud)[simplex_arr]   # shape (3,2)

    # 2) Find the sort order by x, then y
    #    lexsort takes keys in the order (secondary, primary), so:
    order = np.lexsort((triangle_pts[:,1], triangle_pts[:,0]))
    simplex_sorted = simplex_arr[order]
    pts_sorted     = triangle_pts[order]

    # 3) Compute the two edge-vectors out of the “lowest” point
    v1 = pts_sorted[1] - pts_sorted[0]
    v2 = pts_sorted[2] - pts_sorted[0]

    # 4) Use the sign of the 2D cross-product to check orientation:
    #    cross > 0 means v1→v2 is CCW; if it’s < 0, swap pts 1 and 2.
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    if cross < 0:
        simplex_sorted[1], simplex_sorted[2] = simplex_sorted[2], simplex_sorted[1]

    return simplex_sorted

def merge_index_list_loops(loop1: NDArray[np.int32],loop2: NDArray[np.int32], edge:List[int]):
    """ 
    Merges two loops, given as list of indices, which both contain edge in oppossing orientation.
    Edge is pair of indices.
    Returns loop as list of indices. 
    """

    #find position of edge in first loop
    for i in range(len(loop1)):
        if loop1[i] == edge[0]:
            if not (loop1[(i-1) % len(loop1)]==edge[1] or loop1[(i+1) % len(loop1)]==edge[1]):  #deal with the case that one vertex in the edge appears an additional time
                #print("Concat loop edge case triggered")
                continue
            idx0= i
            break
    
    #check which edge appears first
    if loop1[idx0-1]== edge[1]: # type: ignore
        j = 0
        p = edge[1]
        idx0 = (idx0 - 1) % len(loop1) # type: ignore
    else:
        j = 1
        p = edge[0]
    
    for i in range(len(loop2)):
        if loop2[i]==p:
            if not (loop2[(i-1) % len(loop2)]==edge[j] or loop2[(i+1) % len(loop2)]==edge[j]):  #deal with the case that one vertex in the edge appears an additional time
                #print("Concat loop edge case triggered")
                continue
            idx1=i
            break

    #loop = loop1[:idx0] + loop2[idx1:] + loop2[:idx1-1] + loop1[ idx0  :]
    
    if idx0 == len(loop1)-1 and idx1==0: # type: ignore
        loop = np.concatenate((loop1[1:], loop2[idx1+1:], loop2[:idx1])) # type: ignore
    elif idx0 == len(loop1)-1 and idx1!=0: # type: ignore #handle edge case that we have the last index for idx0
        loop = np.concatenate((loop1, loop2[idx1+1:], loop2[:idx1-1]))  # type: ignore
    else: #common case
        loop = np.concatenate((loop1[:idx0], loop2[idx1:], loop2[:idx1], loop1[ idx0+2  :])) # type: ignore

    #print(F'resulting loop {loop}')
    if len(loop) != len(loop1)+len(loop2)-2:
        raise ValueError("Error, concatitnation has wrong length")

    return loop

def edge_birth_time(a, b, simplex_tree):
    # sort the endpoints so you get the same key for [a,b] or [b,a]
    edge = tuple(sorted((a, b)))
    return simplex_tree.filtration(edge)  # returns +inf if not present

def contains_pair_in_order(lst, a, b):
    """
    Returns True iff the list `lst` contains the adjacent subsequence [a, b].
    """

    for i in range(-1,len(lst)):
        if lst[i] == a and lst[i+1 ] == b:
            return True
    return False

def key(simplex):
    """ canonical key: return sorted tuple of vertex ids so orientation doesn’t matter """ 
    return tuple(sorted(simplex))

def split_vertex_loop_with_double_edge(edge, loop: NDArray[np.int32]):
    """splits a loop containg edge twice into two disjoint loops.
    return two loops, one of which might be empty"""

    if not (contains_pair_in_order(lst=loop, a=edge[0], b=edge[1]) and contains_pair_in_order(lst=loop, a=edge[1], b=edge[0])):
        raise ValueError("Input loop does not contain edge in both directions")
    for idx1 in range(len(loop)):
        if loop[idx1] in edge and (loop[idx1+1] in edge or loop[idx1-1] in edge):
            break
    if loop[idx1+1] in edge: # type: ignore #this is only to deal with the case idx = 0
        idx1 +=1 # type: ignore
    
    for idx2 in range(idx1+1, len(loop)): # type: ignore
        if loop[idx2] in edge and (loop[(idx2+1) % (len(loop)-1)] in edge or loop[idx2-1] in edge):
            break
    if idx2 !=len(loop)-1: # type: ignore
        idx2 +=1 # type: ignore

    #deal with edge case where we the edge [a,b] appears as [a,b,a]
    #in this case idx1 is at b and idx2 is the one after the second a 
    if idx2-idx1 == 2: # type: ignore
        loop1 = np.concatenate((loop[:idx1], loop[idx2:])) # type: ignore
        loop2 = []
        #print(f"non-trivial loop returned: {loop1}")
        return loop1, loop2
    #same edge case but this time it is [a,....,a,b] or [b,a,....,a]
    if idx2 - idx1 == len(loop)-2: # type: ignore
        loop1 = loop[idx1:idx2] # type: ignore
        loop2 = []
        #print(f"non-trivial loop returned: {loop1}")
        return loop1, loop2


    loop1 = loop[idx1: idx2-1] # type: ignore
    
    if idx1 == 0: # type: ignore
        loop2 = loop[idx2:len(loop)-1]  # type: ignore
    
    else:
        loop2 = np.concatenate((loop[idx2:], loop[:idx1-1])) # type: ignore

    #Loop is only a single vertex does not count as loop an will be treated as empty
    if len(loop1)==1:
        loop1=[] 
    if len(loop2)==1:
        loop2=[]

    #print(f"edge {edge}, loop {loop}")
    #print(f"idx1 {idx1}, idx2 {idx2}") # type: ignore
    #print(f"loop1 {loop1}")
    #print(f"loop2 {loop2}")


    return loop1, loop2

def loop_xmax(loop, point_cloud):
    return point_cloud[loop, 0].max()

def loop_xmin(loop, point_cloud):
    return point_cloud[loop, 0].min()

def loop_ymax(loop, point_cloud):
    return point_cloud[loop, 1].max()

def loop_ymin(loop, point_cloud):
    return point_cloud[loop, 1].min()

def find_outer_loop(vertex_loop:NDArray[np.int32] ,edge:List[int], point_cloud):
    """ Computes surviving loop in case of double edge in a single loop, first returned loop is survivor loop"""
    #print("tiebreak activated")
    loop1, loop2 = split_vertex_loop_with_double_edge(edge=edge, loop=vertex_loop)


    if len(loop1)<=2 and len(loop2)<=2:
        raise ValueError("Both split loops are trivial")
    elif len(loop1) <=2:
        return loop2, loop1

    elif len(loop2) <=2:
        return loop1, loop2
    
    else:
        #check which loop is contained by the other one
        xmax1=loop_xmax(loop=loop1, point_cloud=point_cloud)
        xmax2=loop_xmax(loop=loop2, point_cloud=point_cloud)

        #could use ymax,  ymin, xmin to check if htey also satisfy the inequalities
        if xmax1>xmax2:
            return loop1, loop2

        if xmax1<xmax2:
            return loop2, loop1

        else:
            raise ValueError("Maximal x value of both loops is equal which should not happen")

def are_dict_keys_sorted(d):
    """Return True if dict keys are in ascending order (linear time)."""
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




# ----------------- class defintions ----------------------
class Loop:
    """
    Loop saved as a list of vertex indices. 
    Point coordinates can be accessed via point_cloud[vertex] where vertex is in vertex_list and point_cloud is the list of points saved in the loop forest
    """
    def __init__(self, vertex_list: NDArray[np.int32], id: int):
        self.vertex_list: NDArray[np.int32] = np.array(vertex_list).astype(np.int32)
        self.id: int = id
        self.active_start: float = float("-inf")           #[active_start, active_end) is interval in which the loop is active as an optimal cycle rep
        self.active_end: float = float("-inf")                 

@dataclass
class Node:
    """ Objects which are the nodes in the LoopForest graph. 
    Each node has a loop representative."""
    id: int #does not need to know its own node
    filt_val: float
    type: Literal["leaf", "root", "merge", "update"]            #a node can be type "merge" and also be a root, it then still appears in the root list
    loop: Loop                                                  #Loops are saved as list of indices of simplex
    children: set[int]                                    #ids of children
    parent: Optional[int] = None
    #is_root: bool = True  #True if it is the root of a tree, also used for bookkeeping of active loops
    _barcode_covered: bool = False

    def __repr__(self) -> str:
        return f"Node(id={self.id}, f={self.filt_val}, type={self.type})"
    
class Bar:
    """ 
    Object which stores a bar in H1 persistence together with a progression of cycle reps.
    Each cycle rep has a an active_start and active_end attribute which is the interval in which this representative is optimal.
    The cycle reps are a strictly decreasing chain w.r.t. inclusion.
    """

    def __init__(self, birth: float, 
                 death: float, 
                 _node_progression: tuple[int,...], 
                 cycle_reps: list[Loop], 
                 is_max_tree_bar: Optional[bool]=None, 
                 root_id: Optional[int]=None):
        self.birth = birth
        self.death = death
        self._node_progression = _node_progression #nodes saved as node_ids
        self.cycle_reps = cycle_reps
        self.is_max_tree_bar = is_max_tree_bar
        self.root_id = root_id

    def loop_at_filtration_value(self, filt_val)->Loop:
        """Binary search to find active loop at filtration value of this bar."""

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

@dataclass
class StepFunctionData:
    """
    Representation of a piecewise-constant function:

    f(t) = vals[i]  on [starts[i], ends[i]]
         = baseline elsewhere.

    All arrays are 1D and aligned by index. `domain` is the global range
    where the function is potentially non-zero (for convenience).
    """
    starts: NDArray[np.float64]
    ends: NDArray[np.float64]
    vals: NDArray[np.float64]
    baseline: float
    domain: Tuple[float, float]
    metadata: Dict[str, object] = field(default_factory=dict)

@dataclass
class PiecewiseLinearFunction:
    """
    Piecewise-linear function specified by breakpoints (xs, ys).
    Between xs[i] and xs[i+1] the function is linear. Outside [xs[0], xs[-1]]
    the function evaluates to 0.0 by default.
    """
    xs: NDArray[np.float64]
    ys: NDArray[np.float64]
    domain: Tuple[float, float]
    metadata: Dict[str, object] = field(default_factory=dict)

    def __call__(self, x: Union[NDArray[np.float64], float]) -> Union[NDArray[np.float64], float]:
        if self.xs.size == 0:
            if np.isscalar(x):
                return 0.0
            x_arr = np.asarray(x, dtype=float)
            return np.zeros_like(x_arr)

        x_arr = np.asarray(x, dtype=float)
        y_arr = np.interp(x_arr, self.xs, self.ys, left=0.0, right=0.0)

        if np.isscalar(x):
            # np.interp returns a scalar np.ndarray for scalar input
            return float(y_arr)
        return y_arr

@dataclass
class GeneralizedLandscapeFamily:
    """
    Container for a whole family of generalized landscapes for a given LoopForest
    and a fixed polyhedral path function.

    - bar_kernels: per-bar kernels (typically already rescaled if mode="pyramid")
    - landscapes: k -> λ_k (k-th landscape) as PiecewiseLinearFunction
    """
    loop_forest_id: str
    poly_func_name: str
    rescaling: str  # e.g. "raw" or "pyramid"
    x_grid: NDArray[np.float64]
    bar_kernels: Dict[int, PiecewiseLinearFunction]
    landscapes: Dict[int, PiecewiseLinearFunction]
    extra_meta: Dict[str, object] = field(default_factory=dict)
    # Optional back-reference for interactive use; safe to ignore for serialization
    loop_forest: Optional["LoopForest"] = None

@dataclass
class LandscapeVectorizer:
    """
    Turn generalized landscapes of LoopForest objects into fixed-size
    feature vectors suitable for machine learning.

    Usage:

        vec = LandscapeVectorizer(
            polyhedral_path_func=path_func,
            max_k=3,
            num_grid_points=256,
            mode="pyramid",
            min_bar_length=0.05,
        )
        vec.fit(training_forests)
        X_train = vec.transform(training_forests)
        X_test  = vec.transform(test_forests)

    After that, X_train and X_test are standard numpy arrays and can be
    used with scikit-learn, PyTorch, etc.
    """
    polyhedral_path_func: Callable[[NDArray[np.float64]], float]
    max_k: int = 3
    num_grid_points: int = 256
    mode: Literal["raw", "pyramid"] = "pyramid"
    min_bar_length: float = 0.0
    t_min: Optional[float] = None
    t_max: Optional[float] = None

    # fitted attributes
    x_grid_: Optional[NDArray[np.float64]] = field(init=False, default=None)
    is_fitted_: bool = field(init=False, default=False)

    def fit(self, forests: Sequence["LoopForest"]) -> "LandscapeVectorizer":
        """
        Decide on a common time grid for all forests.

        If t_min / t_max are given, they are used directly; otherwise, they
        are inferred from the barcodes (respecting min_bar_length).
        """
        if len(forests) == 0:
            raise ValueError("fit() received an empty list of forests")

        # 1. Determine global t_min / t_max from barcodes if not fixed
        if self.t_min is None or self.t_max is None:
            births: List[float] = []
            deaths: List[float] = []
            for lf in forests:
                if not hasattr(lf, "barcode"):
                    raise AttributeError("One of the LoopForest objects has no 'barcode'")
                for bar in lf.barcode:
                    length = bar.death - bar.birth
                    if length >= self.min_bar_length:
                        births.append(bar.birth)
                        deaths.append(bar.death)
            if not births or not deaths:
                raise ValueError(
                    "No bars of sufficient length found in any forest while fitting "
                    f"(min_bar_length={self.min_bar_length})."
                )
            t_min = min(births) if self.t_min is None else self.t_min
            t_max = max(deaths) if self.t_max is None else self.t_max
        else:
            t_min, t_max = self.t_min, self.t_max

        if t_max <= t_min:
            raise ValueError(f"t_max must be > t_min, got [{t_min}, {t_max}]")

        # 2. Build fixed grid and store it
        self.x_grid_ = np.linspace(t_min, t_max, self.num_grid_points)
        self.is_fitted_ = True
        return self

    def _vectorise_single(self, lf: "LoopForest") -> NDArray[np.float64]:
        """
        Compute the feature vector for a single LoopForest on the fixed grid.
        """
        assert self.x_grid_ is not None

        family = lf.compute_generalized_landscape_family(
            self.polyhedral_path_func,
            max_k=self.max_k,
            num_grid_points=self.num_grid_points,
            mode=self.mode,
            min_bar_length=self.min_bar_length,
            x_grid=self.x_grid_,  # enforce consistent grid
        )

        # Flatten [k, x] into one vector in a fixed, documented order:
        # [λ_1(x_1..x_N), λ_2(x_1..x_N), ..., λ_max_k(x_1..x_N)].
        pieces: List[NDArray[np.float64]] = []
        for k in range(1, self.max_k + 1):
            if k in family.landscapes:
                ys = family.landscapes[k].ys
            else:
                # If this forest has fewer bars than k, define λ_k ≡ 0
                ys = np.zeros_like(self.x_grid_)
            pieces.append(ys)

        vec = np.concatenate(pieces, axis=0)  # shape: (max_k * num_grid_points,)
        return vec

    def transform(self, forests: Sequence["LoopForest"]) -> NDArray[np.float64]:
        """
        Transform a sequence of LoopForest objects into a design matrix X.

        X has shape (n_forests, max_k * num_grid_points).
        """
        if not self.is_fitted_ or self.x_grid_ is None:
            raise RuntimeError("LandscapeVectorizer must be fitted before calling transform().")

        n = len(forests)
        if n == 0:
            raise ValueError("transform() received an empty list of forests")

        X = np.zeros((n, self.max_k * self.num_grid_points), dtype=float)
        for i, lf in enumerate(forests):
            X[i, :] = self._vectorise_single(lf)

        return X

    def fit_transform(self, forests: Sequence["LoopForest"]) -> NDArray[np.float64]:
        """
        Convenience method: fit the grid from forests and return X in one call.
        """
        self.fit(forests)
        return self.transform(forests)

@dataclass
class MultiLandscapeVectorizer:
    """
    Vectorise generalized landscapes for multiple path functions into a single
    feature vector suitable for machine learning.

    For each forest and each path function f_j, we:
      - compute landscapes λ_1,...,λ_max_k on a shared x_grid,
      - flatten the sampled values,
      - optionally append L1 and L2 norms of each λ_k.

    Resulting feature vector (conceptually):

        [ samples for f_1 | stats for f_1 | samples for f_2 | stats for f_2 | ... ]

    Usage:
        vec = MultiLandscapeVectorizer(
            poly_funcs=[f_const_one, f_length, ...],
            max_k=3,
            num_grid_points=256,
            mode="pyramid",
            min_bar_length=0.05,
            include_stats=True,
        )

        vec.fit(train_forests)
        X_train = vec.transform(train_forests)
        X_test  = vec.transform(test_forests)
    """
    poly_funcs: Sequence[Callable[[NDArray[np.float64]], float]]
    max_k: int = 5
    num_grid_points: int = 256
    mode: Literal["raw", "pyramid"] = "pyramid"
    min_bar_length: float = 0.0
    t_min: Optional[float] = None
    t_max: Optional[float] = None
    include_stats: bool = False

    # derived
    func_names: Optional[Sequence[str]] = None

    # fitted attributes
    x_grid_: Optional[NDArray[np.float64]] = field(init=False, default=None)
    dx_: Optional[float] = field(init=False, default=None)
    is_fitted_: bool = field(init=False, default=False)
    n_features_: Optional[int] = field(init=False, default=None)

    def __post_init__(self):
        if self.func_names is None:
            self.func_names = [
                getattr(f, "__name__", f"func_{i}")
                for i, f in enumerate(self.poly_funcs)
            ]

    def fit(self, forests: Sequence["LoopForest"]) -> "MultiLandscapeVectorizer":
        """
        Decide on a common time grid for all forests.

        If t_min / t_max are given, they are used directly; otherwise, they
        are inferred from the barcodes (respecting min_bar_length).
        """
        if len(forests) == 0:
            raise ValueError("fit() received an empty list of forests")

        # 1. Determine global t_min / t_max from barcodes if not fixed
        if self.t_min is None or self.t_max is None:
            births: List[float] = []
            deaths: List[float] = []
            for lf in forests:
                if not hasattr(lf, "barcode"):
                    raise AttributeError("One of the LoopForest objects has no 'barcode'")
                for bar in lf.barcode:
                    length = bar.death - bar.birth
                    if length >= self.min_bar_length:
                        births.append(bar.birth)
                        deaths.append(bar.death)
            if not births or not deaths:
                raise ValueError(
                    "No bars of sufficient length found in any forest while fitting "
                    f"(min_bar_length={self.min_bar_length})."
                )
            t_min = min(births) if self.t_min is None else self.t_min
            t_max = max(deaths) if self.t_max is None else self.t_max
        else:
            t_min, t_max = self.t_min, self.t_max

        if t_max <= t_min:
            raise ValueError(f"t_max must be > t_min, got [{t_min}, {t_max}]")

        # 2. Build fixed grid and store it
        self.x_grid_ = np.linspace(t_min, t_max, self.num_grid_points)
        self.dx_ = float(self.x_grid_[1] - self.x_grid_[0])
        self.is_fitted_ = True

        # 3. Precompute feature dimension
        n_funcs = len(self.poly_funcs)
        n_samples_per_func = self.max_k * self.num_grid_points
        n_stats_per_func = 0
        if self.include_stats:
            # L1 and L2 per level => 2 * max_k
            n_stats_per_func = 2 * self.max_k
        self.n_features_ = n_funcs * (n_samples_per_func + n_stats_per_func)

        return self

    def _vectorise_single_for_func(
        self,
        lf: "LoopForest",
        f: Callable[[NDArray[np.float64]], float],
    ) -> NDArray[np.float64]:
        """
        Compute the feature block for one forest and one path function f.
        Block = [samples, (optional) stats].
        """
        assert self.x_grid_ is not None and self.dx_ is not None

        family = lf.compute_generalized_landscape_family(
            polyhedral_path_func=f,
            max_k=self.max_k,
            num_grid_points=self.num_grid_points,
            mode=self.mode,
            min_bar_length=self.min_bar_length,
            x_grid=self.x_grid_,  # enforce consistent grid
            cache=False, # Do not save family to forest
        )

        # Sample part: [λ_1(x_1..x_N), ..., λ_max_k(x_1..x_N)]
        pieces: List[NDArray[np.float64]] = []
        stats: List[float] = []

        for k in range(1, self.max_k + 1):
            if k in family.landscapes:
                ys = family.landscapes[k].ys
            else:
                ys = np.zeros_like(self.x_grid_)
            pieces.append(ys)

            if self.include_stats:
                # L1 norm ~ ∫ |λ_k| dx
                l1 = float(np.sum(np.abs(ys)) * self.dx_)
                # L2 norm ~ (∫ λ_k^2 dx)^(1/2)
                l2 = float(np.sqrt(np.sum(ys ** 2) * self.dx_))
                stats.extend([l1, l2])

        samples_flat = np.concatenate(pieces, axis=0)  # size = max_k * num_grid_points
        if self.include_stats:
            stats_arr = np.asarray(stats, dtype=float)  # size = 2 * max_k
            return np.concatenate([samples_flat, stats_arr], axis=0)
        else:
            return samples_flat

    def transform(self, forests: Sequence["LoopForest"]) -> NDArray[np.float64]:
        """
        Transform a sequence of LoopForest objects into a design matrix X.

        X has shape (n_forests, n_features_).
        """
        if not self.is_fitted_ or self.x_grid_ is None:
            raise RuntimeError("MultiLandscapeVectorizer must be fitted before calling transform().")

        n = len(forests)
        if n == 0:
            raise ValueError("transform() received an empty list of forests")

        if self.n_features_ is None:
            raise RuntimeError("n_features_ not initialised. Call fit() first.")

        X = np.zeros((n, self.n_features_), dtype=float)

        for i, lf in enumerate(forests):
            blocks: List[NDArray[np.float64]] = []
            for f in self.poly_funcs:
                block = self._vectorise_single_for_func(lf, f)
                blocks.append(block)
            X[i, :] = np.concatenate(blocks, axis=0)

        return X

    def fit_transform(self, forests: Sequence["LoopForest"]) -> NDArray[np.float64]:
        """
        Convenience method: fit the grid from forests and return X in one call.
        """
        self.fit(forests)
        return self.transform(forests)

class LoopForest:
    """Object that computes and stores progression of optimal loops for alpha complex of a point cloud in a forest format."""

    def __init__(self, 
                 point_cloud,
                 compute = True,
                 reduce: bool = True,
                 compute_barcode: bool = True,
                 print_info: bool = False) -> None:
        self.point_cloud = np.array(point_cloud) #point cloud is list of 2-dim arrays

        #check if point cloud has correct shape
        if not (self.point_cloud.ndim == 2 and self.point_cloud.shape[1] == 2):
            raise TypeError("Point cloud input has wrong shape, point cloud should be gives as array containing 2dim vectors")

        self._node_id = itertools.count(1)         #used to assign unique id to each node in the forest
        self._loop_id = itertools.count(1)
        self.nodes: Dict[int, Node] = {}
        self.loops: Dict[int, Loop] = {}

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
        #take square root of filtration value since filtation values for alpha complexes are squared in Gudhi
        for simplex, filtration in self.simplex_tree.get_filtration():
            self.simplex_tree.assign_filtration(simplex, (filtration**0.5)*2)
        
        # Extract s filtration up to order 2
        self.filtration = [(simplex,filtration) for simplex, filtration in self.simplex_tree.get_filtration() if len(simplex) <= 3]  # Keep simplices up to 2D
        filtration_time = time.perf_counter()-start
        if print_info:
            print(f"Filtration processed in {filtration_time}")

        self.barcode: set[Bar] = set()

        #compute forest
        if compute:
            self._compute_forest(reduce=reduce, compute_barcode=compute_barcode, print_info = print_info)
            self.reduced = reduce
        

    # ---------- builders ---------

    def generate_loop(self,vertex_list): #necessary to generate id for the loop
        lid= next(self._loop_id)
        return Loop(vertex_list=vertex_list, id=lid)

    def add_leaf(self, triangle: List[int], filt_val: float):
        """
        Create a new leaf in tree. 
        Corresponds to death of a loop in homology and adding a loop in algorithm for computing the LoopForest.
        """

        nid = next(self._node_id)
        lid = next(self._loop_id)

        triangle_counterclockwise = triangle_loop_counterclockwise(simplex=triangle, point_cloud=self.point_cloud) #orient vertices in triangle counterclockwise
        new_loop = Loop(vertex_list=triangle_counterclockwise, id=lid)

        new_node = Node(id=nid,filt_val=filt_val, type='leaf', children=set(), loop=new_loop)

        self.nodes[nid]=new_node
        self.loops[lid]= new_loop
        self._active_node_ids.add(nid)  #roots are active nodes in algorithm, at termination the active nodes are precicely the roots of the forest 

        self.levels.append(filt_val)

        return new_node
    
    def make_root(self, node: Node, filt_val: float):
        """
        Ends this tree in the forest by creating the root as a top node.
        Corresponds to birth in homology and removing an edge of a loop without merging it with another loop in LoopForest computation algorithm.
        """
        nid = next(self._node_id)

        self._active_node_ids.remove(node.id)
        node.parent = nid #new root node is parent of input node

        root_node = Node(id=nid, filt_val=filt_val,type='root', loop=node.loop, children={node.id})


        self.nodes[nid]=root_node
        self.roots.add(root_node.id)

        self.levels.append(filt_val)

        return

    def merge_nodes(self, node1: Node, node2: Node, parent_loop: Loop, filt_val: float):
        """ 
        Creates parent node of node1 and node2 with loop representative parent loop
        Corresponds to a loop being split into two loops in homology and a new bar appearing in barcode
        """

        nid = next(self._node_id)
        node1.parent = nid
        node2.parent = nid

        parent_node = Node(id=nid, filt_val=filt_val, type="merge", children={node1.id,node2.id}, loop = parent_loop)
        self.nodes[nid]=parent_node
        
        self._active_node_ids.add(nid)
        self._active_node_ids.remove(node1.id)
        self._active_node_ids.remove(node2.id)

        self.levels.append(filt_val)

        return
    
    def update_node(self, node: Node, updated_loop: Loop, filt_val:float):
        """ Updates loop representative, corresponds to node with one parent and one child in tree """

        nid = next(self._node_id)
        node.parent=nid

        update_node = Node(id=nid, filt_val=filt_val, type="update", children={node.id},loop=updated_loop)
        self.nodes[nid]=update_node

        self._active_node_ids.add(nid)
        self._active_node_ids.remove(node.id)

        self.levels.append(filt_val)

        return

    def merge_loops(self, edge: List[int], loop1: Loop, loop2: Loop) -> Loop:
        """ Merges two loops which both contain edge in opposing orientations"""

        nid = next(self._loop_id)

        loop_vertex_list = merge_index_list_loops(loop1 = loop1.vertex_list,loop2 = loop2.vertex_list, edge=edge)

        merged_loop = Loop(id=nid, vertex_list=loop_vertex_list)
        self.loops[nid] = merged_loop

        return merged_loop
    

    # ---- helper functions ------
    def nodes_with_loop_containing_edge(self ,edge: List[int], node_ids: set[int]) -> List[Node]:
        """
        Input: List of node ids and edge as pair of indexes

        Finds nodes where the associated loop contains the edge. Duplicates are possible if the edge appears in a loop multiple time, i.e., the node will be listed twice.

        returns list of nodes in one of the following forms: [],[l1],[l1,l2],[l1,l1]
        """

        nodes = [self.nodes[id] for id in node_ids]

        #list of nodes containing the edge "simplex"
        node_list=[]

        if len(edge)!=2:
            raise ValueError("Error, simplex is not an edge")
            return

        #check which loops contain simplex
        for node in nodes:
            vertex_loop = node.loop.vertex_list
            for i in range(len(vertex_loop)):
                if {vertex_loop[i], vertex_loop[(i+1) %  len(vertex_loop)]} == {edge[0], edge[1]}:
                    node_list.append(node)

        if len(node_list)>2:
            raise ValueError(f"Error, too many loops contain edge {edge}")

        return node_list
    

    # ----- methods to work with the forest

    def active_nodes_at(self, filt_val: float) -> List[Node]:
        """
        Return list of active nodes at a given filtration value.

        A node is active at r if:
        - node.filt_val >= r
        - has a parent and parent has filt val < r
        """
        nodes = self.nodes

        active: List[Node] = []
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

    def active_loops_at(self, filt_val: float) -> List[Loop]:
        active_nodes = self.active_nodes_at(filt_val=filt_val)
        return [node.loop for node in active_nodes]

    def leaves_below_node(self, node: Node) -> set[int]:
        """ Returns set of all leaves below a given node (below in tree means higher filtration value) """
        leaf_ids: set[int] = set()

        if node.type == "leaf":
            leaf_ids.add(node.id)
            return leaf_ids
        

        if len(node.children) == 0:
            raise ValueError(f"Node {node} is not a leaf but has no children, this should not happen")
        
        for cid in node.children:
            child = self.nodes[cid]
            leaf_ids.update(self.leaves_below_node(child))


        return leaf_ids

    def leaf_to_node_path(self, leaf: Node, node: Node) -> List[int]:
        """ 
        Returns direct path from a leaf to a node.
        Path is returned as list of node ids

        If there does not exist a path, an error is raised.
        """
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

    def node_to_leaf_path(self, leaf: Node, node: Node) -> List[int]:
        """ 
        Returns direct path from a node to a leaf.
        Path is returned as list of node ids

        If there does not exist a path, an error is raised.
        """
        return list(reversed(self.leaf_to_node_path(leaf=leaf, node=node)))

    def get_root(self, node: Node) -> Node:
        while node.parent != None:
                    pid = node.parent
                    node = self.nodes[pid]

        return node

    def _update_node_list(self, node_id_list: List[int]) -> List[Node]:
        """ Returns list of all current roots of a given list of node IDs"""
        L = [ self.get_root( self.nodes[id] ) for id in node_id_list ]
        return L


    # ----- compute the forest ----------

    def _compute_forest(self, reduce = True, compute_barcode = True, print_info: bool = False):
        """ 
        Computes LoopForest object for a point cloud.
        reduce = True means that multiple changes at the same filtration value is collapsed to a single node.
        compute_barcode = True computes barcodes and stores it in self.barcode as list of bar objects
        """

        loop_forest_start = time.perf_counter()

        edge_loop_dict = {}

        #simplices is already ordered in ascending order by number of simplices 
        for simplex, filt_val in reversed(self.filtration):

            if len(simplex)<=1:
                continue
            
            if len(simplex) == 3:
                new_node = self.add_leaf( triangle=simplex, filt_val=filt_val)

                faces = list(itertools.combinations(simplex, 2))
                for edge in faces:
                    if key(edge) in edge_loop_dict:
                        edge_loop_dict[key(edge)].append(new_node.id)
                    else:
                        edge_loop_dict[key(edge)] = [new_node.id]

            if len(simplex) == 2:
                #L is loops containing nodes, can be of the form [],[l1], [l1,l2], [l1,l1]
                #L is active nodes over L_tmp

                #If key exists, get its value and remove it
                #if key does not exists, get []
                L_tmp_ids = edge_loop_dict.pop(key(simplex), [])

                L = self._update_node_list(L_tmp_ids)


                #if no loop contains edge, nothing happens
                if len(L) == 0:
                    continue

                #if edge is only contained in a single loop and appears only once in that loop once, remove that loop from the active loops 
                elif len(L) == 1:
                    #update the loop dict for all edges which very contained in the loop we just removed
                    vertex_loop = L[0].loop.vertex_list
                    for i in range(len(vertex_loop)):

                            edge = [vertex_loop[i-1], vertex_loop[i]]


                            L_edge_tmp = edge_loop_dict.pop(key(edge), None)
                            if L_edge_tmp is None:
                                continue

                            L_edge = self._update_node_list(L_edge_tmp)

                            if len(L_edge)> 2:
                                raise ValueError("L_edge too long in loop removal process")

                            if len(L_edge)==1:
                                continue
                            elif L_edge[0] != L[0]:
                                edge_loop_dict[key(edge)] = [L_edge[0].id]
                            elif L_edge[1] != L[0]: 
                                edge_loop_dict[key(edge)] = [L_edge[1].id]
                            else:
                                continue
                                
                    self.make_root(node=L[0],filt_val=filt_val)

                    continue

                elif len(L) == 2 and L[0]!=L[1]:
                    parent_loop = self.merge_loops(loop1=L[0].loop ,loop2= L[1].loop, edge=simplex) 
                    self.merge_nodes( node1=L[0], node2=L[1], parent_loop=parent_loop, filt_val=filt_val)
                    if not loop_in_filtration_check(parent_loop.vertex_list, simplex_tree=self.simplex_tree, filt_value=filt_val):
                            print('edge', simplex)
                            print('edge dict entry')
                            print('filtration value', filt_val)
                            print(f'first loop', L[0].loop)
                            print(f'second loop', L[1].loop)
                            raise ValueError("Loop not in simplex, Loop concat Case")


                elif len(L) == 2 and L[0]==L[1]:
                    #Same edge is contained in a loop in both directions, we update the loop
                    vertex_loop, redundant_vertices = find_outer_loop(edge=simplex,
                                                    vertex_loop=L[0].loop.vertex_list, 
                                                    point_cloud=self.point_cloud)
                    if not loop_in_filtration_check(vertex_loop, simplex_tree=self.simplex_tree, filt_value=filt_val):
                                print('edge', simplex)
                                print(f'starting loop', L[0].loop)
                                print(f'outer loop', vertex_loop)
                                raise ValueError("Loop not in simplex, Tiebreak Case")
                    
                    #update dict entries for edges from the redundant loop
                    if len(redundant_vertices)>=2:
                        for i in range(len(redundant_vertices)):
                            edge = [redundant_vertices[i-1], redundant_vertices[i]]

                            L_edge_tmp = edge_loop_dict.pop(key(edge), None)
                            if L_edge_tmp is None:
                                continue

                            L_edge = self._update_node_list(L_edge_tmp)

                            if len(L_edge)> 2:
                                raise ValueError("L_edge too long in loop removal process")

                            if len(L_edge)==1:
                                continue
                            elif L_edge[0] != L[0]:
                                edge_loop_dict[key(edge)] = [L_edge[0].id]
                            elif L_edge[1] != L[0]: 
                                edge_loop_dict[key(edge)] = [L_edge[1].id]
                            else:
                                continue

                    updated_loop = self.generate_loop(vertex_loop)

                    self.update_node(node=L[0], updated_loop=updated_loop, filt_val=filt_val)

                else:
                    raise ValueError("Error, L is of the wrong form")

        loop_forest_time = time.perf_counter() - loop_forest_start
        if print_info:
            print(f"Forest succesfully computed in {loop_forest_time} sec")

        #compute where each loop is active
        self._compute_loop_activity()

        if reduce:
            self._reduce_forest(print_info = print_info)

        if compute_barcode:
            self.compute_barcode(print_info = print_info)
        
        return

    # ----- reduce forest (collapses trivial edges which happen at the same filtration value) -------------

    def _collapse_parent_child(self, parent: Node, child: Node):
        """
        Collapses a parent - child pair into the parent node.
        Intended for parent - child pair with same filtration value 
        """

        parent.children.remove(child.id)
        parent.children.update(child.children)

        #re-parent grandchildren
        for gcid in child.children:
            self.nodes[gcid].parent = parent.id

        #remove child node from forest
        del self.nodes[child.id]

        n = len(parent.children)
        #adapt type of parent node
        if n==0:
            parent.type="leaf"

            #if parent is now isolated point in forest, delete it from forest completely
            if parent.parent == None:
                del self.nodes[parent.id]
                self.roots.remove(parent.id)
        elif n==1:
            parent.type="update"
        else:
            parent.type="merge"

        return

    def _reduce_forest(self, print_info: bool = False):
        """
        Reduce the forest by collapsing every parent–child pair with equal filtration value.

        Collapse rule for an edge (parent p, child c) with p.filt_val == c.filt_val:
        - Keep the *parent* node p (its loop stays as-is).
        - Remove the child node c from the forest.
        - The parent of the resulting (collapsed) node remains p.parent (i.e., the
            parent of the original parent), if any.
        - Children of the resulting node are the union of p.children and c.children,
            minus the removed child c itself.
        - For every grandchild g in c.children, set g.parent = p.id.
        Repeats until no collapsible edges remain.
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
                continue

            for cid in p.children.copy():
                child = self.nodes[cid]
                if child.filt_val == p.filt_val:
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
    # If a merge is also a root, then the merge should be split into 2 seperate roots as the merge only lives for 0 time
    # This is not implemented yet and should not occur for points in general position

    # ------ Add active period of each loops ----------

    def _compute_loop_activity(self):
        """ Computes period in which each loop is an optimal cycle rep and adds it to the loop as attributes """

        for node in self.nodes.values():

            if node.parent == None:
                continue
            
            parent = self.nodes[node.parent]
            node.loop.active_end = node.filt_val
            node.loop.active_start = parent.filt_val


        return

    # ----- Compute barcode sequence ---------

    #recursive barcode not relevant anymore
    def _compute_tree_barcode_recursive(self, node: Node, child_id: int):
        """ Recursively computes barcode of sub-tree below the input node and the child with child_id"""

        #compute longest bar.
        choosen_child = self.nodes[child_id]
        leaf_ids = self.leaves_below_node(node=choosen_child)

        max_leaf_id = max( leaf_ids, key = lambda id: self.nodes[id].filt_val)
        max_leaf = self.nodes[max_leaf_id]

        path = self.node_to_leaf_path(node=node, leaf = max_leaf)
        if len(path)<=1:
            raise ValueError("path for barcode too short, something went wrong")

        cycle_reps = [self.nodes[id].loop for id in path[1:]]
        
        bar = Bar(birth=node.filt_val, death=max_leaf.filt_val, _node_progression=tuple(path), cycle_reps=cycle_reps )
        self.barcode.add(bar)

        #at every merge node, compute barcode of subtree of merge nodes with the other children
        for id in self.leaf_to_node_path(node=node,leaf=max_leaf)[:-2]:   #we do not want to repeat the top node of the tree
            child_node = self.nodes[id]
            pid = child_node.parent
            if pid == None:
                raise ValueError(f"Parent node incounter, this should not happen. Node id {pid}")
            parent_node = self.nodes[pid]

            if parent_node.type == "merge":
                for cid in parent_node.children:
                    if cid == id:
                        continue 
                    else:
                        self._compute_tree_barcode_recursive(node=parent_node, child_id=cid)      

    #recursive barcode not relevant anymore
    def compute_barcode_recursive(self):
        """ Computes H1 barcode of forest and stores it in self.barcode """

        print("Computing Barcode")
        barcode_start = time.perf_counter()

        #compute barcode for each tree
        for root_id in self.roots:
            for child_id in self.nodes[root_id].children:
                self._compute_tree_barcode_recursive(node=self.nodes[root_id], child_id=child_id)

        barcode_time = time.perf_counter() - barcode_start
        print(f"Barcode computation completed in {barcode_time} sec")

        return

    def compute_barcode(self, print_info: bool = False):
        """ Computes H1 barcode of forest and stores it in self.barcode """
        
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
            loop_progression = [node.loop]
            node._barcode_covered=True
            is_max_tree_bar = True
            root_id = self.get_root(node).id

            if node.parent == None:
                raise ValueError("Leaf has no Parent, this should not happen")
            else:
                parent = self.nodes[node.parent]

            #walk up forest until a root or an already _barcode_covered node is discovered
            while parent.parent is not None:
                #check if parent node has already been covered by leaf with larger filtration value
                if parent._barcode_covered == True:
                    is_max_tree_bar = False
                    break

                node_id_progession.append(parent.id)
                loop_progression.append(parent.loop)
                parent._barcode_covered = True

                #move to parent of parent
                parent = self.nodes[parent.parent]

            birth = parent.filt_val


            #reverse lists to get progression which is ascending with respect to filtration value
            bar = Bar(birth=birth,
                      death=death, 
                      _node_progression = tuple(reversed(node_id_progession)), 
                      cycle_reps=list(reversed(loop_progression)), 
                      is_max_tree_bar=is_max_tree_bar,
                      root_id=root_id)
            self.barcode.add(bar)
 

        barcode_time = time.perf_counter() - barcode_start
        if print_info:
            print(f"Barcode computation completed in {barcode_time} sec")
    
        return
         
    def max_bar(self):
        return max(self.barcode, key=lambda bar: bar.lifespan())
    
    def active_bars_at(self, filt_val:float):
        return [bar for bar in self.barcode if (bar.birth<=filt_val and bar.death>filt_val)]

    # ------ generate color scheme  ---------

    def _build_color_map_forest(self, seed: Optional[int] = 39, start_color: Optional[str] = "#ff7f0e",):
        """
        Computes a color map which assign a color to each bar in the barcode. 
        Bars in same tree will have similiar colors. 
        Saved as a dictionary {bar: "#RRGGBB"} in self.color_map_forest
        Based on json
        Seed for randomness
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
        return

    # ----- plotting tools -------

    def plot_at_filtration( 
        self,
        filt_val: float,
        ax=None,
        show: bool = True,
        fill_triangles: bool = True,
        loop_vertex_markers: bool = False,
        figsize: tuple[float, float] = (7, 7), 
        vertex_size: float = 3,
        coloring: Literal['forest','bars'] = "forest",
        title: Optional[str] = None,
    ):
        """
        Plot the 2-D point cloud, all edges/triangles with filtration <= filt_val,
        and overlay the loops of the nodes active at filt_val.

        Notes
        -----
        - GUDHI's AlphaComplex / SimplexTree work
        Pass the same units here.
        - Uses SimplexTree.get_filtration(), which is sorted by increasing filtration.

        Parameters
        ----------
        filt_val : float
            Filtration threshold.
        ax : matplotlib.axes.Axes or None
            Axes to draw on; if None, a new figure+axes are created.
        show : bool
            If True, calls plt.show() when done.
        fill_triangles : bool
            If True, lightly fill triangles present at this filtration.
        loop_vertex_markers : bool
            If True, mark the vertices used in each loop.

        Returns
        -------
        matplotlib.axes.Axes
        """
        
        
        if coloring == "forest":
            #built color map if it has not already been done
            if not hasattr(self,"color_map_forest"):
                self._build_color_map_forest()
            color_map = self.color_map_forest
        elif coloring == "bars":
            if not hasattr(self,"color_map_bars"):
                self._build_color_map_bars()
            color_map = self.color_map_bars

        # --- Prep
        pts = np.asarray(self.point_cloud, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("point_cloud must be an (n_points, 2) array-like.")

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        # --- Collect edges and triangles present at this filtration value
        edges_xy = []      # list of [[x1,y1],[x2,y2]]
        tris_xy = []       # list of [[x1,y1],[x2,y2],[x3,y3]]
        for simplex, f in self.filtration:
            if f > filt_val:
                # Filtration is sorted non-decreasing → safe to stop here
                break
            if len(simplex) == 2:  # edge
                i, j = simplex
                edges_xy.append([pts[i], pts[j]])
            elif len(simplex) == 3:  # triangle
                i, j, k = simplex
                tris_xy.append([pts[i], pts[j], pts[k]])

        # --- Base scatter
        ax.scatter(pts[:, 0], pts[:, 1], s=vertex_size, color="k", zorder=3, label="points", ec="none")

        # --- Draw triangles first (under edges)
        if fill_triangles and tris_xy:
            tri_coll = PolyCollection(
                tris_xy, closed=True, edgecolors="none", facecolors="C0", alpha=0.15, zorder=1
            )
            ax.add_collection(tri_coll)

        # --- Draw edges
        if edges_xy:
            edge_coll = LineCollection(edges_xy, linewidths=0.5, colors="0.65", zorder=2, label="edges")
            ax.add_collection(edge_coll)

        

        # --- Overlay loops from active nodes at filt_val
        for bar in self.barcode:
            if filt_val>=bar.birth and filt_val<bar.death:

                loop = bar.loop_at_filtration_value(filt_val=filt_val)
                vs = loop.vertex_list # type: ignore
                if len(vs) < 2:
                    continue

                loop_xy = pts[vs]  # shape: (m, 2)

                # >>> make a Sequence[ArrayLike] (list of 2x2 arrays) for Pylance
                segments = np.vstack(loop_xy)
                ax.add_patch(Polygon(segments, fc="none", ec=color_map[bar], lw=1, zorder=5))
                # segments = [np.array([loop_xy[i-1], loop_xy[i]]) for i in range(len(loop_xy))]

                # Thicker colored edges along the loop
                # loop_coll = LineCollection(segments, linewidths=1.8, colors=[color_map[bar]], zorder=5)
                # ax.add_collection(loop_coll)

                # Optional vertex markers for the loop
                if loop_vertex_markers:
                    ax.scatter(
                        pts[vs, 0], pts[vs, 1],
                        s=36, color="orange", edgecolors="white", linewidths=0.8, zorder=6
                    )

        # --- Aesthetics
        ax.set_aspect("equal", adjustable="box")
        if title is None:
            ax.set_title(f"α ≤ {filt_val:.4g}  •  edges/triangles in filtration + active loops")
        else:
            ax.set_title(title)
        #ax.set_xlabel("x")
        #ax.set_ylabel("y")
        # A simple legend (points + edges); loop colors are self-explanatory on top
        #handles, labels = ax.get_legend_handles_labels()
        #if handles:
        #    ax.legend(loc="lower right", frameon=True)

        ax.autoscale()  # fit collections
        if show:
            plt.show()
        return ax

    def plot_at_filtration_with_dual( 
        self,
        filt_val: float,
        ax=None,
        show: bool = True,
        fill_triangles: bool = True,
        loop_vertex_markers: bool = False,
        figsize: tuple[float, float] = (7, 7),
        vertex_size: float = 1,
        coloring: Literal['forest','bars'] = "forest",
        dual_vertex_size: float = 1,
    ):
        if coloring == "forest":
            if not hasattr(self, "color_map_forest"):
                self._build_color_map_forest()
            color_map = self.color_map_forest
        elif coloring == "bars":
            if not hasattr(self, "color_map_bars"):
                self._build_color_map_bars()
            color_map = self.color_map_bars
        else:
            raise ValueError("Unsupported coloring option. Use 'forest' or 'bars'.")

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

        ax.scatter(pts[:, 0], pts[:, 1], s=vertex_size, color="k", zorder=3, label="points", marker="o", ec="none")

        if fill_triangles and tris_present:
            tri_coll = PolyCollection(
                tris_present,
                closed=True,
                edgecolors="none",
                facecolors="C0",
                alpha=0.28,
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

        if dual_vertices_present:
            present_arr = np.array(dual_vertices_present)
            ax.scatter(
                present_arr[:, 0],
                present_arr[:, 1],
                s=dual_vertex_size,
                marker = "o",
                c="C3",
                edgecolors="none",
                zorder=2.8,
            )

        if edges_future:
            future_edge_coll = LineCollection(
                edges_future,
                linewidths=0.25,
                alpha=0.25,
                zorder=1.6,
                color='black'
            )
            ax.add_collection(future_edge_coll)

        if edges_present:
            edge_coll = LineCollection(
                edges_present,
                linewidths=0.35,
                colors="0",
                zorder=2,
                label="edges",
            )
            ax.add_collection(edge_coll)

        if dual_edges_future:
            dual_thin_coll = LineCollection(
                dual_edges_future,
                colors="C3",
                linewidths=0.25,
                alpha=0.25,
                # linestyle="dotted",
                zorder=0,
            )
            ax.add_collection(dual_thin_coll)

        if dual_edges_present:
            dual_thick_coll = LineCollection(
                dual_edges_present,
                colors="C3",
                linewidths=0.25,
                alpha=1,
                zorder=4,
                label="dual edges",
            )
            ax.add_collection(dual_thick_coll)

        for bar in self.barcode:
            if filt_val >= bar.birth and filt_val < bar.death:
                loop = bar.loop_at_filtration_value(filt_val=filt_val)
                vs = loop.vertex_list  # type: ignore
                if len(vs) < 2:
                    continue
                loop_xy = pts[vs]
                segments = np.vstack(loop_xy)
                loop_coll = Polygon(segments, edgecolor=color_map[bar], facecolor="none", linewidth=1, zorder=5)
                ax.add_patch(loop_coll)

                if loop_vertex_markers:
                    ax.scatter(
                        pts[vs, 0], pts[vs, 1],
                        s=36, color="orange", edgecolors="white", linewidths=0.8, zorder=6
                    )

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"α = {filt_val:.4g}")
        # handles, labels = ax.get_legend_handles_labels()
        # if handles:
        #     ax.legend(loc="lower right", frameon=True)

        # ax.autoscale()
        return ax

    def plot_dendrogram(
        self,
        ax=None,
        show: bool = True,
        annotate_ids: bool = False,
        leaf_spacing: float = 1.0,
        tree_gap_leaves: int = 1,
        check_reduced: bool = True,
        small_on_top: bool = False,
        threshold: float = 0.0
    ):
        import warnings

        if not self.nodes:
            raise ValueError("LoopForest has no nodes to plot.")

        all_nodes = self.nodes
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
                ax.set_title(f"LoopForest dendrogram (y = filt_val) — no trees exceed threshold {threshold}")
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
        title = "LoopForest dendrogram (y = filt_val)"
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

   
    def plot_barcode(
        self,
        *,
        ax=None,
        sort: str | None = "birth",   # "length" | "birth" | "death" | None
        title: str = "Barcode",
        xlabel: str = "filtration value",
        coloring: Literal["forest", "bars","none"] = "forest",
        max_bars: int = 0,
        min_bar_length: float = 0.0,
        **kwargs
    ):
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
        coloring : {"forest","bars"}
            Which color scheme to use:
            - "forest": use self.color_map_forest (tree-structured colors).
            - "bars":   use self.color_map_bars (ignores tree structure).
            If the chosen color map does not exist yet, it is built as in
            `plot_at_filtration`.
        max_bars : int
            If > 0, display at most this many bars, keeping the longest ones
            (by lifespan). 0 means show all bars.
        min_bar_length : float
            Filter out bars with lifespan < min_bar_length before plotting.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes the barcode was drawn on.
        """
        import math
        import numpy as np
        import matplotlib.pyplot as plt

        if not getattr(self, "barcode", None):
            raise ValueError("No bars to plot: `self.barcode` is empty.")

        # ---- Prepare color map (same logic as plot_at_filtration) ----
        if coloring == "forest":
            if not hasattr(self, "color_map_forest"):
                self._build_color_map_forest()
            color_map = self.color_map_forest
        elif coloring == "bars":
            if not hasattr(self, "color_map_bars"):
                self._build_color_map_bars()
            color_map = self.color_map_bars
        elif coloring =="none":
            color_map = {}
        else:
            raise ValueError("Invalid Coloring input")

        # ---- Work on a copy so we don't mutate original order ----
        bars = list(self.barcode)

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
            bars.sort(key=lambda b: (b.birth, b.death))
        elif sort == "death":
            def dkey(b):
                d = b.death
                return (math.inf if not math.isfinite(d) else d, b.birth)
            bars.sort(key=dkey)
        elif sort == "length":
            def length(b):
                d = b.death
                d_val = math.inf if not math.isfinite(d) else d
                return d_val - b.birth
            bars.sort(key=length, reverse=True)
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
        births = np.array([b.birth for b in bars], dtype=float)
        deaths = np.array([b.death for b in bars], dtype=float)
        finite_deaths = deaths[np.isfinite(deaths)]

        xmin = float(np.nanmin(births))
        if finite_deaths.size:
            xmax = float(np.nanmax(finite_deaths))
        else:
            xmax = float(np.nanmax(births))

        if not np.isfinite(xmax):  # extreme corner case
            xmax = xmin

        pad = (xmax - xmin) * 0.05 if xmax > xmin else 1.0
        ax.set_xlim(xmin - pad, xmax + pad)

        # ---- Draw segments ----
        for i, b in enumerate(bars):
            x0, x1 = float(b.birth), float(b.death)

            # Guard against inverted bars due to numerical issues
            if math.isfinite(x1) and x1 < x0:
                x0, x1 = x1, x0

            color = color_map.get(b, None)
            line_kwargs = {
                "y": i,
                "xmin": x0,
                "xmax": x1 if math.isfinite(x1) else ax.get_xlim()[1] - 0.25 * pad,
                "linewidth": 3.0, 
                **kwargs # thicker bars
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
        ax.set_ylim(-1, n_bars)  # keep bars nicely framed
        fig.tight_layout()

        # If we created the axes, show it immediately (so this works in scripts)
        if created_ax:
            import matplotlib.pyplot as plt
            plt.show()

        return ax

   
    def plot_loops(
            self, 
            vertex_loops: list[list[int]], 
            title = None):
        """
        Plot a 2D point cloud and a set of closed loops (polylines) over it.

        Parameters
        ----------

        loops : list[list[int]] or list[np.ndarray]
            Each loop is a list of indices into `point_cloud`. The path is closed:
            the last vertex connects back to the first.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
            Handles to the created figure and axes.
        """

        point_cloud = self.point_cloud

        if not isinstance(point_cloud, np.ndarray):
            point_cloud = np.asarray(point_cloud)
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 2:
            raise ValueError("point_cloud must be an array of shape (N, 2).")

        fig, ax = plt.subplots()
        fig.set_size_inches(10,10)
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], s=2, c="k", alpha=0.8, label="points")

        # Color map to distinguish loops
        colors = sns.color_palette("tab20", len(vertex_loops))

        for i, loop in enumerate(vertex_loops or []):
            if loop is None or len(loop) == 0:
                continue
            idx = np.asarray(loop, dtype=int)

            # Basic validation: ensure indices are in range
            if np.any(idx < 0) or np.any(idx >= len(point_cloud)):
                raise IndexError(f"Loop {i} contains out-of-range indices.")

            # Close the loop by appending the first index at the end
            closed_idx = np.concatenate([idx, idx[:1]])
            xy = point_cloud[closed_idx]

            color = colors[i]
            ax.plot(xy[:, 0], xy[:, 1], "-", lw=2, color=color, label=f"loop {i+1}")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if title is None:
            ax.set_title("Point Cloud with Loops")
        else:
            ax.set_title(title) 

        ax.grid(True, linestyle=":", alpha=0.5)

        plt.show()

        return fig, ax

    # --------- animation -------------

    def animate_filtration(
        self,
        filename: Optional[str] = None,
        *,
        fps: int = 20,
        frames: int = 200,
        coloring: Literal["forest", "bars"] = "forest",
        with_barcode: bool = False,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        dpi: int = 200,
        cloud_figsize: tuple[float, float] = (6.0, 6.0),
        total_figsize: Optional[tuple[float, float]] = None,
        plot_kwargs: Optional[dict] = None,
        barcode_kwargs: Optional[dict] = None,
    ):
        """
        Create an animation of the loop forest over the filtration.

        Parameters
        ----------
        filename : str | None, optional
            If given, the animation is written to this path.
        fps : int, optional
            Frames per second for the saved animation.
        frames : int, optional
            Number of time steps (frames) sampled between t_min and t_max.
        with_barcode : bool, optional
            If True, show a second panel with the barcode and a moving vertical
            line indicating the current filtration value.
        t_min, t_max : float | None, optional
            Optional lower/upper bounds on the filtration values to animate.
        dpi : int, optional
            DPI for saving the animation.
        cloud_figsize : (float, float), optional
            Size of the point-cloud panel when with_barcode=False.
        total_figsize : (float, float) | None, optional
            Total figure size when with_barcode=True. If None, a reasonable
            default (10, 5) is used.
        plot_kwargs : dict | None, optional
            Extra keyword arguments forwarded to ``plot_at_filtration``.
            For example::
                plot_kwargs=dict(
                    fill_triangles=True,
                    loop_vertex_markers=False,
                    vertex_size=3,
                    coloring="forest",
                )
        barcode_kwargs : dict | None, optional
            Extra keyword arguments forwarded to ``_plot_barcode`` **except**
            ``ax`` and ``coloring``, which are managed by this method.
            For example::
                barcode_kwargs=dict(
                    max_bars=150,
                    min_bar_length=1e-3,
                    sort="length",
                    title="Barcode",
                )

        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            The created animation. If ``filename`` is not None, the animation
            is also saved to disk.
        fig : matplotlib.figure.Figure
            The figure on which the animation is drawn.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, FFMpegWriter

        if not hasattr(self, "filtration") or not self.filtration:
            raise ValueError("LoopForest has no filtration data to animate.")

        # Optional restriction to a time window
        if t_min is None:
            t_min = 0.0
        if t_max is None:
            t_max = max(node.filt_val for node in self.nodes.values())

        # Uniformly spaced in filtration value → uniform speed
        frame_times = np.linspace(t_min, t_max, frames).tolist()

        # ---- Common kwargs for plot_at_filtration (cloud panel) ----
        if plot_kwargs is None:
            plot_kwargs = {}
        # Reasonable defaults (only used if not explicitly overridden)
        plot_kwargs = {
            "fill_triangles": True,
            "loop_vertex_markers": False,
            "vertex_size": 3,
            "coloring": coloring,
            "show": False,   # important: we manage the figure ourselves
            **plot_kwargs,
        }

        # ---- Barcode kwargs & shared color dict ----
        if with_barcode:
            if total_figsize is None:
                total_figsize = (10.0, 5.0)

            fig, (ax_cloud, ax_bar) = plt.subplots(
                1, 2,
                figsize=total_figsize,
                gridspec_kw={"width_ratios": [3, 2]},
            )

            # Draw the (static) barcode once
            if not getattr(self, "barcode", None):
                raise ValueError("`with_barcode=True` but `self.barcode` is empty.")

            if barcode_kwargs is None:
                barcode_kwargs = {}
            # Do not let the caller override ax or coloring here
            barcode_kwargs = {
                k: v for k, v in barcode_kwargs.items()
                if k not in {"ax", "coloring"}
            }
            # Defaults for the barcode panel – user can override sort/title/xlabel
            barcode_kwargs = {
                "sort": "length",
                "title": "Barcode",
                "xlabel": "filtration value",
                **barcode_kwargs,
            }
            # Enforce *same* coloring as in plot_at_filtration
            barcode_kwargs["coloring"] = plot_kwargs[coloring]

            self.plot_barcode(
                ax=ax_bar,
                **barcode_kwargs,
            )

            # Vertical line that will move with the filtration
            current_t0 = frame_times[0]
            barcode_line = ax_bar.axvline(current_t0, color="k", linewidth=2)
        else:
            fig, ax_cloud = plt.subplots(figsize=cloud_figsize)
            ax_bar = None
            barcode_line = None

        # ---- Helper to draw a single frame on the cloud panel ----
        def _draw_frame_at_time(t: float):
            """Helper: clear the cloud axis and redraw for filtration value t."""
            ax_cloud.clear()
            # Delegate the heavy lifting to the existing helper
            self.plot_at_filtration(filt_val=t, ax=ax_cloud, **plot_kwargs)

            # Optional: overlay a small text box with the current filtration value.
            # Comment this out if you prefer only the built-in title.
            ax_cloud.text(
                0.02, 0.98, rf"$\alpha = {t:.3g}$",
                transform=ax_cloud.transAxes,
                va="top", ha="left",
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

        # ---- Animation callbacks ----
        def init():
            _draw_frame_at_time(frame_times[0])
            if with_barcode and barcode_line is not None:
                t0 = frame_times[0]
                barcode_line.set_xdata([t0, t0])
            return []

        def update(frame_idx: int):
            t = frame_times[frame_idx]
            _draw_frame_at_time(t)
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
            fname = str(filename)
            ext = fname.lower().rsplit(".", 1)[-1] if "." in fname else ""
            if ext == "mp4":
                writer = FFMpegWriter(fps=fps, bitrate=2000)
                anim.save(fname, writer=writer, dpi=dpi)
            elif ext in {"gif", "gifv"}:
                try:
                    from matplotlib.animation import PillowWriter
                except ImportError as e:  # optional dependency
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

    #------- generalized landscape ----------------

    Interval = Tuple[float, float]
    Item = Tuple[Interval, Loop]


    #v1
    def _build_piecewise_function(
        self,
        bar: Bar,
        polyhedral_path_func: Callable[[NDArray], float],
        baseline: float = 0.0,
        *,
        eps: float = 1e-12
    ) -> Callable[[float], float]:
        """
        Precompute a non-overlapping (except boundaries) piecewise-constant function.
        
        Returns f(x) that equals interesting_quantity(obj) on [a, b] for each interval [a,b],
        and `baseline` elsewhere.
        """

        if bar not in self.barcode:
            raise ValueError("Bar is not in barcode of loop forest")

        # Build arrays for binary search
        starts = [loop.active_start for loop in bar.cycle_reps]           # ascending
        ends   = [loop.active_end for loop in bar.cycle_reps]           # same order
        vals   = [polyhedral_path_func(self.point_cloud[loop.vertex_list]) for loop in bar.cycle_reps]

        def f(x: float) -> float:
            # Find rightmost interval whose start <= x
            i = bisect.bisect_right(starts, x) - 1 # pyright: ignore[reportArgumentType]
            if i >= 0 and x <= ends[i] + eps:         # pyright: ignore[reportOptionalOperand] # include right boundary
                # If x == start of i+1 (touching boundary), bisect_right ensures we choose the left interval.
                return vals[i]
            return baseline

        return f

    #v2
    def _build_step_function_data(
        self,
        bar: "Bar",
        polyhedral_path_func: Callable[[NDArray[np.float64]], float],
        baseline: float = 0.0,
    ) -> StepFunctionData:
        """
        Build the piecewise-constant function associated to a single bar:

        For each cycle representative 'loop' in bar.cycle_reps, we create an interval
        [loop.active_start, loop.active_end] on which the function takes value
        polyhedral_path_func(point_cloud[loop.vertex_list]).

        Returns a StepFunctionData object.
        """
        if bar not in self.barcode:
            raise ValueError("Bar is not in barcode of loop forest")

        starts = np.array(
            [loop.active_start for loop in bar.cycle_reps],
            dtype=float,
        )
        ends = np.array(
            [loop.active_end for loop in bar.cycle_reps],
            dtype=float,
        )
        vals = np.array(
            [polyhedral_path_func(self.point_cloud[loop.vertex_list]) for loop in bar.cycle_reps],
            dtype=float,
        )

        if starts.size == 0:
            # Degenerate: no representatives
            domain = (float(bar.birth), float(bar.death))
        else:
            domain = (float(starts.min()), float(ends.max()))

        return StepFunctionData(
            starts=starts,
            ends=ends,
            vals=vals,
            baseline=baseline,
            domain=domain,
            metadata={
                "bar_birth": bar.birth,
                "bar_death": bar.death,
                "root_id": getattr(bar, "root_id", None),
            },
        )
    

    def _build_convolution_with_indicator(
            self,
            starts: List[float],
            ends: List[float],
            vals: List[float],
            a: float,
            b: float,
            *,
            tol: float = 1e-12
    ) -> Tuple[Callable[[float], float], List[float], List[float]]:
        """
        Compute h(x) = (f * 1_[a,b])(x) where:
        - f is piecewise-constant: f(t) = vals[i] on [starts[i], ends[i]] and 0 elsewhere
        - 1_[a,b] is the indicator of [a, b] (assumes a <= b)

        Returns:
        (h, xs, ys)
            h  : Callable that evaluates the convolution in O(log n) time via linear interpolation.
            xs : Sorted x 'knot' locations where the slope can change (event points).
            ys : The exact h(x) values at those knots (piecewise-linear nodes).

        Correctness sketch:
        h'(x) = f(x-a) - f(x-b). Each interval [s,e] of f contributes slope jumps of ±v
        at x ∈ {a+s, a+e, b+s, b+e}. Between events, h has constant slope, hence is linear.
        We start from 0 at the left boundary (the sliding window is fully left of support).

        Runtime:
        Building events: O(n)
        Sorting events:  O(n log n) with at most 4n unique points
        One sweep:       O(n)
        Evaluating h(x): O(log n) per query (binary search on xs)
        """
        if b < a:
            raise ValueError("Require a <= b for the indicator [a,b].")
        if not (len(starts) == len(ends) == len(vals)):
            raise ValueError("starts, ends, vals must have equal length.")
        n = len(starts)
        if n == 0:
            # No support: convolution is identically zero
            def h_zero(_: float) -> float: return 0.0
            return h_zero, [], []

        # Helper to merge nearly identical event keys to improve numerical stability
        def _quantize(x: float) -> float:
            if tol <= 0:
                return x
            # snap to nearest multiple of tol
            return round(x / tol) * tol

        # Build slope-change "events"
        events = {}  # x -> delta_slope
        def add_event(x: float, delta: float):
            qx = _quantize(x)
            events[qx] = events.get(qx, 0.0) + float(delta)

        for s, e, v in zip(starts, ends, vals):
            if e < s:
                raise ValueError(f"Encountered interval with end < start: [{s}, {e}].")
            if abs(v) < tol or abs(e - s) < tol:
                # Zero value or degenerate interval contributes nothing
                continue
            add_event(a + s, +v)
            add_event(a + e, -v)
            add_event(b + s, -v)
            add_event(b + e, +v)

        if not events:
            def h_zero(_: float) -> float: return 0.0
            return h_zero, [], []

        xs = sorted(events.keys())
        # We know h(x)=0 for x < xs[0] (the window [x-b, x-a] is fully left of f’s support).
        ys: List[float] = [0.0]

        # Sweep: on (xs[i], xs[i+1]) the slope is constant; update slope at the left endpoint.
        slope = 0.0
        slopes_per_interval: List[float] = []
        for i in range(len(xs) - 1):
            x_i, x_next = xs[i], xs[i + 1]
            slope += events[x_i]              # slope just to the right of x_i
            slopes_per_interval.append(slope) # slope on (x_i, x_{i+1})
            y_next = ys[-1] + slope * (x_next - x_i)
            ys.append(y_next)

        # Consume the last event to bring slope back (should return to ~0)
        slope += events[xs[-1]]
        # Optional check (tolerant to rounding)
        if not math.isclose(slope, 0.0, rel_tol=1e-9, abs_tol=1e-9):
            # Not fatal; tiny residue can appear from floating noise
            pass

        # Build a fast evaluator by linear interpolation on the piecewise-linear segments
        def h(x: float) -> float:
            if not xs:
                return 0.0
            if x <= xs[0] or x >= xs[-1]:
                return 0.0
            i = bisect_right(xs, x) - 1
            # segment i is (xs[i], xs[i+1]) with slope slopes_per_interval[i]
            return ys[i] + slopes_per_interval[i] * (x - xs[i])

        return h, xs, ys

    def compute_convolution_kernel_for_bar(
        self,
        bar: "Bar",
        polyhedral_path_func: Callable[[NDArray[np.float64]], float],
        *,
        tol: float = 1e-12,
    ) -> PiecewiseLinearFunction:
        """
        Use the existing build_convolution_with_indicator to compute

            g(x) = (f * 1_[birth, death])(x)

        where f is the piecewise-constant function from build_step_function_data.
        Returns g as a PiecewiseLinearFunction.
        """
        sf = self._build_step_function_data(bar, polyhedral_path_func, baseline=0.0)

        a, b = bar.birth, bar.death
        h, xs, ys = self._build_convolution_with_indicator(
            sf.starts.tolist(),
            sf.ends.tolist(),
            sf.vals.tolist(),
            a,
            b,
            tol=tol,
        )

        xs_arr = np.asarray(xs, dtype=float)
        ys_arr = np.asarray(ys, dtype=float)

        if xs_arr.size > 1:
            domain = (float(xs_arr[0]), float(xs_arr[-1]))
        else:
            domain = (float(a), float(b))

        return PiecewiseLinearFunction(
            xs=xs_arr,
            ys=ys_arr,
            domain=domain,
            metadata={
                **sf.metadata,
                "poly_func_name": getattr(polyhedral_path_func, "__name__", "anonymous"),
                "kernel_type": "raw_convolution",
            },
        )

    def compute_generalized_interval_landscape(
        self,
        polyhedral_path_func: Callable[[NDArray[np.float64]], float],
        bar: Optional["Bar"] = None,
    ):
        """
        Backwards-compatible helper that computes the (raw) convolution kernel g
        for a single bar and returns (h, xs, ys) as before, but implemented via
        the new PiecewiseLinearFunction.

        Note: this uses the *raw* convolution (no pyramid rescaling).
        For landscapes, prefer compute_generalized_landscape_family.
        """
        if bar is None:
            bar = self.max_bar()

        if bar not in self.barcode:
            raise ValueError("Bar is not in barcode of loop forest")

        kernel = self.compute_convolution_kernel_for_bar(
            bar,
            polyhedral_path_func,
        )

        xs = kernel.xs.tolist()
        ys = kernel.ys.tolist()

        def h(x: float) -> float:
            return float(kernel(x))

        return h, xs, ys

    def compute_landscape_kernel_for_bar(
        self,
        bar: "Bar",
        polyhedral_path_func: Callable[[NDArray[np.float64]], float],
        *,
        mode: Literal["raw", "pyramid"] = "pyramid",
        tol: float = 1e-12,
    ) -> PiecewiseLinearFunction:
        """
        Build the kernel used for landscapes for a single bar.

        - mode="raw":      return g(x) = (f * 1_[birth,death])(x)
        - mode="pyramid":  return λ(x) = 1/2 * g(2x)

        The "pyramid" mode should reproduce the usual persistence landscape
        shape when f ≡ 1.
        """
        raw_kernel = self.compute_convolution_kernel_for_bar(
            bar,
            polyhedral_path_func,
            tol=tol,
        )

        if mode == "raw":
            # Just return a shallow copy with explicit metadata
            return PiecewiseLinearFunction(
                xs=raw_kernel.xs.copy(),
                ys=raw_kernel.ys.copy(),
                domain=raw_kernel.domain,
                metadata={
                    **raw_kernel.metadata,
                    "kernel_type": "raw_convolution",
                },
            )

        # mode == "pyramid"
        xs = raw_kernel.xs
        ys = raw_kernel.ys

        # λ(x) = 1/2 * g(2x)
        xs_scaled = xs / 2.0
        ys_scaled = ys / 2.0

        domain = (float(xs_scaled[0]), float(xs_scaled[-1])) if xs_scaled.size > 1 else raw_kernel.domain

        return PiecewiseLinearFunction(
            xs=xs_scaled,
            ys=ys_scaled,
            domain=domain,
            metadata={
                **raw_kernel.metadata,
                "kernel_type": "pyramid_rescaled",
                "rescaling_formula": "lambda(x) = 0.5 * g(2x)",
            },
        )

    def compute_generalized_landscape_family(
        self,
        polyhedral_path_func: Callable[[NDArray[np.float64]], float],
        *,
        max_k: int = 5,
        num_grid_points: int = 512,
        mode: Literal["raw", "pyramid"] = "pyramid",
        label: Optional[str] = None,
        min_bar_length: float = 0.0,
        x_grid: Optional[NDArray[np.float64]] = None,
        cache: bool = True,
    ) -> GeneralizedLandscapeFamily:
        """
        Compute the generalized landscape family for this LoopForest for a given
        polyhedral_path_func.

        If x_grid is provided, all landscapes are evaluated on that grid
        (and num_grid_points is ignored). This is crucial for consistent
        vectorisations across multiple LoopForest objects.

        Parameters
        ----------
        polyhedral_path_func:
            Function f(points_of_loop) -> scalar.
        max_k:
            Number of landscapes λ_1, ..., λ_max_k to compute.
        num_grid_points:
            Number of grid points used when x_grid is None.
        mode:
            "raw" or "pyramid" (see compute_landscape_kernel_for_bar).
        label:
            Key used to store the family in self.landscape_families.
        min_bar_length:
            Ignore bars with (death - birth) < min_bar_length.
        x_grid:
            Optional fixed grid on which to evaluate the landscapes.
        cache:
            Decides if landscapes is saved to loop forest or only returned.
        """
        if not hasattr(self, "barcode"):
            raise AttributeError("LoopForest has no 'barcode' attribute. Did you compute it?")

        # 1. Filter bars by length
        bars = [
            bar for bar in self.barcode
            if (bar.death - bar.birth) >= min_bar_length
        ]

        if not bars:
            raise ValueError(
                f"No bars with length >= {min_bar_length}. "
                "Increase min_bar_length or check your barcode."
            )

        # 2. Compute kernels for each bar
        bar_kernels: Dict[int, PiecewiseLinearFunction] = {}
        global_min_x = float("inf")
        global_max_x = float("-inf")

        for i, bar in enumerate(bars):
            kernel = self.compute_landscape_kernel_for_bar(
                bar,
                polyhedral_path_func,
                mode=mode,
            )
            bar_kernels[i] = kernel

            global_min_x = min(global_min_x, kernel.domain[0])
            global_max_x = max(global_max_x, kernel.domain[1])

        if not np.isfinite(global_min_x) or not np.isfinite(global_max_x):
            raise RuntimeError("Failed to infer a finite global domain for the kernels.")

        if global_max_x <= global_min_x:
            raise RuntimeError(
                f"Non-positive domain width: [{global_min_x}, {global_max_x}]"
            )

        # 3. Common grid
        if x_grid is None:
            x_grid = np.linspace(global_min_x, global_max_x, num_grid_points)
        else:
            x_grid = np.asarray(x_grid, dtype=float)
            if x_grid.ndim != 1 or x_grid.size < 2:
                raise ValueError("x_grid must be a 1D array with at least 2 points")
        num_grid_points = x_grid.size  # ensure consistency

        # 4. Evaluate all kernels on the grid
        n_bars = len(bars)
        values = np.zeros((n_bars, num_grid_points), dtype=float)
        for i, kernel in bar_kernels.items():
            values[i, :] = kernel(x_grid)

        # 5. Compute order statistics along the bar axis
        # sorted_vals[k, j] = (k+1)-th largest value at x_grid[j]
        sorted_vals = np.sort(values, axis=0)[::-1, :]  # descending along axis 0
        max_possible_k = sorted_vals.shape[0]

        landscapes: Dict[int, PiecewiseLinearFunction] = {}
        for k in range(1, max_k + 1):
            if k <= max_possible_k:
                y_k = sorted_vals[k - 1, :]
            else:
                # If we ask for more landscapes than bars, pad with zeros.
                y_k = np.zeros(num_grid_points, dtype=float)

            landscapes[k] = PiecewiseLinearFunction(
                xs=x_grid.copy(),
                ys=y_k.copy(),
                domain=(float(x_grid[0]), float(x_grid[-1])),
                metadata={
                    "k": k,
                    "poly_func_name": getattr(polyhedral_path_func, "__name__", "anonymous"),
                    "mode": mode,
                    "min_bar_length": min_bar_length,
                },
            )

        # 6. Assemble family object
        loop_forest_id = getattr(self, "name", f"LoopForest@{id(self)}")

        family = GeneralizedLandscapeFamily(
            loop_forest_id=loop_forest_id,
            poly_func_name=getattr(polyhedral_path_func, "__name__", "anonymous"),
            rescaling=mode,
            x_grid=x_grid,
            bar_kernels=bar_kernels,
            landscapes=landscapes,
            extra_meta={
                "num_grid_points": num_grid_points,
                "min_bar_length": min_bar_length,
                "global_min_x": global_min_x,
                "global_max_x": global_max_x,
            },
            loop_forest=self,
        )

        # Cache on the LoopForest instance
        if cache:
            if not hasattr(self, "landscape_families"):
                self.landscape_families: Dict[str, GeneralizedLandscapeFamily] = {}

            key = label or family.poly_func_name
            self.landscape_families[key] = family

        return family

    # ------- landscape plotting tools ----------

    def plot_barcode_measurement(
            self,
            polyhedral_path_func: Callable[[NDArray], float],
            bar: Optional[Bar] = None,
            x_range: Optional[Tuple[float, float]] = None,
            y_range: Optional[Tuple[float, float]] = None,
            *,
            baseline: float = 0.0,
            ax = None,
            linewidth: float = 2.0,
            label: Optional[str] = None,
        ):
        """
        Plot the step graph of the piecewise-constant function defined by `items`.
        - `x_range=(xmin, xmax)` restricts the plot; if None, it's inferred from the data.
        - Regions not covered by any interval are drawn at `baseline` (default 0).
        If no bar is given, bar with max length is used.
        """
        if bar is None:
            bar = self.max_bar()

        if bar not in self.barcode:
            raise ValueError("Bar is not in barcode of loop forest")

        # Build the evaluator
        f = self._build_piecewise_function(bar, polyhedral_path_func, baseline)

        if x_range is not None:
            xmin, xmax = map(float, x_range)
            if xmax <= xmin:
                raise ValueError("x_range must be (xmin, xmax) with xmax > xmin.")
        else:
            start= bar.cycle_reps[0].active_start
            end = bar.cycle_reps[-1].active_end
            xmin = start - 0.05 * (end - start)
            xmax = end + 0.05 * (end - start)

        # compute break points of piecewise constant function
        breaks = {xmin, xmax} #Using breaks as set and then sorting is inefficient since bar is already sorted but I currently dont care
        for loop in bar.cycle_reps:
            # Only keep endpoints that intersect the chosen window
            if loop.active_end >= xmin and loop.active_start <= xmax:
                breaks.add(max(loop.active_start, xmin))
                breaks.add(min(loop.active_end, xmax))

        xs = sorted(breaks)
        if len(xs) < 2:
            # Degenerate range; still produce an empty baseline line
            xs = [xmin, xmax]

        # Compute y on each half-open slice [x_k, x_{k+1})
        ys = []
        for i in range(len(xs) - 1):
            mid = (xs[i] + xs[i + 1]) * 0.5
            ys.append(f(mid))
        # Append last y to match step-api length
        ys.append(ys[-1] if ys else baseline)

        # Plot
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.step(xs, ys, where="post", linewidth=linewidth, label=label or "piecewise")
        ax.set_xlim(xmin, xmax)
        if y_range is not None:
            ax.set_ylim(y_range[0],y_range[1])
        #ax.axhline(baseline, linestyle="--", linewidth=1)
        ax.set_xlabel("$r$")
        # ax.set_ylabel("value")
        if label:
            ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_convolution(
            self,
            starts: List[float],
            ends: List[float],
            vals: List[float],
            a: float,
            b: float,
            *,
            tol: float = 1e-12,
            ax: Optional["matplotlib.axes.Axes"] = None,
            title: Optional[str] = None,
        ):
        """
        Convenience plotting wrapper. Uses the exact piecewise-linear knots (no sampling).

        Returns (ax, (h, xs, ys)) so you can reuse the callable and knots.
        """
        import matplotlib.pyplot as plt

        h, xs, ys = self._build_convolution_with_indicator(starts, ends, vals, a, b, tol=tol)

        if ax is None:
            fig, ax = plt.subplots()

        if xs:
            ax.plot(xs, ys, linewidth=2)
            # Optional: show zero tails to make the shape clear
            left_tail = xs[0] - 0.05 * (xs[-1] - xs[0])
            right_tail = xs[-1] + 0.05 * (xs[-1] - xs[0])
            ax.plot([left_tail, xs[0]], [0.0, 0.0], linestyle="--", linewidth=1)
            ax.plot([xs[-1], right_tail], [ys[-1], 0.0], linestyle="--", linewidth=1)
        else:
            # identically zero
            ax.axhline(0.0, linestyle="--", linewidth=1)

        ax.set_xlabel("$r$")
        # ax.set_ylabel(r"$f * \mathbf{1}_{[a,b]})")
        if title is not None:
            ax.set_title(title)
        ax.grid(True, alpha=0.3)

        return ax, (h, xs, ys)

    #v1
    def plot_generalized_interval_landscape(self,  polyhedral_path_func: Callable[[NDArray],float], bar: Optional[Bar]=None,
                                            ax: Optional["matplotlib.axes.Axes"] =None, title: Optional[str] = None):

        if bar is None:
            bar = self.max_bar()

        if bar not in self.barcode:
            raise ValueError("Bar is not in barcode of loop forest")

        starts = [loop.active_start for loop in bar.cycle_reps]
        ends   = [loop.active_end   for loop in bar.cycle_reps]
        vals   = [polyhedral_path_func(self.point_cloud[loop.vertex_list]) for loop in bar.cycle_reps]
        a  = bar.birth # + bar.lifespan()/4
        b =  bar.death # - bar.lifespan()/4

        ax, (h, xs, ys) = self.plot_convolution(starts, ends, vals, a, b, title = title, ax = ax)

        return ax, (h, xs, ys)

    #v1
    def plot_landscape_subplots(self, 
                                polyhedral_path_funcs: List[Callable[[NDArray],float]],
                                bar: Optional[Bar] = None,
                                column_titles: Optional[List[str]] = None,
                                xrange_dict: Optional[ dict[tuple[int,int], tuple[float,float]] ]= None,
                                yrange_dict: Optional[ dict[tuple[int,int], tuple[float,float]] ] = None,
                                figsize = None
                                ):
        """
        Plot point cloud, barcode measurement and generalized barcode of input bar and measurement function in polyhedral_path_functions
        
        If no bar is given, bar with max length is used.
        xrange_dict and yrange_dict are optional paramters to fix axis size of indiviual subplots
        column_title can be used to specifiy titles of columns for the measurement functions. If column_title=None then function names are used as column titles
        """

        if bar is None:
            bar = self.max_bar()
        
        if bar not in self.barcode:
            raise ValueError("Bar is not in barcode of loop forest")
        if figsize is None:
            figsize = (4*(len(polyhedral_path_funcs)+1),8)

        fig, axes = plt.subplots(nrows = 2, ncols= len(polyhedral_path_funcs)+1, figsize=figsize)

        axes[0,0].scatter(self.point_cloud[:,0], self.point_cloud[:,1], s=1)
        axes[0,0].set_aspect('equal')

        axes[1,0].axis("off")
        axes[0,0].set_title("point cloud")


        for i in range(0, len(polyhedral_path_funcs)):
            self.plot_barcode_measurement(bar=bar,polyhedral_path_func=polyhedral_path_funcs[i], ax=axes[0,i+1])
            self.plot_generalized_interval_landscape(bar = bar, polyhedral_path_func=polyhedral_path_funcs[i], ax = axes[1,i+1])
            
            axes[0,i+1].set_xlabel(None)
            if column_titles is None:
                axes[0,i+1].set_title(polyhedral_path_funcs[i].__name__)
            else:
                if not len(column_titles)==len(polyhedral_path_funcs):
                    raise ValueError("Length of column titles does not match number of input functions")
                axes[0,i+1].set_title(column_titles[i])
            
            if xrange_dict is not None:
                if (0,i+1) in xrange_dict:
                    axes[0,i+1].set_xlim(xrange_dict[(0,i+1)])
                if (1,i+1) in xrange_dict:
                    axes[1,i+1].set_xlim(xrange_dict[(1,i+1)])
            
            if yrange_dict is not None:
                if (0,i+1) in yrange_dict:
                    axes[0,i+1].set_ylim(yrange_dict[(0,i+1)])
                if (1,i+1) in yrange_dict:
                    axes[1,i+1].set_ylim(yrange_dict[(1,i+1)])
        return fig, axes

    #v2
    def plot_landscape_family(
        self,
        label: str,
        ks: Optional[List[int]] = None,
        ax: Optional["matplotlib.axes.Axes"] = None,
        title: Optional[str] = None,
    ):

        if not hasattr(self, "landscape_families"):
            raise AttributeError("No landscape_families attribute on this LoopForest")

        family = self.landscape_families[label]

        if ks is None:
            ks = sorted(family.landscapes.keys())

        if ax is None:
            fig, ax = plt.subplots()

        for k in ks:
            plf = family.landscapes[k]
            ax.plot(plf.xs, plf.ys, label=fr"$\lambda_{k}$")

        ax.set_xlabel("filtration value")
        ax.set_ylabel("landscape value")
        if title is None:
            title = f"Generalized landscapes of {label}"
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax


# --------- Generalized Landscape functions ---------------

def plot_landscape_comparison(
    loop_forests: List[LoopForest],
    label: str,
    k: int = 1,
    ax: Optional["matplotlib.axes.Axes"] = None,
    forest_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
):
    for forest in loop_forests:
        if not hasattr(forest, "landscape_families"):
            raise AttributeError(f"No landscape_families attribute on {forest}")

    families = [forest.landscape_families[label] for forest in loop_forests]

    if ax is None:
        fig, ax = plt.subplots()

    if forest_labels is None:
        forest_labels = [fam.loop_forest_id for fam in families]

    for fam, forest_label in zip(families, forest_labels):
        if k not in fam.landscapes:
            continue
        plf = fam.landscapes[k]
        ax.plot(plf.xs, plf.ys, label=forest_label)

    ax.set_xlabel("filtration value")
    ax.set_ylabel(fr"$\lambda_{k}$")
    if title is None:
        title = fr"Comparison of {label} $\lambda_{k}$ across forests (k={k})"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax

# -------- Animate Comparison -----------------

from typing import Optional, Tuple

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
    Animate two LoopForests side-by-side: for each forest, show the evolving
    cycle representatives in the point cloud together with its barcode.

    The filtration panels and barcodes are styled via `plot_at_filtration` and
    `_plot_barcode`, so they match what `animate_filtration` produces.

    Parameters
    ----------
    forest1, forest2 : LoopForest
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
            writer = FFMpegWriter(fps=fps, bitrate=2000)
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
