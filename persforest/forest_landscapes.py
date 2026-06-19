"""
forest_landscapes.py

Generic "generalized landscape" machinery that works for any forest-like
object which provides:

- forest.point_cloud: np.ndarray of shape (n_points, dim)
- forest.barcode: iterable of bar objects, where each bar has
      bar.birth: float
      bar.death: float (can be math.inf)
      bar.cycle_reps: iterable of cycle representatives

- each cycle representative has
      rep.active_start: float
      rep.active_end: float

The *only* object-specific part is how we turn a cycle representative
into a number. This is supplied by the user as:

    cycle_value_func(rep, point_cloud) -> float

For PersistenceForest, `rep` is a SignedChain with `signed_simplices`.
Other forest classes may provide their own cycle representative type as long
as the supplied `cycle_value_func` knows how to evaluate it.

"""

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, Optional, Tuple, Dict, Union, Literal,List
from numpy.typing import NDArray
import matplotlib.axes
import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_right
import math


CycleValueFunc = Callable[[Any, np.ndarray], float]
"""Takes (cycle_rep, point_cloud) and returns a scalar feature."""


@dataclass
class StepFunctionData:
    """
    Piecewise-constant function data.

    f(t) = vals[i]  on [starts[i], ends[i])
         = baseline elsewhere.

    All arrays are 1D and aligned by index. `domain` is the global range
    where the function is potentially non-zero (for convenience).

    We assume that ends[i] = starts[i+1] for all i, i.e., no gaps or overlaps between intervals.
    """
    starts: NDArray[np.float64]
    ends: NDArray[np.float64]
    vals: NDArray[np.float64]
    baseline: float
    domain: Tuple[float, float]
    metadata: Dict[str, object] = field(default_factory=dict)

    def plot(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        *,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        show_baseline: bool = False,
        title: Optional[str] = None,
        baseline_kwargs: Optional[Dict[str, Any]] = None,
        **line_kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """
        Plot the step function represented by this object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to draw on. If None, a new figure and axis are created.
        x_range : (float, float), optional
            Restrict the x-axis to the given range; defaults to the stored domain.
        y_range : (float, float), optional
            Restrict the y-axis to the given range; auto-scales when omitted.
        show_baseline : bool, default False
            Whether to draw a dashed line at `baseline`.
        title : str, optional
            Title for the plot.
        baseline_kwargs : dict, optional
            Extra keyword arguments forwarded to the baseline line plot.
        **line_kwargs :
            Keyword arguments forwarded to `ax.plot` for the step curve.

        Returns
        -------
        matplotlib.axes.Axes
            The axis with the rendered step function.
        """
        if ax is None:
            _, ax = plt.subplots()

        starts = np.asarray(self.starts, dtype=float).ravel()
        ends = np.asarray(self.ends, dtype=float).ravel()
        vals = np.asarray(self.vals, dtype=float).ravel()

        if not (starts.size == ends.size == vals.size):
            raise ValueError("starts, ends and vals must have the same length.")

        # Determine plotting domain
        if x_range is not None:
            xmin, xmax = x_range
        else:
            xmin, xmax = self.domain

        if xmax <= xmin:
            raise ValueError("x_range / domain must satisfy xmin < xmax.")

        # Build breakpoints inside [xmin, xmax]
        xs_candidates = [xmin, xmax]
        xs_candidates.extend(starts.tolist())
        xs_candidates.extend(ends.tolist())
        xs_unique = sorted({x for x in xs_candidates if xmin <= x <= xmax})

        if len(xs_unique) < 2:
            xs_unique = [xmin, xmax]

        def _value_at(t: float) -> float:                               #this is a correct but inefficient way to do it since we do not allow overlaping intervals, fix later
            """Evaluate the step function at a single point t."""
            mask = (starts <= t) & (t < ends)
            if not np.any(mask):
                return float(self.baseline)
            idx = np.nonzero(mask)[0]
            if idx.size > 1:
                # If intervals overlap, fall back to summing their values.
                return float(vals[idx].sum())
            return float(vals[idx[0]])

        xs_plot: list[float] = []
        ys_plot: list[float] = []

        for left, right in zip(xs_unique[:-1], xs_unique[1:]):
            mid = 0.5 * (left + right)
            y = _value_at(mid)
            xs_plot.extend([left, right])
            ys_plot.extend([y, y])

        # Defaults that can be overridden by **line_kwargs
        ax.plot(xs_plot, ys_plot, **line_kwargs)

        if show_baseline:
            bl_kwargs = {
                "linewidth": 1.0,
                "linestyle": "--",
            }
            if baseline_kwargs is not None:
                bl_kwargs.update(baseline_kwargs)
            ax.hlines(self.baseline, xmin, xmax, **bl_kwargs)

        # ax.grid(True, alpha=0.3)

        ax.set_xlim(xmin, xmax)
        if y_range is not None:
            ax.set_ylim(*y_range)

        if title is not None:
            ax.set_title(title)

        return ax
    
    def eval_on_grid(self, grid: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the step function on a given grid of points.

        Parameters
        ----------
        grid : numpy.ndarray
            1D array of points at which to evaluate the function.
            must be non-empty and sorted in non-decreasing order.

        Returns
        -------
        numpy.ndarray
            Array of function values at the provided grid points.
        """
        if grid.size == 0:
            raise ValueError("x_grid must be non-empty")
        if not np.all(np.isfinite(grid)):
            raise ValueError("x_grid contains non-finite values")
        if np.any(np.diff(grid) < 0):
            raise ValueError("x_grid must be sorted in non-decreasing order")

        x_grid = np.asarray(grid, dtype=float).ravel()
        y_vals = np.full_like(grid, fill_value=self.baseline, dtype=float)

        if self.starts.size == 0:
            return np.full_like(x_grid, fill_value=float(self.baseline), dtype=float)
        if not (self.starts.size == self.ends.size == self.vals.size):
            raise ValueError("starts, ends, vals must have the same length")

        i = 0
        for j,x in enumerate(x_grid):
            if x < self.starts[0]:
                continue
            if x >= self.ends[-1]:
                break

            while i < self.starts.size and x >= self.ends[i]:
                i += 1
            if i < self.starts.size and self.starts[i] <= x < self.ends[i]:
                y_vals[j] = self.vals[i]

        return y_vals

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

    def __call__(self, x: Union[NDArray[np.float64], float]):
        """
        Evaluate the piecewise-linear function at one or many points.

        Parameters
        ----------
        x : float or array-like of float
            Locations at which to sample the function.

        Returns
        -------
        float or numpy.ndarray
            Scalar if `x` is scalar, otherwise an array of the same shape as `x`.
        """
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
class BarcodeFunctionals:
    """
    Container for per-bar step functions (StepFunctionData) for a fixed cycle_func.
    Aligned with the bar ordering used in compute_generalized_landscape_family.
    """
    forest_id: str
    label: str
    baseline: float
    bars: List[Any]  # PFBar or bar-like objects
    step_functions: Dict[int, StepFunctionData]  # index -> StepFunctionData
    extra_meta: Dict[str, Any] = field(default_factory=dict)

    # built once for convenience
    _bar_to_index: Dict[Any, int] = field(init=False, repr=False)

    def __post_init__(self):
        """Build the lookup from bar object to stored row index."""
        self._bar_to_index = {bar: i for i, bar in enumerate(self.bars)}

    def get(self, bar_or_index: Union[int, Any]) -> StepFunctionData:
        """
        Return the step function for a bar object or stored integer index.
        """
        if isinstance(bar_or_index, int):
            return self.step_functions[bar_or_index]
        return self.step_functions[self._bar_to_index[bar_or_index]]

    def __getitem__(self, bar_or_index: Union[int, Any]) -> StepFunctionData:
        """Alias for ``get`` to support bracket access."""
        return self.get(bar_or_index)
    
    def evaluate_on_grid(
        self,
        grid: NDArray[np.float64],
        bars: Optional[Sequence[Union[int, Any]]] = None,
    ) -> NDArray[np.float64]:
        """Evaluate stored per-bar step functions on a common grid.

        Parameters
        ----------
        grid:
            One-dimensional, monotonically non-decreasing sample grid.
        bars:
            Optional subset of bars (bar objects) or indices to evaluate.
            If omitted, evaluates for all bars in `self.bars` order.

        Returns
        -------
        numpy.ndarray
            Array of shape (B, G) where G=len(grid) and B is the number of
            requested bars.
        """
        grid_arr = np.asarray(grid, dtype=float).ravel()
        if grid_arr.size == 0:
            raise ValueError("grid must be non-empty")
        if not np.all(np.isfinite(grid_arr)):
            raise ValueError("grid contains non-finite values")
        if np.any(np.diff(grid_arr) < 0):
            raise ValueError("grid must be sorted in non-decreasing order")

        if bars is None:
            indices = list(range(len(self.bars)))
        else:
            indices = []
            for b in bars:
                if isinstance(b, int):
                    indices.append(b)
                else:
                    indices.append(self._bar_to_index[b])

        out = np.empty((len(indices), grid_arr.size), dtype=float)
        for row, idx in enumerate(indices):
            if idx not in self.step_functions:
                raise KeyError(f"No StepFunctionData stored for bar index {idx}")
            out[row, :] = self.step_functions[idx].eval_on_grid(grid_arr)

        return out

@dataclass
class GeneralizedLandscapeFamily:
    """
    Container for a family of generalized landscapes for one forest and cycle
    functional.

    - bar_kernels: per-bar kernels (typically already rescaled if mode="pyramid")
    - landscapes: k -> λ_k (k-th landscape) as PiecewiseLinearFunction
    """
    forest_id: str 
    label: str
    rescaling: str  # "raw" or "pyramid", where "pyramid" is default
    x_grid: NDArray[np.float64]
    bar_kernels: Dict[int, PiecewiseLinearFunction]
    landscapes: Dict[int, PiecewiseLinearFunction]
    extra_meta: Dict[str, object] = field(default_factory=dict)
    
    def evaluate_on_grid(
        self,
        grid: NDArray[np.float64],
        levels: Union[int, Sequence[int]] = 1,
        *,
        fill_value: float = 0.0,
    ) -> NDArray[np.float64]:
        """Evaluate selected generalized landscape levels on a common grid.

        Parameters
        ----------
        grid:
            One-dimensional, monotonically non-decreasing sample grid.
        levels:
            If an int m is given, evaluates levels k=1..m. If a sequence is
            given, evaluates those k-values in the provided order.
        fill_value:
            Value used when a requested landscape level is unavailable.

        Returns
        -------
        numpy.ndarray
            Array of shape (L, G) where G=len(grid) and L is the number of
            requested levels.
        """
        grid_arr = np.asarray(grid, dtype=float).ravel()
        if grid_arr.size == 0:
            raise ValueError("grid must be non-empty")
        if not np.all(np.isfinite(grid_arr)):
            raise ValueError("grid contains non-finite values")
        if np.any(np.diff(grid_arr) < 0):
            raise ValueError("grid must be sorted in non-decreasing order")

        if isinstance(levels, int):
            if levels < 1:
                raise ValueError("levels must be >= 1")
            ks = list(range(1, levels + 1))
        else:
            ks = [int(k) for k in levels]
            if any(k < 1 for k in ks):
                raise ValueError("all requested landscape levels must be >= 1")

        out = np.full((len(ks), grid_arr.size), float(fill_value), dtype=float)
        for i, k in enumerate(ks):
            f = self.landscapes.get(k)
            if f is None:
                continue
            out[i, :] = np.asarray(f(grid_arr), dtype=float) # evaluate landscape f on grid with method __call__ in PiecewiseLinearFunction

        return out


def _build_step_function_data(
        forest,
        bar,
        cycle_func: CycleValueFunc,
        baseline: float = 0.0,
    ) -> StepFunctionData:
        """
        Build the piecewise-constant measurement function for one bar.

        Each cycle representative contributes an interval
        `[cycle.active_start, cycle.active_end]` on which the function takes the
        value `cycle_func(cycle, forest.point_cloud)`. Outside these intervals,
        the function equals `baseline`.

        Parameters
        ----------
        forest :
            Forest-like object providing ``point_cloud`` and ``barcode``.
        bar :
            Bar instance taken from ``forest.barcode``.
        cycle_func : CycleValueFunc
            Callable that assigns a scalar to each cycle representative.
        baseline : float, default 0.0
            Value of the function outside the active intervals.

        Returns
        -------
        StepFunctionData
            Structured representation of the piecewise-constant function.
        """
        if bar not in forest.barcode:
            raise ValueError("Bar is not in barcode of forest")

        starts = np.array(
            [cycle.active_start for cycle in bar.cycle_reps],
            dtype=float,
        )
        ends = np.array(
            [cycle.active_end for cycle in bar.cycle_reps],
            dtype=float,
        )
        vals = np.array(
            [cycle_func(cycle, forest.point_cloud) for cycle in bar.cycle_reps],
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

def compute_barcode_functionals(
    forest,
    cycle_func: CycleValueFunc,
    label: str,
    *,
    min_bar_length: float = 0.0,
    baseline: float = 0.0,
    cache: bool = True,
) -> BarcodeFunctionals:
    """
    Compute per-bar step functions for one cycle functional.

    Parameters
    ----------
    forest :
        Forest-like object with ``barcode``, ``point_cloud`` and
        ``barcode_functionals`` when caching is enabled.
    cycle_func : CycleValueFunc
        Callable ``(cycle_rep, point_cloud) -> float``.
    label : str
        Cache key and label for the resulting container.
    min_bar_length : float
        Ignore bars with lifespan below this threshold.
    baseline : float
        Value outside each bar's representative intervals.
    cache : bool
        If True, store the result in ``forest.barcode_functionals[label]``.

    Returns
    -------
    BarcodeFunctionals
        Step-function data for the selected bars.
    """

    bars = [bar for bar in forest.barcode if (bar.death - bar.birth) >= min_bar_length]
    step_functions: Dict[int, StepFunctionData] = {}

    for i, bar in enumerate(bars):
        sf = _build_step_function_data(forest=forest, bar=bar, cycle_func=cycle_func, baseline=baseline)
        step_functions[i] = sf

    bf = BarcodeFunctionals(
        forest_id=getattr(forest, "id", str(id(forest))),
        label=label,
        baseline=baseline,
        bars=bars,
        step_functions=step_functions,
        extra_meta={"min_bar_length": min_bar_length},
    )

    if cache:
        key = label
        forest.barcode_functionals[key] = bf

    return bf

def plot_barcode_measurement_generic(
        forest,
        cycle_func: CycleValueFunc,
        bar=None,
        *,
        ax: Optional["matplotlib.axes.Axes"] = None,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        label: Optional[str] = None,
        show_baseline: bool = True,
        show: bool = False,
        **kwargs,
    ) -> Tuple["matplotlib.axes.Axes", StepFunctionData]:
    """
    Plot the cycle measurement step function for a single bar.

    Parameters
    ----------
    forest :
        Forest-like object with ``barcode``, ``point_cloud`` and ``max_bar``.
    cycle_func : CycleValueFunc
        Callable (cycle_rep, point_cloud) -> float, used to assign a scalar
        value to each cycle representative of the bar.
    bar :
        Bar object from `forest.barcode`. If None, `forest.max_bar()` is used.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. If None, a new figure and axis are created.
    x_range : (float, float), optional
        If given, set the x-limits to this range.
    y_range : (float, float), optional
        If given, set the y-limits to this range.
    title : str, optional
        Title for the axes. If None, a default title is constructed.
    label : str, optional
        Label used in the fallback title when ``title`` is omitted.
    show_baseline : bool, default True
        Whether to draw the baseline of the step function.
    show : bool, default False
        Whether to call `plt.show()` after plotting.
    **kwargs
        Forwarded to ``StepFunctionData.plot`` as line style options.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the plot.
    step_func : StepFunctionData
        The piecewise-constant function induced by `cycle_func`.
    """

    if ax is None:
        fig, ax = plt.subplots()

    if bar is None:
        bar = forest.max_bar()

    step_func = _build_step_function_data(forest=forest, bar=bar,cycle_func=cycle_func)

    step_func.plot(x_range=x_range, y_range=y_range, ax= ax, title = title,show_baseline=show_baseline, **kwargs)

    if title is None:
        ax.set_title(f"{label} progression in max bar")

    if show:
        plt.show()

    return (ax, step_func)

def _build_convolution_with_indicator(
            starts: List[float],
            ends: List[float],
            vals: List[float],
            a: float,
            b: float,
            *,
            tol: float = 1e-12
    ) -> Tuple[Callable[[float], float], List[float], List[float]]:
        """
        Compute ``h(x) = (f * 1_[a,b])(x)``.

        ``f`` is piecewise constant with values ``vals[i]`` on
        ``[starts[i], ends[i]]`` and zero elsewhere. ``1_[a,b]`` is the
        indicator of ``[a, b]``.

        Parameters
        ----------
        starts, ends, vals : list of float
            Interval boundaries and values describing the step function f.
        a, b : float
            Bounds of the indicator kernel; must satisfy a <= b.
        tol : float, default 1e-12
            Quantisation tolerance for merging nearly coincident event points.

        Returns
        -------
        tuple
            ``(h, xs, ys)`` where ``h`` evaluates the piecewise-linear
            convolution, ``xs`` are knot locations and ``ys`` are values at
            those knots.
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
        forest,
        bar,
        cycle_func: CycleValueFunc,
        *,
        tol: float = 1e-12,
        sf: Optional[StepFunctionData] = None,
    ) -> PiecewiseLinearFunction:
        """
        Compute the raw convolution kernel for one bar.

        The kernel is ``g(x) = (f * 1_[birth, death])(x)``, where ``f`` is the
        step function induced by ``cycle_func`` on the bar's cycle
        representatives.

        Parameters
        ----------
        forest :
            Forest-like object with `barcode` and `point_cloud`.
        bar :
            Bar instance from `forest.barcode`.
        cycle_func : CycleValueFunc
            Callable that assigns a scalar value to each cycle representative.
        tol : float, default 1e-12
            Numerical tolerance used when merging breakpoint events.
        sf : StepFunctionData, optional
            If given, reuse this step function instead of rebuilding it.

        Returns
        -------
        PiecewiseLinearFunction
            Piecewise-linear convolution kernel defined on the bar interval.
        """
        if sf is None:
            sf = _build_step_function_data(forest=forest,
                                           bar=bar, 
                                           cycle_func=cycle_func, 
                                           baseline=0.0)

        a, b = bar.birth, bar.death
        h, xs, ys = _build_convolution_with_indicator(
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
                "kernel_type": "raw_convolution",
            },
        )

def compute_generalized_interval_landscape(
        forest,
        cycle_func: CycleValueFunc,
        bar,
        sf: Optional[StepFunctionData] = None,
    ):
        """
        Helper that computes the (raw) convolution kernel g
        for a single bar and returns (h, xs, ys) as before, but implemented via
        the new PiecewiseLinearFunction.

        Note: this uses the *raw* convolution (no pyramid rescaling).
        For landscapes, prefer compute_generalized_landscape_family.

        Parameters
        ----------
        forest :
            Forest-like object with `barcode`.
        cycle_func : CycleValueFunc
            Scalar-valued functional on cycle representatives.
        bar :
            Bar from the forest barcode; if None, `forest.max_bar()` is used.
        sf : StepFunctionData, optional
            If given, reuse this step function instead of rebuilding it.

        Returns
        -------
        tuple
            `(h, xs, ys)` where `h` is a callable evaluator and `xs`, `ys`
            describe the breakpoints of the piecewise-linear kernel.
        """
        if bar is None:
            bar = forest.max_bar()

        if bar not in forest.barcode:
            raise ValueError("Bar is not in barcode of forest")

        kernel = compute_convolution_kernel_for_bar(
            forest=forest,
            bar=bar,
            cycle_func=cycle_func,
            sf=sf,
        )

        xs = kernel.xs.tolist()
        ys = kernel.ys.tolist()

        def h(x: float) -> float:
            return float(kernel(x))

        return h, xs, ys

def compute_landscape_kernel_for_bar(
        forest,
        bar,
        cycle_func: CycleValueFunc,
        *,
        mode: Literal["raw", "pyramid"] = "pyramid",
        tol: float = 1e-12,
        sf: Optional[StepFunctionData] = None,
    ) -> PiecewiseLinearFunction:
        """
        Build the kernel used for landscapes for a single bar.

        - mode="raw":      return g(x) = (f * 1_[birth,death])(x)
        - mode="pyramid":  return λ(x) = 1/2 * g(2x)

        The "pyramid" mode should reproduce the usual persistence landscape
        shape when f ≡ 1.

        Parameters
        ----------
        forest :
            Forest-like object with `barcode`.
        bar :
            Bar from the forest barcode.
        cycle_func : CycleValueFunc
            Functional used to evaluate each cycle representative.
        mode : {"raw", "pyramid"}, default "pyramid"
            Whether to return the raw convolution or the pyramid-rescaled kernel.
        tol : float, default 1e-12
            Tolerance forwarded to the convolution helper.
        sf : StepFunctionData, optional
            If given, reuse this step function instead of rebuilding it.

        Returns
        -------
        PiecewiseLinearFunction
            Kernel ready to be sampled on a grid.
        """
        raw_kernel = compute_convolution_kernel_for_bar(
            forest = forest,
            bar=bar,
            cycle_func=cycle_func,
            tol=tol,
            sf=sf,
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
        forest,
        cycle_func: CycleValueFunc,
        label: str,
        *,
        max_k: int = 5,
        num_grid_points: int = 512,
        mode: Literal["raw", "pyramid"] = "pyramid",
        min_bar_length: float = 0.0,
        x_grid: Optional[NDArray[np.float64]] = None,
        cache: bool = True,
        cache_functionals: bool = False,
        functionals_label: Optional[str] = None,
        compute_functionals: bool = True,
    ) -> GeneralizedLandscapeFamily:
        """
        Compute a generalized landscape family for a forest.

        If x_grid is provided, all landscapes are evaluated on that grid
        (and num_grid_points is ignored). This is crucial for consistent
        vectorisations across multiple forest objects.

        Parameters
        ----------
        forest:
            Forest-like object with ``barcode``, ``point_cloud``,
            ``landscape_families`` and ``barcode_functionals``.
        cycle_func:
            Function ``f(cycle_rep, point_cloud) -> scalar``.
        label:
            Key used to store the family in ``forest.landscape_families``.
        max_k:
            Number of landscapes λ_1, ..., λ_max_k to compute.
        num_grid_points:
            Number of grid points used when x_grid is None.
        mode:
            "raw" or "pyramid" (see compute_landscape_kernel_for_bar).
        min_bar_length:
            Ignore bars with (death - birth) < min_bar_length.
        x_grid:
            Optional fixed grid on which to evaluate the landscapes.
        cache:
            If True, save the result to ``forest.landscape_families[label]``.
        cache_functionals:
            If True, cache computed barcode functionals on the forest.
        functionals_label:
            Label used to cache/retrieve barcode functionals.
            If None, uses `label`.
        compute_functionals:
            If True, compute barcode functionals. If False, reuse cached
            functionals with ``functionals_label``.

        Returns
        -------
        GeneralizedLandscapeFamily
            Collection of per-bar kernels and λ_k landscapes on the shared grid.
        """
        if not hasattr(forest, "barcode"):
            raise AttributeError("LoopForest has no 'barcode' attribute. Did you compute it?")

        # 1. Filter bars by length
        bars = [
            bar for bar in forest.barcode
            if (bar.death - bar.birth) >= min_bar_length
        ]

        if functionals_label is None:
            functionals_label = label
        if cache_functionals is None:
            cache_functionals = cache


        if not bars:
            # --- Empty barcode (after min_bar_length filtering): return 0-landscapes instead of error ---
            # Common grid (still needed for a well-formed family / vectorization)
            if x_grid is None:
                # No bars => no natural domain; choose a stable default
                x_grid = np.linspace(0.0, 1.0, num_grid_points)
            else:
                x_grid = np.asarray(x_grid, dtype=float)
                if x_grid.ndim != 1 or x_grid.size < 2:
                    raise ValueError("x_grid must be a 1D array with at least 2 points")

            # Build zero landscapes on the chosen grid
            xmin, xmax = float(x_grid[0]), float(x_grid[-1])

            def _zero_plf(k: int) -> PiecewiseLinearFunction:
                return PiecewiseLinearFunction(
                    xs=x_grid,
                    ys=np.zeros_like(x_grid),
                    domain=(xmin, xmax),
                    metadata={"k": k, "mode": mode, "empty_barcode": True},
                )

            landscapes = {k: _zero_plf(k) for k in range(1, max_k + 1)}

            fam = GeneralizedLandscapeFamily(
                forest_id=getattr(forest, "id", str(id(forest))),
                label=label,
                rescaling=mode,
                x_grid=x_grid,
                bar_kernels={},  # no bars => no kernels
                landscapes=landscapes,
                extra_meta={
                    "min_bar_length": min_bar_length,
                    "n_bars": 0,
                    "empty_barcode": True,
                    "functionals_label": functionals_label,
                },
            )

            if cache:
                if not hasattr(forest, "landscape_families"):
                    forest.landscape_families = {}
                forest.landscape_families[label] = fam

            return fam

        if not compute_functionals:
            bf =forest.barcode_functionals.get(functionals_label, None)
            if bf is None:
                raise ValueError(
                    f"Requested to reuse cached barcode functionals "
                    f"with label '{functionals_label}', but none found."
                )
        else:
            bf = compute_barcode_functionals(
                forest=forest,
                cycle_func=cycle_func,
                label=functionals_label,
                min_bar_length=min_bar_length,
                cache=cache_functionals,
            )

        if cache_functionals and compute_functionals:
            # Ensure cached functionals are available
            if not hasattr(forest, "barcode_functionals"):
                forest.barcode_functionals = {}
            key = functionals_label
            forest.barcode_functionals[key] = bf

        bar_kernels: Dict[int, PiecewiseLinearFunction] = {}
        global_min_x = float("inf")
        global_max_x = float("-inf")

        for i, bar in enumerate(bars):
            kernel = compute_landscape_kernel_for_bar(
                forest,
                bar,
                cycle_func,
                mode=mode,
                sf=bf[bar] #access precomputed step function for this bar
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
                    "mode": mode,
                    "min_bar_length": min_bar_length,
                },
            )

        # 6. Assemble family object
        forest_id = getattr(forest, "name", f"Forest@{id(forest)}")

        family = GeneralizedLandscapeFamily(
            forest_id=forest_id,
            label=label,
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
        )

        # Cache on the forest instance.
        if cache:
            if not hasattr(forest, "landscape_families"):
                forest.landscape_families = {}

            key = label
            forest.landscape_families[key] = family

        return family

def plot_landscape_family(
        forest,
        label: str,
        ks: Optional[List[int]] = None,
        ax: Optional["matplotlib.axes.Axes"] = None,
        title: Optional[str] = None,
        show: bool = True,
        *,
        show_legend: Optional[bool] = None,
        linewidth: Optional[float] = None,
    ):
    """
    Plot selected landscapes λ_k from a stored family on a given forest.

    Parameters
    ----------
    forest :
        Forest-like object carrying `landscape_families`.
    label : str
        Key of the desired family inside `forest.landscape_families`.
    ks : list of int, optional
        Which landscape levels to plot; defaults to all available.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; a new one is created if omitted.
    title : str, optional
        Custom plot title.
    show : bool
        If True, call ``plt.show()`` after plotting.
    show_legend : bool, optional
        Whether to show the legend. Defaults to showing it for fewer than
        10 plotted landscapes and hiding it otherwise.
    linewidth : float, optional
        Line width for the landscape plots. If omitted, Matplotlib's default
        line width is used.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the plot.
    """

    if not hasattr(forest, "landscape_families"):
        raise AttributeError("No landscape_families attribute on this LoopForest")

    family = forest.landscape_families[label]

    if ks is None:
        ks = sorted(family.landscapes.keys())

    if ax is None:
        fig, ax = plt.subplots()

    zorders = {k: zorder for zorder, k in enumerate(sorted(ks, reverse=True), start=1)}
    for k in ks:
        plf = family.landscapes[k]
        plot_kwargs = {
            "label": fr"$\lambda_{{{k}}}$",
            "zorder": zorders[k],
        }
        if linewidth is not None:
            plot_kwargs["linewidth"] = linewidth
        ax.plot(plf.xs, plf.ys, **plot_kwargs)

    ax.set_xlabel("filtration value")
    ax.set_ylabel("landscape value")
    if title is None:
        title = f"Generalized landscapes of {label}"
    ax.set_title(title)
    if show_legend is None:
        show_legend = len(ks) < 10
    if show_legend:
        ax.legend()
    ax.set_ylim(bottom=0)

    if show:
        plt.show()

    return ax

def plot_landscape_comparison_between_functionals(forest,
    labels: list[str],
    k: int = 1,
    ax: Optional["matplotlib.axes.Axes"] = None,
    title: Optional[str] = None,
):
    """
    Compare the k-th landscape across multiple functionals on a single forest.

    Parameters
    ----------
    forest :
        Forest-like object carrying `landscape_families`.
    labels : list of str
        Names of the families/functionals to compare.
    k : int, default 1
        Landscape level to plot.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; a new one is created if omitted.
    title : str, optional
        Custom plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the comparison plot.
    """

    if not hasattr(forest, "landscape_families"):
        raise AttributeError(f"No landscape_families attribute on {forest}")


    families = [forest.landscape_families[label] for label in labels]

    if ax is None:
        fig, ax = plt.subplots()

    for fam, label in zip(families, labels):
        if k not in fam.landscapes:
            continue
        plf = fam.landscapes[k]
        ax.plot(plf.xs, plf.ys, label=label)

    ax.set_xlabel("filtration value")
    ax.set_ylabel(fr"$\lambda_{k}$")
    if title is None:
        title = fr"Comparison of $\lambda_{k}$"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax

def plot_landscape_comparison(
    forests: List,
    label: str,
    k: int = 1,
    ax: Optional["matplotlib.axes.Axes"] = None,
    forest_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    **kwargs
):
    """
    Compare the k-th landscape across multiple forests for a single functional.

    Parameters
    ----------
    forests : list
        Forest-like objects that have `landscape_families`.
    label : str
        Family/functional name to extract from each forest.
    k : int, default 1
        Landscape level to plot.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; a new one is created if omitted.
    forest_labels : list of str, optional
        Labels to use in the legend; defaults to each family's `forest_id`.
    title : str, optional
        Custom plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the comparison plot.
    """
    for forest in forests:
        if not hasattr(forest, "landscape_families"):
            raise AttributeError(f"No landscape_families attribute on {forest}")

    families = [forest.landscape_families[label] for forest in forests]

    if ax is None:
        fig, ax = plt.subplots()

    if forest_labels is None:
        forest_labels = [fam.forest_id for fam in families]

    for fam, forest_label in zip(families, forest_labels):
        if k not in fam.landscapes:
            continue
        plf = fam.landscapes[k]
        ax.plot(plf.xs, plf.ys, label=forest_label, lw=1, **kwargs)

    ax.set_xlabel("filtration value")
    ax.set_ylabel(fr"$\lambda_{k}$")
    if title is None:
        title = fr"Comparison of {label} $\lambda_{k}$"
    ax.set_title(title)
    # ax.legend()
    # ax.grid(True, alpha=0.3)

    return ax

def animate_barcode_measurement_generic(
        forest,
        cycle_func: CycleValueFunc,
        bar=None,
        *,
        filename: Optional[str] = None,
        fps: int = 20,
        frames: int = 200,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        dpi: int = 200,
        total_figsize: Tuple[float, float] = (12.0, 5.0),
        filtration_kwargs: Optional[dict] = None,
        measurement_kwargs: Optional[dict] = None,
    ):
    """
    Animate the filtration of a forest together with a barcode measurement.

    The left panel shows ``forest.plot_at_filtration`` at a moving
    filtration value :math:`alpha`. The right panel shows the step
    function obtained from ``_build_step_function_data`` for a single
    bar together with a vertical line that tracks the current value of
    :math:`alpha`.

    Parameters
    ----------
    forest :
        Forest-like object with attributes

        - ``point_cloud`` (``np.ndarray``),
        - ``barcode`` (iterable of Bar objects),
        - ``filtration`` (iterable of ``(simplex, filt_val)``),
        - a method ``plot_at_filtration(filt_val, ax=None, **kwargs)``.

    cycle_func : CycleValueFunc
        Callable ``(cycle_rep, point_cloud) -> float`` used to assign a
        scalar to each cycle representative of ``bar``.
    bar :
        Bar object in ``forest.barcode``. If ``None``, ``forest.max_bar()``
        is used.
    filename : str or None, optional
        If given, the animation is written to this path. The file extension
        determines the writer (".mp4" uses FFMpeg, ".gif" uses Pillow).
    fps : int, optional
        Frames per second for the saved animation.
    frames : int, optional
        Number of time steps (frames) in the animation.
    t_min, t_max : float or None, optional
        Optional lower/upper bounds on the filtration values to animate.
        If ``None``, ``t_min = 0`` and ``t_max`` is the maximum bar death
        in ``forest.barcode``.
    dpi : int, optional
        DPI used when saving the animation.
    total_figsize : (float, float), optional
        Overall figure size for the two-panel figure.
    filtration_kwargs : dict or None, optional
        Extra keyword arguments forwarded to
        ``forest.plot_at_filtration``. Useful keys include e.g.
        ``show_complex``, ``vertex_size``, ``coloring``.
    measurement_kwargs : dict or None, optional
        Extra keyword arguments forwarded to
        ``StepFunctionData.plot``, **except** for ``ax`` which is
        managed by this function. This is the place to pass e.g.
        ``x_range``, ``y_range``, ``show_baseline``, ``title`` or any
        valid ``matplotlib`` line style keyword.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The created animation. If ``filename`` is not ``None``, the
        animation is also saved to disk.
    fig : matplotlib.figure.Figure
        The figure on which the animation is drawn.
    """
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    if not hasattr(forest, "filtration") or not forest.filtration:
        raise ValueError("Forest has no filtration data to animate.")

    # Choose bar if not specified
    if bar is None:
        bar = forest.max_bar()

    # Optional restriction to a time window
    if t_min is None:
        t_min = 0.0
    if t_max is None:
        # Use the largest death time in the barcode
        deaths = [b.death for b in forest.barcode]
        if not deaths:
            raise ValueError("Forest has an empty barcode.")
        t_max = max(deaths)

    if t_max <= t_min: # type: ignore
        raise ValueError("t_max must be strictly larger than t_min for animation.")

    # Uniformly spaced in filtration value → uniform speed
    frame_times = np.linspace(t_min, t_max, frames).tolist()  # type: ignore

    # ---- Common kwargs for plot_at_filtration (cloud panel) ----
    if filtration_kwargs is None:
        filtration_kwargs = {}
    # Reasonable defaults (only used if not explicitly overridden)
    filtration_kwargs = {
        "show_complex": True,
        "vertex_size": 3,
        "coloring": "forest",
        "title": "Alpha Filtration",
        "show": False,   # important: we manage the figure ourselves
        **filtration_kwargs,
    }

    # ---- Build the step function data once ----
    step_func = _build_step_function_data(
        forest=forest,
        bar=bar,
        cycle_func=cycle_func,
    )

    # ---- Figure and axes ----
    fig, (ax_cloud, ax_meas) = plt.subplots(
        1, 2,
        figsize=total_figsize,
        gridspec_kw={"width_ratios": [3, 4]},
    )

    # ---- Measurement panel: draw the static step function ----
    if measurement_kwargs is None:
        measurement_kwargs = {}

    # Do not let the caller override the axis here
    measurement_kwargs = {
        k: v for k, v in measurement_kwargs.items()
        if k not in {"ax"}
    }

    # Provide sensible defaults, but allow user overrides
    meas_defaults = {
        "x_range": (0, t_max*1.05), # type: ignore
        "show_baseline": True,
        "title": "Barcode measurement",
    }
    # Only fill in keys that the user did not specify
    for k, v in meas_defaults.items():
        measurement_kwargs.setdefault(k, v)

    step_func.plot(ax=ax_meas, **measurement_kwargs)

    # Vertical line that will move with the filtration
    current_t0 = frame_times[0]
    meas_line = ax_meas.axvline(current_t0, color="k", linewidth=2)

    # ---- Helper to draw a single frame on the cloud panel ----
    def _draw_frame_at_time(t: float):
        """Clear the cloud axis and redraw for filtration value t."""
        ax_cloud.clear()
        forest.plot_at_filtration(filt_val=t, ax=ax_cloud, **filtration_kwargs)

        # Optional: overlay a small text box with the current filtration value.
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
        t0 = frame_times[0]
        meas_line.set_xdata([t0, t0])
        return []

    def update(frame_idx: int):
        t = frame_times[frame_idx]
        _draw_frame_at_time(t)
        meas_line.set_xdata([t, t])
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
