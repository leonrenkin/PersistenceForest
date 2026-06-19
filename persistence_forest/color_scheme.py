from __future__ import annotations
import json, math, random, datetime
from typing import Dict, List, Sequence, Optional, Callable, Any, Iterable
import numpy as np
from matplotlib import colors as mcolors

# ======================
# Color helpers & Lab
# ======================

def _to_hex(c):
    """
    Convert a Matplotlib-compatible color to a hex string.

    Parameters
    ----------
    c : Any
        Color accepted by Matplotlib, such as a name, hex string or RGB tuple.

    Returns
    -------
    str
        Normalized hex color string.
    """
    if isinstance(c, str):
        return mcolors.to_hex(mcolors.to_rgb(c))
    return mcolors.to_hex(c)



def _mix(rgb1, rgb2, t):
    """Linear interpolation between two RGB colors."""
    return tuple((1 - t) * a + t * b for a, b in zip(rgb1, rgb2))

def _luminance(rgb):
    """Approximate perceived luminance of an RGB triple in [0,1]."""
    r, g, b = rgb
    return 0.2126*r + 0.7152*g + 0.0722*b

def _hue_shift(rgb, delta_h):
    """Shift hue of an RGB triple by delta_h (in HSV hue space)."""
    h, s, v = mcolors.rgb_to_hsv(rgb)
    h = (h + delta_h) % 1.0
    return tuple(mcolors.hsv_to_rgb((h, s, v)))

# --- sRGB -> Lab (D65) utilities (no external deps) ---

# Reference white (D65)
_Xn, _Yn, _Zn = 0.95047, 1.00000, 1.08883

def _srgb_to_linear(c):
    """Convert sRGB channel(s) in [0,1] to linear RGB."""
    # c in [0,1]
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def _f_lab(t):
    """CIE Lab helper f(t) used in XYZ-to-Lab conversion."""
    # CIE standard helper
    delta = 6/29
    return np.where(t > delta**3, np.cbrt(t), t/(3*delta**2) + 4/29)
   

def _rgb_to_xyz(rgb):
    """Convert an RGB triple in [0,1] to XYZ (D65)."""
    # rgb in [0,1], sRGB, D65
    r, g, b = _srgb_to_linear(np.array(rgb))
    X = 0.4124564*r + 0.3575761*g + 0.1804375*b
    Y = 0.2126729*r + 0.7151522*g + 0.0721750*b
    Z = 0.0193339*r + 0.1191920*g + 0.9503041*b
    return X, Y, Z


    
def _rgb_to_lab(rgb):
    """Convert an RGB triple in [0,1] to Lab (D65)."""
    X, Y, Z = _rgb_to_xyz(rgb)
    fx, fy, fz = _f_lab(X/_Xn), _f_lab(Y/_Yn), _f_lab(Z/_Zn)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return np.array([L, a, b])

    

def _lab_distance(lab1, lab2):
    """Compute delta-E 76 distance between two Lab vectors."""
    d = lab1 - lab2
    return float(np.sqrt(np.dot(d, d)))
 
# ======================
# Base color generator
# ======================

def _hue_sequence(n: int, start: float) -> Iterable[float]:
    """Yield n hues spaced by the golden ratio to cover HSV space uniformly."""
    phi = 0.61803398875
    h = start
    for _ in range(n):
        h = (h + phi) % 1.0
        yield h

def _generate_candidates(
    num_hues: int = 360,
    sats: Sequence[float] = (0.65, 0.85, 1.0),
    vals: Sequence[float] = (0.80, 0.95),
    hue_start: float = 0.11,   # near orange by default
    L_bounds: Sequence[float] = (30.0, 92.0),  # keep mid/bright; avoid near-black/white
) -> List[tuple]:
    """
    Generate candidate colors in HSV, convert to hex/Lab, and filter by lightness.

    Parameters
    ----------
    num_hues : int
        Number of hue samples (via golden stepping).
    sats : Sequence[float]
        Saturation levels to combine with each hue.
    vals : Sequence[float]
        Value/brightness levels to combine with each hue.
    hue_start : float
        Initial hue in [0,1] for the golden sequence.
    L_bounds : Sequence[float]
        Inclusive (L_min, L_max) bounds in Lab lightness for keeping candidates.

    Returns
    -------
    list[tuple[str, np.ndarray]]
        List of (hex_color, lab_color) pairs.
    """
    candidates = []
    for h in _hue_sequence(num_hues, hue_start):
        for s in sats:
            for v in vals:
                rgb = tuple(mcolors.hsv_to_rgb((h, s, v)))
                lab = _rgb_to_lab(rgb)
                if L_bounds[0] <= lab[0] <= L_bounds[1]:
                    candidates.append((_to_hex(rgb), lab))
    # Deduplicate very close colors (rare with golden stepping, but cheap)
    uniq = []
    for hx, lab in candidates:
        if not uniq or _lab_distance(lab, uniq[-1][1]) > 0.5:
            uniq.append((hx, lab))
    return uniq

def distinct_base_colors(
    n_sets: int,
    *,
    seed: Optional[int] = None,
    prefer_start: Optional[str] = None,
    num_hues: int = 360,
    sats: Sequence[float] = (0.65, 0.85, 1.00),
    vals: Sequence[float] = (0.80, 0.95),
) -> List[str]:
    """
    Choose base colors by greedy farthest-point sampling in Lab space.

    Parameters
    ----------
    n_sets : int
        Number of base colors to return.
    seed : int | None, optional
        Seed controlling candidate ordering and initial pick.
    prefer_start : str | None, optional
        Hex/Matplotlib color to bias the first chosen color toward.
    num_hues : int, optional
        Number of hues to consider when generating candidates.
    sats : sequence of float, optional
        Saturation levels for candidate generation.
    vals : sequence of float, optional
        Value/brightness levels for candidate generation.

    Returns
    -------
    list[str]
        Hex color strings for each base color.
    """
    rng = random.Random(seed)
    hue_start = rng.random() if seed is not None else 0.11  # randomize starting phase if seeded
    cand = _generate_candidates(num_hues=num_hues, sats=sats, vals=vals, hue_start=hue_start)
    if len(cand) == 0:
        raise RuntimeError("No color candidates generated; adjust sats/vals/L_bounds.")

    # Choose first color:
    if prefer_start is not None:
        target_lab = _rgb_to_lab(mcolors.to_rgb(prefer_start))
        first_idx = min(range(len(cand)), key=lambda i: _lab_distance(cand[i][1], target_lab))
    else:
        first_idx = rng.randrange(len(cand)) if seed is not None else 0

    selected_idx = [first_idx]
    selected_lab = [cand[first_idx][1]]

    # Precompute distances to speed up greedy FPS
    min_dists = np.array([_lab_distance(c[1], selected_lab[0]) for c in cand], dtype=float)

    while len(selected_idx) < min(n_sets, len(cand)):
        # pick the candidate with the largest distance to the current selected set
        next_idx = int(np.argmax(min_dists))
        selected_idx.append(next_idx)
        sel_lab = cand[next_idx][1]
        # update distances
        for i, (hx, lab) in enumerate(cand):
            d = _lab_distance(lab, sel_lab)
            if d < min_dists[i]:
                min_dists[i] = d



    return [cand[ selected_idx[i % len(selected_idx) ] ][0] for i in range(n_sets)]

# ======================
# Within-set variants
# ======================

def variants_for_set(base_hex: str,
                     n: int,
                     tint_strength: float = 0.8,
                     shade_strength: float = 0.7,
                     hue_jitter: float = 0.04,
                     order: str = "light_to_dark") -> List[str]:
    """
    Generate n related colors around a base shade.

    Parameters
    ----------
    base_hex : str
        Base color (hex or Matplotlib-compatible) to vary.
    n : int
        Number of variants to produce.
    tint_strength : float, optional
        Blend amount toward white for lighter variants.
    shade_strength : float, optional
        Blend amount toward black for darker variants.
    hue_jitter : float, optional
        Max absolute hue shift applied across the sequence.
    order : {"light_to_dark","dark_to_light"}, optional
        Order of returned colors.

    Returns
    -------
    list[str]
        Hex strings for the generated variants.
    """
    base_rgb = mcolors.to_rgb(base_hex)
    L = _luminance(base_rgb)
    white, black = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)

    if L < 0.55:
        mixes = np.linspace(0.15, tint_strength, n)
        seq = [_mix(base_rgb, white, t) for t in mixes]
        seq = seq if order == "light_to_dark" else list(reversed(seq))
    elif L > 0.80:
        mixes = np.linspace(0.10, shade_strength, n)
        seq = [_mix(base_rgb, black, t) for t in mixes]
        seq = seq if order == "dark_to_light" else list(reversed(seq))
    else:
        tints  = [_mix(base_rgb, white, t) for t in np.linspace(0.0, tint_strength * 0.6, math.ceil(n/2))]
        shades = [_mix(base_rgb, black, t) for t in np.linspace(0.0, shade_strength * 0.5, n - len(tints))]
        seq = tints + list(reversed(shades))
        if order == "dark_to_light":
            seq = list(reversed(seq))

    if n > 1 and hue_jitter > 0:
        offsets = np.linspace(-hue_jitter, hue_jitter, n)
        seq = [_hue_shift(rgb, dh) for rgb, dh in zip(seq, offsets)]

    return [_to_hex(rgb) for rgb in seq]

# ======================
# Scheme build/save/load
# ======================

def build_color_scheme(set_sizes: Sequence[int],
                       set_ids: Sequence[str],
                       seed: Optional[int] = None,
                       order_within_set: str = "light_to_dark",
                       prefer_start: Optional[str] = None,
                       num_hues: int = 360,
                       sats: Sequence[float] = (0.65, 0.85, 1.00),
                       vals: Sequence[float] = (0.80, 0.95),
                       max_variants_per_set: int = 20) -> Dict[str, Any]:
    """
    Build a full color scheme with base colors and within-set variants.

    Parameters
    ----------
    set_sizes : sequence of int
        Number of colors required for each set.
    set_ids : sequence of str
        Identifiers matching each set size.
    seed : int | None, optional
        Seed for reproducible base selection.
    order_within_set : {"light_to_dark","dark_to_light"}, optional
        Ordering for variants inside each set.
    prefer_start : str | None, optional
        Optional color bias for the first base color.
    num_hues, sats, vals : see ``distinct_base_colors``
        Parameters controlling candidate generation.
    max_variants_per_set : int, optional
        Reserved for future use; currently variants are sized exactly by
        ``set_sizes``.

    Returns
    -------
    dict
        JSON-serializable scheme containing ``_meta`` and ``sets`` entries.
    """
    if len(set_sizes) != len(set_ids):
        raise ValueError("set_sizes and set_ids must have the same length")
    bases = distinct_base_colors(
        len(set_ids),
        seed=seed,
        prefer_start=prefer_start,
        num_hues=num_hues,
        sats=sats,
        vals=vals,
    )
    sets = {
        sid: {
            "base": base,
            "colors": variants_for_set(base, sz, order=order_within_set)
        }
        for sid, base, sz in zip(set_ids, bases, set_sizes)
    }
    return {
        "_meta": {
            "algorithm": "Lab-FPS over HSV rings",
            "seed": seed,
            "params": {"num_hues": num_hues, "sats": list(sats), "vals": list(vals)},
            "base_colors": bases,
        },
        "sets": sets,
    }


# -----------------------
# forest objects entry point
# -----------------------

def _sid(ob) -> int:
    """
    Return the integer color-set id for a bar-like object.

    PersistenceForest bars use ``root_id`` to group related bars into color
    families.
    """
    return int(getattr(ob, "root_id"))

def _stable_sid_key(sid: Any) -> Any:
    """
    Return a stable, immutable key for a set id.
    If sid is already a simple immutable (str/int/tuple), use it.
    Otherwise fall back to id(sid) so it won't change if sid's state mutates.
    """
    if isinstance(sid, (str, int, float, tuple)):
        return sid
    return ("objid", id(sid))

def build_scheme_from_bars(
    bars: Iterable,
    *,
    seed: Optional[int] = None,
    order_within_set: str = "light_to_dark",
    prefer_start: Optional[str] = "#ff7f0e",  # e.g., "#ff7f0e" to bias first base toward orange
    num_hues: int = 360,
    sats: Sequence[float] = (0.65, 0.85, 1.00),
    vals: Sequence[float] = (0.80, 0.95),
    max_variants_per_set: int = 16,
) -> Dict[str, Any]:
    """
    Build a JSON-friendly color scheme directly from bar-like objects.

    Parameters
    ----------
    bars : iterable
        Objects exposing ``root_id`` to define their color set.
    seed : int | None, optional
        Seed for reproducible base selection.
    order_within_set : {"light_to_dark","dark_to_light"}, optional
        Ordering for variants inside each set.
    prefer_start : str | None, optional
        Optional bias for the first base color.
    num_hues, sats, vals : see ``distinct_base_colors``
        Parameters controlling candidate generation.
    max_variants_per_set : int, optional
        Maximum colors generated per set (caps very large groups).

    Returns
    -------
    dict
        Color scheme suitable for JSON serialization, keyed by set id.
    """
    # Group objects by set
    groups: Dict[Any, List] = {}
    for bar in bars:
        sid = _sid(bar)
        groups.setdefault(_stable_sid_key(sid), []).append(bar)


    # Stable set id order: by first appearance (insertion order of groups)
    set_ids = list(groups.keys())
    set_sizes_capped = [min(len(groups[sid]), max_variants_per_set) for sid in set_ids]


    scheme = build_color_scheme(
        set_sizes=set_sizes_capped,
        set_ids=set_ids,
        seed=seed,
        order_within_set=order_within_set,
        prefer_start=prefer_start,
        num_hues=num_hues,
        sats=sats,
        vals=vals
    )
    return scheme

def color_map_for_bars(
    bars: Iterable,
    seed: Optional[int]=None,
    prefer_start: Optional[str] = "#ff7f0e",
    *,
    by_id: bool = False,
) -> Dict[object, str]:
    """
    Build a color map for barcode bars.

    Bars with the same ``root_id`` receive related color variants. This is the
    color-family map used by ``PersistenceForest._build_color_map_forest``.

    Parameters
    ----------
    bars : iterable
        Bar-like objects exposing ``root_id`` and usable as dictionary keys
        unless ``by_id=True``.
    seed : int | None, optional
        Seed for reproducible scheme generation.
    prefer_start : str | None, optional
        Optional bias for the first base color.
    by_id : bool, optional
        If True, return ``{id(obj): color}`` instead of ``{obj: color}``.

    Returns
    -------
    dict
        Mapping from object (or id) to hex color string.
    """
    scheme = build_scheme_from_bars(
        bars,
        seed=seed,
        prefer_start=prefer_start
    )

    # Recreate groups to align colors with the same within-set order
    groups: Dict[str, List] = {}
    for bar in bars:
        sid = _sid(bar)
        groups.setdefault(_stable_sid_key(sid), []).append(bar)

    color_map = {}
    for sid_key, obs in groups.items():
        colors = list(scheme["sets"][sid_key]["colors"])

        # Always cycle on shortfall
        for i, ob in enumerate(obs):
            color_map[id(ob) if by_id else ob] = colors[i % len(colors)]
    return color_map
