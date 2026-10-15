"""
Cross-section geometry: total tube length, RG (geometric complexity) and Omega.

Every cross-section is normalised so the area enclosed by its OUTERMOST side is a
constant 5625 mm^2 (=> square outer side 75 mm, circle outer radius 42.31 mm,
hexagon outer side 46.53 mm, ...). The total tube length is the drawn skeleton
length scaled into that fixed-area frame (scale-invariant):

    total_length_mm = 75 * skeleton_length_px / sqrt(outer_area_px)

RG (non-dimensional geometric complexity index, Eq. 1 of the paper):

    RG = L_i / L_SC_S

where L_SC_S is the total side length of the single-celled SQUARE tube (SC_S =
"S1"), the ONE reference tube for every cross-section regardless of outer shape:

    L_SC_S = 4 * 75 = 300 mm.

So RG = total_length_mm / 300 for square, circular, hexagonal, octagonal, ...
tubes alike (a plain circle therefore has RG < 1, a plain hexagon RG < 1, etc.).

Omega = sqrt(RT) / RG, where RT is the total membrane energy. RT and RG are
each rounded to 2 decimal places before this division, and the Omega result
is itself rounded to 2 decimal places, so Omega always matches what you'd get
by hand from the displayed (2dp) RT and RG values.
"""

import math
from decimal import Decimal, ROUND_HALF_UP

OUTER_AREA_MM2 = 5625.0
OUTER_SIDE_MM = math.sqrt(OUTER_AREA_MM2)          # 75.0

# The single reference tube for ALL cross-sections: the single square tube S1.
REFERENCE_LENGTH_MM = 4.0 * OUTER_SIDE_MM          # L_SC_S = 300.0


def round_half_up(x, ndigits=2):
    """Round x to ndigits decimal places, with an exact .5 at the last digit
    always rounding UP (away from zero) -- unlike Python's built-in round(),
    which rounds half-to-even AND is further skewed by binary floating-point
    representation (e.g. round(1.005, 2) gives 1.0, not 1.01, because 1.005
    is not exactly representable in binary; it's stored as very slightly
    less). Going through the number's own decimal string first (str(x), the
    shortest decimal that round-trips to that float -- i.e. what you'd
    actually see printed) removes that representation issue, so a value that
    reads as X.XX5 always rounds up to X.(XX+1), the ordinary-rounding rule."""
    if x is None:
        return None
    q = Decimal(1).scaleb(-ndigits)
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


def compute_geometry(shape_type, skel_len_px, outer_area_px, RT):
    """Return total length, reference length, RG and Omega for one section.
    RG is always relative to the single square tube S1 (300 mm)."""
    if not outer_area_px or outer_area_px <= 0:
        return {"total_length_mm": None, "reference_length_mm": REFERENCE_LENGTH_MM,
                "RG": None, "omega": None, "scale_mm_per_px": None}
    scale = math.sqrt(OUTER_AREA_MM2 / outer_area_px)
    total_len = skel_len_px * scale
    rg = total_len / REFERENCE_LENGTH_MM
    # Omega is computed from RT and RG rounded to 2dp first (matching the
    # 2dp values shown on screen), then the result is itself rounded to 2dp.
    # round_half_up (not round()) so an exact .5 always rounds up, never
    # down or to even.
    _rt_r = round_half_up(RT, 2) if RT is not None else None
    _rg_r = round_half_up(rg, 2)
    omega = (round_half_up(math.sqrt(_rt_r) / _rg_r, 2)
             if (_rg_r and _rt_r is not None and _rt_r >= 0) else None)
    return {
        "total_length_mm": total_len,
        "reference_length_mm": REFERENCE_LENGTH_MM,
        "RG": rg,
        "omega": omega,
        "scale_mm_per_px": scale,
    }