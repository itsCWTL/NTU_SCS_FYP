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

Omega = sqrt(RT) / RG, where RT is the total membrane energy.
"""

import math

OUTER_AREA_MM2 = 5625.0
OUTER_SIDE_MM = math.sqrt(OUTER_AREA_MM2)          # 75.0

# The single reference tube for ALL cross-sections: the single square tube S1.
REFERENCE_LENGTH_MM = 4.0 * OUTER_SIDE_MM          # L_SC_S = 300.0


def compute_geometry(shape_type, skel_len_px, outer_area_px, RT):
    """Return total length, reference length, RG and Omega for one section.
    RG is always relative to the single square tube S1 (300 mm)."""
    if not outer_area_px or outer_area_px <= 0:
        return {"total_length_mm": None, "reference_length_mm": REFERENCE_LENGTH_MM,
                "RG": None, "omega": None, "scale_mm_per_px": None}
    scale = math.sqrt(OUTER_AREA_MM2 / outer_area_px)
    total_len = skel_len_px * scale
    rg = total_len / REFERENCE_LENGTH_MM
    omega = (math.sqrt(RT) / rg) if (rg and RT is not None and RT >= 0) else None
    return {
        "total_length_mm": total_len,
        "reference_length_mm": REFERENCE_LENGTH_MM,
        "RG": rg,
        "omega": omega,
        "scale_mm_per_px": scale,
    }