"""
Membrane energy of junction elements.

Each junction element has a membrane energy of the form

        E_m = coeff(angle) * (M * H^2 / t)

where coeff(angle) is the dimensionless factor defined by the element's formula
(from "19 types of junction elements for membrane energy formula"). This module
computes coeff(angle) directly from the element's TRUE angle (no lookup table).

The total membrane energy of a cross-section (in units of M*H^2/t), called RT,
is the sum over all elements of coeff * count, plus one 8*pi term per detected
full circle ("circular element").

All formulas were verified against the reference coefficient values:
  2-panel-60 -> 3.826,  k-45 -> 19.314,  X-60 -> 41.569,  Y-120 -> 13.279,
  trident-60 -> 13.565,  claw-90 (3-T) -> 12.40,  circular -> 8*pi = 25.133.
"""

import math
import re

# energy of one detected full circle ("circular element")
CIRCLE_ENERGY = 8.0 * math.pi          # = 25.1327...


def _t(deg):
    return math.tan(math.radians(deg))


def _s(deg):
    return math.sin(math.radians(deg))


def _c(deg):
    return math.cos(math.radians(deg))


# ---- per-family dimensionless coefficient formulas ----

def e_2panel(a):
    """2-panel-corner:  4 * [ 1.1 tan(a/2) / (tan(a/2) + 0.05/tan(a/2)) ]."""
    t = _t(a / 2.0)
    return 4.0 * (1.1 * t / (t + 0.05 / t))


def e_claw(a):
    """3-panel-claw:  4 * [ 1.1 tan(a)/(tan(a)+0.05/tan(a)) + 2 tan(a/2) ].
    At a = 90 the first term tends to 1.1, giving the 3-T / 3e value 12.40."""
    t = _t(a)
    return 4.0 * (1.1 * t / (t + 0.05 / t) + 2.0 * _t(a / 2.0))


def e_Y(a):
    """3-panel-Y:  2 * [ 4 tan(a/4) + 2 sin(a/2) + 3 sin(a) ]."""
    return 2.0 * (4.0 * _t(a / 4.0) + 2.0 * _s(a / 2.0) + 3.0 * _s(a))


def e_k(a):
    """4-panel-K (and 4-cir-k, same formula):  8 * (1 + 1/cos(a))."""
    return 8.0 * (1.0 + 1.0 / _c(a))


def e_trident(a):
    """4-panel-trident:  4 * [ 1.1 tan(a)/(tan(a)+0.05/tan(a)) + 4 tan(a/2) ]."""
    t = _t(a)
    return 4.0 * (1.1 * t / (t + 0.05 / t) + 4.0 * _t(a / 2.0))


def e_X(a):
    """4-panel-X:  8 * [ tan(a/2) + 4/cos(a/2) ]."""
    return 8.0 * (_t(a / 2.0) + 4.0 / _c(a / 2.0))


# fixed-value elements (angle is implied by the name)
E_4PANEL_90 = 16.0
E_6PANEL_60 = 48.0
E_8PANEL_45 = 33.181
E_12PANEL = 48.0                 # 12 * 4
E_16PANEL = 64.0                 # 16 * 4
E_3E_ARC = 12.40                 # = e_claw(90)
E_5PANEL_PLAIN = 20.846          # generic 5-panel fallback
E_3T = 12.40                     # 3-panel-claw-90 (3-T)


def _nums(name):
    """All numbers appearing in a shape name, in order, as floats."""
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", name)]


def element_energy(name):
    """Dimensionless membrane-energy coefficient for a detected element name
    (e.g. '4-cir-k-32', '3-panel-Y-40', '5-crc-panel-(3T+2panel90)').
    Returns None if the name is not recognised."""
    s = name.strip()

    # --- composites first (their names contain other family keywords) ---
    if s.startswith("8-panel-(2*(4-panel-(2panel"):
        n = _nums(s)                      # [8,2,4,2,hi,2,lo]  -> take the two panel angles
        # the two 2panel angles are the last two numbers before trailing ')'
        panels = _nums(s.split("2panel", 1)[1]) if "2panel" in s else n
        # robust: pull numbers that directly follow each '2panel'
        angs = [float(m) for m in re.findall(r"2panel(\d+(?:\.\d+)?)", s)]
        if len(angs) >= 2:
            return 2.0 * (e_2panel(angs[0]) + e_2panel(angs[1]))
        return None
    if s.startswith("6-panel-(2*k-"):
        a = _nums(s.split("2*k-", 1)[1])
        return 2.0 * e_k(a[0]) if a else None
    if s.startswith("6-panel-(4panel"):
        # doc formula: 16 (a 4-panel-90 base) + 2-panel(corner angle)
        angs = re.findall(r"2panel(\d+(?:\.\d+)?)", s)
        return E_4PANEL_90 + e_2panel(float(angs[0])) if angs else None
    if s.startswith("5-crc-panel-(3T+2panel") or s.startswith("5-panel-(3T+2panel"):
        angs = re.findall(r"2panel(\d+(?:\.\d+)?)", s)
        return E_3T + e_2panel(float(angs[0])) if angs else None
    if s.startswith("4-panel-(3claw"):
        angs = re.findall(r"3claw(\d+(?:\.\d+)?)", s)
        return e_claw(float(angs[0])) + 4.0 if angs else None

    # --- fixed-value elements ---
    if s.startswith("3-panel-claw-90"):     # 3-T
        return E_3T
    if s == "3e-arc":
        return E_3E_ARC
    if s.startswith("3f-arc"):
        # asymmetric-Y (two angles) is not carried in the name; use the 3e value
        # as a stand-in until the two angles are exposed.
        return E_3E_ARC
    if s.startswith("4-panel-90"):
        return E_4PANEL_90
    if s.startswith("6-panel-60"):
        return E_6PANEL_60
    if s.startswith("8-panel-45"):
        return E_8PANEL_45
    if s.startswith("12-panel"):
        return E_12PANEL
    if s.startswith("16-panel"):
        return E_16PANEL
    if s == "5-panel" or s.startswith("5-cir-panel"):
        return E_5PANEL_PLAIN
    if s == "circular":
        return CIRCLE_ENERGY

    # --- single-angle families (angle is the trailing number) ---
    n = _nums(s)
    a = n[-1] if n else None
    if a is None:
        return None
    if s.startswith("2-panel"):
        return e_2panel(a)
    if s.startswith("3-panel-claw"):
        return e_claw(a)
    if s.startswith("3-panel-Y"):
        return e_Y(a)
    if s.startswith("4-panel-k") or s.startswith("4-cir-k"):
        return e_k(a)
    if s.startswith("4-panel-trident"):
        return e_trident(a)
    if s.startswith("4-panel-X") or s.startswith("4-cir-X"):
        return e_X(a)
    if s.startswith("8-panel"):
        return E_8PANEL_45
    return None


def compute_energy(shape_counts, circle_count=0):
    """Given the detector's shape_counts dict and the number of full circles,
    return a summary:
        {
          "per_element": [ {shape, count, energy_each, energy_total, known}, ... ],
          "RT": <float>,            # total membrane energy (units of M H^2 / t)
          "unknown": [shape, ...],  # shapes with no formula
        }
    """
    rows = []
    RT = 0.0
    unknown = []
    for shape, cnt in sorted(shape_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        e = element_energy(shape)
        if e is None:
            unknown.append(shape)
            rows.append({"shape": shape, "count": cnt, "energy_each": None,
                         "energy_total": None, "known": False})
            continue
        tot = e * cnt
        RT += tot
        rows.append({"shape": shape, "count": cnt, "energy_each": e,
                     "energy_total": tot, "known": True})

    if circle_count:
        e = CIRCLE_ENERGY
        tot = e * circle_count
        RT += tot
        rows.append({"shape": "circular elements", "count": circle_count,
                     "energy_each": e, "energy_total": tot, "known": True})

    return {"per_element": rows, "RT": RT, "unknown": unknown}