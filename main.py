import cv2
import numpy as np
from skimage.morphology import skeletonize
import math
import re
import energy as _energy
import geometry as _geometry


# ============================================================
# ELEMENT COLOUR
# ============================================================

def assign_shape_colors(names):
    """Give every distinct shape name its own DISTINCT colour by spreading hues
    evenly across all shapes present (max separation). Names are sorted first so
    the same set of shapes always yields the same colours. Variants like
    '4-cir-k-30' and '4-cir-k-45' are separate names -> separate colours."""
    _uniq = sorted(set(names))
    _n = max(1, len(_uniq))
    _out = {}
    for _i, _nm in enumerate(_uniq):
        _hue = int(180.0 * _i / _n)                       # 0..179 evenly spaced
        _sat = 235 if _i % 2 == 0 else 200                # alternate to separate
        _val = 255 if _i % 3 else 215                     #   neighbouring hues
        _hsv = np.uint8([[[_hue, _sat, _val]]])
        _b = cv2.cvtColor(_hsv, cv2.COLOR_HSV2BGR)[0][0]
        _out[_nm] = (int(_b[0]), int(_b[1]), int(_b[2]))
    return _out

# ============================================================
# NUMERIC TEMPLATE EXTRACTION FROM ELEMENT LABELS
# ============================================================

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _numeric_template(name):
    """Replace every number in a name with a placeholder and return
    (template, tuple_of_numbers). '4-panel-(3claw46+1)' ->
    ('\\x00-panel-(\\x00claw\\x00+\\x00)', (4.0, 3.0, 46.0, 1.0)). Two shapes
    with the same template are the same element differing only in their numbers."""
    nums = []

    def _repl(m):
        nums.append(float(m.group()))
        return "\x00"

    return _NUM_RE.sub(_repl, name), tuple(nums)

# ============================================================
# ELEMENT LABEL CONSOLIDATION BY ANGLE CLUSTERING
# ============================================================

def consolidate_shape_angles(shape_counts, node_records, tol=5.0):
    """Merge shapes of the SAME element whose measured angles are close, accounting 
    for measurement noise at symmetric nodes. Examples include 4-cir-k-31/32 and 4-panel-(3claw45/46+1).
    Shapes are grouped by their numeric template (identical structure with numbers removed), 
    then clustered when all corresponding numbers differ by no more than tol=5.0. 
    Each cluster is assigned a representative label by replacing each numeric position with 
    its count-weighted MEAN."""

    groups = {}
    for _sh, _cnt in shape_counts.items():
        _tmpl, _nums = _numeric_template(_sh)
        if not _nums:
            continue
        groups.setdefault(_tmpl, []).append((_nums, _cnt, _sh))

    remap = {}
    for _tmpl, _items in groups.items():
        #labels merge if only if their first number is the same 
        #difference of remaining values cannot more than tol
        #single-linkage clustering: If A is close enough to B, 
        #and B is close enough to C, all three can end up in one 
        #cluster even if A and C differ by more than 5.
        _n = len(_items)
        _parent = list(range(_n))

        def _find(_a):
            while _parent[_a] != _a:
                _parent[_a] = _parent[_parent[_a]]
                _a = _parent[_a]
            return _a

        for _i in range(_n):
            for _j in range(_i + 1, _n):
                _ni, _nj = _items[_i][0], _items[_j][0]
                if (_ni[0] == _nj[0] and
                        all(abs(a - b) <= tol for a, b in zip(_ni[1:], _nj[1:]))):
                    _ra, _rb = _find(_i), _find(_j)
                    if _ra != _rb:
                        _parent[_rb] = _ra

        _clusters = {}
        for _i in range(_n):
            _clusters.setdefault(_find(_i), []).append(_items[_i])

        for _cl in _clusters.values():
            if len(_cl) < 2:
                continue
            _tot = sum(c for _, c, _ in _cl)
            _reps = []
            for _p in range(len(_cl[0][0])):
                _reps.append(round(sum(n[_p] * c for n, c, _ in _cl) / _tot))
            _newname = _tmpl
            for _v in _reps:
                _newname = _newname.replace("\x00", _afmt(_v), 1)
            for _nums, _c, _name in _cl:
                if _name != _newname:
                    remap[_name] = _newname

    if not remap:
        return shape_counts
    for _rec in node_records:
        if _rec["shape"] in remap:
            _rec["shape"] = remap[_rec["shape"]]
    new_counts = {}
    for _sh, _cnt in shape_counts.items():
        _k = remap.get(_sh, _sh)
        new_counts[_k] = new_counts.get(_k, 0) + _cnt
    return new_counts

# ============================================================
# convert float to string 
# ============================================================

def _afmt(a):
    """convert float to string with no trailing zeros, e.g. 3.0 -> '3', 3.5 -> '3.5'"""
    return ("%g" % a)

# ============================================================
# nosie correction: canonical angle snapping
# ============================================================
def _snap_canon(a, canon=(45.0, 60.0, 90.0, 120.0, 135.0), tol=2.0):
    """Snap a measured angle to the nearest canonical design angle when it is
    within `tol` degrees; otherwise leave it unchanged. Removes the 1-degree
    pixel-discretisation bias"""
    best = min(canon, key=lambda c: abs(c - a))
    return best if abs(best - a) <= tol else a


# ============================================================
# LENGTH & AREA MEASUREMENT 
# ============================================================

def skeleton_length_px(skel):
    """Calculate the total length (px) of a 1-pixel skeleton. Sums the 8-adjacency
    edges (orthogonal = 1, diagonal = sqrt2) and drops redundant diagonals in a
    staircase so 45-degree / curved lines are measured at their true length."""
    ys, xs = np.where(skel)
    S = set(zip(ys.tolist(), xs.tolist()))
    Lo = 0
    Ld = 0
    for (r, c) in S:
        if (r, c + 1) in S:
            Lo += 1
        if (r + 1, c) in S:
            Lo += 1
        if (r + 1, c + 1) in S and not ((r, c + 1) in S and (r + 1, c) in S):
            Ld += 1
        if (r + 1, c - 1) in S and not ((r, c - 1) in S and (r + 1, c) in S):
            Ld += 1
    return Lo + Ld * math.sqrt(2.0)


def analytic_polygon_len_px(skel, dp_eps=2.0):
    """Calculate the total length (px) of a polygon-like skeleton.
    Traces paths between endpoints and junctions, and identifies closed loops.
    Simplifies each path using the Douglas-Peucker algorithm before summing
    the Euclidean lengths of the resulting line segments. This reduces the
    overestimation caused by pixel staircases along straight, slanted walls.
    `dp_eps` controls the simplification tolerance in pixels. Intended for
    straight-walled polygon sections without circular features.  
    """
    ys, xs = np.where(skel > 0)
    S = set(zip(ys.tolist(), xs.tolist()))
    if not S:
        return 0.0

    def _nbrs(p):
        r, c = p
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and (r + dr, c + dc) in S:
                    out.append((r + dr, c + dc))
        return out

    _deg = {p: len(_nbrs(p)) for p in S}
    _nodes = set(p for p in S if _deg[p] != 2)

    def _straight(path, closed):
        if len(path) < 2:
            return 0.0
        pts = np.array([[c, r] for r, c in path], dtype=np.int32).reshape(-1, 1, 2)
        ap = [tuple(x[0]) for x in cv2.approxPolyDP(pts, dp_eps, closed)] # Douglas-Peucker distance
        rng = range(len(ap)) if closed else range(1, len(ap))
        return sum(math.hypot(ap[i][0] - ap[i - 1][0], ap[i][1] - ap[i - 1][1])
                   for i in rng)

    total = 0.0
    visited = set()
    edge_used = set()
    for n in _nodes:
        for m in _nbrs(n):
            if frozenset((n, m)) in edge_used:
                continue
            path = [n]
            prev, cur = n, m
            while True:
                path.append(cur)
                if cur in _nodes:
                    break
                nxt = [q for q in _nbrs(cur) if q != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
            edge_used.add(frozenset((n, m)))
            edge_used.add(frozenset((path[-1], path[-2])))
            for q in path:
                visited.add(q)
            total += _straight(path, False)
    for p in S:
        if p in visited:
            continue
        path = [p]
        visited.add(p)
        nb = _nbrs(p)
        if not nb:
            continue
        prev, cur = p, nb[0]
        while cur not in visited:
            path.append(cur)
            visited.add(cur)
            nxt = [q for q in _nbrs(cur) if q != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
        total += _straight(path, True)
    return total


def composite_reconstruct_len_px(skel, circles, shape_type):
    """Length correction for the 'circle inscribed in a polygon' composite
    (e.g. octagon O2, square S15): a full circle sits inside a polygon and is
    TANGENT to every polygon edge at its mid-point. At each tangent point the
    circle and the straight edge come closer than the wall thickness, so the
    skeleton merges them into a single centre-line and the total length is
    under-measured.

        length = 2*pi*r  +  polygon perimeter  +  spokes (centre -> vertex)

    Calculates the length as the circle circumference, polygon perimeter,
    and total centre-to-vertex spoke length. Returns None if the checks
    fail, allowing the normal skeleton-length method to be used instead."""
    if not circles or shape_type not in ("Quadrilateral", "Hexagon", "Octagon"):
        return None
    N = EXPECTED_SIDES.get(shape_type, -1)
    cont, _ = cv2.findContours((skel > 0).astype(np.uint8) * 255,
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cont:
        return None
    c = max(cont, key=cv2.contourArea)
    ap = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(ap) != N:
        return None
    cx, cy, r = max(circles, key=lambda z: z[2])
    if r <= 0:
        return None
    Rc = float(np.mean([math.hypot(v[0] - cx, v[1] - cy) for v in ap]))   # circumradius
    inr = float(np.mean([math.hypot((ap[i][0] + ap[(i + 1) % N][0]) / 2.0 - cx,
                                     (ap[i][1] + ap[(i + 1) % N][1]) / 2.0 - cy)
                         for i in range(N)]))                              # in-radius
    if not (abs(r - inr) / r < 0.06 and Rc > r * 1.03):
        return None
    # verify N radial spokes are present inside the circle (guards against a
    # polygon+circle that is not this spoked composite)
    ys, xs = np.where(skel > 0)
    d = np.hypot(xs - cx, ys - cy)
    inner = np.zeros_like(skel)
    m = d < 0.9 * r
    inner[ys[m], xs[m]] = 1
    Lin = skeleton_length_px(inner)
    if N * 0.9 * r <= 0 or abs(Lin - N * 0.9 * r) / (N * 0.9 * r) >= 0.20:
        return None

    # Spokes must actually run to the VERTICES (length Rc), not to the edge
    # midpoints (length r). Sample each centre->vertex ray in the annulus just
    # OUTSIDE the circle (1.15 r .. 0.9 Rc); a spoke to a corner leaves skeleton
    # there, a centre-to-midpoint cross does not. Require most vertices covered.
    Sset = set(zip(ys.tolist(), xs.tolist()))
    def _ray_has_skel(vx, vy):
        ux, uy = (vx - cx), (vy - cy)
        L = math.hypot(ux, uy) or 1.0
        ux, uy = ux / L, uy / L
        hit = 0; tot = 0
        rr = 1.15 * r
        while rr <= 0.9 * Rc:
            tot += 1
            px, py = cx + ux * rr, cy + uy * rr
            found = any((int(round(py)) + dr, int(round(px)) + dc) in Sset
                        for dr in (-2, -1, 0, 1, 2) for dc in (-2, -1, 0, 1, 2))
            if found:
                hit += 1
            rr += 2.0
        return tot == 0 or hit / tot >= 0.6
    if Rc > 1.15 * r:                      # only meaningful when corners stick out
        covered = sum(1 for (vx, vy) in ap if _ray_has_skel(vx, vy))
        if covered < max(1, int(round(0.75 * N))):
            return None                    # spokes end at midpoints -> not this composite

    Poly = sum(math.hypot(ap[i][0] - ap[(i + 1) % N][0],
                          ap[i][1] - ap[(i + 1) % N][1]) for i in range(N))
    return 2.0 * math.pi * r + Poly + N * Rc


def circle_packing_len_px(skel, circles):
    """The pixel-based method tends to overestimate smooth circular lengths due to 
    digitisation (the staircase of the digitised curve is longer than the true arc), 
    while merged skeletons at tangent contacts may cause slight length loss.
    
    Reconstruct the total length (px) of a skeleton dominated by circular
    features. Refits each detected circle using least squares to estimate
    its circumference, then adds the remaining non-circle skeleton length.
    
        length = sum(2*pi*r_i for each circle) + length of non-circle skeleton
    
    Applied ONLY when at least 92% of the skeleton lies on detected circles (a
    near-pure circle packing); For sections containing a mixture of circular and 
    straight walls, the standard skeleton-length method is retained because reliably 
    separating the two types of segments is difficult. Returns the reconstructed 
    length in pixels, or None if the correction criteria are not met"""
    if not circles:
        return None
    ys, xs = np.where(skel > 0)
    if len(ys) < 20: # too few pixels to fit a circle reliably
        return None
    fx = xs.astype(np.float64)
    fy = ys.astype(np.float64)
    on_circle = np.zeros(len(ys), dtype=bool)
    csum = 0.0
    for (cx, cy, r) in circles:
        sel = np.abs(np.hypot(fx - cx, fy - cy) - r) < max(4.0, 0.06 * r)
        on_circle |= sel
        if int(sel.sum()) >= 20: # least-squares refit
            px, py = fx[sel], fy[sel]
            A = np.c_[2 * px, 2 * py, np.ones(len(px))]
            b = px * px + py * py
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            rr = math.sqrt(max(0.0, sol[2] + sol[0] ** 2 + sol[1] ** 2))
            csum += 2.0 * math.pi * rr
        else:
            csum += 2.0 * math.pi * r
    total = skeleton_length_px(skel)
    non = np.zeros_like(skel)
    non[ys[~on_circle], xs[~on_circle]] = 1
    non_circle_len = skeleton_length_px(non)
    if total <= 0 or (total - non_circle_len) / total < 0.92:
        return None
    return csum + non_circle_len


def isolated_circle_len_px(skel, circles, dp_eps=2.0):
    """Length for a section whose walls are STRAIGHT lines plus one or more
    circles that do NOT touch each other (most of the circular sections). 
    Each circle is refitted to its skeleton pixels using least squares, and its 
    circumference is calculated as 2πr. Straight segments are traced and simplified 
    using the Douglas-Peucker algorithm, then measured by summing the Euclidean 
    distances between consecutive points. 

        length = 2*pi*r  +  non-circular paths

    Returns the length in px, or None when the section is NOT this case: no
    circles; any two circles tangent to one another (a packing, where the merged
    skeleton would double-count -- handled by step-count / circle_packing); or an
    almost-pure circle section (<=8% straight, left to circle_packing)."""
    if not circles:
        return None
    # reject if any two circles touch (external or internal tangency -> packing)
    for _i in range(len(circles)):
        for _j in range(_i + 1, len(circles)):
            _c1, _c2 = circles[_i], circles[_j]
            _d = math.hypot(_c1[0] - _c2[0], _c1[1] - _c2[1])
            _tol = 0.08 * max(_c1[2], _c2[2])
            if (abs(_d - (_c1[2] + _c2[2])) < _tol
                    or abs(_d - abs(_c1[2] - _c2[2])) < _tol):
                return None
    ys, xs = np.where(skel > 0)
    if len(ys) < 20:
        return None
    fx = xs.astype(np.float64)
    fy = ys.astype(np.float64)
    ref = []
    csum = 0.0
    on_any = np.zeros(len(ys), dtype=bool)
    for (cx, cy, r) in circles:
        sel = np.abs(np.hypot(fx - cx, fy - cy) - r) < max(4.0, 0.06 * r)
        on_any |= sel
        if int(sel.sum()) >= 20:
            px, py = fx[sel], fy[sel]
            A = np.c_[2 * px, 2 * py, np.ones(len(px))]
            b = px * px + py * py
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            rr = math.sqrt(max(0.0, sol[2] + sol[0] ** 2 + sol[1] ** 2))
            if abs(rr - r) > 0.25 * r:
                rr = r
        else:
            rr = r
        ref.append((cx, cy, rr))
        csum += 2.0 * math.pi * rr
    total_px = skeleton_length_px(skel)
    non = np.zeros_like(skel)
    non[ys[~on_any], xs[~on_any]] = 1
    if total_px <= 0 or skeleton_length_px(non) / total_px <= 0.08:
        return None                     # near-pure circle -> circle_packing
    S = set(zip(ys.tolist(), xs.tolist()))

    def _nbrs(p):
        r, c = p
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and (r + dr, c + dc) in S:
                    out.append((r + dr, c + dc))
        return out

    _deg = {p: len(_nbrs(p)) for p in S}
    _nodes = set(p for p in S if _deg[p] != 2)

    def _is_arc(path):
        if len(path) <= 10:
            return False
        for (cx, cy, r) in ref:
            on = sum(1 for (rr, cc) in path
                     if abs(math.hypot(cc - cx, rr - cy) - r) < max(4.0, 0.06 * r))
            if on >= 0.8 * len(path):
                return True
        return False

    def _straight(path, closed):
        pts = np.array([[c, r] for r, c in path], dtype=np.int32).reshape(-1, 1, 2)
        ap = [tuple(x[0]) for x in cv2.approxPolyDP(pts, dp_eps, closed)]
        rng = range(len(ap)) if closed else range(1, len(ap))
        return sum(math.hypot(ap[i][0] - ap[i - 1][0], ap[i][1] - ap[i - 1][1])
                   for i in rng)

    total = csum
    visited = set()
    edge_used = set()
    for n in _nodes:
        for m in _nbrs(n):
            if frozenset((n, m)) in edge_used:
                continue
            path = [n]
            prev, cur = n, m
            while True:
                path.append(cur)
                if cur in _nodes:
                    break
                nxt = [q for q in _nbrs(cur) if q != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
            edge_used.add(frozenset((n, m)))
            edge_used.add(frozenset((path[-1], path[-2])))
            for q in path:
                visited.add(q)
            if not _is_arc(path):
                total += _straight(path, False)
    for p in S:
        if p in visited:
            continue
        path = [p]
        visited.add(p)
        nb = _nbrs(p)
        if not nb:
            continue
        prev, cur = p, nb[0]
        while cur not in visited:
            path.append(cur)
            visited.add(cur)
            nxt = [q for q in _nbrs(cur) if q != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
        if not _is_arc(path):
            total += _straight(path, True)
    return total


def enclosed_area_px(skel, fallback_area=0.0):
    """Calculate the area (px^2) enclosed by the skeleton using flood-fill.
    Flood-fills from a corner to identify the outside area. If the loop is
    closed, the fill cannot enter the enclosed region."""
    h, w = skel.shape
    free = (skel == 0).astype(np.uint8) #1: background; 0: skeleton
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(free, mask, (0, 0), 2)          # 4-connected exterior fill
    enclosed = h * w - int((free == 2).sum()) 
    if fallback_area > 0 and enclosed < 0.5 * fallback_area:
        return float(fallback_area)
    return float(enclosed)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def remove_small_components(binary_img, min_area_ratio=0.02):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img)
    if num_labels <= 1:
        return binary_img

    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return binary_img
    largest_area = int(areas.max())
    min_area = min_area_ratio * largest_area

    cleaned = np.zeros_like(binary_img)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def remove_border_artifacts(binary_img, border_width=2, max_border_area_ratio=0.05):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img)
    if num_labels <= 1:
        return binary_img

    largest_area = int(stats[1:, cv2.CC_STAT_AREA].max())
    max_removable = max_border_area_ratio * largest_area

    border_mask = np.zeros_like(binary_img, dtype=bool)
    border_mask[:border_width, :] = True
    border_mask[-border_width:, :] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True

    cleaned = binary_img.copy()
    for i in range(1, num_labels):
        comp_mask = (labels == i)
        touches_border = np.any(comp_mask & border_mask)
        is_small = stats[i, cv2.CC_STAT_AREA] < max_removable
        if touches_border and is_small:
            cleaned[comp_mask] = 0
    return cleaned


def preprocess_for_skeleton(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    binary = remove_border_artifacts(binary, border_width=2, max_border_area_ratio=0.05)
    binary = remove_small_components(binary, min_area_ratio=0.02)
    return binary


def prune_skeleton_spurs(skeleton, min_branch_length=4):
    skel = skeleton.copy().astype(np.uint8)
    neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    for _ in range(min_branch_length):
        neighbors = cv2.filter2D(skel, -1, neighbor_kernel)
        endpoints = ((neighbors == 1) & (skel == 1)).astype(np.uint8)
        if endpoints.sum() == 0:
            break
        skel = skel & ~endpoints
    return skel.astype(np.uint8)


def prune_dead_end_branches(skeleton, max_dead_end_length=25):
    skel = skeleton.copy().astype(np.uint8)
    neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

    changed = True
    while changed:
        changed = False
        neighbors = cv2.filter2D(skel, -1, neighbor_kernel)
        endpoint_rc = np.argwhere((neighbors == 1) & (skel == 1))

        for (r, c) in endpoint_rc:
            if not skel[r, c]:
                continue 
            path = [(int(r), int(c))]
            prev = None
            curr = (int(r), int(c))
            hit_junction = False

            while len(path) <= max_dead_end_length:
                nr, nc = curr
                next_cell = None
                count_neighbors = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = nr + dr, nc + dc
                        if (0 <= rr < skel.shape[0] and 0 <= cc < skel.shape[1]
                                and skel[rr, cc] and (rr, cc) != prev):
                            count_neighbors += 1
                            if next_cell is None:
                                next_cell = (rr, cc)
                if count_neighbors == 0:
                    break
                if count_neighbors >= 2:
                    hit_junction = True
                    break
                prev = curr
                curr = next_cell
                path.append(curr)

            if hit_junction and len(path) <= max_dead_end_length:
                for (pr, pc) in path:
                    skel[pr, pc] = 0
                changed = True

    return skel.astype(np.uint8)


# ============================================================
# OUTER SHAPE & VERTEX DETECTION
# ============================================================

def get_refined_shape(cnt):
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return "Unknown", []

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)
    hull_circularity = (4 * np.pi * hull_area) / (hull_perimeter ** 2) if hull_perimeter > 0 else 0

    (_, _), min_circle_radius = cv2.minEnclosingCircle(cnt)
    enclosing_circle_area = np.pi * (min_circle_radius ** 2)
    hull_extent = hull_area / enclosing_circle_area if enclosing_circle_area > 0 else 0

    hull_eps = 0.018 * hull_perimeter
    approx = cv2.approxPolyDP(hull, hull_eps, True)
    num_v = len(approx)
    vertices = [tuple(p[0]) for p in approx]

    if hull_circularity > 0.93 and hull_extent > 0.93:
        return "Circle", []

    if hull_circularity > 0.97:
        return "Circle", []

    shape_map = {
        3: "Triangle", 4: "Quadrilateral", 5: "Pentagon",
        6: "Hexagon", 7: "Heptagon", 8: "Octagon",
    }
    shape_name = shape_map.get(num_v, f"Polygon ({num_v} sides)")
    return shape_name, vertices


def detect_shape_vertices(binary_img, expected_sides=None):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)

    if expected_sides is not None:
        best_match_diff = float('inf')
        vertices = []
        for epsilon_factor in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(contour, epsilon_factor * perimeter, True)
            diff = abs(len(approx) - expected_sides)
            if diff < best_match_diff:
                best_match_diff = diff
                vertices = [tuple(p[0]) for p in approx]
            if len(approx) == expected_sides:
                break

        if best_match_diff > 0:
            epsilon_factor = 0.02
            approx = cv2.approxPolyDP(contour, epsilon_factor * perimeter, True)
            tries = 0
            while len(approx) != expected_sides and tries < 50:
                if len(approx) > expected_sides:
                    epsilon_factor += 0.002
                else:
                    epsilon_factor -= 0.001
                if epsilon_factor <= 0:
                    break
                approx = cv2.approxPolyDP(contour, epsilon_factor * perimeter, True)
                tries += 1
            if abs(len(approx) - expected_sides) < best_match_diff:
                vertices = [tuple(p[0]) for p in approx]
    else:
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = [tuple(p[0]) for p in approx]

    if len(vertices) >= 3:
        cx_avg = np.mean([v[0] for v in vertices])
        cy_avg = np.mean([v[1] for v in vertices])
        vertices = sorted(
            vertices,
            key=lambda v: math.atan2(v[1] - cy_avg, v[0] - cx_avg)
        )
    return vertices


EXPECTED_SIDES = {
    "Triangle": 3, "Quadrilateral": 4, "Pentagon": 5,
    "Hexagon": 6, "Heptagon": 7, "Octagon": 8,
}


# ============================================================
# CIRCLE DETECTION & FITTING
# ============================================================

def detect_all_circles(binary, min_radius_px=10):
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    out = []
    for c in contours:
        if len(c) < 20:
            continue
        area = cv2.contourArea(c)
        per = cv2.arcLength(c, True)
        if per < 30 or area < 50:
            continue
        circularity = 4 * np.pi * area / (per * per)
        if circularity < 0.85:
            continue
            
        # NEW: Reject polygons (like octagons) that mimic circles
        approx = cv2.approxPolyDP(c, 0.01 * per, True)
        if len(approx) <= 8:
            continue
            
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r < min_radius_px:
            continue
        pts = c.reshape(-1, 2).astype(np.float64)
        ds = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        if ds.mean() < 1:
            continue
        if ds.std() / ds.mean() > 0.08:
            continue
        out.append((float(cx), float(cy), float(r)))
        
    dedup = []
    for cx, cy, r in out:
        keep = True
        for cx2, cy2, r2 in dedup:
            if abs(r - r2) < max(8.0, 0.1 * r) and (cx - cx2) ** 2 + (cy - cy2) ** 2 < (0.15 * r) ** 2:
                keep = False
                break
        if keep:
            dedup.append((cx, cy, r))
    return dedup


def _fit_circle_3pts(p1, p2, p3):
    ax, ay = p1
    bx, by = p2
    cx_, cy_ = p3
    d = 2 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
    if abs(d) < 1e-6:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy_)
          + (bx * bx + by * by) * (cy_ - ay)
          + (cx_ * cx_ + cy_ * cy_) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx_ - bx)
          + (bx * bx + by * by) * (ax - cx_)
          + (cx_ * cx_ + cy_ * cy_) * (bx - ax)) / d
    r = math.hypot(ax - ux, ay - uy)
    return (ux, uy, r)


def ransac_circles_on_skeleton(skeleton, min_radius, max_radius,
                               num_iter=2500, inlier_dist=2.5,
                               min_inlier_frac=0.55):
    ys, xs = np.where(skeleton)
    n = len(xs)
    if n < 30:
        return []
    pts = np.column_stack([xs, ys]).astype(np.float64)
    rng = np.random.default_rng(0)
    H, W = skeleton.shape
    accepted = []
    margin = max(20, min(H, W) // 50)

    for _ in range(num_iter):
        i1 = rng.integers(0, n)
        p1 = pts[i1]
        
        # 50% global random, 50% localized sampling (finds small intersecting circles)
        if rng.random() < 0.5:
            idx = rng.choice(n, size=2, replace=False)
            p2, p3 = pts[idx[0]], pts[idx[1]]
        else:
            search_radius = rng.uniform(min_radius, max_radius) * 2.5
            dx = np.abs(pts[:, 0] - p1[0])
            dy = np.abs(pts[:, 1] - p1[1])
            local_mask = (dx < search_radius) & (dy < search_radius)
            local_pts = pts[local_mask]
            
            if len(local_pts) < 3:
                continue
            idx = rng.choice(len(local_pts), size=2, replace=False)
            p2, p3 = local_pts[idx[0]], local_pts[idx[1]]

        if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) < 10: continue
        if abs(p1[0] - p3[0]) + abs(p1[1] - p3[1]) < 10: continue
        if abs(p2[0] - p3[0]) + abs(p2[1] - p3[1]) < 10: continue
            
        c = _fit_circle_3pts(p1, p2, p3)
        if c is None: continue
        cx, cy, r = c
        
        if r < min_radius or r > max_radius: continue
        if cx < -margin or cx > W + margin: continue
        if cy < -margin or cy > H + margin: continue
            
        ds = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        n_in = int(np.sum(ds < inlier_dist))
        circumference = max(1, int(2 * math.pi * r))
        frac = n_in / circumference
        
        if frac < min_inlier_frac: continue
            
        is_dup = False
        for (ax, ay, ar) in accepted:
            if (abs(ar - r) < max(8.0, 0.1 * r)
                    and (cx - ax) ** 2 + (cy - ay) ** 2 < (0.15 * r) ** 2):
                is_dup = True
                break
        if not is_dup:
            accepted.append((cx, cy, r))
            
    return accepted


def detect_all_circles_combined(binary, skeleton, gray_img, min_radius_px):
    H, W = skeleton.shape
    # FIX 1: Allow circles that stretch all the way to the image corners
    max_rad = int(max(H, W) * 0.8) 
    
    # 1. Standard Contour Detection
    contour_c = detect_all_circles(binary, min_radius_px=min_radius_px)
    
    # 2. RANSAC Skeleton
    ransac_c = ransac_circles_on_skeleton(
        skeleton, min_radius=min_radius_px, max_radius=max_rad,
        num_iter=2000, inlier_dist=3.0, min_inlier_frac=0.55,
    )
    
    # 3. Hough Transform (param2=30 is the perfect sweet spot for sketches)
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    hough = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.0, minDist=min_radius_px * 1.5,
        param1=50, param2=30, minRadius=min_radius_px, maxRadius=max_rad
    )
    
    hough_c = []
    if hough is not None:
        for (x, y, r) in hough[0, :]:
            hough_c.append((float(x), float(y), float(r)))
            
    # Combine and Deduplicate
    combined = list(contour_c)
    
    ys, xs = np.where(skeleton)
    pts = np.column_stack([xs, ys]).astype(np.float64) if len(xs) > 0 else np.array([])
    
    for cand_list in [ransac_c, hough_c]:
        for cx, cy, r in cand_list:
            is_dup = False
            for (ax, ay, ar) in combined:
                if (abs(ar - r) < max(8.0, 0.15 * r)
                        and (cx - ax) ** 2 + (cy - ay) ** 2 < (0.20 * r) ** 2):
                    is_dup = True
                    break
            
            if not is_dup and len(pts) > 0:
                # FIX 2: Slightly wider pixel tolerance (4.0) to account for thick sketch lines
                ds = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
                n_in = int(np.sum(ds < 4.0)) 
                circumference = max(1, int(2 * math.pi * r))
                
                # FIX 3: 50% coverage. (Allows the heavily intersected center circle to pass!)
                if n_in / circumference > 0.40:
                    combined.append((cx, cy, r))
                    
    return combined


def is_on_any_circle(x, y, circles, tol):
    for (cx, cy, r) in circles:
        if abs(math.hypot(x - cx, y - cy) - r) < tol:
            return True
    return False


def snap_to_skeleton(skeleton, r, c, R):
    if 0 <= r < skeleton.shape[0] and 0 <= c < skeleton.shape[1] and skeleton[r, c]:
        return r, c
    h, w = skeleton.shape
    best = None
    best_d = float('inf')
    for rr in range(max(0, r - R), min(h, r + R + 1)):
        for cc in range(max(0, c - R), min(w, c + R + 1)):
            if skeleton[rr, cc]:
                d = (rr - r) ** 2 + (cc - c) ** 2
                if d < best_d:
                    best_d = d
                    best = (rr, cc)
    return best if best is not None else (r, c)


def add_concentric_circles(skeleton, circles, min_r, step=4,
                          cov_thresh=0.95, tol=5, samples=360):
    """Find circles missed because they are concentric with another circle
    (Hough suppresses same-centre circles via minDist). For every detected
    centre, scan radii and keep any with near-full skeleton coverage."""
    if not circles:
        return circles
    srcm = (skeleton == 0).astype(np.uint8)
    dt = cv2.distanceTransform(srcm, cv2.DIST_L2, 3)
    H, W = skeleton.shape

    def cov(cx, cy, r):
        hit = 0
        for d in range(samples):
            a = 2.0 * math.pi * d / samples
            px = int(round(cx + r * math.cos(a)))
            py = int(round(cy + r * math.sin(a)))
            if 0 <= px < W and 0 <= py < H and dt[py, px] < tol:
                hit += 1
        return hit / samples

    out = list(circles)
    # unique centres
    centres = []
    for (cx, cy, r) in circles:
        if all((cx - ox) ** 2 + (cy - oy) ** 2 > 12 * 12 for ox, oy in centres):
            centres.append((cx, cy))
    for (cx, cy) in centres:
        existing = [r for (ox, oy, r) in out
                    if (cx - ox) ** 2 + (cy - oy) ** 2 <= 12 * 12]
        rmax = int(max(existing)) if existing else 0
        r = min_r
        while r < rmax - 8:
            if all(abs(r - er) >= 8 for er in existing) and cov(cx, cy, r) >= cov_thresh:
                out.append((float(cx), float(cy), float(r)))
                existing.append(r)
            r += step
    return out


def _circle_tangent_misalign(skeleton, cx, cy, r, snap=4, win=5, step=6,
                             dev_thresh=22.0):
    """Fraction of the circumference where the local skeleton direction does
    NOT run tangent to the circle. A real circle's skeleton is tangent
    everywhere (~0); a circle fitted through straight grid edges only touches
    tangentially at a few points, so it is high (~0.5). Returns -1 if too few
    samples to judge.

    Also returns the MISS fraction (of the full sweep, samples where no
    skeleton pixel is even within `snap`) via the second return value: a
    polygon circumscribing this candidate (vertices poking out past it) has
    NO nearby skeleton for a wide angular band around each corner -- those
    samples are silently skipped by the alignment ratio below (which only
    looks at samples it COULD find), so a high miss fraction is itself a
    strong "this is not a real circle" signal that the ratio alone misses."""
    H, W = skeleton.shape
    bad = 0; tot = 0; total_samples = 0; miss = 0
    for d in range(0, 360, step):
        total_samples += 1
        a = math.radians(d)
        px = cx + r * math.cos(a); py = cy + r * math.sin(a)
        best = None; bd = snap + 0.5
        for rr in range(-snap, snap + 1):
            for cc in range(-snap, snap + 1):
                yy = int(round(py)) + rr; xx = int(round(px)) + cc
                if 0 <= yy < H and 0 <= xx < W and skeleton[yy, xx]:
                    dd = math.hypot(cc, rr)
                    if dd < bd:
                        bd = dd; best = (xx, yy)
        if best is None:
            miss += 1
            continue
        bx, by = best; pts = []
        for rr in range(-win, win + 1):
            for cc in range(-win, win + 1):
                yy = by + rr; xx = bx + cc
                if 0 <= yy < H and 0 <= xx < W and skeleton[yy, xx]:
                    pts.append((cc, rr))
        if len(pts) < 4:
            continue
        P = np.asarray(pts, float); P -= P.mean(0)
        try:
            _, _, vt = np.linalg.svd(P, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        sd = vt[0]
        tan = (-math.sin(a), math.cos(a))
        dot = max(0.0, min(1.0, abs(sd[0] * tan[0] + sd[1] * tan[1])))
        tot += 1
        if math.degrees(math.acos(dot)) > dev_thresh:
            bad += 1
    miss_frac = miss / total_samples if total_samples else 1.0
    if tot < 20:
        return -1.0, miss_frac
    return bad / tot, miss_frac


def _circle_shape_ok(skeleton, cx, cy, r, max_miss_frac=0.05):
    """True if the candidate's skeleton really behaves like a circle.
    Rejects: grid-inscribed circles (high tangent misalignment) and polygon
    rings (many vertices). A genuine circle is tangent everywhere with few
    sharp turns; rings that pass through other shapes (real) keep a low
    vertex count even if misalignment is slightly raised.

    Also rejects a polygon whose vertices poke out past this candidate: a
    wide angular band around each such corner has NO skeleton pixel near the
    candidate's own radius (the tight, un-widened snap search misses it
    entirely rather than flagging misalignment), so a high MISS fraction over
    the sweep is itself proof this isn't a real circle everywhere, even when
    the misalignment ratio computed only from the samples that DID find a
    match looks deceptively low.

    max_miss_frac must be loosened by a caller whose candidate is KNOWN to be
    legitimately gap-heavy -- e.g. a big ring that passes behind several
    smaller circles/spokes drawn over it (find_center_rings) is missing real
    skeleton over a wide swath even though it is a genuine circle, which the
    default (tuned for a fully-drawn circle) would wrongly reject."""
    ma, miss_frac = _circle_tangent_misalign(skeleton, cx, cy, r)
    if miss_frac > max_miss_frac:
        return False
    if ma > 0.30:
        return False
    vc = _circle_vertex_count(skeleton, cx, cy, r)
    # graded: more vertices => require lower misalignment to accept
    if vc >= 10 and ma >= 0.12:
        return False
    if vc >= 9 and ma >= 0.18:
        return False
    return True


def _circle_vertex_count(skeleton, cx, cy, r, snap=6, step=4, turn_thresh=22):
    """Count sharp turns (vertices) of the skeleton around a candidate circle.
    A real circle curves smoothly (few turns); a polygon ring of straight
    edges (e.g. a hexagon of triangle edges) has many turns."""
    H, W = skeleton.shape
    pts = []
    for d in range(0, 360, step):
        a = math.radians(d)
        px = cx + r * math.cos(a); py = cy + r * math.sin(a)
        best = None; bd = snap + 0.5
        for rr in range(-snap, snap + 1):
            for cc in range(-snap, snap + 1):
                yy = int(round(py)) + rr; xx = int(round(px)) + cc
                if 0 <= yy < H and 0 <= xx < W and skeleton[yy, xx]:
                    dd = math.hypot(cc, rr)
                    if dd < bd:
                        bd = dd; best = (xx, yy)
        pts.append(best)
    valid = [p for p in pts if p is not None]
    if len(valid) < 8:
        return 99
    turns = 0; n = len(valid)
    for i in range(n):
        a = valid[i - 1]; bp = valid[i]; c = valid[(i + 1) % n]
        v1 = (bp[0] - a[0], bp[1] - a[1]); v2 = (c[0] - bp[0], c[1] - bp[1])
        n1 = math.hypot(*v1); n2 = math.hypot(*v2)
        if n1 < 2 or n2 < 2:
            continue
        dot = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        if math.degrees(math.acos(dot)) > turn_thresh:
            turns += 1
    return turns


def _circle_coverage(dt, H, W, cx, cy, r, tol, samples=720):
    hit = 0
    for d in range(samples):
        a = 2.0 * math.pi * d / samples
        px = int(round(cx + r * math.cos(a)))
        py = int(round(cy + r * math.sin(a)))
        if 0 <= px < W and 0 <= py < H and dt[py, px] < tol:
            hit += 1
    return hit / samples


def _refine_circle_fit(skeleton, cx, cy, r, tol=8):
    """Least-squares refit of (cx,cy,r) using skeleton points already close to
    the initial estimate. A Hough/RANSAC/contour candidate is often correct in
    shape but off by a handful of px in centre/radius (e.g. from an
    anti-aliased or slightly hand-drawn circle) -- just enough to fail the
    strict cov_thresh=0.95 coverage check below at tol=5, wrongly rejecting a
    genuine circle. Tightening the fit first (same lstsq technique already
    used for LENGTH measurement in circle_packing_len_px) fixes that without
    loosening the coverage threshold itself."""
    ys, xs = np.where(skeleton > 0)
    if len(xs) == 0:
        return cx, cy, r
    fx = xs.astype(np.float64); fy = ys.astype(np.float64)
    d = np.abs(np.hypot(fx - cx, fy - cy) - r)
    sel = d < tol
    if int(sel.sum()) < 20:
        return cx, cy, r
    px, py = fx[sel], fy[sel]
    A = np.c_[2 * px, 2 * py, np.ones(len(px))]
    b = px * px + py * py
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    ncx, ncy = float(sol[0]), float(sol[1])
    nr = math.sqrt(max(0.0, sol[2] + ncx ** 2 + ncy ** 2))
    return ncx, ncy, nr


def filter_real_circles(skeleton, circles, cov_thresh=0.95, **_):
    """Strict filter for circles used in NODE logic: keep only nearly fully
    covered circles (clean drawn circles). Partial big rings are handled
    separately for counting (find_center_rings)."""
    if not circles:
        return []
    src = (skeleton == 0).astype(np.uint8)
    dt = cv2.distanceTransform(src, cv2.DIST_L2, 3)
    H, W = skeleton.shape
    out = []
    for (cx, cy, r) in circles:
        rcx, rcy, rr = _refine_circle_fit(skeleton, cx, cy, r)
        if (_circle_coverage(dt, H, W, rcx, rcy, rr, 5) >= cov_thresh
                and _circle_shape_ok(skeleton, rcx, rcy, rr)):
            out.append((rcx, rcy, rr))
    return out


def find_center_rings(skeleton, cx0, cy0, min_r, max_r, step=4):
    """Find big concentric rings centred near (cx0,cy0) that the other circle
    detectors miss (they are partly covered because they pass through the
    smaller circles). Uses the curvature-based test so polygon outlines are
    not mistaken for rings."""
    src = (skeleton == 0).astype(np.uint8)
    dt = cv2.distanceTransform(src, cv2.DIST_L2, 3)
    H, W = skeleton.shape
    out = []
    r = min_r
    while r <= max_r:
        best = (-1.0, cx0, cy0)
        for dcy in range(-8, 9, 4):
            for dcx in range(-8, 9, 4):
                c5 = _circle_coverage(dt, H, W, cx0 + dcx, cy0 + dcy, r, 5, samples=360)
                if c5 > best[0]:
                    best = (c5, cx0 + dcx, cy0 + dcy)
        c5, bcx, bcy = best
        if c5 >= 0.60:
            c12 = _circle_coverage(dt, H, W, bcx, bcy, r, 12, samples=360)
            if (c12 - c5) <= 0.08 and all(abs(r - rr) > 15 for (_, _, rr) in out):
                out.append((bcx, bcy, r))
                r += 20
                continue
        r += step
    return out


# ============================================================
# JUNCTION / NODE DETECTION
# ============================================================

def _atomic_junctions(skeleton, dil=5):
    nk = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    nb = cv2.filter2D(skeleton, -1, nk)
    junc = np.where((nb >= 3) & (skeleton == 1), 255, 0).astype(np.uint8)
    blob = cv2.dilate(junc, np.ones((dil, dil), np.uint8))
    n, _, _, cents = cv2.connectedComponentsWithStats(blob)
    pts = []
    for i in range(1, n):
        x, y = int(cents[i][0]), int(cents[i][1])
        sy, sx = snap_to_skeleton(skeleton, y, x, 8)
        pts.append((sx, sy))
    return pts


def _ring_branch_count(skeleton, cx, cy, R):
    H, W = skeleton.shape
    on = False; t = 0; first = None; last = 0
    steps = max(360, int(2 * math.pi * R * 2))
    for d in range(steps):
        a = 2 * math.pi * d / steps
        r = int(round(cy + R * math.sin(a)))
        c = int(round(cx + R * math.cos(a)))
        v = 1 if (0 <= r < H and 0 <= c < W and skeleton[r, c]) else 0
        if first is None:
            first = v
        last = v
        if v and not on:
            t += 1; on = True
        elif not v:
            on = False
    if first and last and t > 0:
        t -= 1
    return t


def _atom_local_degree(skeleton, cx, cy, R=10):
    """Local crossing degree: 3 = T-junction, 4+ = real crossing."""
    return _ring_branch_count(skeleton, cx, cy, R)


def _bfs_skeleton_path(skel, src, dst, blocked, limit):
    from collections import deque
    H, W = skel.shape
    if src == dst:
        return [src]
    q = deque([src]); prev = {src: None}
    while q:
        r, c = q.popleft()
        if (r - src[0]) ** 2 + (c - src[1]) ** 2 > limit * limit:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if not skel[rr, cc] or (rr, cc) in prev:
                    continue
                if (rr, cc) != dst and (rr, cc) in blocked:
                    continue
                prev[(rr, cc)] = (r, c)
                if (rr, cc) == dst:
                    path = [(rr, cc)]
                    while path[-1] is not None and path[-1] != src:
                        path.append(prev[path[-1]])
                    return path[::-1] if path[-1] == src else None
                q.append((rr, cc))
    return None


def _segment_is_straight(path, thresh=0.012):
    if path is None or len(path) < 5:
        return False
    pts = np.array([[c, r] for r, c in path], dtype=np.float64)
    span = math.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1])
    if span < 6:
        return False
    cen = pts - pts.mean(0)
    try:
        _, s, _ = np.linalg.svd(cen, full_matrices=False)
    except np.linalg.LinAlgError:
        return False
    if len(s) < 2:
        return True
    return (s[-1] / math.sqrt(len(pts)) / span) < thresh


def _path_follows_circle(path, circles, tol):
    """True if the BFS path linking two junction atoms runs almost entirely
    along ONE detected circle. Used to catch a tangent/grazing circle that
    scatters 2-3 extra T-junctions a short arc apart: they are connected by a
    curved (not straight) segment, so _segment_is_straight misses them, but
    they are still the same physical junction, just spread along the curve."""
    if not path or not circles:
        return False
    for (cx, cy, r) in circles:
        on = sum(1 for (r_, c_) in path
                 if abs(math.hypot(c_ - cx, r_ - cy) - r) < tol)
        if on >= 0.8 * len(path):
            return True
    return False


def refine_junction_nodes(skeleton, max_link_dist=60, circles=None, circle_tol=12):
    """Returns list of (cx, cy, node_type):
       'through' -> square (collapsed tangency cluster with a through-line)
       'corner'  -> circle (collapsed cluster WITHOUT a through-line)
       'single'  -> decide square/circle later by its own through-line."""
    atoms = _atomic_junctions(skeleton, dil=5)
    n = len(atoms)
    if n == 0:
        return []
    deg = [_atom_local_degree(skeleton, x, y) for (x, y) in atoms]

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    core = 3
    atomcore = {}
    allcore = set()
    for k, (x, y) in enumerate(atoms):
        s = {(y + dr, x + dc) for dr in range(-core, core + 1)
             for dc in range(-core, core + 1)}
        atomcore[k] = s; allcore |= s

    md2 = max_link_dist * max_link_dist
    # A circle tangent to a straight wall (e.g. a circle inscribed in a
    # polygon, touching at each edge midpoint) can scatter its ONE tangency
    # point into two T-junctions straddling the true contact point, well
    # apart (the two walls stay within stroke-width of each other over a
    # stretch near tangency, not just a few px) -- so a pair that both sit on
    # the SAME detected circle AND is linked by a genuinely STRAIGHT wall
    # gets a much longer leash than an ordinary tangency scatter.
    circle_link_dist = max_link_dist * 4
    circle_md2 = circle_link_dist * circle_link_dist
    atom_circles = [frozenset(ci for ci, (ccx, ccy, rr) in enumerate(circles or [])
                              if abs(math.hypot(x - ccx, y - ccy) - rr) < circle_tol)
                    for (x, y) in atoms]
    # A wide exclusion disk around every atom that sits on NO detected circle
    # (e.g. a polygon corner just outside an inscribed circle). Two on-circle
    # atoms from DIFFERENT tangent zones (different polygon edges) are joined
    # only by routing past that corner -- the corner's own tiny 3px atomcore
    # does not reliably block that route (the corner's own branch can sit just
    # off the direct edge-to-edge line), so without a wider barrier here the
    # long-leash straight-wall check below can bridge past it and, edge by
    # edge, collapse every tangent zone around the whole circle into one.
    off_circle_barrier = set()
    barrier_r = max(10, circle_tol) * 3
    for k, (x, y) in enumerate(atoms):
        if not atom_circles[k]:
            for dr in range(-barrier_r, barrier_r + 1):
                for dc in range(-barrier_r, barrier_r + 1):
                    if dr * dr + dc * dc <= barrier_r * barrier_r:
                        off_circle_barrier.add((y + dr, x + dc))
    for i in range(n):
        # only T-junctions (deg<=3) can be tangency scatter of one node
        if deg[i] > 3:
            continue
        xi, yi = atoms[i]
        for j in range(i + 1, n):
            if deg[j] > 3:
                continue
            xj, yj = atoms[j]
            d2 = (xi - xj) ** 2 + (yi - yj) ** 2
            blocked = allcore - atomcore[i] - atomcore[j]
            if atom_circles[i] and atom_circles[j]:
                blocked = blocked | (off_circle_barrier - atomcore[i] - atomcore[j])
            merged = False
            # An atom sitting ON some detected circle and an atom sitting on
            # NONE of them must never auto-merge here, even if close and
            # straight/circle-linked: that is not skeleton noise around one
            # point, it is a genuinely different feature a short real wall
            # segment away (e.g. a polygon corner just past where a spoke
            # crosses an inscribed circle -- the corner sits ~20-30px outside
            # the circle, well within reach of a short hop). Without this
            # guard a single such hop can chain transitively all the way
            # around a circle-in-polygon shape (each edge bridging one
            # tangent zone to the next) and collapse the whole boundary into
            # one bogus node.
            circle_mismatch = bool(atom_circles[i]) != bool(atom_circles[j])
            # 1) ordinary tangency scatter (straight OR curved connector),
            #    kept at the SHORT radius -- a curved connector merely means
            #    "the path runs along a circle", which is also true of the
            #    long arc between two genuinely distinct junctions on the
            #    same circle, so it must not get the long leash below.
            if d2 <= md2 and not circle_mismatch:
                path = _bfs_skeleton_path(skeleton, (yi, xi), (yj, xj),
                                          blocked, max_link_dist + 6)
                if _segment_is_straight(path) or _path_follows_circle(
                        path, circles, circle_tol):
                    union(i, j)
                    merged = True
            # 2) straight-wall tangent to a shared circle: the long leash,
            #    but ONLY for a genuinely STRAIGHT connector (a flat wall) --
            #    never merely circle-following, which would just be the
            #    boundary arc between two distinct real junctions.
            if not merged and d2 <= circle_md2 and (atom_circles[i] & atom_circles[j]):
                path = _bfs_skeleton_path(skeleton, (yi, xi), (yj, xj),
                                          blocked, circle_link_dist + 6)
                if _segment_is_straight(path):
                    union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    nodes = []
    for idxs in groups.values():
        if len(idxs) >= 2:
            cx = int(round(sum(atoms[i][0] for i in idxs) / len(idxs)))
            cy = int(round(sum(atoms[i][1] for i in idxs) / len(idxs)))
            if _has_straight_through(skeleton, cx, cy):
                nodes.append((cx, cy, 'through'))
            elif circles and any(
                    all(abs(math.hypot(atoms[i][0] - ccx, atoms[i][1] - ccy) - rr)
                        < circle_tol for i in idxs)
                    for (ccx, ccy, rr) in circles):
                # every atom in this cluster lies on the SAME detected circle:
                # a grazing tangency scatters several T-junctions along one
                # short arc, but it is really ONE (big) node, not several
                # separate corners -- collapse to a single node at the
                # cluster centroid (mirrors the composite length reconstruction
                # used for a circle tangent to a polygon, e.g. S15).
                nodes.append((cx, cy, 'corner'))
            else:
                for i in idxs:
                    nodes.append((atoms[i][0], atoms[i][1], 'corner'))
        else:
            i = idxs[0]
            nodes.append((atoms[i][0], atoms[i][1], 'single'))

    return _dedup_nodes(nodes, min_dist=25, circles=circles,
                        circle_tol=circle_tol)


def _dedup_nodes(nodes, min_dist=25, circles=None, circle_tol=12):
    """Merge near-duplicate detections (same junction found 2-3 times a few
    px apart). Exception: if a cluster straddles TWO different circles (a
    straight spoke crossing two overlapping circles), keep it as two square
    nodes at the cluster's extremes instead of merging to one."""
    m = len(nodes)
    if m == 0:
        return nodes
    parent = list(range(m))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    md2 = min_dist * min_dist
    for i in range(m):
        for j in range(i + 1, m):
            if (nodes[i][0] - nodes[j][0]) ** 2 + (nodes[i][1] - nodes[j][1]) ** 2 <= md2:
                union(i, j)
    groups = {}
    for i in range(m):
        groups.setdefault(find(i), []).append(i)

    def distinct_circles(idxs):
        if not circles:
            return set()
        d = set()
        for i in idxs:
            x, y = nodes[i][0], nodes[i][1]
            for ci, (cx, cy, r) in enumerate(circles):
                if abs(math.hypot(x - cx, y - cy) - r) < circle_tol:
                    d.add(ci)
        return d

    def circles_separate(idxset):
        # Are two spanned circles clearly SEPARATE (a real gap between them)?
        # Separate => spoke crosses each at a distinct point => TWO junctions.
        # Tangent/overlapping => the circles meet => ONE junction.
        cl = [circles[i] for i in idxset]
        for a in range(len(cl)):
            for b in range(a + 1, len(cl)):
                (x1, y1, r1), (x2, y2, r2) = cl[a], cl[b]
                d = math.hypot(x1 - x2, y1 - y2)
                if d <= abs(r1 - r2) + 3:
                    continue                      # one inside the other (big circle)
                if d - (r1 + r2) > 9.0:           # clear gap between the circles
                    return True
        return False

    rank = {'through': 3, 'single': 2, 'corner': 1}
    out = []
    for idxs in groups.values():
        cx = int(round(sum(nodes[i][0] for i in idxs) / len(idxs)))
        cy = int(round(sum(nodes[i][1] for i in idxs) / len(idxs)))
        dc = distinct_circles(idxs) if len(idxs) >= 2 else set()
        if len(dc) >= 2 and circles_separate(dc):
            # spoke crosses two SEPARATE circles -> two degree-4 square nodes
            best = None; bd = -1
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    pa, pb = nodes[idxs[a]], nodes[idxs[b]]
                    dd = (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2
                    if dd > bd:
                        bd = dd; best = (pa, pb)
            out.append((best[0][0], best[0][1], 'xcross'))
            out.append((best[1][0], best[1][1], 'xcross'))
            continue
        if len(dc) >= 2:
            nt = 'circ6'   # tangent/overlapping circles -> ONE degree-6 square
        else:
            nt = max((nodes[i][2] for i in idxs), key=lambda t: rank.get(t, 0))
        out.append((cx, cy, nt))

    # Merge the two corners of a circle-circle lens (both points lie on the
    # SAME two circles) into one node -> a single junction where the circles
    # meet (classified later as a degree-4 circle).
    if circles:
        out = _merge_lens_corners(out, circles, circle_tol)
    return out


def _merge_lens_corners(out, circles, tol, max_dist=130):
    n = len(out)
    if n < 2:
        return out
    def mem(x, y):
        return frozenset(i for i, (cx, cy, r) in enumerate(circles)
                         if abs(math.hypot(x - cx, y - cy) - r) < tol)
    M = [mem(o[0], o[1]) for o in out]
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    def genuine_lens(shared):
        # circles meet (overlap / tangent / one inside other) -> a real lens.
        # Only EXTERNALLY SEPARATE circles (clear gap) are two junctions.
        sc = [circles[k] for k in shared]
        for a in range(len(sc)):
            for b in range(a + 1, len(sc)):
                (x1, y1, r1), (x2, y2, r2) = sc[a], sc[b]
                d = math.hypot(x1 - x2, y1 - y2)
                if d - (r1 + r2) <= 9.0:
                    return True
        return False

    md2 = max_dist * max_dist
    for i in range(n):
        for j in range(i + 1, n):
            shared = M[i] & M[j]
            if len(shared) >= 2 and genuine_lens(shared) and \
               (out[i][0] - out[j][0]) ** 2 + (out[i][1] - out[j][1]) ** 2 <= md2:
                union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    res = []
    for idxs in groups.values():
        if len(idxs) == 1:
            res.append(out[idxs[0]])
        else:
            cx = int(round(sum(out[i][0] for i in idxs) / len(idxs)))
            cy = int(round(sum(out[i][1] for i in idxs) / len(idxs)))
            res.append((cx, cy, 'single'))
    return res


def degree_from_ring(skeleton, cx, cy, node_pts, hard_cap=75, rmin=14):
    """Count distinct skeleton branches on a ring around (cx,cy).
    Sampled out to where overlapping curves (e.g. a polygon edge and a
    tangent circle arc) separate, so a diameter crossing an edge-midpoint
    is correctly counted as degree 5 (edge+edge+spoke+arc+arc) rather
    than 3. Radius is capped at ~half the distance to the nearest node
    to avoid leaking into neighbouring structure."""
    from collections import Counter
    H, W = skeleton.shape
    nd = min((math.hypot(cx - x, cy - y) for (x, y) in node_pts
              if (x, y) != (cx, cy)), default=200)
    Rmax = int(min(0.48 * nd, hard_cap))
    if Rmax < rmin + 6:
        Rmax = rmin + 6

    def ring(R):
        on = False; t = 0; first = None; last = 0
        for d in range(0, 1440):
            a = math.radians(d / 4.0)
            r = int(round(cy + R * math.sin(a)))
            c = int(round(cx + R * math.cos(a)))
            v = 1 if (0 <= r < H and 0 <= c < W and skeleton[r, c]) else 0
            if first is None:
                first = v
            last = v
            if v and not on:
                t += 1; on = True
            elif not v:
                on = False
        if first and last and t > 0:
            t -= 1
        return t

    counts = [ring(R) for R in range(rmin, Rmax + 1, 2)]
    if not counts:
        return 0
    cnt = Counter(counts)
    cand = [v for v in cnt if cnt[v] >= 3]   # ignore transient grazing spikes
    return max(cand) if cand else max(counts)


def detect_corner_nodes(skeleton, existing, circles=None, circle_tol=12,
                        min_dist=24, bend_dot=-0.85):
    """Find genuine degree-2 corners (a sharp bend where two straight edges
    meet) that are neither junctions nor polygon vertices, e.g. the inner
    corners of a star outline. Smooth curves/circles have no sharp bend, and
    points lying on a detected circle are excluded (a circle is not a corner)."""
    nk = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    nb = cv2.filter2D(skeleton, -1, nk)
    ys, xs = np.where((skeleton == 1) & (nb == 2))
    md2 = min_dist * min_dist
    circles = circles or []
    found = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        if any((x - cx) ** 2 + (y - cy) ** 2 < md2 for cx, cy in existing):
            continue
        if any((x - fx) ** 2 + (y - fy) ** 2 < md2 for fx, fy, _ in found):
            continue
        if any(abs(math.hypot(x - cx, y - cy) - r) < circle_tol
               for (cx, cy, r) in circles):
            continue                                   # on a circle, not a corner
        arms = _trace_arms_simple(skeleton, y, x, core=3, length=18)
        if len(arms) != 2 or not (arms[0][1] and arms[1][1]):
            continue
        d1, d2 = arms[0][0], arms[1][0]
        if d1[0] * d2[0] + d1[1] * d2[1] > bend_dot:   # sharp bend
            found.append((x, y, 'corner_pt'))
    return found



# ============================================================
# ARM TRACING
# ============================================================

def _trace_arms_simple(skel, cy, cx, core=5, length=45):
    H, W = skel.shape
    visited = np.zeros_like(skel, bool)
    for r in range(max(0, cy - core - 2), min(H, cy + core + 3)):
        for c in range(max(0, cx - core - 2), min(W, cx + core + 3)):
            if (r - cy) ** 2 + (c - cx) ** 2 <= core * core:
                visited[r, c] = True
    exits = []
    for r in range(max(0, cy - core - 3), min(H, cy + core + 4)):
        for c in range(max(0, cx - core - 3), min(W, cx + core + 4)):
            d2 = (r - cy) ** 2 + (c - cx) ** 2
            if d2 <= core * core or d2 > (core + 3) ** 2:
                continue
            if skel[r, c]:
                exits.append((r, c))
    ded = []
    for (r, c) in exits:
        if all((r - r2) ** 2 + (c - c2) ** 2 >= 4 for r2, c2 in ded):
            ded.append((r, c))
    arms = []
    for (sr, sc) in ded:
        if visited[sr, sc]:
            continue
        path = [(sr, sc)]; visited[sr, sc] = True
        for _ in range(length):
            pr, pc = path[-1]; best = None; bd = -1
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = pr + dr, pc + dc
                    if 0 <= rr < H and 0 <= cc < W and skel[rr, cc] and not visited[rr, cc]:
                        d = (rr - cy) ** 2 + (cc - cx) ** 2
                        if d > bd:
                            bd = d; best = (rr, cc)
            if best is None:
                break
            visited[best] = True; path.append(best)
        if len(path) >= 4:
            er, ec = path[-1]; dx = ec - cx; dy = er - cy; nrm = math.hypot(dx, dy)
            if nrm > 1:
                pts = np.array([[c, r] for r, c in path], float)
                span = math.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1])
                straight = True
                if span >= 6 and len(pts) >= 5:
                    m = pts.mean(0); cen = pts - m
                    try:
                        _, s, _ = np.linalg.svd(cen, full_matrices=False)
                        straight = (s[-1] / math.sqrt(len(pts)) / span) < 0.06
                    except np.linalg.LinAlgError:
                        straight = True
                arms.append(((dx / nrm, dy / nrm), straight))
    return arms


def _node_has_straight_arm(skel, cx, cy, circles, tol, core=5, length=40):
    """True if a STRAIGHT arm (a spoke) leaves the node that does not run
    along a detected circle. Used to tell a spoke-bearing ring corner from
    a spoke-less circle-circle lens."""
    H, W = skel.shape
    sy, sx = snap_to_skeleton(skel, cy, cx, 12)
    visited = np.zeros_like(skel, bool)
    for r in range(max(0, sy - core - 2), min(H, sy + core + 3)):
        for c in range(max(0, sx - core - 2), min(W, sx + core + 3)):
            if (r - sy) ** 2 + (c - sx) ** 2 <= core * core:
                visited[r, c] = True
    exits = []
    for r in range(max(0, sy - core - 3), min(H, sy + core + 4)):
        for c in range(max(0, sx - core - 3), min(W, sx + core + 4)):
            d2 = (r - sy) ** 2 + (c - sx) ** 2
            if core * core < d2 <= (core + 3) ** 2 and skel[r, c]:
                exits.append((r, c))
    ded = []
    for (r, c) in exits:
        if all((r - r2) ** 2 + (c - c2) ** 2 >= 4 for r2, c2 in ded):
            ded.append((r, c))
    for (sr, sc) in ded:
        if visited[sr, sc]:
            continue
        path = [(sr, sc)]; visited[sr, sc] = True
        for _ in range(length):
            pr, pc = path[-1]; best = None; bd = -1
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = pr + dr, pc + dc
                    if 0 <= rr < H and 0 <= cc < W and skel[rr, cc] and not visited[rr, cc]:
                        d = (rr - sy) ** 2 + (cc - sx) ** 2
                        if d > bd:
                            bd = d; best = (rr, cc)
            if best is None:
                break
            visited[best] = True; path.append(best)
        if len(path) < 6:
            continue
        pts = np.array([[c, r] for r, c in path], float)
        span = math.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1])
        if span < 8:
            continue
        m = pts.mean(0); cen = pts - m
        try:
            _, s, _ = np.linalg.svd(cen, full_matrices=False)
            straight = (s[-1] / math.sqrt(len(pts)) / span) < 0.05
        except np.linalg.LinAlgError:
            straight = True
        if not straight:
            continue
        on = 0
        for (r, c) in path:
            for (ccx, ccy, rr) in circles:
                if abs(math.hypot(c - ccx, r - ccy) - rr) < tol:
                    on += 1; break
        if on < 0.6 * len(path):      # straight AND not a circle arc -> a spoke
            return True
    return False


def _has_straight_through(skel, cx, cy):
    """True if two straight arms leave (cx,cy) in opposite (collinear)
    directions -> a straight line passes through the node."""
    sy, sx = snap_to_skeleton(skel, cy, cx, 12)
    arms = _trace_arms_simple(skel, sy, sx)
    sa = [a[0] for a in arms if a[1]]
    for i in range(len(sa)):
        for j in range(i + 1, len(sa)):
            if sa[i][0] * sa[j][0] + sa[i][1] * sa[j][1] < -0.9:
                return True
    return False


def _is_real_corner(skel, vx, vy):
    """A genuine polygon corner: exactly two STRAIGHT edges meeting at an
    angle. Rejects points lying on a smooth curve / circle (no corner)."""
    sy, sx = snap_to_skeleton(skel, vy, vx, 15)
    arms = _trace_arms_simple(skel, sy, sx)
    if len(arms) != 2:
        return False
    if not (arms[0][1] and arms[1][1]):          # both arms must be straight
        return False
    d1, d2 = arms[0][0], arms[1][0]
    dot = d1[0] * d2[0] + d1[1] * d2[1]
    return dot > -0.95                            # a real bend, not a straight line





# ============================================================
# JUNCTION CLASSIFICATION 
# ============================================================

def classify_node_shape(dirs, straight, degree=None, on_circle=False, wedge_angs=None,
                        marker_circle=False):
    """Classify a junction into the taxonomy type name, e.g. '2-panel-90',
    '3-panel-Y-120', '3-panel-claw-45', '4-panel-X-60', '4-panel-90',
    '4-panel-k-45', '4-cir-k-45', '4-panel-trident-60', '8-panel-45'.
    The angle is the smallest panel (gap between adjacent arms), snapped to
    the allowed set for that shape."""
    n = len(dirs)
    if n == 0:
        return "isolated"
    if degree is None:
        degree = n
    curved = any(not s for s in straight)

    # If the panel structure (wedges) shows MORE arms than were traced, rebuild
    # the arm directions from the wedge angles so the true arm count is used.
    # (A '+' centre is sometimes traced as 2 collinear arms while its 4 90-panels
    # are correctly measured -> it must classify as 4-panel-90, not 2-panel-90.)
    if wedge_angs and len(wedge_angs) > n and len(wedge_angs) >= 3:
        _cum = 0.0
        dirs = []
        for _w in wedge_angs:
            _r = math.radians(_cum)
            dirs.append((math.cos(_r), math.sin(_r)))
            _cum += float(_w)
        n = len(dirs)

    # panels (gaps between adjacent arms). Prefer the same wedge angles that
    # are displayed, so the shape label matches the on-image angles exactly.
    if wedge_angs and len(wedge_angs) >= 2:
        gaps = [float(g) for g in wedge_angs]
    else:
        angs = sorted(math.degrees(math.atan2(d[1], d[0])) % 360.0 for d in dirs)
        gaps = [(angs[(i + 1) % n] - angs[i]) % 360.0 for i in range(n)]
    if not gaps:
        gaps = [360.0]
    min_gap = min(gaps)
    max_gap = max(gaps)

    # number of opposite arm pairs ("through lines")
    used = [False] * n
    through = 0
    for i in range(n):
        if used[i]:
            continue
        for j in range(i + 1, n):
            if used[j]:
                continue
            if dirs[i][0] * dirs[j][0] + dirs[i][1] * dirs[j][1] < -0.9:
                through += 1
                used[i] = used[j] = True
                break

    if on_circle:
        if degree == 3:
            if max_gap >= 200.0:
                return "3-panel-claw-%s" % _afmt(round(min_gap))
            # The arc is the ~180 panel; the spoke splits the remaining ~180.
            # Equal halves (90/90) = spoke perpendicular to the arc -> 3e-arc;
            # unequal halves (e.g. 120/60) = spoke slanted -> 3-panel-asymY-arc,
            # whose angle (the smaller spoke-side panel) is carried in the name so
            # the asymmetric-Y energy formula can be evaluated per instance.
            _rest = sorted(gaps)[:2]        # the two spoke-side panels
            if len(_rest) == 2 and abs(_rest[0] - _rest[1]) < 20.0:
                return "3-panel-claw-arc-90(3e-arc)"
            return "3-panel-asymY-arc-%s" % _afmt(round(min(_rest)))
        if degree == 5:
            # 5-crc-panel = a 3T element + a 2-panel corner. The label carries the
            # corner opening.
            #   * If a dominant near-straight arc/line is present (a point on a big
            #     circle, widest gap >= 150), that gap is the arc, and the 2-panel
            #     corner flanks the spoke -> the sum of the two SMALLEST panels
            #     (e.g. a pentagram point: 29 + 29 = 58).
            #   * Otherwise there is no wide arc gap and the corner opening IS the
            #     largest panel itself (e.g. a hexagon-type 120 corner).
            # Prefer the same `gaps` the rest of this function already uses
            # (wedge_angs when given, i.e. the on-image corrected angles) --
            # NOT a fresh recompute from the raw traced `dirs`, which are a
            # CHORD approximation over the arc near a point-on-circle and
            # drift a degree or two off true (e.g. 88+88=176 instead of the
            # true, displayed 90+90=180).
            _cyc5 = gaps
            _mx5 = max(_cyc5)
            if _mx5 >= 150.0:
                # dominant arc gap -> the 3T is (2 arc arms + spoke). Remove it:
                # the remaining two arms are the 2-panel corner, and the spoke
                # sits between them, so their opening is the sum of the two panels
                # FLANKING the spoke = the two MIDDLE panels of the inner fan
                # (e.g. blue node 45 + 45 = 90; pentagram point 29 + 29 = 58).
                _mi5 = _cyc5.index(_mx5)
                fan = _cyc5[_mi5 + 1:] + _cyc5[:_mi5]     # panels between the arcs
                if len(fan) >= 4:
                    # round EACH flanking panel then add, so the label matches the
                    # displayed per-panel angles (29 + 29 = 58, not round(57.2)=57)
                    base = round(fan[len(fan) // 2 - 1]) + round(fan[len(fan) // 2])
                elif fan:
                    base = sum(round(g) for g in fan)
                else:
                    base = round(_mx5)
            else:
                # no dominant arc: the corner opening IS the largest panel itself
                # (e.g. a hexagon-type 120 corner).
                base = _mx5
            return "5-crc-panel-(3T+2panel%s)" % _afmt(round(base))

    if n == 1:
        return "endpoint"

    if n == 2:
        # opening = the angle between the two arms, taken from the arm
        # DIRECTIONS themselves. Do not use wedge_angs/gaps[0] here: at boundary
        # nodes compute_wedge_angles can carry spurious extra wedges, making
        # gaps[0] meaningless (it split a clean 120 into e.g. 26+97).
        dot = max(-1.0, min(1.0, dirs[0][0] * dirs[1][0] + dirs[0][1] * dirs[1][1]))
        opening = math.degrees(math.acos(dot))
        name = "2-panel-%s" % _afmt(round(opening))
        return name + " (arc)" if curved else name

    if n == 3:
        # A T-junction is a straight pass-through LINE (max panel ~180) plus a
        # perpendicular branch, giving a ~90 panel. BOTH must hold: a right-angle
        # panel alone is not enough (a Y like 135/135/90 also has a ~90 panel but
        # has no straight line). Requiring the line (max_gap >= 150) also survives
        # noise that distorts the collinear arm (e.g. 180/90/90 -> 211/90/59).
        if max_gap >= 150.0 and any(abs(g - 90.0) < 8.0 for g in gaps):
            return "3-panel-claw-90 (3-T)"
        if curved:
            if max_gap >= 200.0:          # bunched curved arms = claw
                return "3-panel-claw-%s" % _afmt(round(min_gap))
            return "3-panel-claw-arc-90(3e-arc)"   # balanced curved 3-node
        # claw = arms bunched, NO straight line through (widest panel > ~200)
        if max_gap >= 200.0:
            return "3-panel-claw-%s" % _afmt(round(min_gap))
        # A straight line passing through the node (a ~180 panel, drawn with a
        # SQUARE marker) plus a slanted branch is an ASYMMETRIC Y, a different
        # element from the symmetric three-pronged Y. It is named by X = the
        # SMALLEST angle of the junction (the acute branch panel). Its membrane
        # energy uses the fixed relation omega = alpha/2 (see energy.e_asymY), so
        # no separate orientation angle is measured or stored.
        # A node with no straight line (symmetric, e.g. 120/120/120) stays a Y.
        if max_gap >= 150.0:
            return "3-panel-asymY-%s" % _afmt(round(min_gap))
        return "3-panel-Y-%s" % _afmt(round(min_gap))

    if n == 4:
        if max_gap >= 150.0:
            if curved or marker_circle:
                if through >= 2:
                    # two circles CROSSING (each passes through -> 2 through
                    # lines). Treating the arcs as straight lines makes an X.
                    _xs = sorted(gaps)[:2]
                    _xa = sum(_xs) / len(_xs) if _xs else min_gap
                    if abs(_xa - 90.0) < 12.0:
                        return "4-panel-90"
                    return "4-panel-X-%s" % _afmt(round(_xa))
                # circle marker = a curved arc passes through, so this is the
                # curved-k family (straight line + 2 arcs), never a polygon
                # (3claw+1). Its name must carry the 'cir' arc tag. Label by the
                # SIDE angle = a panel ADJACENT to the wide arc gap (the spoke-to-
                # arc angle), not the middle panel. gaps are in cyclic order, so
                # the two neighbours of the widest gap are the sides; take the
                # larger of the two.
                _mi = gaps.index(max(gaps))
                _ng = len(gaps)
                _sidek = max(gaps[(_mi + 1) % _ng], gaps[(_mi - 1) % _ng])
                # arc (not a straight line) passes through -> curved-k family
                return "4-panel-K-arc-%s" % _afmt(round(_sidek))
            if max_gap > 184.0:
                # widest gap is not a straight line -> 3-claw + 1 separate arm
                # The claw angle comes from the two internal claw gaps. If they
                # are CLOSE (a clean symmetric claw, e.g. 45/46) use their AVERAGE
                # so the value tracks the design angle (45). If FAR apart, the
                # smaller is a noisy middle arm, so use the larger (stable) gap.
                _cg = sorted(gaps)[:2]
                _claw = (sum(_cg) / 2.0) if (_cg[1] - _cg[0]) <= 8.0 else _cg[1]
                return "4-panel-(3claw%s+1)" % _afmt(round(_claw))
            # A genuine straight line (~180) is present. Split the panel cycle
            # at that line -> 3 remaining panels form the arc from one line-end
            # to the other. k is symmetric (line + 2 arms mirrored about the
            # perpendicular): those 3 panels read small-BIG-small, i.e. the
            # MIDDLE panel is the largest (~90). (3claw+1) has 3 bunched arms +
            # 1: the 3 panels read small-small-BIG, i.e. the largest is at an
            # END (adjacent to the straight line).
            asort = sorted(math.degrees(math.atan2(d[1], d[0])) % 360.0 for d in dirs)
            cyc = [(asort[(i + 1) % n] - asort[i]) % 360.0 for i in range(n)]
            im = cyc.index(max(cyc))
            rest = [cyc[(im + 1 + k) % n] for k in range(n - 1)]  # 3 panels, linear
            mid_is_max = abs(rest[len(rest) // 2] - max(rest)) < 8.0
            if not mid_is_max:                      # asymmetric -> 3claw + 1
                # The claw angle comes from the two internal claw gaps. If they
                # are CLOSE (a clean symmetric claw, e.g. 45/46) use their AVERAGE
                # so the value tracks the design angle (45). If FAR apart, the
                # smaller is a noisy middle arm, so use the larger (stable) gap.
                _cg = sorted(gaps)[:2]
                _claw = (sum(_cg) / 2.0) if (_cg[1] - _cg[0]) <= 8.0 else _cg[1]
                return "4-panel-(3claw%s+1)" % _afmt(round(_claw))
            # a genuine straight line (~180) + two arms symmetric on one side = k
            return "4-panel-k-%s" % _afmt(round(_snap_canon(min_gap)))
        if through >= 2:                            # two lines crossing (no 180 panel)
            # X panels alternate small/large (e.g. 45/135/45/135). Label by the
            # AVERAGE of the two small (opposite) panels, so the value matches the
            # displayed per-panel angle (45) instead of dipping to the single
            # smallest (44).
            _xs = sorted(gaps)[:2]
            _xang = sum(_xs) / len(_xs) if _xs else min_gap
            if abs(_xang - 90.0) < 12.0:
                return "4-panel-90"
            return "4-panel-X-%s" % _afmt(round(_xang))
        # spread arms (three prongs + a stem) = trident
        return "4-panel-trident-%s" % _afmt(round(min_gap))

    if n == 5:
        if curved:
            return "5-cir-panel"
        pd = sorted(gaps, reverse=True)                # [~180, ...]
        a2 = round(pd[1] + pd[2])
        return "5-panel-(3T+2panel%s)" % _afmt(a2)
    if n == 6:
        # composite: an X/plus (two through lines = 4 arms) + a 2-panel corner
        paired = [False] * n
        xidx = []
        for i in range(n):
            if paired[i]:
                continue
            for j in range(i + 1, n):
                if not paired[j] and \
                        dirs[i][0] * dirs[j][0] + dirs[i][1] * dirs[j][1] < -0.985:
                    paired[i] = paired[j] = True
                    xidx += [i, j]
                    break
        corner = [i for i in range(n) if not paired[i]]
        if len(xidx) >= 4 and len(corner) == 2:
            xang = sorted(math.degrees(math.atan2(dirs[i][1], dirs[i][0])) % 360.0
                          for i in xidx)
            xgaps = [(xang[(k + 1) % len(xang)] - xang[k]) % 360.0
                     for k in range(len(xang))]
            a = round(min(xgaps))
            cdot = max(-1.0, min(1.0, dirs[corner[0]][0] * dirs[corner[1]][0] +
                                 dirs[corner[0]][1] * dirs[corner[1]][1]))
            b = round(math.degrees(math.acos(cdot)))
            return "6-panel-(4panel%s+2panel%s)" % (_afmt(a), _afmt(b))
        # Three straight lines through the node with panels [a,a,2a,a,a,2a]
        # (four small + two = 2x small) = two overlapping k-shape elements
        # -> 6-panel-(2*k-a).
        _c6 = sorted(math.degrees(math.atan2(d[1], d[0])) % 360.0 for d in dirs)
        _cg6 = sorted((_c6[(i + 1) % 6] - _c6[i]) % 360.0 for i in range(6))
        _a6 = sum(_cg6[:4]) / 4.0
        _b6 = sum(_cg6[4:]) / 2.0
        if (max(abs(x - _a6) for x in _cg6[:4]) < 12.0
                and max(abs(x - _b6) for x in _cg6[4:]) < 12.0
                and abs(_b6 - 2.0 * _a6) < 15.0):
            return "6-panel-(2*k-%s)" % _afmt(round(_a6))
        return "6-panel-%s" % _afmt(360.0 / 6)
    if n == 8:
        # cyclic panels. Alternating two values (e.g. 30/60) = two overlapping
        # 4-panel-(2panelX+2panelY) elements, where the sub-element angles are
        # TWICE the two panel values. Even panels give a regular 8-star.
        asort = sorted(math.degrees(math.atan2(d[1], d[0])) % 360.0 for d in dirs)
        cg = [(asort[(i + 1) % 8] - asort[i]) % 360.0 for i in range(8)]
        _ev, _od = cg[0::2], cg[1::2]
        _me = sum(_ev) / len(_ev)
        _mo = sum(_od) / len(_od)
        if (max(abs(x - _me) for x in _ev) < 12.0
                and max(abs(x - _mo) for x in _od) < 12.0
                and abs(_me - _mo) > 12.0):
            _hi = round(2.0 * max(_me, _mo))
            _lo = round(2.0 * min(_me, _mo))
            return ("8-panel-(2*(4-panel-(2panel%s+2panel%s)))"
                    % (_afmt(_hi), _afmt(_lo)))
        return "8-panel-%s" % _afmt(360.0 / 8)
    if n in (12, 16):
        return "%d-panel-%s" % (n, _afmt(360.0 / n))   # regular star: 360/n
    return "%d-panel-%d" % (n, int(round(min_gap)))


# ============================================================
# MAIN DETECTION PIPELINE
# ============================================================

def detect_grid_patterns_robust(img, prune_length=4):
    if img is None:
        return None
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- PHASE 1: preprocess the raw image into a clean 1px skeleton ----
    binary = preprocess_for_skeleton(img)

    raw_skel = skeletonize(binary // 255).astype(np.uint8)
    if prune_length > 0:
        raw_skel = prune_skeleton_spurs(raw_skel, min_branch_length=prune_length)
    raw_skel = prune_dead_end_branches(raw_skel, max_dead_end_length=25)

    line_thickness_est = max(3, int(np.sqrt(binary.sum() / 255 / max(raw_skel.sum(), 1)) * 2))
    rebuild_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (line_thickness_est, line_thickness_est))
    binary = cv2.dilate(raw_skel * 255, rebuild_kernel)

    # ---- PHASE 2: detect every circle in the figure ----
    img_h, img_w = img.shape
    detected_circles = detect_all_circles_combined(
        binary, raw_skel, img,  
        min_radius_px=max(10, min(img_h, img_w) // 40),
    )

    # Recover circles concentric with another (Hough suppresses same-centre).
    detected_circles = add_concentric_circles(
        raw_skel, detected_circles, min_r=max(10, min(img_h, img_w) // 40),
    )
    # Reject false positives: keep only clean circles for NODE logic.
    detected_circles = filter_real_circles(raw_skel, detected_circles)

    # Recover big concentric rings centred on the figure (they pass through the
    # smaller circles so they are only partly covered -> excluded from node
    # logic above, but still real circles that must be COUNTED).
    extra_rings = []
    if len(detected_circles) >= 3:
        cxm = sum(c[0] for c in detected_circles) / len(detected_circles)
        cym = sum(c[1] for c in detected_circles) / len(detected_circles)
        small_r = min(c[2] for c in detected_circles)
        for (rx, ry, rr) in find_center_rings(
                raw_skel, cxm, cym,
                min_r=int(small_r * 1.6), max_r=int(max(img_h, img_w) * 0.5)):
            if (_circle_shape_ok(raw_skel, rx, ry, rr, max_miss_frac=0.6) and
                all(abs(rr - c[2]) > max(10, 0.1 * rr) or
                    (rx - c[0]) ** 2 + (ry - c[1]) ** 2 > (0.2 * rr) ** 2
                    for c in detected_circles)):
                extra_rings.append((rx, ry, rr))

    # Total circle count = clean circles (used for nodes) + extra rings.
    total_circles_found = len(detected_circles) + len(extra_rings)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return None

    # ---- PHASE 3: identify the outer shape type and its vertices ----
    main_contour = max(contours, key=cv2.contourArea)
    shape_type, shape_vertices = get_refined_shape(main_contour)

    if shape_type in EXPECTED_SIDES:
        expected = EXPECTED_SIDES[shape_type]
        forced = detect_shape_vertices(binary, expected_sides=expected)
        if len(forced) == expected:
            shape_vertices = forced

    skeleton = raw_skel

    img_diag = int(np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2))
    marker_radius = max(8, img_diag // 80)
    marker_thickness = max(2, img_diag // 400)

    # ---- PHASE 4: detect every junction node ----
    # Junction nodes via straight-through-line aware refinement:
    #   straight pass-through chain -> ONE 'through' node (square)
    #   corner crossing            -> separate 'cross' nodes (circles)
    all_nodes = refine_junction_nodes(
        skeleton, max_link_dist=max(35, img_diag // 26),
        circles=detected_circles, circle_tol=max(10, img_diag // 70),
    )

    if shape_type != "Circle" and len(shape_vertices) > 0:
        dedup_distance = max(20, img_diag // 30)
        for (vx, vy) in shape_vertices:
            is_duplicate = False
            for (jx, jy, _) in all_nodes:
                if np.sqrt((vx - jx) ** 2 + (vy - jy) ** 2) < dedup_distance:
                    is_duplicate = True
                    break
            # only inject a real polygon corner (not a point on a smooth curve)
            if not is_duplicate and _is_real_corner(skeleton, vx, vy):
                all_nodes.append((vx, vy, 'vertex'))

    # Split an outer-ring corner where a spoke both meets the big outer circle
    # AND crosses a petal just inside it: the skeleton merges these into one
    # junction, but they are two nodes -> a degree-3 circle on the ring and a
    # degree-4 square at the petal. Guarded by "has a straight spoke" so a
    # spoke-less circle-circle lens (e.g. C14) is never split here.
    if detected_circles:
        ocx, ocy, oR = max(detected_circles, key=lambda c: c[2])
        ring_tol = max(10, img_diag // 70)
        split_nodes = []
        for (cx, cy, nt) in all_nodes:
            d_out = math.hypot(cx - ocx, cy - ocy)
            petal = None
            if abs(d_out - oR) < ring_tol and nt not in ('vertex',):
                for c in detected_circles:
                    if c[2] >= oR - 1:
                        continue
                    if abs(math.hypot(cx - c[0], cy - c[1]) - c[2]) < ring_tol:
                        petal = c
                        break
            if (petal is not None and d_out > 1
                    and _node_has_straight_arm(skeleton, cx, cy,
                                               detected_circles, ring_tol)):
                ux, uy = (cx - ocx) / d_out, (cy - ocy) / d_out
                ring_pt = (int(round(ocx + oR * ux)), int(round(ocy + oR * uy)))
                px, py, rp = petal
                bx, by = px - ocx, py - ocy
                B = ux * bx + uy * by
                Cc = bx * bx + by * by - rp * rp
                disc = B * B - Cc
                if disc >= 0:
                    t = B + math.sqrt(disc)          # petal crossing nearest the ring
                    petal_pt = (int(round(ocx + t * ux)), int(round(ocy + t * uy)))
                else:
                    petal_pt = (cx, cy)
                if (ring_pt[0] - petal_pt[0]) ** 2 + (ring_pt[1] - petal_pt[1]) ** 2 >= 16:
                    split_nodes.append((ring_pt[0], ring_pt[1], 'ring3'))
                    split_nodes.append((petal_pt[0], petal_pt[1], 'pcross4'))
                    continue
            split_nodes.append((cx, cy, nt))
        all_nodes = split_nodes

    # Add genuine degree-2 corners (sharp bends) not already captured.
    _existing_pts = [(p[0], p[1]) for p in all_nodes]
    all_nodes = all_nodes + detect_corner_nodes(
        skeleton, _existing_pts, circles=detected_circles,
        circle_tol=max(10, img_diag // 70), min_dist=max(20, img_diag // 45))

    # ---- PHASE 5: per-node measurement setup ----
    # Output canvas, shared tolerances, and the nested helpers every node
    # below draws on to trace its arms, count them, measure the wedge angle
    # between each pair, and test straightness / circle-following.
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    pattern_counts = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, "other": 0}

    edge_tol = max(4, img_diag // 200)
    vertex_tol = max(20, img_diag // 30)      
    core_radius_px = max(4, img_diag // 120)    
    arm_trace_length = max(20, img_diag // 22)  
    collinear_cos_threshold = -0.85             

    _neighbor_kernel_3x3 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    _nb_count = cv2.filter2D(skeleton, -1, _neighbor_kernel_3x3)
    is_junction_pixel = (skeleton == 1) & (_nb_count >= 3)

    def point_segment_distance(px, py, ax, ay, bx, by):
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        seg_len_sq = vx * vx + vy * vy
        if seg_len_sq == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len_sq))
        projx, projy = ax + t * vx, ay + t * vy
        return math.hypot(px - projx, py - projy)

    def on_straight_edge(cx, cy):
        if shape_type == "Circle" or len(shape_vertices) < 3:
            return False
        for (vx, vy) in shape_vertices:
            if math.hypot(cx - vx, cy - vy) < vertex_tol:
                return False
        n = len(shape_vertices)
        for i in range(n):
            ax, ay = shape_vertices[i]
            bx, by = shape_vertices[(i + 1) % n]
            if point_segment_distance(cx, cy, ax, ay, bx, by) < edge_tol:
                return True
        return False

    # -- arm counting & measurement: trace every arm leaving a node and fit
    # -- its precise sub-pixel direction (used below to count how many arms
    # -- a node has and measure the angle between each neighbouring pair) --
    def get_arm_directions(cy, cx):
        h, w = skeleton.shape
        visited = np.zeros_like(skeleton, dtype=bool)
        rmin = max(0, cy - core_radius_px - 2)
        rmax = min(h, cy + core_radius_px + 3)
        cmin = max(0, cx - core_radius_px - 2)
        cmax = min(w, cx + core_radius_px + 3)
        for r in range(rmin, rmax):
            for c in range(cmin, cmax):
                if (r - cy) ** 2 + (c - cx) ** 2 <= core_radius_px ** 2:
                    visited[r, c] = True

        exit_pts = []
        for r in range(rmin, rmax):
            for c in range(cmin, cmax):
                d2 = (r - cy) ** 2 + (c - cx) ** 2
                if d2 <= core_radius_px ** 2 or d2 > (core_radius_px + 3) ** 2:
                    continue
                if not skeleton[r, c]:
                    continue
                attached = False
                for ddr in (-1, 0, 1):
                    for ddc in (-1, 0, 1):
                        if ddr == 0 and ddc == 0:
                            continue
                        r2, c2 = r + ddr, c + ddc
                        if (0 <= r2 < h and 0 <= c2 < w
                                and skeleton[r2, c2]
                                and (r2 - cy) ** 2 + (c2 - cx) ** 2 <= core_radius_px ** 2):
                            attached = True
                            break
                    if attached:
                        break
                if attached:
                    exit_pts.append((r, c))

        deduped = []
        for (r, c) in exit_pts:
            keep = True
            for (r2, c2) in deduped:
                if (r - r2) ** 2 + (c - c2) ** 2 < 4:
                    keep = False
                    break
            if keep:
                deduped.append((r, c))

        arms = []
        for (sr, sc) in deduped:
            if visited[sr, sc]:
                continue
            path = [(sr, sc)]
            visited[sr, sc] = True
            for step in range(arm_trace_length):
                pr, pc = path[-1]
                if step > 2 and is_junction_pixel[pr, pc]:
                    break
                best = None
                best_d = -1
                for ddr in (-1, 0, 1):
                    for ddc in (-1, 0, 1):
                        if ddr == 0 and ddc == 0:
                            continue
                        rr, cc = pr + ddr, pc + ddc
                        if (0 <= rr < h and 0 <= cc < w
                                and skeleton[rr, cc] and not visited[rr, cc]):
                            d = (rr - cy) ** 2 + (cc - cx) ** 2
                            if d > best_d:
                                best_d = d
                                best = (rr, cc)
                if best is None:
                    break
                visited[best] = True
                path.append(best)
            if len(path) >= 3:
                er, ec = path[-1]
                ox, oy = ec - cx, er - cy            # outward reference
                if len(path) >= 5:
                    ppts = np.asarray([[c, r] for r, c in path], dtype=np.float64)
                    ppts -= ppts.mean(0)
                    try:
                        _, _, vt = np.linalg.svd(ppts, full_matrices=False)
                        pdir = vt[0]                  # best-fit line direction
                        if pdir[0] * ox + pdir[1] * oy < 0:
                            pdir = -pdir
                        dx, dy = float(pdir[0]), float(pdir[1])
                    except np.linalg.LinAlgError:
                        dx, dy = ox, oy
                else:
                    dx, dy = ox, oy
                # Sub-pixel refinement: walk the arm and find the darkness-
                # weighted centre of the stroke across its width (perpendicular
                # to the current direction). Averaging the anti-aliased stroke
                # removes skeleton staircase noise -> tighter angles. On thin
                # crisp lines the centre already equals the skeleton, so this is
                # a no-op there (safe).
                nnx, nny = -dy, dx
                sub = []
                for (pr, pc) in path[2:]:
                    num = den = 0.0
                    for tt in (-3, -2, -1, 0, 1, 2, 3):
                        sx = int(round(pc + nnx * tt)); sy = int(round(pr + nny * tt))
                        if 0 <= sy < h and 0 <= sx < w:
                            wv = 255.0 - float(img[sy, sx])
                            num += tt * wv; den += wv
                    if den > 0:
                        off = num / den
                        sub.append((pc + nnx * off, pr + nny * off))
                if len(sub) >= 5:
                    sp = np.asarray(sub, dtype=np.float64); sp -= sp.mean(0)
                    try:
                        _, _, vt2 = np.linalg.svd(sp, full_matrices=False)
                        p2 = vt2[0]
                        if p2[0] * ox + p2[1] * oy < 0:
                            p2 = -p2
                        dx, dy = float(p2[0]), float(p2[1])
                    except np.linalg.LinAlgError:
                        pass
                norm = math.hypot(dx, dy)
                if norm > 1e-6:
                    arms.append(((dx / norm, dy / norm), path))
        return arms

    def arm_is_straight(path, residual_thresh=0.00025):
        if len(path) < 6:
            return True 

        max_segment = 40
        trimmed = path[:max_segment] if len(path) > max_segment else path
        if len(trimmed) < 6:
            trimmed = path

        pts = np.asarray(trimmed, dtype=np.float64)
        diag = math.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1])
        if diag < 4:
            return True

        mean = pts.mean(axis=0)
        centered = pts - mean
        try:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return True
        if len(s) < 2:
            return True
        mean_sq_dist = (s[-1] ** 2) / len(pts)
        normalized = mean_sq_dist / (diag * diag + 1e-6)
        return normalized < residual_thresh

    def any_arm_curved(arms):
        for _direction, path in arms:
            if not arm_is_straight(path):
                return True
        return False

    def has_through_line(cx, cy, arms=None):
        if arms is None:
            arms = get_arm_directions(cy, cx)
        if len(arms) < 2:
            return False
        straight_arms = [a for a in arms if arm_is_straight(a[1])]
        for i in range(len(straight_arms)):
            for j in range(i + 1, len(straight_arms)):
                d1 = straight_arms[i][0]
                d2 = straight_arms[j][0]
                dot = d1[0] * d2[0] + d1[1] * d2[1]
                if dot < collinear_cos_threshold:
                    return True
        return False

    def _arm_follows_circle(path):
        # True if the traced arm runs along a detected (curved) circle.
        if not detected_circles or not path:
            return False
        ctol = max(6, img_diag // 110)
        on = 0
        for (r, c) in path:
            for (ccx, ccy, cr) in detected_circles:
                if abs(math.hypot(c - ccx, r - ccy) - cr) < ctol:
                    on += 1
                    break
        return on >= 0.6 * len(path)

    def has_straight_chord(arms):
        # Through-line made of STRAIGHT arms that are NOT circle arcs.
        if arms is None or len(arms) < 2:
            return False
        sa = [a for a in arms
              if arm_is_straight(a[1]) and not _arm_follows_circle(a[1])]
        for i in range(len(sa)):
            for j in range(i + 1, len(sa)):
                d1 = sa[i][0]; d2 = sa[j][0]
                if d1[0] * d2[0] + d1[1] * d2[1] < collinear_cos_threshold:
                    return True
        return False

    def draw_marker(img_out, cx, cy, radius, color, thickness, use_square):
        if use_square:
            cv2.rectangle(
                img_out,
                (cx - radius, cy - radius),
                (cx + radius, cy + radius),
                color, thickness,
            )
        else:
            cv2.circle(img_out, (cx, cy), radius, color, thickness)

    # ---- angle measurement: turn a node's traced arms into the wedge
    # ---- angle between each neighbouring pair, and label them on the image
    def _wedges_from_dirs(dirs):
        if len(dirs) < 2:
            return []
        arm_sorted = sorted(
            ((d, math.atan2(d[1], d[0]) % (2 * math.pi)) for d in dirs),
            key=lambda x: x[1],
        )
        results = []
        n = len(arm_sorted)
        for i in range(n):
            d1, a1 = arm_sorted[i]
            d2, a2 = arm_sorted[(i + 1) % n]
            wedge = math.degrees((a2 - a1) % (2 * math.pi))
            bx, by = d1[0] + d2[0], d1[1] + d2[1]
            bn = math.hypot(bx, by)
            if bn < 1e-6:
                bx, by = -d1[1], d1[0]
                bn = 1.0
            else:
                bx, by = bx / bn, by / bn
                bn = 1.0
                if wedge > 180.0:
                    bx, by = -bx, -by
            results.append((wedge, bx, by))
        return results

    def compute_wedge_angles(cx, cy):
        corner_dirs = None
        if shape_type != "Circle" and len(shape_vertices) >= 3:
            match_tol = max(20, img_diag // 40)
            best_i = -1
            best_d = match_tol
            for i, (vx, vy) in enumerate(shape_vertices):
                d = math.hypot(vx - cx, vy - cy)
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i >= 0:
                prev_v = shape_vertices[(best_i - 1) % len(shape_vertices)]
                next_v = shape_vertices[(best_i + 1) % len(shape_vertices)]
                px, py = prev_v[0] - cx, prev_v[1] - cy
                nx, ny = next_v[0] - cx, next_v[1] - cy
                np_ = math.hypot(px, py); nn = math.hypot(nx, ny)
                if np_ >= 1 and nn >= 1:
                    corner_dirs = [(px / np_, py / np_), (nx / nn, ny / nn)]

        trace_cy, trace_cx = cy, cx
        if not skeleton[cy, cx]:
            search_r = max(15, img_diag // 60)
            h_s, w_s = skeleton.shape
            best_dist = float('inf')
            for r in range(max(0, cy - search_r), min(h_s, cy + search_r + 1)):
                for c in range(max(0, cx - search_r), min(w_s, cx + search_r + 1)):
                    if skeleton[r, c]:
                        d2 = (r - cy) ** 2 + (c - cx) ** 2
                        if d2 < best_dist:
                            best_dist = d2
                            trace_cy, trace_cx = r, c

        traced = get_arm_directions(trace_cy, trace_cx)
        traced_dirs = [a[0] for a in traced]

        if corner_dirs is not None and traced_dirs:
            # Refine each TRACED arm to a nearby corner (polygon-edge) direction
            # when one aligns, but never ADD a corner direction that matches no
            # traced arm. Adding unmatched corner dirs inflated 2-arm boundary
            # nodes (e.g. a honeycomb vertex whose "adjacent shape corners" are
            # far away) into 3-4 spurious wedges like 27/24/96.
            min_sep_cos = math.cos(math.radians(25))
            refined = []
            for td in traced_dirs:
                pick = td
                best_c = min_sep_cos
                for cd in corner_dirs:
                    c = td[0] * cd[0] + td[1] * cd[1]
                    if c > best_c:
                        best_c = c
                        pick = cd
                refined.append(pick)
            return _wedges_from_dirs(refined)

        return _wedges_from_dirs(traced_dirs)

    def draw_wedge_labels(img_out, cx, cy, wedge_entries, marker_radius):
        if not wedge_entries:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.4, img_diag / 2800.0)
        thickness = max(1, int(scale * 2))
        label_radius = marker_radius + max(14, int(img_diag / 60))
        h_img, w_img = img_out.shape[:2]
        margin = 4
        for (angle, bx, by) in wedge_entries:
            text = f"{int(round(angle))}"
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            lx = cx + bx * label_radius
            ly = cy + by * label_radius
            tries = 0
            while tries < 30:
                left = lx - tw / 2.0
                right = lx + tw / 2.0
                top = ly - th / 2.0
                bot = ly + th / 2.0
                if (left >= margin and right < w_img - margin
                        and top >= margin and bot < h_img - margin):
                    break
                lx -= bx * label_radius * 0.1
                ly -= by * label_radius * 0.1
                tries += 1
            org = (int(lx - tw / 2), int(ly + th / 2))
            cv2.putText(img_out, text, org, font, scale, (255, 255, 255),
                        thickness + 2, cv2.LINE_AA)
            cv2.putText(img_out, text, org, font, scale, (0, 0, 0),
                        thickness, cv2.LINE_AA)

    node_records = []
    shape_counts = {}

    circle_geom = None
    if shape_type == "Circle":
        (ccx, ccy), c_radius = cv2.minEnclosingCircle(main_contour)
        circle_geom = (ccx, ccy, c_radius)

    all_node_pts = [(p[0], p[1]) for p in all_nodes]

    # ---- PHASE 6: the main per-node loop ----
    # For every detected node: work out its degree, count and measure its
    # arms (wedge angles between them), then classify the result into a
    # taxonomy shape label and record it.
    for (cx, cy, node_type) in all_nodes:
        is_shape_vertex = (node_type == 'vertex')

        scy, scx = snap_to_skeleton(skeleton, cy, cx, max(10, img_diag // 70))
        arms = get_arm_directions(scy, scx)

        # --- Fixed-radii tangent for circular arms --------------------------
        # For an arm that follows a detected circle, replace its noisy per-arm
        # SVD tangent with the exact chord direction implied by the fitted
        # circle: the node tangent leaned toward the circle centre by L/(2r)
        # (L = traced arc length, r = radius). Symmetric arc nodes (e.g. C14)
        # then give symmetric, noise-free angles. Straight-wall arms follow no
        # circle and are left unchanged.
        if detected_circles and arms:
            _ctol = max(6, img_diag // 110)
            _fixed = []
            for (_d, _p) in arms:
                _best = None; _beston = 0.0
                for (_ccx, _ccy, _cr) in detected_circles:
                    if _cr <= 1e-6:
                        continue
                    _on = 0
                    for (_r, _c) in _p:
                        if abs(math.hypot(_c - _ccx, _r - _ccy) - _cr) < _ctol:
                            _on += 1
                    if _on >= 0.6 * len(_p) and _on > _beston:
                        _beston = _on; _best = (_ccx, _ccy, _cr)
                if _best is not None:
                    _ccx, _ccy, _cr = _best
                    _ux, _uy = scx - _ccx, scy - _ccy
                    _un = math.hypot(_ux, _uy)
                    if _un > 1e-6:
                        _ux, _uy = _ux / _un, _uy / _un
                        _tx, _ty = -_uy, _ux
                        _sgn = 1.0 if (_d[0] * _tx + _d[1] * _ty) >= 0 else -1.0
                        _L = 0.0
                        for _k in range(1, len(_p)):
                            _L += math.hypot(_p[_k][0] - _p[_k - 1][0],
                                             _p[_k][1] - _p[_k - 1][1])
                        _th = _L / (2.0 * _cr)
                        _nx = _sgn * math.cos(_th) * _tx - math.sin(_th) * _ux
                        _ny = _sgn * math.cos(_th) * _ty - math.sin(_th) * _uy
                        _nn = math.hypot(_nx, _ny)
                        if _nn > 1e-6:
                            _d = (_nx / _nn, _ny / _nn)
                _fixed.append((_d, _p))
            arms = _fixed
        # --------------------------------------------------------------------

        # A node lying on TWO circles with no straight line through it is a
        # circle-circle meeting point (a "lens"): one degree-4 circle marker.
        _chord = has_straight_chord(arms)
        _lens_tol = max(10, img_diag // 70)
        _on_circ = sum(1 for (ccx, ccy, rr) in detected_circles
                       if abs(math.hypot(cx - ccx, cy - ccy) - rr) < _lens_tol)
        # A spoke (straight non-arc arm) means this is NOT a pure lens: e.g. a
        # spoke meeting the outer ring where a petal is tangent = degree 5.
        _has_spoke = _node_has_straight_arm(skeleton, cx, cy,
                                            detected_circles, _lens_tol)
        is_lens = (node_type not in ('circ6', 'xcross', 'vertex')
                   and _on_circ >= 2 and not _chord and not _has_spoke)

        # A straight wall running exactly tangent to ONE detected circle (a
        # circle inscribed in a polygon, touching at an edge midpoint, e.g.
        # S15): right at the contact point the edge and the circle are
        # indistinguishable, so the arm tracer only ever finds the edge's own
        # 2 (near-collinear) exits, never the circle's. Treat it as a degree-4
        # node (2 straight edge arms + 2 circle-arc arms) instead of letting it
        # fall through as a plain 2-panel corner.
        _edge_tangent_circle = None
        if not is_lens and _on_circ == 1 and len(arms) == 2:
            # Near a true tangency the traced arm can run close enough to the
            # circle over its whole length that _arm_follows_circle (used by
            # has_straight_chord) mistakes the STRAIGHT edge arm for a circle
            # arc and discards it -- so check straightness directly here
            # instead of relying on has_straight_chord/_chord.
            (_d0, _p0), (_d1, _p1) = arms[0], arms[1]
            _opposite = (_d0[0] * _d1[0] + _d0[1] * _d1[1]) < -0.85
            if (_opposite and arm_is_straight(_p0)
                    and arm_is_straight(_p1)):
                for (ccx, ccy, rr) in detected_circles:
                    if abs(math.hypot(cx - ccx, cy - ccy) - rr) < _lens_tol:
                        _edge_tangent_circle = (ccx, ccy, rr)
                        break
        is_edge_tangent = _edge_tangent_circle is not None

        # Degree from a ring count (sees curves that overlap near the node
        # but diverge further out, e.g. a circle arc tangent to an edge).
        if node_type == 'ring3':
            degree = 3          # spoke meets the outer ring: 1 + 2 arcs
        elif node_type == 'pcross4':
            degree = 4          # spoke crosses a petal: 2 + 2
        elif node_type == 'xcross':
            degree = 4          # straight line crossing a circle, by construction
        elif node_type == 'circ6':
            degree = 6          # spoke through two circles: 2 + 2 + 2
        elif is_lens:
            degree = 4          # two circles meeting: 2 + 2 arcs
        elif is_edge_tangent:
            degree = 4          # straight edge tangent to a circle: 2 + 2 arcs
        else:
            degree = degree_from_ring(skeleton, cx, cy, all_node_pts)
            if degree <= 0:
                degree = len(arms)

        # Retained: Custom degree tracking for types
        if is_shape_vertex and degree < 2:
            degree = 2

        # Degree-2 is not a junction. Keep it only if it is a genuine polygon
        # corner (two straight edges at an angle) AND not lying on a circle /
        # smooth curve (a point on a circle is not a node).
        if degree == 2:
            on_c = any(abs(math.hypot(cx - ccx, cy - ccy) - rr) < _lens_tol
                       for (ccx, ccy, rr) in detected_circles)
            if on_c or not _is_real_corner(skeleton, cx, cy):
                continue

        on_circle_perimeter = False
        if circle_geom is not None:
            dist_to_center = math.hypot(cx - circle_geom[0], cy - circle_geom[1])
            # Tolerance scales with the circle's OWN radius (not a flat
            # image-size fraction): a vertex meant to sit on a big circle can
            # be a few percent of that radius off (drawing/generation
            # imprecision) and should still count as "on the circle", same
            # as every other on-circle check in this file already does
            # relative to its own feature size.
            if abs(dist_to_center - circle_geom[2]) < max(8, 0.05 * circle_geom[2]):
                on_circle_perimeter = True

        circle_tol = max(8, img_diag // 80)
        on_any_circle = is_on_any_circle(cx, cy, detected_circles, circle_tol)
        if not on_any_circle and detected_circles:
            on_any_circle = any(
                abs(math.hypot(cx - ccx, cy - ccy) - rr) < max(8, 0.05 * rr)
                for (ccx, ccy, rr) in detected_circles)

        straight_through = has_through_line(scx, scy, arms)
        on_polygon_edge = on_straight_edge(cx, cy)
        any_curved = any_arm_curved(arms)

        # Marker shape: SQUARE iff a straight line passes through the node.
        if node_type == 'ring3':
            use_square = False           # spoke meeting the curved ring -> circle
        elif node_type == 'pcross4':
            use_square = True            # spoke through a petal -> square
        elif is_lens:
            use_square = False           # circle-circle meeting -> circle
        elif node_type in ('xcross', 'circ6'):
            use_square = True
        elif node_type in ('through', 'corner'):
            # collapsed cluster: square iff a straight line really passes
            # through. 'through' already means "refine_junction_nodes found
            # a straight connector in this cluster" and is square in the
            # ordinary case -- but on a node that ALSO sits on a detected
            # circle, a diamond/polygon vertex can get mislabelled 'through'
            # even though its own two edges meet at a genuine corner, not a
            # straight pass-through, so re-check with the same has_straight_
            # chord test 'corner' already uses rather than trusting the
            # upstream label blindly.
            use_square = has_straight_chord(arms)
        elif node_type == 'vertex':
            if on_circle_perimeter or on_any_circle:
                use_square = False
            elif any_curved:
                use_square = False
            elif on_polygon_edge:
                use_square = True
            else:
                use_square = straight_through
        else:  # 'single': square iff a STRAIGHT (non-arc) line passes through
            use_square = has_straight_chord(arms)

        wedges = compute_wedge_angles(cx, cy)

        _lens_ang = None
        _lens_cross = False
        _crc5_panel_ang = None   # degree-5 node that sits on 2 DIFFERENT circles
        if is_lens:
            # Two circles meeting at this node. Nothing here uses
            # marker_radius (a cosmetic drawing-size constant unrelated to
            # the drawn geometry) -- every quantity below comes only from
            # the two circles' own fitted centres and radii.
            _lc = [(ccx, ccy, rr) for (ccx, ccy, rr) in detected_circles
                   if abs(math.hypot(cx - ccx, cy - ccy) - rr) < _lens_tol]
            if len(_lc) >= 2:
                _lc.sort(key=lambda c: c[2])
                (_c1x, _c1y, _r1), (_c2x, _c2y, _r2) = _lc[0], _lc[-1]
                # Each circle's own EXACT tangent at the node is perpendicular
                # to that circle's own radius into the node. To that, lean
                # each circle's tangent toward its own centre by a small tilt
                # -- a fixed 10-degree budget split between the two circles
                # in proportion to the OTHER circle's radius, so the smaller,
                # sharper-curving circle always takes the larger share of the
                # tilt and the bigger, gentler circle takes the smaller share,
                # and the reading is never exactly 0 even at perfect tangency.
                # Only r1 and r2 drive the SPLIT; the 10-degree budget itself
                # is a chosen display constant (not a pixel one like
                # marker_radius), picked to keep the numbers legible.
                _tilt_budget = math.radians(10.0)
                _rsum = _r1 + _r2
                _dlt1 = _tilt_budget * (_r2 / _rsum) if _rsum > 0 else 0.0
                _dlt2 = _tilt_budget * (_r1 / _rsum) if _rsum > 0 else 0.0

                _r1len = math.hypot(cx - _c1x, cy - _c1y) or 1.0
                _r1xu, _r1yu = (cx - _c1x) / _r1len, (cy - _c1y) / _r1len  # outward radial
                _t1xu, _t1yu = -_r1yu, _r1xu                                 # tangent

                _r2len = math.hypot(cx - _c2x, cy - _c2y) or 1.0
                _r2xu, _r2yu = (cx - _c2x) / _r2len, (cy - _c2y) / _r2len
                _t2xu, _t2yu = -_r2yu, _r2xu

                # K vs X: which way the two arcs curve relative to each
                # other. Each circle's arc bends toward its OWN centre, so
                # the direction from the node to that centre is the side it
                # curves to. Opposite directions (dot product negative) ->
                # the arcs bow away from one another, a genuine X crossing.
                # Same direction (dot product >= 0) -> the arcs hug the same
                # side, a K element, e.g. a small circle nested just inside
                # a bigger one. Computed BEFORE the lean below because the
                # lean's own sign now depends on it.
                _in1xu, _in1yu = -_r1xu, -_r1yu
                _in2xu, _in2yu = -_r2xu, -_r2yu
                _lens_cross = (_in1xu * _in2xu + _in1yu * _in2yu) < 0.0

                # Lean each tangent toward its own centre by its share of the
                # budget. For a K pair both leans turn the SAME absolute way,
                # so the result is the DIFFERENCE |dlt1-dlt2| (varies with
                # the radius ratio). For an X pair (centres opposite) the
                # second lean is flipped to the OPPOSITE absolute sense, so
                # the two tangents fan apart instead of sliding together,
                # giving the SUM dlt1+dlt2.
                _dlt2_signed = -_dlt2 if _lens_cross else _dlt2

                _cd1, _sd1 = math.cos(_dlt1), math.sin(_dlt1)
                _lt1xu = _cd1 * _t1xu - _sd1 * _r1xu   # leaned toward circle 1's centre
                _lt1yu = _cd1 * _t1yu - _sd1 * _r1yu

                _cd2, _sd2 = math.cos(_dlt2_signed), math.sin(_dlt2_signed)
                _lt2xu = _cd2 * _t2xu - _sd2 * _r2xu   # leaned toward circle 2's centre
                _lt2yu = _cd2 * _t2yu - _sd2 * _r2yu

                _ld = [(_lt1xu, _lt1yu), (-_lt1xu, -_lt1yu),
                       (_lt2xu, _lt2yu), (-_lt2xu, -_lt2yu)]
                wedges = _wedges_from_dirs(_ld)
                _sm = sorted(w[0] for w in wedges)[:2]   # the two splay panels
                _lens_ang = sum(_sm) / len(_sm) if _sm else None
        elif is_edge_tangent:
            # A straight wall exactly tangent to one circle: reconstruct the
            # true 4 arms analytically instead of trusting the arm tracer
            # (which only ever finds the 2 straight edge exits here). 2 EXACT
            # edge directions (the wall is straight) + 2 circle-arc directions
            # leaning toward the circle's own centre by the curvature tilt
            # (same construction used for the circle-circle lens above).
            _ecx, _ecy, _er = _edge_tangent_circle
            _rlen = math.hypot(cx - _ecx, cy - _ecy) or 1.0
            _rxu, _ryu = (cx - _ecx) / _rlen, (cy - _ecy) / _rlen  # outward radial
            _txu, _tyu = -_ryu, _rxu                                 # tangent = edge dir
            _dlt = min(marker_radius / _er, 0.05)   # capped, see note above
            _cd, _sd = math.cos(_dlt), math.sin(_dlt)
            wedges = _wedges_from_dirs([
                (_txu, _tyu),
                (_cd * _txu - _sd * _rxu, _cd * _tyu - _sd * _ryu),
                (-_txu, -_tyu),
                (-_cd * _txu - _sd * _rxu, -_cd * _tyu - _sd * _ryu),
            ])
            _tang_sm = sorted(w[0] for w in wedges)[:2]   # the two splay panels
            _tang_ang = sum(_tang_sm) / len(_tang_sm) if _tang_sm else None
        elif (_on_circ >= 1
              and (degree in (3, 5) or (degree == 4 and not use_square))
              and node_type not in ('ring3', 'pcross4', 'xcross', 'circ6')):
            # A node on a detected circle where 2 of its arms are that same
            # circle's own arcs (3e-arc at degree 3; a spoke + 2 extra chords
            # meeting the circle at degree 5, e.g. a star point -> "5-crc-panel";
            # a polygon VERTEX sitting on the circle at degree 4, e.g. a square
            # inscribed in a circle, where the vertex's own 2 edges are NOT a
            # through-line -- use_square is already False there, so this never
            # overlaps the separate "spoke crosses circle" degree-4 correction
            # below, which only fires when use_square is True).
            # The circle's two tangent directions at any single point are by
            # definition exactly opposite (one tangent line, 180 apart) --
            # but the traced direction is a CHORD over a long arc (wrong on
            # small/busy circles), so rebuild the wedges from the TRUE TANGENT
            # at the node for arms that run along a detected circle -> the two
            # arcs read as ~collinear (e.g. a spoke at the top of a circle:
            # 180/90/90) instead of the arc gap drifting off true (e.g. 196,
            # or a square vertex's corner reading 38/90/38/193 instead of the
            # true 45/90/45/180).
            #
            # EXCEPT: a degree-5 node that sits directly on TWO DIFFERENT
            # circles (e.g. a small circle nested almost-tangent just inside
            # a much bigger one -- the same pair a direct circle-circle lens
            # would see) isn't really "a point on one circle plus a spoke" --
            # it's where two circles themselves meet, with a spoke on top.
            # This is the SAME small-vs-big circle tilt angle a direct lens
            # meeting measures (lean each circle's own tangent toward its own
            # centre by the curvature-share construction, independently, not
            # forced onto one shared line) -- just read off the WIDE pair of
            # panels this time instead of the narrow one (is_lens's own
            # label uses the narrow pair, e.g. a K-arc-6 at a similar C14
            # pair is really 6/174/6/174 -- the 174 is this same number).
            # The spoke sits exactly perpendicular to the original tangent,
            # i.e. exactly on the midline of that wide span, so it splits it
            # into two EQUAL flanking panels -- shown that way rather than
            # at the spoke's own slightly-off raw direction, since the two
            # circles' curvature difference is already fully carried by the
            # (now wide, not narrow) angle itself.
            _crc5_lc = ([(ccx, ccy, rr) for (ccx, ccy, rr) in detected_circles
                         if abs(math.hypot(cx - ccx, cy - ccy) - rr) < _lens_tol]
                        if degree == 5 else [])
            if len(_crc5_lc) >= 2:
                _crc5_lc.sort(key=lambda c: c[2])
                (_p1x, _p1y, _pr1), (_p2x, _p2y, _pr2) = _crc5_lc[0], _crc5_lc[-1]
                _tilt_budget5 = math.radians(10.0)
                _prsum = _pr1 + _pr2
                _pdlt1 = _tilt_budget5 * (_pr2 / _prsum) if _prsum > 0 else 0.0
                _pdlt2 = _tilt_budget5 * (_pr1 / _prsum) if _prsum > 0 else 0.0

                _p1len = math.hypot(cx - _p1x, cy - _p1y) or 1.0
                _p1xu, _p1yu = (cx - _p1x) / _p1len, (cy - _p1y) / _p1len
                _pt1xu, _pt1yu = -_p1yu, _p1xu
                _pcd1, _psd1 = math.cos(_pdlt1), math.sin(_pdlt1)
                _plt1x = _pcd1 * _pt1xu - _psd1 * _p1xu
                _plt1y = _pcd1 * _pt1yu - _psd1 * _p1yu

                _p2len = math.hypot(cx - _p2x, cy - _p2y) or 1.0
                _p2xu, _p2yu = (cx - _p2x) / _p2len, (cy - _p2y) / _p2len
                _pt2xu, _pt2yu = -_p2yu, _p2xu
                _pcd2, _psd2 = math.cos(_pdlt2), math.sin(_pdlt2)
                _plt2x = _pcd2 * _pt2xu - _psd2 * _p2xu
                _plt2y = _pcd2 * _pt2yu - _psd2 * _p2yu

                # Identify the spoke (the raw traced arm LEAST aligned with
                # either circle's tangent line), purely to tell which of the
                # two equal wide gaps it actually falls in.
                _spoke_dir = None
                _best_score = None
                for (_adir, _) in arms:
                    _s1 = abs(_adir[0] * _pt1xu + _adir[1] * _pt1yu)
                    _s2 = abs(_adir[0] * _pt2xu + _adir[1] * _pt2yu)
                    _score = max(_s1, _s2)
                    if _best_score is None or _score < _best_score:
                        _best_score = _score
                        _spoke_dir = _adir

                _p4 = _wedges_from_dirs([(_plt1x, _plt1y), (-_plt1x, -_plt1y),
                                         (_plt2x, _plt2y), (-_plt2x, -_plt2y)])
                if _spoke_dir is not None and _p4:
                    # Pick whichever of the 4 gaps the spoke actually sits
                    # in, by nearest bisector -- not by first filtering to
                    # "the wide ones", which is numerically fragile (the two
                    # wide gaps are only equal by symmetry; in float they can
                    # differ by an epsilon, so an exact-equality filter can
                    # keep just one of them, which may be the wrong one).
                    _picked = max(
                        _p4, key=lambda w: w[1] * _spoke_dir[0] + w[2] * _spoke_dir[1])
                    _wbx, _wby = _picked[1], _picked[2]
                    _crc5_panel_ang = _picked[0]
                    wedges = _wedges_from_dirs([
                        (_plt1x, _plt1y), (-_plt1x, -_plt1y),
                        (_plt2x, _plt2y), (-_plt2x, -_plt2y),
                        (_wbx, _wby)])
            else:
                _td = []
                _ctol = max(6, img_diag // 110)
                for (_dir, _path) in arms:
                    _fx = _dir
                    for (ccx, ccy, rr) in detected_circles:
                        if _path and sum(1 for (pr, pc) in _path
                                         if abs(math.hypot(pc - ccx, pr - ccy) - rr)
                                         < _ctol) >= 0.7 * len(_path):
                            _rx, _ry = scx - ccx, scy - ccy
                            _tx, _ty = -_ry, _rx
                            if _tx * _dir[0] + _ty * _dir[1] < 0:
                                _tx, _ty = -_tx, -_ty
                            _n = math.hypot(_tx, _ty)
                            if _n > 1e-6:
                                _fx = (_tx / _n, _ty / _n)
                            break
                    _td.append(_fx)
                if len(_td) >= 2:
                    wedges = _wedges_from_dirs(_td)
        elif (degree == 4 and not is_lens and use_square
              and node_type not in ('circ6', 'pcross4')):
            # A straight line crossing a circle. The spoke is the most-opposite
            # pair of arms (the through-line); the other two arms are the arcs.
            # Treat the arcs as straight lines (circle tangent): if that makes a
            # perpendicular '+', it is 4-panel-90. Oblique crossings keep their
            # original (trident/X) reading.
            _nn = len(arms)
            # Identify the 2 arc arms by whether they genuinely follow a
            # detected circle, NOT by first guessing the spoke as "whichever
            # pair is most opposite". On a SMALL circle close to another
            # junction, one spoke arm's trace can get cut short (it hits the
            # next junction only a few px out) and its SVD-fit direction ends
            # up noticeably off-axis -- that corrupted arm can then look MORE
            # "opposite" to an arc arm than the true opposite spoke arm does,
            # misidentifying which 2 arms are the arc in the first place.
            # A short, truncated trace (e.g. a spoke arm cut off after a
            # handful of points because the next junction sits only a few px
            # away) can spuriously satisfy _arm_follows_circle by chance --
            # near a tangent point two different circles both pass close by,
            # so a short segment can look like it "follows" one of them even
            # though it is really the corrupted spoke arm. Require a decent
            # minimum length before trusting the circle-following verdict.
            _arc = [k for k in range(_nn)
                    if arms[k][1] and len(arms[k][1]) >= 15
                    and _arm_follows_circle(arms[k][1])]
            _spoke = [k for k in range(_nn) if k not in _arc]
            if len(_arc) == 2 and len(_spoke) == 2:
                _cc = max(6, img_diag // 110)
                _nd = [a[0] for a in arms]
                for _k in _arc:
                    _p = arms[_k][1]; _d = arms[_k][0]
                    for (ccx, ccy, rr) in detected_circles:
                        if sum(1 for (pr, pc) in _p
                               if abs(math.hypot(pc - ccx, pr - ccy) - rr)
                               < _cc) >= 0.6 * len(_p):
                            _tx, _ty = -(scy - ccy), (scx - ccx)
                            if _tx * _d[0] + _ty * _d[1] < 0:
                                _tx, _ty = -_tx, -_ty
                            _n = math.hypot(_tx, _ty)
                            if _n > 1e-6:
                                _nd[_k] = (_tx / _n, _ty / _n)
                            break
                # The spoke is a single straight line through the node, so its
                # two measured directions should be opposite. If they aren't,
                # one was mistraced (cut short near another junction) -- trust
                # whichever spoke arm traced the LONGER path (more points to
                # fit, so a more reliable direction) and set the other to its
                # exact opposite instead of its own corrupted reading.
                _s0, _s1 = _spoke
                _dot_spoke = (arms[_s0][0][0] * arms[_s1][0][0]
                              + arms[_s0][0][1] * arms[_s1][0][1])
                if _dot_spoke >= -0.9:
                    _len0 = len(arms[_s0][1]) if arms[_s0][1] else 0
                    _len1 = len(arms[_s1][1]) if arms[_s1][1] else 0
                    _trusted, _other = (_s0, _s1) if _len0 >= _len1 else (_s1, _s0)
                    _td = _nd[_trusted]
                    _nd[_other] = (-_td[0], -_td[1])
                # Accept only if the straightened arc gives a perpendicular '+'
                # (a radial spoke crossing a circle). Oblique crossings keep
                # their original (trident/X) reading.
                _w2 = _wedges_from_dirs(_nd)
                if _w2 and abs(min(w[0] for w in _w2) - 90.0) < 15.0:
                    arms = [(_nd[i], arms[i][1]) for i in range(len(arms))]
                    wedges = _w2
        elif node_type == 'circ6':
            # A straight spoke running through TWO tangent/overlapping
            # circles: 2 EXACT spoke directions (the line is straight) + 2
            # EXACT tangent directions PER circle (perpendicular to the
            # radius from EACH circle's own centre to the node) -- 6 arms
            # total, same exact-tangent construction as the is_lens fix
            # above (no cosmetic lean: since the two circles are themselves
            # tangent to each other here, their tangent lines are genuinely
            # near-identical, so the "small" panels correctly come out near 0
            # and the "big" panels near 90, both as exact as the measured
            # circle fit allows). The raw arm tracer only ever finds ~4: right
            # at the shared tangent point the two circles' own tangent
            # directions are nearly identical, so it can't tell them apart as
            # separate arcs.
            _c6 = [(ccx, ccy, rr) for (ccx, ccy, rr) in detected_circles
                   if abs(math.hypot(cx - ccx, cy - ccy) - rr) < _lens_tol]
            _spoke_dir = None
            for (_d, _p) in arms:
                if arm_is_straight(_p) and not _arm_follows_circle(_p):
                    _spoke_dir = _d
                    break
            if _spoke_dir is None and arms:
                _spoke_dir = arms[0][0]
            if _spoke_dir is not None and len(_c6) >= 2:
                _sx, _sy = _spoke_dir
                _dirs6 = [(_sx, _sy), (-_sx, -_sy)]
                for (ccx, ccy, rr) in _c6[:2]:
                    _rlen6 = math.hypot(cx - ccx, cy - ccy) or 1.0
                    _rxu6, _ryu6 = (cx - ccx) / _rlen6, (cy - ccy) / _rlen6
                    _txu6, _tyu6 = -_ryu6, _rxu6          # EXACT tangent, no lean
                    _dirs6.append((_txu6, _tyu6))
                    _dirs6.append((-_txu6, -_tyu6))
                wedges = _wedges_from_dirs(_dirs6)
        elif node_type == 'pcross4':
            # A spoke crossing a petal, perpendicular: treat the arc as a straight
            # line and show a clean '+' from the spoke direction -> 4-panel-90.
            _sp = None
            for (_d, _p) in arms:
                if arm_is_straight(_p) and not _arm_follows_circle(_p):
                    _sp = _d
                    break
            if _sp is None and arms:
                _sp = arms[0][0]
            if _sp is not None:
                _pxx, _pyy = -_sp[1], _sp[0]
                wedges = _wedges_from_dirs(
                    [_sp, (-_sp[0], -_sp[1]), (_pxx, _pyy), (-_pxx, -_pyy)])

        if 2 <= degree <= 12:
            pattern_counts[degree] += 1
        else:
            pattern_counts["other"] += 1

        draw_wedge_labels(output_img, cx, cy, wedges, marker_radius)

        # Classify the node's geometric shape (X, +, T, Y, curved X, ...).
        if node_type == 'circ6':
            # spoke through two meeting circles = a symmetric six-way crossing
            # (six 60 panels) -> the regular 6-panel element.
            shape = "6-panel-60"
        elif node_type == 'pcross4':
            # A straight spoke crossing a petal, perpendicular: the arc as a
            # straight line makes a '+' -> 4-panel-90 (wedges already rebuilt).
            shape = "4-panel-90"
        elif node_type == 'ring3':
            shape = "3-panel-claw-arc-90(3e-arc)"
        elif is_lens:
            # No floor here: 0 is a real, meaningful reading (near-perfect
            # tangency) and must match the on-image panel numbers exactly,
            # which are never floored either.
            if _lens_cross:
                shape = "4-panel-X-%s" % _afmt(
                    round(_lens_ang) if _lens_ang is not None else 90)
            else:
                shape = "4-panel-K-arc-%s" % _afmt(
                    round(_lens_ang) if _lens_ang is not None else 45)
        elif is_edge_tangent:
            # straight wall tangent to a circle: a degenerate 4-panel-K where
            # the "second line" is the circle grazing the wall at (near) 0.
            shape = "4-panel-K-arc-%s" % _afmt(
                max(1, round(_tang_ang)) if _tang_ang is not None else 5)
        elif _crc5_panel_ang is not None:
            # The two-circle lean+spoke construction above already built
            # `wedges` to match -- the label just reads the same number.
            shape = "5-crc-panel-(3T+2panel%s)" % _afmt(round(_crc5_panel_ang))
        else:
            _adirs = [a[0] for a in arms]
            _astr = [arm_is_straight(a[1]) for a in arms]
            shape = classify_node_shape(
                _adirs, _astr, degree, on_circle=(_on_circ >= 1),
                wedge_angs=[w[0] for w in wedges],
                marker_circle=(not use_square))
        # A 3-T element has a straight line passing straight through it (the
        # crossbar), so it is always a SQUARE marker. At grid-boundary nodes the
        # arm tracer can miss the through-line, so key this off the (reliable)
        # shape label instead.
        if shape.startswith("3-panel-claw-90"):
            use_square = True


        shape_counts[shape] = shape_counts.get(shape, 0) + 1

        node_records.append({
            "x": int(cx),
            "y": int(cy),
            "degree": degree,
            "marker": "square" if use_square else "circle",
            "shape": shape,
            "angles_deg": [round(float(w[0]), 1) for w in wedges],
        })

    # ---- PHASE 7: consolidate labels, then compute RT / RG / Omega ----
    # Merge near-equal measured angles in the same family (symmetry / noise),
    # e.g. 4-cir-k-31 and 4-cir-k-32 -> one category.
    shape_counts = consolidate_shape_angles(shape_counts, node_records)

    # ---- crashworthiness numbers (RT membrane energy, RG complexity, Omega) ----
    # Total wall length is measured by the 8-adjacency pixel-count method: it sums
    # orthogonal (=1) and diagonal (=sqrt2) skeleton steps and drops redundant
    # staircase diagonals, so straight AND curved walls are measured at their true
    # length. Validated on squares (S1=300, S2=450, S4=512) AND circular tubes
    # (C14, C20, C22 within ~1%). The earlier analytical r*angle arc method
    # systematically under-measured circular arcs by ~5%.
    _skel_len_px = skeleton_length_px(skeleton)
    # For the 'circle inscribed in a polygon' composite (e.g. O2, S15) the circle
    # is tangent to every polygon edge, and skeletonisation merges the two walls
    # at each tangent point, under-measuring the length. Rebuild the outer walls
    # from the clean detected primitives when that exact geometry is present.
    _recon_len = composite_reconstruct_len_px(skeleton, detected_circles, shape_type)
    if _recon_len is None:
        # near-pure circle packing (e.g. C1, C14): measure circles from a
        # least-squares fit (exact 2*pi*r) instead of the pixel count, which
        # over-measures smooth curves.
        _recon_len = circle_packing_len_px(skeleton, detected_circles)
    if _recon_len is None:
        # isolated circle(s) + straight walls (e.g. C3, C4, C10): measure the
        # circle exactly (2*pi*r) and the straight walls end-to-end, so the
        # circle's staircase over-measurement is removed.
        _recon_len = isolated_circle_len_px(skeleton, detected_circles)
    if _recon_len is None and not detected_circles and shape_type in ("Quadrilateral", "Hexagon"):
        # plain square / hexagon sections (straight walls, no circles): measure
        # each wall end-to-end (analytic) so the slanted edges are not
        # over-measured by the pixel staircase.
        _recon_len = analytic_polygon_len_px(skeleton)
    if _recon_len is not None:
        _skel_len_px = _recon_len
    _outer_area_px = enclosed_area_px(skeleton, float(cv2.contourArea(main_contour)))
    _en = _energy.compute_energy(shape_counts, total_circles_found)
    _RT = _en["RT"]
    _geo = _geometry.compute_geometry(shape_type, _skel_len_px, _outer_area_px, _RT)
    _RG = _geo["RG"]
    _OMEGA = _geo["omega"]

    # ---- colour every marker by its SHAPE type + draw a legend ----
    shape_colors = assign_shape_colors(shape_counts.keys())
    for _rec in node_records:
        _deg = _rec["degree"]
        _mr = marker_radius if 2 <= _deg <= 12 else max(4, marker_radius // 2)
        _mt = marker_thickness if 2 <= _deg <= 12 else max(1, marker_thickness // 2)
        draw_marker(output_img, _rec["x"], _rec["y"], _mr,
                    shape_colors.get(_rec["shape"], (0, 0, 0)), _mt,
                    _rec["marker"] == "square")

    # ---- RT / RG / Omega panel in the TOP-RIGHT corner of the figure ----
    def _fmt(_v):
        # round_half_up (not plain %.2f, which rounds half-to-even and is
        # skewed by binary floating-point) so an exact .XX5 always displays
        # rounded UP, matching how RT/RG feed into Omega above.
        return "--" if _v is None else ("%.2f" % _geometry.round_half_up(_v, 2))
    _pf = max(0.5, img_diag / 2300.0)
    _pth = max(1, int(_pf * 2))
    _pfont = cv2.FONT_HERSHEY_SIMPLEX
    _lines = ["RT = %s" % _fmt(_RT),
              "RG = %s" % _fmt(_RG),
              "Omega = %s" % _fmt(_OMEGA)]
    _tsz = [cv2.getTextSize(_l, _pfont, _pf, _pth)[0] for _l in _lines]
    _tw = max(_w for _w, _h in _tsz)
    _th0 = max(_h for _w, _h in _tsz)
    _plh = int(_th0 * 1.9)
    _pad = int(10 * _pf)
    _Wimg = output_img.shape[1]
    _bx1 = _Wimg - _pad
    _bx0 = _bx1 - _tw - 2 * _pad
    _by0 = _pad
    _by1 = _by0 + len(_lines) * _plh + _pad
    cv2.rectangle(output_img, (_bx0, _by0), (_bx1, _by1), (255, 255, 255), -1)
    cv2.rectangle(output_img, (_bx0, _by0), (_bx1, _by1), (60, 60, 60), _pth)
    for _i, _l in enumerate(_lines):
        _ty = _by0 + _pad + int(_th0) + _i * _plh
        cv2.putText(output_img, _l, (_bx0 + _pad, _ty), _pfont, _pf,
                    (0, 0, 0), _pth, cv2.LINE_AA)

    if shape_colors or total_circles_found:
        _fs = max(0.42, img_diag / 2700.0)
        _lh = int(38 * _fs) + 6
        _sw = int(20 * _fs)
        _x0, _y0 = 8, 8
        _font = cv2.FONT_HERSHEY_SIMPLEX
        _rows = sorted(shape_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        _th = max(1, int(_fs * 2))
        # Draw the legend in its OWN white banner ABOVE the figure so it never
        # overlaps the drawing. The banner is stacked on top of output_img.
        _bh = _y0 + (1 + len(_rows)) * _lh + _y0
        _W = output_img.shape[1]
        banner = np.full((_bh, _W, 3), 255, dtype=np.uint8)
        # first line: the detected circles, labelled "circular elements"
        # (text only, no marker symbol in front)
        cv2.putText(banner, "circular elements : %d" % total_circles_found,
                    (_x0, _y0 + _sw - 2), _font, _fs, (0, 0, 0), _th, cv2.LINE_AA)
        for _i, (_sh, _cnt) in enumerate(_rows, start=1):
            _yy = _y0 + _i * _lh
            _col = shape_colors[_sh]
            _sq = any(r["shape"] == _sh and r["marker"] == "square"
                      for r in node_records)
            if _sq:
                cv2.rectangle(banner, (_x0, _yy), (_x0 + _sw, _yy + _sw),
                              _col, max(2, int(_fs * 3)))
            else:
                cv2.circle(banner, (_x0 + _sw // 2, _yy + _sw // 2),
                           _sw // 2, _col, max(2, int(_fs * 3)))
            cv2.putText(banner, "%s : %d" % (_sh, _cnt),
                        (_x0 + _sw + 8, _yy + _sw - 2), _font, _fs,
                        (0, 0, 0), _th, cv2.LINE_AA)
        cv2.line(banner, (0, _bh - 1), (_W, _bh - 1), (210, 210, 210), 1)
        output_img = np.vstack([banner, output_img])

    return {
        "output_img": output_img,
        "shape_type": shape_type,
        "pattern_counts": pattern_counts,
        "nodes": node_records,
        "circle_count": total_circles_found,
        "shape_counts": shape_counts,
        "shape_colors": shape_colors,
        "skel_len_px": _skel_len_px,
        "outer_area_px": _outer_area_px,
        "RT": _RT,
        "RG": _RG,
        "omega": _OMEGA,
    }


def detect(img, min_dimension=900):
    """
    Processes the entire image assuming one main shape.
    Outputs the core visual pattern data and the number of circles detected.
    """
    if img is None:
        return None
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if min_dimension > 0:
        h, w = img.shape
        smaller = min(h, w)
        if smaller < min_dimension:
            scale = float(min_dimension) / smaller
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    h, w = img.shape
    pad = max(100, min(h, w) // 8)
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

    result = detect_grid_patterns_robust(img)
    return result
