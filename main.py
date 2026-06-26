import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import math

DEGREE_COLORS = {
    2:  (200, 0, 200),    # purple
    3:  (0, 0, 255),      # red
    4:  (255, 0, 0),      # blue
    5:  (0, 165, 255),    # orange
    6:  (0, 255, 0),      # green
    7:  (255, 255, 0),    # cyan
    8:  (0, 255, 255),    # yellow
    9:  (255, 0, 255),    # magenta
    10: (128, 255, 0),    # spring green
    11: (255, 128, 0),    # azure
    12: (0, 128, 255),    # amber
}

#preprocessing

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

    h, w = binary_img.shape
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

def is_straight_line(skeleton, r, c, radius=6):
    h, w = skeleton.shape
    points = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and skeleton[rr, cc]:
                points.append([dc, dr])
    if len(points) < 5:
        return True

    pts = np.array(points, dtype=np.float32)
    pts -= np.mean(pts, axis=0)
    cov = np.cov(pts.T)
    eigvals, _ = np.linalg.eig(cov)
    min_eigval = min(eigvals)
    if min_eigval < 1e-6:
        return True
    return max(eigvals) / min_eigval > 15

def count_branches_robust(skeleton_img, center_r, center_c, radius=15):
    height, width = skeleton_img.shape
    r_start = max(0, center_r - radius)
    r_end = min(height, center_r + radius + 1)
    c_start = max(0, center_c - radius)
    c_end = min(width, center_c + radius + 1)

    window = skeleton_img[r_start:r_end, c_start:c_end]
    if window.shape[0] < 3 or window.shape[1] < 3:
        return 0

    top = window[0, :]
    bottom = window[-1, :][::-1]
    if window.shape[0] > 1:
        right = window[1:-1, -1]
        left = window[1:-1, 0][::-1]
    else:
        right = np.array([], dtype=window.dtype)
        left = np.array([], dtype=window.dtype)

    perimeter = np.concatenate([top, right, bottom, left])

    transitions = 0
    is_on_line = False
    for v in perimeter:
        if v > 0:
            if not is_on_line:
                transitions += 1
                is_on_line = True
        else:
            is_on_line = False

    if perimeter[0] > 0 and perimeter[-1] > 0:
        transitions -= 1
    return transitions

def count_branches_standard(skeleton_img, center_r, center_c, radius=10):
    return count_branches_robust(skeleton_img, center_r, center_c, radius=radius)

def get_refined_shape(cnt):
    area = cv2.contourArea(cnt)
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
        best_match = None
        best_match_diff = float('inf')
        vertices = []
        for epsilon_factor in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(contour, epsilon_factor * perimeter, True)
            diff = abs(len(approx) - expected_sides)
            if diff < best_match_diff:
                best_match_diff = diff
                best_match = approx
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
# NEW HELPERS (node-detection improvements)
# ============================================================

def detect_all_circles(binary, min_radius_px=10):
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
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

def _merge_tangent_bridge_pairs(points, skeleton, max_dist):
    n = len(points)
    if n == 0:
        return []
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def snap(x, y):
        H, W = skeleton.shape
        for radius in (0, 2, 4, 6, 8):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr * dr + dc * dc > radius * radius:
                        continue
                    rr, cc = y + dr, x + dc
                    if 0 <= rr < H and 0 <= cc < W and skeleton[rr, cc]:
                        return rr, cc
        return None

    H, W = skeleton.shape
    block_dist = 6 
    snapped = [snap(p[0], p[1]) for p in points]

    def path_exists(i, j, limit):
        from collections import deque
        src = snapped[i]; dst = snapped[j]
        if src is None or dst is None:
            return False
        if src == dst:
            return True
        forbidden = set()
        for k in range(n):
            if k == i or k == j:
                continue
            xk, yk = points[k]
            for dr in range(-block_dist, block_dist + 1):
                for dc in range(-block_dist, block_dist + 1):
                    if dr * dr + dc * dc > block_dist * block_dist:
                        continue
                    rr, cc = yk + dr, xk + dc
                    forbidden.add((rr, cc))
        q = deque([(src[0], src[1], 0)])
        seen = {src}
        while q:
            r, c, d = q.popleft()
            if d >= limit:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < H and 0 <= cc < W):
                        continue
                    if not skeleton[rr, cc]:
                        continue
                    if (rr, cc) in seen:
                        continue
                    if (rr, cc) in forbidden:
                        continue
                    if (rr, cc) == dst:
                        return True
                    seen.add((rr, cc))
                    q.append((rr, cc, d + 1))
        return False

    md2 = max_dist * max_dist
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            if (xi - xj) ** 2 + (yi - yj) ** 2 > md2:
                continue
            if path_exists(i, j, max_dist + 4):
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(points[i])
    merged = []
    for pts in groups.values():
        mx = int(round(sum(p[0] for p in pts) / len(pts)))
        my = int(round(sum(p[1] for p in pts) / len(pts)))
        merged.append((mx, my, 'junction'))
    return merged

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

def find_tangent_junctions(skeleton, img_diag):
    neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skeleton, -1, neighbor_kernel)
    junction_pixels = np.where((neighbors >= 3) & (skeleton == 1), 255, 0).astype(np.uint8)
    return junction_pixels

def _merge_close_points(points, max_dist, node_type='junction'):
    n = len(points)
    if n == 0:
        return []
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    md2 = max_dist * max_dist
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            if (xi - xj) ** 2 + (yi - yj) ** 2 <= md2:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(points[i])

    merged = []
    for pts in groups.values():
        mx = int(round(sum(p[0] for p in pts) / len(pts)))
        my = int(round(sum(p[1] for p in pts) / len(pts)))
        merged.append((mx, my, node_type))
    return merged

def _merge_shared_circle_pairs(points, circles, circle_tol, max_dist):
    n = len(points)
    if n == 0 or not circles:
        return list(points)

    def memberships(x, y):
        s = set()
        for ci, (cx, cy, r) in enumerate(circles):
            if abs(math.hypot(x - cx, y - cy) - r) < circle_tol:
                s.add(ci)
        return s

    mem = [memberships(p[0], p[1]) for p in points]

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    md2 = max_dist * max_dist
    for i in range(n):
        if not mem[i]:
            continue
        xi, yi = points[i][0], points[i][1]
        for j in range(i + 1, n):
            if not mem[j]:
                continue
            xj, yj = points[j][0], points[j][1]
            if (xi - xj) ** 2 + (yi - yj) ** 2 > md2:
                continue
            if len(mem[i] & mem[j]) >= 2:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for idxs in groups.values():
        pts = [points[i] for i in idxs]
        mx = int(round(sum(p[0] for p in pts) / len(pts)))
        my = int(round(sum(p[1] for p in pts) / len(pts)))
        nt = pts[0][2] if len(pts[0]) > 2 else 'junction'
        merged.append((mx, my, nt))
    return merged

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

# ============================================================
# JUNCTION REFINEMENT (straight-through-line aware)
# ============================================================
# Goals:
#   * detect EVERY junction (never drop nodes)
#   * a node where a STRAIGHT line passes through  -> SQUARE
#   * a corner / circle-only / curved point         -> CIRCLE
#   * the tight cluster of junctions that a tangent circle creates at
#     one crossing (circle grazes an edge => 2-3 extra T-junctions a few
#     dozen px apart) collapses to ONE node; genuine separate crossings
#     and far-apart nodes stay separate.

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


def _split_through_by_circles(atoms, idxs, circles, tol):
    """If a straight-through cluster crosses TWO OR MORE distinct circles,
    return one 'through' node per circle (the pure spoke-circle crossing).
    A cluster sitting on a single circle (tangent edge-midpoint) returns
    None so it stays a single node."""
    if not circles:
        return None
    mem = {}
    for k in idxs:
        x, y = atoms[k]
        mem[k] = {ci for ci, (cx, cy, r) in enumerate(circles)
                  if abs(math.hypot(x - cx, y - cy) - r) < tol}
    distinct = set().union(*mem.values()) if mem else set()
    if len(distinct) < 2:
        return None
    picks = []
    for ci in distinct:
        cands = [k for k in idxs if ci in mem[k]]
        if not cands:
            continue
        # the purest crossing of this circle = on the fewest circles
        best = min(cands, key=lambda k: len(mem[k]))
        picks.append(atoms[best])
    out = []
    for (x, y) in picks:
        if all((x - ox) ** 2 + (y - oy) ** 2 >= 15 * 15 for ox, oy, _ in out):
            out.append((x, y, 'through'))
    return out if len(out) >= 2 else None


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
    for i in range(n):
        # only T-junctions (deg<=3) can be tangency scatter of one node
        if deg[i] > 3:
            continue
        xi, yi = atoms[i]
        for j in range(i + 1, n):
            if deg[j] > 3:
                continue
            xj, yj = atoms[j]
            if (xi - xj) ** 2 + (yi - yj) ** 2 > md2:
                continue
            blocked = allcore - atomcore[i] - atomcore[j]
            path = _bfs_skeleton_path(skeleton, (yi, xi), (yj, xj),
                                      blocked, max_link_dist + 6)
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


def _circle_coverage(dt, H, W, cx, cy, r, tol, samples=720):
    hit = 0
    for d in range(samples):
        a = 2.0 * math.pi * d / samples
        px = int(round(cx + r * math.cos(a)))
        py = int(round(cy + r * math.sin(a)))
        if 0 <= px < W and 0 <= py < H and dt[py, px] < tol:
            hit += 1
    return hit / samples

def _is_real_circle(dt, H, W, cx, cy, r):
    """Curvature-based test. A real circle's skeleton lies ON it: most of the
    circumference is covered at a tight tolerance, and loosening the tolerance
    barely adds coverage (gaps are genuine absences). A polygon outline or a
    circle fitted through scattered intersections only matches at a loose
    tolerance, so its coverage JUMPS when the tolerance grows."""
    c5 = _circle_coverage(dt, H, W, cx, cy, r, 5)
    c12 = _circle_coverage(dt, H, W, cx, cy, r, 12)
    return c5 >= 0.60 and (c12 - c5) <= 0.08

def filter_real_circles(skeleton, circles, cov_thresh=0.95, **_):
    """Strict filter for circles used in NODE logic: keep only nearly fully
    covered circles (clean drawn circles). Partial big rings are handled
    separately for counting (find_center_rings)."""
    if not circles:
        return []
    src = (skeleton == 0).astype(np.uint8)
    dt = cv2.distanceTransform(src, cv2.DIST_L2, 3)
    H, W = skeleton.shape
    return [(cx, cy, r) for (cx, cy, r) in circles
            if _circle_coverage(dt, H, W, cx, cy, r, 5) >= cov_thresh]

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


def detect_grid_patterns_robust(img, prune_length=4):
    if img is None:
        return None
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary = preprocess_for_skeleton(img)

    raw_skel = skeletonize(binary // 255).astype(np.uint8)
    if prune_length > 0:
        raw_skel = prune_skeleton_spurs(raw_skel, min_branch_length=prune_length)
    raw_skel = prune_dead_end_branches(raw_skel, max_dead_end_length=25)

    line_thickness_est = max(3, int(np.sqrt(binary.sum() / 255 / max(raw_skel.sum(), 1)) * 2))
    rebuild_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (line_thickness_est, line_thickness_est))
    binary = cv2.dilate(raw_skel * 255, rebuild_kernel)

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
            if all(abs(rr - c[2]) > max(10, 0.1 * rr) or
                   (rx - c[0]) ** 2 + (ry - c[1]) ** 2 > (0.2 * rr) ** 2
                   for c in detected_circles):
                extra_rings.append((rx, ry, rr))

    # Total circle count = clean circles (used for nodes) + extra rings.
    total_circles_found = len(detected_circles) + len(extra_rings)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return None

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
    line_thickness = max(2, int(round(binary.sum() / 255 / max(raw_skel.sum(), 1))))
    merge_kernel_size = max(7, img_diag // 70)

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
                dx = ec - cx
                dy = er - cy
                norm = math.hypot(dx, dy)
                if norm > 1:
                    arms.append(((dx / norm, dy / norm), path))
        return arms

    def arm_is_straight(path, cy, cx, residual_thresh=0.00025):
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

    def any_arm_curved(arms, cy, cx):
        for direction, path in arms:
            if not arm_is_straight(path, cy, cx):
                return True
        return False

    def has_through_line(cx, cy, arms=None):
        if arms is None:
            arms = get_arm_directions(cy, cx)
        if len(arms) < 2:
            return False
        straight_arms = [a for a in arms if arm_is_straight(a[1], cy, cx)]
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

    def has_straight_chord(cx, cy, arms):
        # Through-line made of STRAIGHT arms that are NOT circle arcs.
        if arms is None or len(arms) < 2:
            return False
        sa = [a for a in arms
              if arm_is_straight(a[1], cy, cx) and not _arm_follows_circle(a[1])]
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

    def compute_wedge_angles(cx, cy, is_shape_vertex):
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

        if corner_dirs is not None:
            merged = list(corner_dirs)
            min_sep_cos = math.cos(math.radians(20))
            for td in traced_dirs:
                is_new = True
                for md in merged:
                    if td[0] * md[0] + td[1] * md[1] > min_sep_cos:
                        is_new = False
                        break
                if is_new:
                    merged.append(td)
            return _wedges_from_dirs(merged)

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

    circle_geom = None
    if shape_type == "Circle":
        (ccx, ccy), c_radius = cv2.minEnclosingCircle(main_contour)
        circle_geom = (ccx, ccy, c_radius)

    all_node_pts = [(p[0], p[1]) for p in all_nodes]

    for (cx, cy, node_type) in all_nodes:
        is_shape_vertex = (node_type == 'vertex')

        scy, scx = snap_to_skeleton(skeleton, cy, cx, max(10, img_diag // 70))
        arms = get_arm_directions(scy, scx)

        # A node lying on TWO circles with no straight line through it is a
        # circle-circle meeting point (a "lens"): one degree-4 circle marker.
        _chord = has_straight_chord(scx, scy, arms)
        _lens_tol = max(10, img_diag // 70)
        _on_circ = sum(1 for (ccx, ccy, rr) in detected_circles
                       if abs(math.hypot(cx - ccx, cy - ccy) - rr) < _lens_tol)
        # A spoke (straight non-arc arm) means this is NOT a pure lens: e.g. a
        # spoke meeting the outer ring where a petal is tangent = degree 5.
        _has_spoke = _node_has_straight_arm(skeleton, cx, cy,
                                            detected_circles, _lens_tol)
        is_lens = (node_type not in ('circ6', 'xcross', 'vertex')
                   and _on_circ >= 2 and not _chord and not _has_spoke)

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
            if abs(dist_to_center - circle_geom[2]) < max(8, img_diag // 60):
                on_circle_perimeter = True

        circle_tol = max(8, img_diag // 80)
        on_any_circle = is_on_any_circle(cx, cy, detected_circles, circle_tol)

        straight_through = has_through_line(scx, scy, arms)
        on_polygon_edge = on_straight_edge(cx, cy)
        any_curved = any_arm_curved(arms, scy, scx)

        # Marker shape: SQUARE iff a straight line passes through the node.
        if node_type == 'ring3':
            use_square = False           # spoke meeting the curved ring -> circle
        elif node_type == 'pcross4':
            use_square = True            # spoke through a petal -> square
        elif is_lens:
            use_square = False           # circle-circle meeting -> circle
        elif node_type in ('through', 'xcross', 'circ6'):
            use_square = True
        elif node_type == 'corner':
            use_square = False           # collapsed corner cluster -> circle
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
            use_square = has_straight_chord(scx, scy, arms)

        wedges = compute_wedge_angles(cx, cy, is_shape_vertex)

        if degree in DEGREE_COLORS:
            pattern_counts[degree] += 1
            draw_marker(output_img, cx, cy, marker_radius,
                        DEGREE_COLORS[degree], marker_thickness, use_square)
        else:
            pattern_counts["other"] += 1
            draw_marker(output_img, cx, cy, max(4, marker_radius // 2),
                        (128, 128, 128), max(1, marker_thickness // 2),
                        use_square)

        draw_wedge_labels(output_img, cx, cy, wedges, marker_radius)
        node_records.append({
            "x": int(cx),
            "y": int(cy),
            "degree": degree,
            "marker": "square" if use_square else "circle",
            "angles_deg": [round(float(w[0]), 1) for w in wedges],
        })

    return {
        "output_img": output_img,
        "shape_type": shape_type,
        "pattern_counts": pattern_counts,
        "nodes": node_records,
        "circle_count": total_circles_found,
    }


# ============================================================
# AUTO-DISPATCH (Updated for Single-Shape Processing)
# ============================================================

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

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    img_path = "images/circle1.png"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Could not load image: {img_path}")
        raise SystemExit(1)

    result = detect(img)
    if result is None:
        print("No detection.")
        raise SystemExit(0)

    # Simplified terminal outputs as requested
    print("\n" + "=" * 50)
    print("DETECTION SUMMARY:")
    print("=" * 50)
    print(f"CIRCLES DETECTED: {result['circle_count']}")
    
    print("\nDEGREES FOUND:")
    for k, v in result["pattern_counts"].items():
        if v > 0: 
            print(f"  Degree {k}: {v}")

    # Simplified matplotlib window title
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(result["output_img"], cv2.COLOR_BGR2RGB))
    plt.title(f"Circles detected: {result['circle_count']}")
    plt.axis('off')
    plt.show()