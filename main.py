import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import math

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
    """
    Otsu threshold + light close + noise removal.
    """
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
                continue  # already removed in this pass
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
                # Delete the dead-end path but keep the junction
                for (pr, pc) in path:
                    skel[pr, pc] = 0
                changed = True

    return skel.astype(np.uint8)

def count_distinct_shapes(img, min_area_ratio=0.01, dilate_kernel=15):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    merged = cv2.dilate(binary, kernel, iterations=2)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(merged)

    min_area = min_area_ratio * h * w
    count = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            count += 1
    return count


def segment_shapes(img, min_area_ratio=0.01, padding=10, dilate_kernel=15):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    merged = cv2.dilate(binary, kernel, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged)

    min_area = min_area_ratio * h * w
    sub_images = []

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue

        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(w, x + bw + padding)
        y1 = min(h, y + bh + padding)

        crop = img[y0:y1, x0:x1].copy()
        crop_labels = labels[y0:y1, x0:x1]
        other_shape = (crop_labels != 0) & (crop_labels != i)
        crop[other_shape] = 255
        sub_images.append((crop, (x0, y0)))

    return sub_images

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

    # Hull-based metrics: capture the outer silhouette only.
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
    merge_kernel_size = max(9, img_diag // 60)

    neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skeleton, -1, neighbor_kernel)
    junction_pixels = np.where((neighbors * skeleton) >= 3, 255, 0).astype(np.uint8)

    junction_blobs = cv2.dilate(
        junction_pixels,
        np.ones((merge_kernel_size, merge_kernel_size), np.uint8),
        iterations=2,
    )
    num_labels, _, _, centroids = cv2.connectedComponentsWithStats(junction_blobs)

    all_nodes = []
    for i in range(1, num_labels):
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        all_nodes.append((cx, cy, 'junction'))

    if shape_type != "Circle" and len(shape_vertices) > 0:
        dedup_distance = max(25, img_diag // 25)
        for (vx, vy) in shape_vertices:
            is_duplicate = False
            for (jx, jy, _) in all_nodes:
                if np.sqrt((vx - jx) ** 2 + (vy - jy) ** 2) < dedup_distance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                all_nodes.append((vx, vy, 'vertex'))

    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    pattern_counts = {2: 0, 3: 0, 4: 0, 6: 0, 8: 0, "other": 0}

    # straight line test
    edge_tol = max(4, img_diag // 200)        # how close to an edge counts as "on it"
    vertex_tol = max(20, img_diag // 30)      # how close to a vertex still counts as a corner
    core_radius_px = max(5, img_diag // 120)    # core to exclude when tracing arms
    arm_trace_length = max(25, img_diag // 25)  # how far to walk along each arm
    collinear_cos_threshold = -0.85             # arms within ~32° of opposite

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
        # Reject if near any vertex (that's a corner, not a straight stretch).
        for (vx, vy) in shape_vertices:
            if math.hypot(cx - vx, cy - vy) < vertex_tol:
                return False
        # Accept if close to any edge.
        n = len(shape_vertices)
        for i in range(n):
            ax, ay = shape_vertices[i]
            bx, by = shape_vertices[(i + 1) % n]
            if point_segment_distance(cx, cy, ax, ay, bx, by) < edge_tol:
                return True
        return False

    def get_arm_directions(cy, cx):
        """
        Trace each skeleton arm leaving (cy, cx). Returns a list of (direction,
        path_points). The core (small disc around the node) is masked out so
        the arms become separate. Direction is end-point minus center,
        normalized. The path is returned so callers can verify straightness.
        """
        h, w = skeleton.shape
        visited = np.zeros_like(skeleton, dtype=bool)
        # Mark core as visited so the walk cannot pass through it.
        rmin = max(0, cy - core_radius_px - 2)
        rmax = min(h, cy + core_radius_px + 3)
        cmin = max(0, cx - core_radius_px - 2)
        cmax = min(w, cx + core_radius_px + 3)
        for r in range(rmin, rmax):
            for c in range(cmin, cmax):
                if (r - cy) ** 2 + (c - cx) ** 2 <= core_radius_px ** 2:
                    visited[r, c] = True

        # Find exit pixels: skeleton pixels just outside the core that have
        # a neighbor inside the core (so they're attached to this node).
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

        # Dedup nearby exit points so one fat arm doesn't get counted twice.
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
            for _ in range(arm_trace_length):
                pr, pc = path[-1]
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

    def arm_is_straight(path, cy, cx, min_cos=0.92):
        """
        Check that the arm itself is straight, not a curved arc. We compare
        the direction at the start of the arm (from center to early points)
        with the direction at the end (from late points to end). A circle
        perimeter would curve noticeably while a polygon edge or spoke stays
        on the same line, so the dot product near 1 means straight.
        """
        if len(path) < 6:
            return True  # too short to tell, assume straight
        # First-half direction: center -> midpoint
        mid_idx = len(path) // 2
        mr, mc = path[mid_idx]
        v1x = mc - cx
        v1y = mr - cy
        n1 = math.hypot(v1x, v1y)
        # Second-half direction: midpoint -> end
        er, ec = path[-1]
        v2x = ec - mc
        v2y = er - mr
        n2 = math.hypot(v2x, v2y)
        if n1 < 1 or n2 < 1:
            return True
        dot = (v1x * v2x + v1y * v2y) / (n1 * n2)
        return dot >= min_cos

    def has_through_line(cx, cy):
        """True if any pair of *straight* arms is roughly opposite."""
        arms = get_arm_directions(cy, cx)
        if len(arms) < 2:
            return False
        # Filter to arms that are themselves straight.
        straight_arms = [a for a in arms if arm_is_straight(a[1], cy, cx)]
        for i in range(len(straight_arms)):
            for j in range(i + 1, len(straight_arms)):
                d1 = straight_arms[i][0]
                d2 = straight_arms[j][0]
                dot = d1[0] * d2[0] + d1[1] * d2[1]
                if dot < collinear_cos_threshold:
                    return True
        return False

    def draw_marker(img_out, cx, cy, radius, color, thickness, use_square):
        """Square for nodes on a straight edge of the outer shape, circle otherwise."""
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
        """Given a list of unit vectors, return wedge entries around the circle."""
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
            # Bisector of the minor arc between d1 and d2 is the sum of the
            # two unit vectors. For a reflex wedge (>180 between arms going
            # around the long way), flip it so the label sits on the correct
            # side of the node.
            bx, by = d1[0] + d2[0], d1[1] + d2[1]
            bn = math.hypot(bx, by)
            if bn < 1e-6:
                # d1 and d2 are exactly opposite -- pick the perpendicular
                # that puts the bisector on the wedge's own side.
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
        """
        Return list of (wedge_angle_deg, bisector_dx, bisector_dy) around the
        node. At a polygon corner we anchor the two adjacent polygon-edge
        directions (so the corner shape angle is exact) and add any extra
        skeleton arms (diagonals) on top so every wedge is labeled.
        """
        # Snap to polygon corner if close enough.
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

        # For arm tracing, snap to the nearest skeleton pixel within a small
        # search radius. The polygon vertex from approxPolyDP can sit a few
        # pixels off the actual skeleton corner (the line has thickness, and
        # the skeleton tracks the line's center, not its outer edge).
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
            # Start with the polygon-edge directions (exact), then add any
            # skeleton arms that point in a substantially different direction
            # (e.g. an interior diagonal). Dedup tolerance is generous because
            # the traced arm directions are approximate.
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
        """Place one number per wedge along its bisector."""
        if not wedge_entries:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.4, img_diag / 2800.0)
        thickness = max(1, int(scale * 2))
        # Place labels just outside the marker, along each wedge's bisector.
        label_radius = marker_radius + max(14, int(img_diag / 60))
        h_img, w_img = img_out.shape[:2]
        margin = 4
        for (angle, bx, by) in wedge_entries:
            text = f"{int(round(angle))}"
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            lx = cx + bx * label_radius
            ly = cy + by * label_radius
            # Pull the label back along the bisector if it falls outside the
            # visible canvas (or its text box would).
            tries = 0
            while tries < 30:
                left = lx - tw / 2.0
                right = lx + tw / 2.0
                top = ly - th / 2.0
                bot = ly + th / 2.0
                if (left >= margin and right < w_img - margin
                        and top >= margin and bot < h_img - margin):
                    break
                # Pull 10% of label_radius closer to node each step.
                lx -= bx * label_radius * 0.1
                ly -= by * label_radius * 0.1
                tries += 1
            org = (int(lx - tw / 2), int(ly + th / 2))
            # White outline for legibility over dark lines.
            cv2.putText(img_out, text, org, font, scale, (255, 255, 255),
                        thickness + 2, cv2.LINE_AA)
            cv2.putText(img_out, text, org, font, scale, (0, 0, 0),
                        thickness, cv2.LINE_AA)

    node_records = []  # collect for return value

    for (cx, cy, node_type) in all_nodes:
        is_shape_vertex = (node_type == 'vertex')
        if is_shape_vertex:
            degree = count_branches_standard(skeleton, cy, cx, radius=10)
        else:
            degree = count_branches_robust(skeleton, cy, cx, radius=15)

        # Shape vertices that come back as degree 0/1 (because the skeleton
        # has been pruned away from the very corner) should be treated as 2.
        if is_shape_vertex and degree < 2:
            degree = 2

        # Square marker if on a straight outer edge OR a straight line
        # passes through this node (opposite arms).
        # Exception: for a Circle shape, perimeter nodes lie on a curve, so
        # the local "two opposite arms along the perimeter" pattern is just
        # the curve, not a straight line. Skip the through-line test there.
        on_circle_perimeter = False
        if shape_type == "Circle":
            (ccx, ccy), c_radius = cv2.minEnclosingCircle(main_contour)
            dist_to_center = math.hypot(cx - ccx, cy - ccy)
            if abs(dist_to_center - c_radius) < max(8, img_diag // 60):
                on_circle_perimeter = True

        if on_circle_perimeter:
            use_square = False
        else:
            use_square = on_straight_edge(cx, cy) or has_through_line(cx, cy)

        # Wedge angles around this node, each with its bisector for label placement.
        wedges = compute_wedge_angles(cx, cy, is_shape_vertex)

        if degree == 2:
            pattern_counts[2] += 1
            draw_marker(output_img, cx, cy, marker_radius,
                        (200, 0, 200), marker_thickness, use_square)
        elif degree == 3:
            pattern_counts[3] += 1
            draw_marker(output_img, cx, cy, marker_radius,
                        (0, 0, 255), marker_thickness, use_square)
        elif degree == 4:
            pattern_counts[4] += 1
            draw_marker(output_img, cx, cy, marker_radius,
                        (255, 0, 0), marker_thickness, use_square)
        elif degree == 6:
            pattern_counts[6] += 1
            draw_marker(output_img, cx, cy, marker_radius,
                        (0, 255, 0), marker_thickness, use_square)
        elif degree == 8:
            pattern_counts[8] += 1
            draw_marker(output_img, cx, cy, marker_radius,
                        (255, 255, 0), marker_thickness, use_square)
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

    if shape_type != "Circle" and len(shape_vertices) >= 3:
        for i in range(len(shape_vertices)):
            pt1 = shape_vertices[i]
            pt2 = shape_vertices[(i + 1) % len(shape_vertices)]
            cv2.line(output_img, pt1, pt2, (0, 255, 255), marker_thickness)
    elif shape_type == "Circle":
        (x, y), radius = cv2.minEnclosingCircle(main_contour)
        cv2.circle(output_img, (int(x), int(y)), int(radius),
                   (0, 255, 255), marker_thickness)

    return {
        "output_img": output_img,
        "shape_type": shape_type,
        "pattern_counts": pattern_counts,
        "nodes": node_records,
    }


# ============================================================
# AUTO-DISPATCH
# ============================================================

def detect(img, force_segment=None, min_dimension=800):
    """
    Auto-decides between single-shape and multi-shape processing.
    Also auto-upscales small images so the pipeline has enough pixels
    to work with reliably.

    Parameters
    ----------
    img : np.ndarray
        Input image (grayscale or BGR).
    force_segment : True / False / None
        True  -> always segment
        False -> never segment
        None  -> auto (default: segment if more than 1 shape detected)
    min_dimension : int
        If either width or height is below this, the image is upscaled
        with cubic interpolation. 600 is a good default for line drawings.
        Set to 0 to disable upscaling.
    """
    if img is None:
        return None
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Auto-upscale tiny images. This is the single biggest fix for low-res
    # inputs: it gives skeletonization enough pixels to resolve thin features
    # and gives segmentation enough space to separate adjacent shapes.
    if min_dimension > 0:
        h, w = img.shape
        smaller = min(h, w)
        if smaller < min_dimension:
            scale = float(min_dimension) / smaller
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Pad with white border so labels for exterior wedge angles (e.g. the
    # 270 outside a square corner) have somewhere to sit. Background is
    # white because Otsu thresholding inverts (line drawing on light).
    h, w = img.shape
    pad = max(100, min(h, w) // 8)
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

    if force_segment is None:
        do_segment = count_distinct_shapes(img) > 1
    else:
        do_segment = force_segment

    if not do_segment:
        result = detect_grid_patterns_robust(img)
        if result is None:
            return None
        return {
            "output_img": result["output_img"],
            "shapes": [{
                "shape_index": 0,
                "shape_type": result["shape_type"],
                "pattern_counts": result["pattern_counts"],
                "nodes": result.get("nodes", []),
                "offset": (0, 0),
            }],
            "total_counts": result["pattern_counts"],
            "num_shapes": 1,
        }

    sub_images = segment_shapes(img)
    if not sub_images:
        print("No shapes found.")
        return None

    combined_output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    combined_counts = {2: 0, 3: 0, 4: 0, 6: 0, 8: 0, "other": 0}
    shape_results = []

    for idx, (sub_img, (x_off, y_off)) in enumerate(sub_images):
        result = detect_grid_patterns_robust(sub_img)
        if result is None:
            continue
        sh, sw = result["output_img"].shape[:2]
        combined_output[y_off:y_off + sh, x_off:x_off + sw] = result["output_img"]
        for k, v in result["pattern_counts"].items():
            combined_counts[k] += v
        # Translate node coords into the combined image frame.
        translated_nodes = []
        for n in result.get("nodes", []):
            n2 = dict(n)
            n2["x"] = n["x"] + x_off
            n2["y"] = n["y"] + y_off
            translated_nodes.append(n2)
        shape_results.append({
            "shape_index": idx,
            "shape_type": result["shape_type"],
            "pattern_counts": result["pattern_counts"],
            "nodes": translated_nodes,
            "offset": (x_off, y_off),
        })

    return {
        "output_img": combined_output,
        "shapes": shape_results,
        "total_counts": combined_counts,
        "num_shapes": len(shape_results),
    }

#main
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

    print("\n" + "=" * 50)
    print(f"DETECTED {result['num_shapes']} SHAPE(S)")
    print("=" * 50)
    for s in result["shapes"]:
        print(f"\nShape {s['shape_index']}: {s['shape_type']}")
        for k, v in s["pattern_counts"].items():
            print(f"  Degree {k}: {v}")
    print("\n" + "-" * 50)
    print("COMBINED TOTALS:")
    print("-" * 50)
    for k, v in result["total_counts"].items():
        print(f"  Degree {k}: {v}")

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(result["output_img"], cv2.COLOR_BGR2RGB))
    plt.title(f"{result['num_shapes']} shape(s) detected")
    plt.axis('off')
    plt.show()