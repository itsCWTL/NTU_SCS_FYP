import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import math

def is_straight_line(skeleton, r, c, radius=6):
    """
    Returns True if degree 2 node lies on a straight line
    """
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

    # PCA
    cov = np.cov(pts.T)
    eigvals, _ = np.linalg.eig(cov)

    # Handle division by zero
    min_eigval = min(eigvals)
    if min_eigval < 1e-6:
        return False
    
    # Strongly linear -> straight line
    return max(eigvals) / min_eigval > 15

def count_branches_robust(skeleton_img, center_r, center_c, radius=15):
    """
    Counts branches. 
    A larger radius ensures we count the arms *leaving* the intersection,
    ignoring the messy 'knot' that often forms at the center of 6 way nodes
    """
    height, width = skeleton_img.shape
    
    # Extract window
    r_start = max(0, center_r - radius)
    r_end = min(height, center_r + radius + 1)
    c_start = max(0, center_c - radius)
    c_end = min(width, center_c + radius + 1)
    
    window = skeleton_img[r_start:r_end, c_start:c_end]
    
    if window.shape[0] < 3 or window.shape[1] < 3:
        return 0

    # Get perimeter pixels
    top = window[0, :]
    bottom = window[-1, :][::-1]
    
    if window.shape[0] > 1:
        right = window[1:-1, -1]
        left = window[1:-1, 0][::-1]
    else:
        right = []
        left = []
        
    perimeter = np.concatenate([top, right, bottom, left])
    
    # Count transitions from 0->1 or 1->0
    # Dividing transitions by 2 gives the number of lines
    transitions = 0
    is_on_line = False
    
    for i in range(len(perimeter)):
        if perimeter[i] > 0:
            if not is_on_line:
                transitions += 1
                is_on_line = True
        else:
            is_on_line = False
            
    # Handle wrap around
    if perimeter[0] > 0 and perimeter[-1] > 0:
        transitions -= 1
        
    return transitions

def count_branches_standard(skeleton_img, center_r, center_c, radius=10):
    """
    Standard branch counting for corners and regular intersections
    """
    return count_branches_robust(skeleton_img, center_r, center_c, radius=radius)

def get_refined_shape(cnt):
    """
    Distinguishes circles from high-order polygons using multiple metrics.
    """
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0: 
        return "Unknown", []

    # 1. circularity metric (perfect circle = 1.0)
    circularity = (4 * np.pi * area) / (perimeter**2)
    
    # 2. extent metric (how much it fills its minimum enclosing circle)
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    enclosing_circle_area = np.pi * (radius**2)
    extent = area / enclosing_circle_area

    # 3. vertex approximation (small epsilon for high detail)
    epsilon = 0.018 * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    num_v = len(approx)
    vertices = [tuple(p[0]) for p in approx]

    # An octagon/hexagon has high circularity, but low 'extent' compared to a true circle.
    # A circle usually has circularity > 0.9 AND extent > 0.9.
    if circularity > 0.88 and extent > 0.91:
        return "Circle", []
    
    shape_map = {3: "Triangle", 4: "Quadrilateral", 5: "Pentagon", 6: "Hexagon", 7: "Heptagon", 8: "Octagon"}
    shape_name = shape_map.get(num_v, f"Polygon ({num_v} sides)")
    
    return shape_name, vertices

def detect_shape_vertices(binary_img, expected_sides=None):
    """
    Detect vertices of any polygon shape using contour approximation
    If expected_sides is provided, it will try to find that many vertices
    """
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    
    if expected_sides is not None:
        # Try to get the expected number of vertices
        vertices = []
        best_match = None
        best_match_diff = float('inf')
        
        # Try different epsilon values
        for epsilon_factor in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]:
            epsilon = epsilon_factor * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            num_vertices = len(approx)
            diff = abs(num_vertices - expected_sides)
            
            if diff < best_match_diff:
                best_match_diff = diff
                best_match = approx
                vertices = [tuple(point[0]) for point in approx]
            
            if num_vertices == expected_sides:
                break
        
        # If we have close to expected sides, use those vertices
        if best_match_diff <= 2:  # allow some tolerance
            vertices = [tuple(point[0]) for point in best_match]
        else:
            # Force approximation to expected sides
            epsilon = 0.03 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Adjust epsilon to get close to expected sides
            epsilon_factor = 0.03
            while len(approx) > expected_sides and epsilon_factor < 0.1:
                epsilon_factor += 0.01
                epsilon = epsilon_factor * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
            
            while len(approx) < expected_sides and epsilon_factor > 0.005:
                epsilon_factor -= 0.005
                epsilon = epsilon_factor * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)
            
            vertices = [tuple(point[0]) for point in approx]
    else:
        # Use adaptive epsilon based on perimeter
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = [tuple(point[0]) for point in approx]
    
    # Sort vertices in clockwise order
    if len(vertices) >= 3:
        center_x = np.mean([v[0] for v in vertices])
        center_y = np.mean([v[1] for v in vertices])
        vertices = sorted(vertices,
                         key=lambda v: math.atan2(v[1] - center_y, v[0] - center_x))
    
    return vertices

def detect_and_verify_corners(binary_img, skeleton, expected_sides=None):
    """
    Detect corners using shape detection AND verify with skeleton analysis
    Returns list of (x, y) coordinates of verified corners
    """
    # First detect shape vertices
    shape_vertices = detect_shape_vertices(binary_img, expected_sides)
    
    if not shape_vertices or len(shape_vertices) < 3:
        return []
    
    # Now verify each vertex with skeleton analysis
    verified_corners = []
    
    for i, (x, y) in enumerate(shape_vertices):
        # Check skeleton at this location
        degree = count_branches_standard(skeleton, y, x, radius=10)
        
        # Only accept if it's a degree-2 node AND not a straight line
        if degree == 2 and not is_straight_line(skeleton, y, x, radius=8):
            verified_corners.append((x, y))
        else:
            # If skeleton doesn't confirm, still keep it if it's part of a shape
            # but mark it as less certain
            verified_corners.append((x, y))
    
    # Remove duplicates
    unique_corners = []
    for corner in verified_corners:
        is_duplicate = False
        for unique_corner in unique_corners:
            distance = np.sqrt((corner[0] - unique_corner[0])**2 + 
                              (corner[1] - unique_corner[1])**2)
            if distance < 15:
                is_duplicate = True
                # Update to average position for better accuracy
                unique_corner = ((unique_corner[0] + corner[0]) // 2, 
                                (unique_corner[1] + corner[1]) // 2)
                break
        if not is_duplicate:
            unique_corners.append(corner)
    
    return unique_corners

def identify_shape_type(num_vertices, circularity=0, extent=0):
    """
    Identify the shape based on number of vertices
    Includes circle detection logic 
    """
    # Check if it's a circle first (using code A logic)
    if circularity > 0.88 and extent > 0.91:
        return "Circle"
    
    if num_vertices == 3:
        return "Triangle"
    elif num_vertices == 4:
        return "Quadrilateral"
    elif num_vertices == 5:
        return "Pentagon"
    elif num_vertices == 6:
        return "Hexagon"
    elif num_vertices == 7:
        return "Heptagon"
    elif num_vertices == 8:
        return "Octagon"
    elif num_vertices > 8:
        return f"Polygon ({num_vertices} sides)"
    else:
        return "Unknown"

# def detect_grid_patterns_robust(image_path):
#     # 1. Load and Preprocess 
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         print(f"Error: Could not find image at {image_path}")
#         return

#     # Invert (Lines = White) 
#     _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
#     # Morphological Close to fill gaps for skeletonization 
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
#     binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

#     # 2. Detect Shape 
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         print("No contours found.")
#         return
    
#     main_contour = max(contours, key=cv2.contourArea)
#     shape_type, shape_vertices = get_refined_shape(main_contour)
#     # print(f"Detected Outer Shape: {shape_type}")
    
#     # Calculate circularity and extent for shape identification
#     area = cv2.contourArea(main_contour)
#     perimeter = cv2.arcLength(main_contour, True)
#     circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
#     (x, y), radius = cv2.minEnclosingCircle(main_contour)
#     enclosing_circle_area = np.pi * (radius**2)
#     extent = area / enclosing_circle_area if enclosing_circle_area > 0 else 0

#     # 3. Skeletonize and find Junctions 
#     skeleton = skeletonize(binary_closed // 255).astype(np.uint8)
    
#     # Detect junction pixels (neighbor count > 2) 
#     neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
#     neighbors = cv2.filter2D(skeleton, -1, neighbor_kernel)
#     junction_pixels = np.where((neighbors * skeleton) >= 3, 255, 0).astype(np.uint8)
    
#     # Merge nearby junction pixels into single blobs 
#     junction_blobs = cv2.dilate(junction_pixels, np.ones((9,9), np.uint8), iterations=2)
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(junction_blobs)
    
#     # Create combined list of ALL nodes (junctions + shape vertices)
#     all_nodes = []
    
#     # First add all junction centroids
#     for i in range(1, num_labels):
#         cx, cy = int(centroids[i][0]), int(centroids[i][1])
#         all_nodes.append((cx, cy, 'junction'))
    
#     # Then add shape vertices (if polygon)
#     if shape_type != "Circle" and len(shape_vertices) > 0:
#         for (vx, vy) in shape_vertices:
#             # Check if vertex is already close to a junction
#             is_duplicate = False
#             for (jx, jy, _) in all_nodes:
#                 distance = np.sqrt((vx - jx)**2 + (vy - jy)**2)
#                 if distance < 20:  # If vertex is too close to existing junction
#                     is_duplicate = True
#                     break
            
#             if not is_duplicate:
#                 all_nodes.append((vx, vy, 'vertex'))
    
#     # Update total nodes count
#     total_nodes = len(all_nodes)

#     # 4. If shape is polygon, get vertices for verification
#     if shape_type != "Circle" and len(shape_vertices) > 0:
#         # Use shape vertices from code A
#         print(f"Using {len(shape_vertices)} vertices from shape detection")
#     else:
#         # Fall back to shape vertex detection 
#         shape_vertices = detect_shape_vertices(binary)
#         if len(shape_vertices) >= 3:
#             # Verify corners with skeleton analysis
#             verified_corners = detect_and_verify_corners(binary, skeleton)
#             if len(verified_corners) >= 3:
#                 shape_vertices = verified_corners
#                 # print(f"Verified corners found: {len(shape_vertices)} vertices")

#     # 5. Process Result Image
#     output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
#     # Initialize counters
#     pattern_counts = {2: 0, 3: 0, 4: 0, 6: 0, 8: 0, "other": 0}
    
#     # Process each node (junctions + vertices)
#     for node in all_nodes:
#         cx, cy, node_type = node
#         is_shape_vertex = (node_type == 'vertex')
        
#         # Determine degree with appropriate method
#         if is_shape_vertex:
#             # For shape vertices - use standard method
#             degree = count_branches_standard(skeleton, cy, cx, radius=10)
#         else:
#             # For junctions - use robust method
#             degree = count_branches_robust(skeleton, cy, cx, radius=15)
        
#         # Classification and coloring - ALL degree2 nodes are counted as 2
#         if degree == 2:
#             if is_shape_vertex:
#                 color = (255, 0, 255)  # Magenta for shape vertices
#             else:
#                 # All other degree-2 nodes (curves, corners)
#                 color = (200, 0, 200)  # Lighter magenta
#             pattern_counts[2] += 1  # Always count as degree 2
            
#             # Draw shape vertices with green dot AND magenta circle
#             if is_shape_vertex:
#                 cv2.circle(output_img, (cx, cy), 8, (0, 255, 0), -1)  # Green dot
#                 cv2.circle(output_img, (cx, cy), 12, color, 2)  # Magenta circle
#                 cv2.putText(output_img, str(degree), (cx-15, cy-15), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#             else:
#                 cv2.circle(output_img, (cx, cy), 10, color, 2)
#                 cv2.putText(output_img, str(degree), (cx-15, cy-15), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
#         elif degree == 3:
#             color = (0, 0, 255)  # Red
#             pattern_counts[3] += 1
#             cv2.circle(output_img, (cx, cy), 10, color, 2)
#             cv2.putText(output_img, str(degree), (cx-15, cy-15), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#         elif degree == 4:
#             color = (255, 0, 0)  # Blue
#             pattern_counts[4] += 1
#             cv2.circle(output_img, (cx, cy), 10, color, 2)
#             cv2.putText(output_img, str(degree), (cx-15, cy-15), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#         elif degree == 6: 
#             color = (0, 255, 0)  # Green
#             pattern_counts[6] += 1
#             cv2.circle(output_img, (cx, cy), 10, color, 2)
#             cv2.putText(output_img, str(degree), (cx-15, cy-15), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#         elif degree == 8:  
#             color = (255, 255, 0)  # Cyan
#             pattern_counts[8] += 1
#             cv2.circle(output_img, (cx, cy), 10, color, 2)
#             cv2.putText(output_img, str(degree), (cx-15, cy-15), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#         else:
#             pattern_counts["other"] += 1
#             color = (128, 128, 128)  # Gray for other nodes
#             cv2.circle(output_img, (cx, cy), 6, color, 1)
#             cv2.putText(output_img, str(degree), (cx-10, cy-10), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
#     # 6. Draw shape outline if it's a polygon
#     if shape_type != "Circle" and len(shape_vertices) >= 3:
#         # Draw connecting lines
#         for i in range(len(shape_vertices)):
#             pt1 = shape_vertices[i]
#             pt2 = shape_vertices[(i + 1) % len(shape_vertices)]
#             cv2.line(output_img, pt1, pt2, (0, 255, 255), 2)  # Yellow lines
#     elif shape_type == "Circle":
#         # Draw circle outline
#         (x, y), radius = cv2.minEnclosingCircle(main_contour)
#         center = (int(x), int(y))
#         radius = int(radius)
#         cv2.circle(output_img, center, radius, (0, 255, 255), 2)  # Yellow circle

#     # 7. Display results
#     plt.figure(figsize=(10, 8))
#     plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
#     plt.title(f"Outer Shape: {shape_type}")
#     plt.axis('off')
#     plt.show()

#     # 8. Detailed Report
#     print("\n" + "=" * 60)
#     print("SHAPE AND GRID PATTERN ANALYSIS")
#     print("=" * 60)
#     print(f"Detected Shape: {shape_type}")
    
#     # if shape_type != "Circle":
#     #     print(f"Number of vertices: {len(shape_vertices)}")
        
#     #     if len(shape_vertices) >= 3:
#     #         print("\nVertex coordinates:")
#     #         for i, (x, y) in enumerate(shape_vertices):
#     #             print(f"  Vertex {i+1}: ({x}, {y})")
            
#     #         # Calculate side lengths
#     #         print("\nSide lengths:")
#     #         side_lengths = []
#     #         for i in range(len(shape_vertices)):
#     #             x1, y1 = shape_vertices[i]
#     #             x2, y2 = shape_vertices[(i + 1) % len(shape_vertices)]
#     #             length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#     #             side_lengths.append(length)
#     #             print(f"  Side {i+1}: {length:.1f} pixels")
            
#     #         avg_side = np.mean(side_lengths)
#     #         std_side = np.std(side_lengths)
#     #         print(f"\nAverage side length: {avg_side:.1f} pixels")
#     #         print(f"Standard deviation: {std_side:.1f} pixels")
            
#     #         if std_side / avg_side < 0.15:
#     #             print("Shape appears regular")
#     #         else:
#     #             print("Shape appears irregular")
#     # else:
#     #     print(f"Circle center: ({int(x)}, {int(y)})")
#     #     print(f"Circle radius: {radius:.1f} pixels")
#     #     print(f"Circularity: {circularity:.3f}")
#     #     print(f"Extent (area/enclosing circle): {extent:.3f}")
    
#     # print("\nINTERSECTION ANALYSIS:")
#     # print(f"Total detected nodes: {total_nodes}")
#     # shape_vertex_count = len([n for n in all_nodes if n[2] == 'vertex'])
#     # junction_count = len([n for n in all_nodes if n[2] == 'junction'])
#     # print(f"  - Shape vertices: {shape_vertex_count}")
#     # print(f"  - Internal junctions: {junction_count}")
#     print("\nRESULT:")
#     print(f"  Degree 2: {pattern_counts[2]}")
#     print(f"  Degree 3: {pattern_counts[3]}")
#     print(f"  Degree 4: {pattern_counts[4]}")
#     print(f"  Degree 6: {pattern_counts[6]}")
#     print(f"  Degree 8: {pattern_counts[8]}")
#     print(f"  Other/Unclassified: {pattern_counts['other']}")

def detect_grid_patterns_robust(img):
    """
    img: numpy array (grayscale)
    Returns dict with output image, shape type, and degree counts
    """
    if img is None:
        print("Error: No image provided")
        return

    # 1. Preprocess
    # Ensure image is grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert (Lines = White) 
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological Close to fill gaps for skeletonization 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 2. Detect Shape 
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return
    
    main_contour = max(contours, key=cv2.contourArea)
    shape_type, shape_vertices = get_refined_shape(main_contour)
    
    # Circularity and extent
    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, True)
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
    (x, y), radius = cv2.minEnclosingCircle(main_contour)
    enclosing_circle_area = np.pi * (radius**2)
    extent = area / enclosing_circle_area if enclosing_circle_area > 0 else 0

    # 3. Skeletonize and find Junctions
    skeleton = skeletonize(binary_closed // 255).astype(np.uint8)
    
    neighbor_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skeleton, -1, neighbor_kernel)
    junction_pixels = np.where((neighbors * skeleton) >= 3, 255, 0).astype(np.uint8)
    
    junction_blobs = cv2.dilate(junction_pixels, np.ones((9,9), np.uint8), iterations=2)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(junction_blobs)
    
    all_nodes = []
    for i in range(1, num_labels):
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        all_nodes.append((cx, cy, 'junction'))
    
    if shape_type != "Circle" and len(shape_vertices) > 0:
        for (vx, vy) in shape_vertices:
            is_duplicate = False
            for (jx, jy, _) in all_nodes:
                if np.sqrt((vx - jx)**2 + (vy - jy)**2) < 20:
                    is_duplicate = True
                    break
            if not is_duplicate:
                all_nodes.append((vx, vy, 'vertex'))
    
    total_nodes = len(all_nodes)

    # 5. Process Result Image
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    pattern_counts = {2:0,3:0,4:0,6:0,8:0,"other":0}
    
    for node in all_nodes:
        cx, cy, node_type = node
        is_shape_vertex = (node_type == 'vertex')
        
        if is_shape_vertex:
            degree = count_branches_standard(skeleton, cy, cx, radius=10)
        else:
            degree = count_branches_robust(skeleton, cy, cx, radius=15)
        
        if degree == 2:
            color = (0,255,0) if is_shape_vertex else (200,0,200)
            pattern_counts[2] += 1
            cv2.circle(output_img, (cx, cy), 10, color, 2)
        elif degree == 3:
            color = (0,0,255)
            pattern_counts[3] += 1
            cv2.circle(output_img, (cx, cy), 10, color, 2)
        elif degree == 4:
            color = (255,0,0)
            pattern_counts[4] += 1
            cv2.circle(output_img, (cx, cy), 10, color, 2)
        elif degree == 6:
            color = (0,255,0)
            pattern_counts[6] += 1
            cv2.circle(output_img, (cx, cy), 10, color, 2)
        elif degree == 8:
            color = (255,255,0)
            pattern_counts[8] += 1
            cv2.circle(output_img, (cx, cy), 10, color, 2)
        else:
            color = (128,128,128)
            pattern_counts["other"] += 1
            cv2.circle(output_img, (cx, cy), 6, color, 1)
    
    if shape_type != "Circle" and len(shape_vertices) >= 3:
        for i in range(len(shape_vertices)):
            pt1 = shape_vertices[i]
            pt2 = shape_vertices[(i+1)%len(shape_vertices)]
            cv2.line(output_img, pt1, pt2, (0,255,255),2)
    elif shape_type=="Circle":
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(output_img, center, radius, (0,255,255),2)
    
    return {"output_img": output_img, "shape_type": shape_type, "pattern_counts": pattern_counts}


# Testing 
if __name__ == "__main__":
    detect_grid_patterns_robust('images/circle3.png')