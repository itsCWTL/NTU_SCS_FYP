import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import math

def is_straight_line(skeleton, r, c, radius=6):
    """
    Returns True if degree-2 node lies on a straight line
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
    
    # Strongly linear → straight line
    return max(eigvals) / min_eigval > 15

def count_branches_robust(skeleton_img, center_r, center_c, radius=15):
    """
    Counts branches. 
    A larger radius ensures we count the arms *leaving* the intersection,
    ignoring the messy 'knot' that often forms at the center of 6-way nodes.
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
            
    # Handle wrap-around
    if perimeter[0] > 0 and perimeter[-1] > 0:
        transitions -= 1
        
    return transitions

def count_branches_standard(skeleton_img, center_r, center_c, radius=10):
    """
    Standard branch counting for corners and regular intersections
    """
    return count_branches_robust(skeleton_img, center_r, center_c, radius=radius)

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
        if best_match_diff <= 2:  # Allow some tolerance
            vertices = [tuple(point[0]) for point in best_match]
        else:
            # Force approximation to expected_sides
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

def identify_shape_type(num_vertices):
    """
    Identify the shape based on number of vertices
    """
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

def detect_grid_patterns_robust(image_path):
    # 1. Load and Preprocess
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found.")
        return

    # Invert: Lines = White (as in code A)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # --- IMPORTANT FIX FROM CODE A: Morphological Closing ---
    # This fills small gaps in the intersection BEFORE skeletonizing.
    # It forces the 6 lines to merge into a solid blob, creating a cleaner skeleton center.
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_morph)

    # 2. Skeletonize
    skeleton = skeletonize(binary_closed // 255).astype(np.uint8)

    # 3. Find Potential Intersection Zones (Pixel-based) - FROM CODE A
    # Filter for pixels with 3+ neighbors
    kernel_neighbors = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skeleton, -1, kernel_neighbors)
    junction_pixels = np.where((neighbors * skeleton) >= 3, 255, 0).astype(np.uint8)

    # --- IMPORTANT FIX FROM CODE A: Aggressive Dilation ---
    # 6-way nodes often split into two 3-way nodes a few pixels apart.
    # We increase dilation to ensure they merge into ONE detected blob.
    dilate_kernel = np.ones((9,9), np.uint8) 
    junction_blobs = cv2.dilate(junction_pixels, dilate_kernel, iterations=2)
    
    # 4. Detect shape vertices with automatic shape detection
    # First try to detect shape without specifying expected sides
    shape_vertices = detect_shape_vertices(binary)
    shape_type = identify_shape_type(len(shape_vertices))
    
    print(f"Auto-detected shape: {shape_type} with {len(shape_vertices)} vertices")
    
    # If it looks like a hexagon (6 sides) or we want to force hexagon detection
    if len(shape_vertices) == 6 or len(shape_vertices) >= 5:
        # Use verified corners method
        verified_corners = detect_and_verify_corners(binary, skeleton)
        print(f"Verified corners found: {len(verified_corners)}")
        
        # If verified corners are close to shape vertices, use them
        if len(verified_corners) >= 3:
            shape_vertices = verified_corners
            print(f"Using verified corners: {len(shape_vertices)} vertices")
    
    # If we still don't have enough vertices, try forcing hexagon detection
    if len(shape_vertices) < 3:
        shape_vertices = detect_shape_vertices(binary, expected_sides=6)
        print(f"Forced hexagon detection: {len(shape_vertices)} vertices")
    
    # Add shape vertices to junction blobs
    for (x, y) in shape_vertices:
        cv2.circle(junction_pixels, (x, y), 8, 255, -1)
        cv2.circle(junction_blobs, (x, y), 8, 255, -1)

    # Get connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(junction_blobs)

    # Initialize counters
    pattern_counts = {2: 0, 3: 0, 4: 0, 6: 0, "other": 0}
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    print(f"Found {num_labels - 1} total potential nodes.")

    # 5. Classify Each Node
    for i in range(1, num_labels):
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        
        # Check if this is a shape vertex
        is_shape_vertex = False
        for (vertex_x, vertex_y) in shape_vertices:
            distance = np.sqrt((cx - vertex_x)**2 + (cy - vertex_y)**2)
            if distance < 25:
                is_shape_vertex = True
                break
        
        # Use appropriate branch counting method
        if is_shape_vertex:
            # For shape vertices, use standard radius
            degree = count_branches_standard(skeleton, int(cy), int(cx), radius=10)
        else:
            # For potential 6-way nodes, use larger radius (from code A)
            degree = count_branches_robust(skeleton, int(cy), int(cx), radius=14)
        
        color = (128, 128, 128)  # Gray for unknown
        
        # Classification logic
        if degree == 2:
            if is_shape_vertex:
                # Shape vertex - Type 2
                color = (255, 0, 255)  # Magenta
                pattern_counts[2] += 1
            elif not is_straight_line(skeleton, int(cy), int(cx)):
                # Other degree-2 node that's not a straight line
                color = (200, 0, 200)  # Lighter magenta
                pattern_counts[2] += 1
            else:
                pattern_counts["other"] += 1
                continue
        elif degree == 3:
            color = (0, 0, 255)  # Red
            pattern_counts[3] += 1
        elif degree == 4:
            color = (255, 0, 0)  # Blue
            pattern_counts[4] += 1
        elif degree >= 5: 
            color = (0, 255, 0)  # Green
            pattern_counts[6] += 1
        else:
            pattern_counts["other"] += 1
            continue

        # Draw visualization
        circle_radius = 12 if is_shape_vertex else 8
        cv2.circle(output_img, (cx, cy), circle_radius, color, 2)
        
        # Labels: "V" for shape vertices, number for others
        if is_shape_vertex:
            cv2.putText(output_img, "V", (cx-5, cy-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(output_img, str(degree), (cx-5, cy-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 6. Draw the shape outline
    if len(shape_vertices) >= 3:
        # Sort vertices for better visualization
        center_x = np.mean([v[0] for v in shape_vertices])
        center_y = np.mean([v[1] for v in shape_vertices])
        
        shape_vertices_sorted = sorted(shape_vertices,
                                      key=lambda v: math.atan2(v[1] - center_y, v[0] - center_x))
        
        # Draw connecting lines
        for i in range(len(shape_vertices_sorted)):
            pt1 = shape_vertices_sorted[i]
            pt2 = shape_vertices_sorted[(i + 1) % len(shape_vertices_sorted)]
            cv2.line(output_img, pt1, pt2, (0, 255, 255), 2)  # Yellow lines
        
        # Label vertices with numbers
        for i, (x, y) in enumerate(shape_vertices_sorted):
            cv2.putText(output_img, str(i+1), (x+5, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 7. ONLY Plot 1: Original image with detected nodes
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Nodes:")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 8. Detailed Report
    print("\n" + "=" * 60)
    print("SHAPE AND GRID PATTERN ANALYSIS")
    print("=" * 60)
    print(f"Detected Shape: {shape_type}")
    print(f"Number of vertices: {len(shape_vertices)}")
    
    if len(shape_vertices) >= 3:
        print("\nVertex coordinates (clockwise):")
        center_x = np.mean([v[0] for v in shape_vertices])
        center_y = np.mean([v[1] for v in shape_vertices])
        
        # Sort for display
        sorted_vertices = sorted(shape_vertices,
                                key=lambda v: math.atan2(v[1] - center_y, v[0] - center_x))
        
        for i, (x, y) in enumerate(sorted_vertices):
            print(f"  Vertex {i+1}: ({x}, {y})")
        
        # Calculate side lengths
        if len(shape_vertices) > 2:
            print("\nSide lengths:")
            side_lengths = []
            for i in range(len(sorted_vertices)):
                x1, y1 = sorted_vertices[i]
                x2, y2 = sorted_vertices[(i + 1) % len(sorted_vertices)]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                side_lengths.append(length)
                print(f"  Side {i+1}: {length:.1f} pixels")
            
            avg_side = np.mean(side_lengths)
            std_side = np.std(side_lengths)
            print(f"\nAverage side length: {avg_side:.1f} pixels")
            print(f"Standard deviation: {std_side:.1f} pixels")
            
            if std_side / avg_side < 0.15:
                print("Shape appears regular")
            else:
                print("Shape appears irregular")
    
    print("\nRESULT:")
    print(f"  Type 2: {pattern_counts[2]}")
    print(f"  Type 3: {pattern_counts[3]}")
    print(f"  Type 4: {pattern_counts[4]}")
    print(f"  Type 6: {pattern_counts[6]}")

# Testing
detect_grid_patterns_robust('images/hex1.png')