"""
Cross-section helpers for the interactive "Draw a section" tab.

Every shape's OUTER wall is sized to enclose a fixed 5625 mm^2 area (the
membrane-energy reference). The user's freehand drawing is clipped inside that
boundary and read back by main.detect().
"""

import cv2
import numpy as np
import math

CANVAS_SIZE = 900
OUTER_AREA_MM2 = 5625.0          # fixed cross-sectional area
_AREA_FRAC = 0.60                # outer area as a fraction of the canvas size^2


def mm_per_px(disp_size):
    """Millimetres per pixel for a canvas of side `disp_size`, given the fixed
    5625 mm^2 outer area (so drawn lengths can be reported in mm)."""
    return math.sqrt(OUTER_AREA_MM2 / (_AREA_FRAC * disp_size * disp_size))


def _regular_pts(cx, cy, R, n, off_deg=0.0):
    """Vertices of a regular n-gon with circumradius R (off_deg rotates it)."""
    return [(cx + R * math.cos(math.radians(360.0 * k / n + off_deg)),
             cy + R * math.sin(math.radians(360.0 * k / n + off_deg)))
            for k in range(n)]


def _outer(shape, size):
    """(cx, cy, polygon_or_None, radius_or_None) for the outer wall, sized so
    every shape encloses the SAME area (-> the fixed 5625 mm^2)."""
    cx = cy = size / 2.0
    A = _AREA_FRAC * size * size
    s = str(shape).lower()
    if s.startswith("cir") or s.startswith("cyl"):
        return cx, cy, None, math.sqrt(A / math.pi)              # A = pi r^2
    if s.startswith("hex"):
        R = math.sqrt(A / (1.5 * math.sqrt(3.0)))                # A = 3sqrt3/2 R^2
        return cx, cy, np.array(_regular_pts(cx, cy, R, 6, 0.0), np.int32), None
    if s.startswith("oct"):
        R = math.sqrt(A / (2.0 * math.sqrt(2.0)))                # A = 2sqrt2 R^2
        return cx, cy, np.array(_regular_pts(cx, cy, R, 8, 22.5), np.int32), None
    L = math.sqrt(A)                                             # square: A = L^2
    m = (size - L) / 2.0
    return cx, cy, np.array([[m, m], [m + L, m], [m + L, m + L], [m, m + L]],
                            np.int32), None


def outer_dimensions_mm(shape):
    """Physical outer dimensions of the shape at the fixed 5625 mm^2 area."""
    s = str(shape).lower()
    if s.startswith("cir"):
        return "circle radius = %.1f mm" % math.sqrt(OUTER_AREA_MM2 / math.pi)
    if s.startswith("hex"):
        R = math.sqrt(OUTER_AREA_MM2 / (1.5 * math.sqrt(3.0)))
        return "hexagon side = %.1f mm" % R                      # side = R for hexagon
    if s.startswith("oct"):
        R = math.sqrt(OUTER_AREA_MM2 / (2.0 * math.sqrt(2.0)))
        return "octagon side = %.1f mm" % (2.0 * R * math.sin(math.pi / 8))
    return "square side = %.1f mm" % math.sqrt(OUTER_AREA_MM2)


def outer_boundary_bgr(shape, size, thickness=3):
    """A white BGR image with ONLY the fixed-area outer wall drawn — used as the
    background the user draws inside on the canvas."""
    img = np.full((int(size), int(size), 3), 255, np.uint8)
    cx, cy, poly, R = _outer(shape, size)
    if poly is not None:
        cv2.polylines(img, [poly], True, (0, 0, 0), thickness)
    else:
        cv2.circle(img, (int(cx), int(cy)), int(R), (0, 0, 0), thickness)
    return img


def render_drawing(stroke_gray, shape, thickness=2):
    """Composite the user's drawn strokes with the outer wall and return a
    GREYSCALE image (dark = wall, white = background). We keep the anti-aliased
    strokes rather than hard-binarising them, so smooth curves stay smooth --
    main.detect() then does its own Otsu threshold + skeletonize (exactly as for
    an uploaded image), giving a clean centre-line instead of a staircase.

    stroke_gray: 2D array, higher value = more 'wall' (e.g. the canvas alpha)."""
    size = stroke_gray.shape[0]
    canvas = (255 - np.clip(stroke_gray, 0, 255)).astype(np.uint8)  # AA walls

    cx, cy, poly, R = _outer(shape, size)
    mask = np.zeros((size, size), np.uint8)
    if poly is not None:
        cv2.fillPoly(mask, [poly], 255)
        canvas[mask == 0] = 255                       # clip drawing to inside
        cv2.polylines(canvas, [poly], True, 0, thickness, cv2.LINE_AA)
    else:
        cv2.circle(mask, (int(cx), int(cy)), int(R), 255, -1)
        canvas[mask == 0] = 255
        cv2.circle(canvas, (int(cx), int(cy)), int(R), 0, thickness, cv2.LINE_AA)
    return canvas