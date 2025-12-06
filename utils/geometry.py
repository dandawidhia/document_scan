"""
utils/geometry.py
Geometric calculations dan validations
"""

import cv2
import numpy as np
from typing import Tuple


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate angle at p2 formed by p1-p2-p3.
    
    Args:
        p1, p2, p3: Points as numpy arrays [x, y]
    
    Returns:
        Angle in degrees
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    return angle


def calculate_rectangularity_error(quad: np.ndarray) -> float:
    """
    Calculate how much quad deviates from perfect rectangle.
    Lower is better (0 = perfect rectangle).
    
    Args:
        quad: 4 corner points
    
    Returns:
        Mean deviation from 90 degrees
    """
    from core.transformer import order_points
    
    ordered = order_points(quad)
    angles = []
    
    for i in range(4):
        p1 = ordered[(i - 1) % 4]
        p2 = ordered[i]
        p3 = ordered[(i + 1) % 4]
        angle = calculate_angle(p1, p2, p3)
        angles.append(abs(angle - 90.0))
    
    return np.mean(angles)


def is_valid_receipt_quad(
    quad: np.ndarray,
    img_width: int,
    img_height: int
) -> bool:
    """
    Validate apakah quad adalah kandidat receipt yang valid.
    
    Args:
        quad: 4 corner points
        img_width: Image width
        img_height: Image height
    
    Returns:
        True if valid
    """
    # 1. Check bounds (dengan sedikit tolerance)
    if (np.any(quad < -10) or
        np.any(quad[:, 0] >= img_width + 10) or
        np.any(quad[:, 1] >= img_height + 10)):
        return False
    
    # 2. Check area
    area = cv2.contourArea(quad)
    min_area = img_width * img_height * 0.003
    max_area = img_width * img_height * 0.95
    if area < min_area or area > max_area:
        return False
    
    # 3. Check sides length
    from core.transformer import order_points
    ordered = order_points(quad)
    sides = [np.linalg.norm(ordered[i] - ordered[(i + 1) % 4]) for i in range(4)]
    
    if min(sides) < 5:
        return False
    
    # 4. Check opposite sides ratio
    ratio1 = max(sides[0], sides[2]) / max(1e-6, min(sides[0], sides[2]))
    ratio2 = max(sides[1], sides[3]) / max(1e-6, min(sides[1], sides[3]))
    
    if ratio1 > 2.0 or ratio2 > 2.0:
        return False
    
    return True


def score_quad_geometry(quad: np.ndarray, img_w: int, img_h: int) -> float:
    """
    Score quad berdasarkan geometrinya.
    
    Args:
        quad: 4 corner points
        img_w: Image width
        img_h: Image height
    
    Returns:
        Score 0.0-1.0 (higher is better)
    """
    from core.transformer import order_points
    
    quad = quad.astype(np.float32)
    rect = order_points(quad)
    
    # 1. Orthogonality score (sudut mendekati 90 derajat)
    angles = [calculate_angle(rect[(i-1) % 4], rect[i], rect[(i+1) % 4]) for i in range(4)]
    angle_deviations = [abs(a - 90.0) for a in angles]
    mean_deviation = np.mean(angle_deviations)
    ortho_score = max(0.0, 1.0 - (mean_deviation / 35.0))
    
    # 2. Opposite sides score (sisi berlawanan panjangnya mirip)
    s01 = np.linalg.norm(rect[1] - rect[0])
    s12 = np.linalg.norm(rect[2] - rect[1])
    s23 = np.linalg.norm(rect[3] - rect[2])
    s30 = np.linalg.norm(rect[0] - rect[3])
    
    def opposite_ratio(a, b):
        m = max(a, b)
        n = max(1e-6, min(a, b))
        return m / n
    
    r1 = opposite_ratio(s01, s23)
    r2 = opposite_ratio(s12, s30)
    opp_score = max(0.0, 1.0 - ((max(r1, r2) - 1.0) / 1.2))
    
    # 3. Position score (lebih bagus jika dekat center)
    center = rect.mean(axis=0)
    img_center = np.array([img_w / 2.0, img_h / 2.0], dtype=np.float32)
    dist_norm = np.linalg.norm(center - img_center) / (np.linalg.norm(img_center) + 1e-6)
    pos_score = 1.0 - min(1.0, dist_norm) * 0.2
    
    # 4. Bounds score (all points dalam gambar)
    in_bounds = (
        np.all(quad[:, 0] >= 0) and np.all(quad[:, 0] < img_w) and
        np.all(quad[:, 1] >= 0) and np.all(quad[:, 1] < img_h)
    )
    bound_score = 1.0 if in_bounds else 0.0
    
    # Combined score
    final_score = (
        0.4 * ortho_score +
        0.3 * opp_score +
        0.2 * pos_score +
        0.1 * bound_score
    )
    
    return max(0.0, min(1.0, final_score))


def calculate_aspect_ratio(quad: np.ndarray) -> float:
    """
    Calculate aspect ratio of quad.
    
    Args:
        quad: 4 corner points
    
    Returns:
        Aspect ratio (width / height)
    """
    from core.transformer import order_points
    
    ordered = order_points(quad)
    width = np.linalg.norm(ordered[1] - ordered[0])
    height = np.linalg.norm(ordered[3] - ordered[0])
    
    return width / max(height, 1e-6)


def calculate_quad_area(quad: np.ndarray) -> float:
    """
    Calculate area of quad.
    
    Args:
        quad: 4 corner points
    
    Returns:
        Area in pixels
    """
    return cv2.contourArea(quad)


def get_quad_dimensions(quad: np.ndarray) -> Tuple[float, float]:
    """
    Get width and height of quad.
    
    Args:
        quad: 4 corner points
    
    Returns:
        (width, height) tuple
    """
    from core.transformer import order_points
    
    ordered = order_points(quad)
    
    # Average of opposite sides
    width = (np.linalg.norm(ordered[1] - ordered[0]) +
             np.linalg.norm(ordered[2] - ordered[3])) / 2
    
    height = (np.linalg.norm(ordered[3] - ordered[0]) +
              np.linalg.norm(ordered[2] - ordered[1])) / 2
    
    return width, height


def point_to_line_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    """
    Calculate perpendicular distance from point to line.
    
    Args:
        point: Point [x, y]
        line_start: Line start point [x, y]
        line_end: Line end point [x, y]
    
    Returns:
        Distance
    """
    # Line vector
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    
    if line_len < 1e-6:
        return np.linalg.norm(point - line_start)
    
    # Normalized line vector
    line_unitvec = line_vec / line_len
    
    # Vector from line start to point
    point_vec = point - line_start
    
    # Project point onto line
    proj_length = np.dot(point_vec, line_unitvec)
    
    # Clamp to line segment
    proj_length = max(0, min(line_len, proj_length))
    
    # Closest point on line
    closest_point = line_start + proj_length * line_unitvec
    
    # Distance
    return np.linalg.norm(point - closest_point)


def snap_to_grid(quad: np.ndarray, grid_size: int = 10) -> np.ndarray:
    """
    Snap quad points to grid untuk alignment.
    
    Args:
        quad: 4 corner points
        grid_size: Grid cell size in pixels
    
    Returns:
        Snapped quad
    """
    snapped = np.round(quad / grid_size) * grid_size
    return snapped.astype(np.float32)


def expand_quad(quad: np.ndarray, expansion: float) -> np.ndarray:
    """
    Expand quad outward dari center.
    
    Args:
        quad: 4 corner points
        expansion: Expansion factor (e.g., 1.1 = 10% larger)
    
    Returns:
        Expanded quad
    """
    center = quad.mean(axis=0)
    expanded = center + (quad - center) * expansion
    return expanded.astype(np.float32)


def contract_quad(quad: np.ndarray, contraction: float) -> np.ndarray:
    """
    Contract quad inward ke center.
    
    Args:
        quad: 4 corner points
        contraction: Contraction factor (e.g., 0.9 = 10% smaller)
    
    Returns:
        Contracted quad
    """
    return expand_quad(quad, contraction)