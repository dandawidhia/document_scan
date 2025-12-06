"""
core/transformer.py
Perspective transformation dan enhancement
"""

import cv2
import numpy as np
from typing import Tuple


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points dalam urutan: top-left, top-right, bottom-right, bottom-left
    
    Args:
        pts: Array of 4 points (x, y)
    
    Returns:
        Ordered array of 4 points
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum: top-left has smallest, bottom-right has largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    
    # Diff: top-right has smallest, bottom-left has largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Perspective transform dari 4 titik ke rectangle.
    
    Args:
        image: Input image
        pts: 4 corner points
    
    Returns:
        Warped/transformed image
    """
    # Order points
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Calculate width
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    
    # Calculate height
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    
    # Hindari ukuran 0
    maxWidth = max(1, maxWidth)
    maxHeight = max(1, maxHeight)
    
    # Destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # Warp
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped


def add_padding(
    quad: np.ndarray,
    pad_frac: float,
    img_width: int,
    img_height: int
) -> np.ndarray:
    """
    Add padding ke quad points.
    
    Args:
        quad: 4 corner points
        pad_frac: Padding fraction (e.g., 0.03 = 3%)
        img_width: Image width for clipping
        img_height: Image height for clipping
    
    Returns:
        Padded quad points
    """
    center = quad.mean(axis=0)
    padded = center + (quad - center) * (1 + pad_frac)
    
    # Clip to image boundaries
    padded[:, 0] = np.clip(padded[:, 0], 0, img_width - 1)
    padded[:, 1] = np.clip(padded[:, 1], 0, img_height - 1)
    
    return padded.astype(np.float32)


def enhance_scanned_image(
    image: np.ndarray,
    mode: str = 'auto'
) -> np.ndarray:
    """
    Enhance hasil scan dengan berbagai mode.
    
    Args:
        image: Input warped image
        mode: 'auto', 'bw' (black & white), 'grayscale', 'color'
    
    Returns:
        Enhanced image
    """
    if mode == 'color':
        # Enhance colors
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return enhanced
    
    elif mode == 'grayscale':
        # Simple grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    elif mode == 'bw':
        # Black & white (binary)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    else:  # auto
        # Automatically decide based on content
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check if mostly text (high contrast)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        if edge_density > 0.05:  # Likely text document
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        else:  # Photo or mixed content
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            return enhanced


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotate image by 90, 180, or 270 degrees.
    
    Args:
        image: Input image
        angle: Rotation angle (90, 180, 270)
    
    Returns:
        Rotated image
    """
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return image


def resize_for_display(
    image: np.ndarray,
    max_width: int = 800,
    max_height: int = 600
) -> np.ndarray:
    """
    Resize image untuk display tanpa mengubah aspect ratio.
    
    Args:
        image: Input image
        max_width: Maximum width
        max_height: Maximum height
    
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    # Calculate scale
    scale_w = max_width / w if w > max_width else 1.0
    scale_h = max_height / h if h > max_height else 1.0
    scale = min(scale_w, scale_h)
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized
    
    return image


def draw_detection_overlay(
    image: np.ndarray,
    quad: np.ndarray,
    score: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3
) -> np.ndarray:
    """
    Draw detection overlay pada image.
    
    Args:
        image: Input image
        quad: 4 corner points
        score: Detection score
        color: Line color (B, G, R)
        thickness: Line thickness
    
    Returns:
        Image with overlay
    """
    overlay = image.copy()
    
    # Draw polygon
    cv2.polylines(overlay, [quad.astype(np.int32)], True, color, thickness)
    
    # Draw corner points
    corner_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    for i, pt in enumerate(quad):
        cv2.circle(overlay, tuple(pt.astype(int)), 8, corner_colors[i], -1)
        cv2.putText(
            overlay, str(i),
            tuple((pt + 12).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )
    
    # Draw score
    h = image.shape[0]
    score_text = f"Score: {score:.3f}"
    cv2.putText(
        overlay, score_text,
        (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (255, 255, 255), 2
    )
    
    return overlay