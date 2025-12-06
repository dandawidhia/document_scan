"""
core/detector.py
Main detection logic for document scanning
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from .preprocessor import preprocess_image, detect_white_regions, detect_skin
from .transformer import order_points
from utils.geometry import (
    is_valid_receipt_quad,
    calculate_rectangularity_error,
    score_quad_geometry
)


def detect_receipt_contour(
    image: np.ndarray,
    debug: bool = False,
    fast_mode: bool = False
) -> List[Tuple[np.ndarray, float, Dict]]:
    """
    Deteksi kontur dokumen/kwitansi dari gambar.
    
    Args:
        image: Input image (BGR format)
        debug: Enable debug logging
        fast_mode: Use faster but less accurate detection (for real-time)
    
    Returns:
        List of (quad_points, confidence_score, details_dict)
        Sorted by confidence score (highest first)
    """
    h, w = image.shape[:2]
    
    # Resize untuk processing jika terlalu besar
    if fast_mode:
        max_dim = 640  # Lebih kecil untuk real-time
    else:
        max_dim = 800
    
    img_small, scale = _resize_for_processing(image, max_dim)
    
    # Preprocessing
    edges, enhanced = preprocess_image(img_small, debug=debug, fast_mode=fast_mode)
    white_mask = detect_white_regions(img_small, debug=debug)
    skin_mask = detect_skin(img_small)
    
    # Deteksi kontur kandidat
    receipt_contours = _find_receipt_contours(
        img_small, white_mask, skin_mask, debug=debug, fast_mode=fast_mode
    )
    
    if debug:
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        print(f"[DEBUG] Found {len(receipt_contours)} potential receipt contours")
        print(f"[DEBUG] Average brightness: {np.mean(gray):.1f}")
    
    # Konversi ke quad dan scoring
    candidates = []
    for contour in receipt_contours:
        quad = _contour_to_quad(contour, img_small.shape)
        if quad is not None:
            # Straightening jika perlu
            straightened_quad = _straighten_quad(quad, img_small.shape)
            if straightened_quad is not None:
                quad = straightened_quad
            
            # Scale kembali ke ukuran original
            quad_orig = (quad / scale).astype(np.float32)
            
            # Scoring
            score, details = _score_receipt_quad(
                quad_orig, 
                image,
                cv2.resize(white_mask, (w, h)),
                cv2.resize(skin_mask, (w, h))
            )
            candidates.append((quad_orig, score, details))
    
    # Sort by score
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Filter kandidat
    filtered = _filter_candidates(candidates, debug=debug)
    
    # Fallback jika tidak ada kandidat bagus
    if not filtered:
        fallback = _fallback_detection(image, white_mask, skin_mask, debug=debug)
        if fallback:
            return [fallback]
    
    return filtered if filtered else candidates[:1]


def _resize_for_processing(img: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    """Resize image untuk processing dengan maintain aspect ratio"""
    h, w = img.shape[:2]
    scale = 1.0
    
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        img_small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_small = img.copy()
    
    return img_small, scale


def _find_receipt_contours(
    img: np.ndarray,
    white_mask: np.ndarray,
    skin_mask: np.ndarray,
    debug: bool = False,
    fast_mode: bool = False
) -> List[np.ndarray]:
    """Temukan kontur kandidat dari white mask"""
    h, w = img.shape[:2]
    
    # Remove skin dari white mask
    clean_white = cv2.bitwise_and(white_mask, cv2.bitwise_not(skin_mask))
    
    # Find contours
    contours, _ = cv2.findContours(clean_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    img_area = w * h
    
    if debug:
        print(f"[DEBUG] Found {len(contours)} total contours")
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area
        min_area = img_area * (0.05 if fast_mode else 0.03)
        max_area = img_area * 0.9
        if area < min_area or area > max_area:
            continue
        
        # Bounding rectangle checks
        x, y, rw, rh = cv2.boundingRect(contour)
        
        # Minimum dimensions
        min_dimension = min(w, h) * (0.12 if fast_mode else 0.10)
        if min(rw, rh) < min_dimension:
            continue
        
        # Aspect ratio check
        aspect_ratio = max(rw, rh) / max(min(rw, rh), 1)
        if aspect_ratio > 8 or aspect_ratio < 1.02:
            continue
        
        # Fill ratio
        rect_area = rw * rh
        fill_ratio = area / (rect_area + 1e-6)
        if fill_ratio < 0.45:
            continue
        
        # Compactness (circularity)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            compactness = (perimeter * perimeter) / (4 * np.pi * area)
            if compactness > 12:
                continue
        
        # Size check
        if rw < 60 or rh < 80:
            continue
        
        candidates.append(contour)
        
        if debug:
            print(f"[DEBUG] Candidate: area={area:.0f}, dims=({rw}x{rh}), aspect={aspect_ratio:.2f}")
    
    if debug:
        print(f"[DEBUG] Final candidates: {len(candidates)}")
    
    return candidates


def _contour_to_quad(contour: np.ndarray, img_shape: Tuple) -> Optional[np.ndarray]:
    """Konversi contour ke quadrilateral (4 titik)"""
    h, w = img_shape[:2]
    perimeter = cv2.arcLength(contour, True)
    
    # Try different epsilon values untuk approximation
    for eps in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06]:
        approx = cv2.approxPolyDP(contour, eps * perimeter, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            if is_valid_receipt_quad(quad, w, h):
                return quad
    
    # Fallback: minimum area rectangle
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    if is_valid_receipt_quad(box, w, h):
        return box
    
    # Last resort: bounding rect
    x, y, rw, rh = cv2.boundingRect(contour)
    return np.array([
        [x, y],
        [x + rw, y],
        [x + rw, y + rh],
        [x, y + rh]
    ], dtype=np.float32)


def _straighten_quad(quad: np.ndarray, img_shape: Tuple) -> Optional[np.ndarray]:
    """Straighten quad yang miring untuk lebih rectangular"""
    h, w = img_shape[:2]
    
    rect_error = calculate_rectangularity_error(quad)
    if rect_error < 8:  # Sudah cukup lurus
        return None
    
    ordered = order_points(quad)
    straightened = ordered.copy()
    
    # Rata-rata Y untuk top dan bottom
    top_y = (ordered[0][1] + ordered[1][1]) / 2
    straightened[0][1] = straightened[1][1] = top_y
    
    bottom_y = (ordered[2][1] + ordered[3][1]) / 2
    straightened[2][1] = straightened[3][1] = bottom_y
    
    # Rata-rata X untuk left dan right
    left_x = (ordered[0][0] + ordered[3][0]) / 2
    straightened[0][0] = straightened[3][0] = left_x
    
    right_x = (ordered[1][0] + ordered[2][0]) / 2
    straightened[1][0] = straightened[2][0] = right_x
    
    # Clip ke boundaries
    straightened[:, 0] = np.clip(straightened[:, 0], 0, w - 1)
    straightened[:, 1] = np.clip(straightened[:, 1], 0, h - 1)
    
    # Validate
    if is_valid_receipt_quad(straightened, w, h):
        new_error = calculate_rectangularity_error(straightened)
        if new_error < rect_error * 0.9:
            return straightened
    
    return None


def _score_receipt_quad(
    quad: np.ndarray,
    img: np.ndarray,
    white_mask: np.ndarray,
    skin_mask: np.ndarray
) -> Tuple[float, Dict]:
    """
    Score quad berdasarkan berbagai faktor.
    Return: (score, details_dict)
    """
    h, w = img.shape[:2]
    
    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
    mask_pixels = np.sum(mask > 0)
    
    if mask_pixels == 0:
        return 0.0, {}
    
    # Brightness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    quad_brightness = np.mean(gray[mask > 0])
    
    # Dimensions
    ordered = order_points(quad)
    quad_width = np.linalg.norm(ordered[1] - ordered[0])
    quad_height = np.linalg.norm(ordered[3] - ordered[0])
    quad_area = cv2.contourArea(quad)
    
    # 1. White Score
    white_in_quad = cv2.bitwise_and(mask, white_mask)
    white_ratio = np.sum(white_in_quad > 0) / mask_pixels
    
    if quad_brightness < 80:
        white_score = min(1.0, white_ratio * 3.5)
    elif quad_brightness < 120:
        white_score = min(1.0, white_ratio * 2.8)
    elif quad_brightness < 160:
        white_score = min(1.0, white_ratio * 2.2)
    else:
        white_score = min(1.0, white_ratio * 1.5)
    
    # 2. Skin Penalty
    skin_in_quad = cv2.bitwise_and(mask, skin_mask)
    skin_ratio = np.sum(skin_in_quad > 0) / mask_pixels
    
    if skin_ratio > 0.6:
        skin_penalty = 0.4
    elif skin_ratio > 0.4:
        skin_penalty = 0.7
    else:
        skin_penalty = max(0.8, 1.0 - skin_ratio * 1.0)
    
    # 3. Rectangularity Score
    rect_error = calculate_rectangularity_error(quad)
    geometry_score = score_quad_geometry(quad, w, h)
    
    if rect_error > 40:
        rect_score = 0.3
    elif rect_error > 30:
        rect_score = 0.5
    elif rect_error > 20:
        rect_score = 0.7
    elif rect_error > 15:
        rect_score = 0.85
    else:
        rect_score = 1.0
    
    combined_rect_score = 0.6 * rect_score + 0.4 * geometry_score
    
    # 4. Size Score
    area_ratio = quad_area / (w * h)
    if 0.05 <= area_ratio <= 0.7:
        size_score = 1.0
        if 0.10 <= area_ratio <= 0.5:
            size_score = 1.2
    elif 0.02 <= area_ratio < 0.05:
        size_score = area_ratio / 0.05 * 0.8
    elif 0.7 < area_ratio <= 0.9:
        size_score = 0.95
    else:
        size_score = 0.3
    
    # 5. Aspect Score
    aspect = max(quad_width, quad_height) / (min(quad_width, quad_height) + 1e-6)
    if 1.2 <= aspect <= 5:
        aspect_score = 1.0
    elif 1.0 <= aspect < 1.2:
        aspect_score = 0.9
    elif 5 < aspect <= 8:
        aspect_score = 0.8
    else:
        aspect_score = max(0.3, 1.0 - (aspect - 8) * 0.1)
    
    # 6. Position Score
    center = quad.mean(axis=0)
    img_center = np.array([w / 2, h / 2])
    dist_from_center = np.linalg.norm(center - img_center) / np.linalg.norm(img_center)
    position_score = 0.6 + 0.4 * (1 - min(1.0, dist_from_center))
    
    # 7. Brightness Variance
    quad_region = img[mask > 0]
    if len(quad_region) > 0:
        brightness_std = np.std(cv2.cvtColor(
            quad_region.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY
        ))
        threshold = 140.0 if quad_brightness < 100 else 120.0
        brightness_score = max(0.6, 1.0 - brightness_std / threshold)
    else:
        brightness_score = 0.5
    
    # 8. Dimension Score
    min_quad_dim = min(quad_width, quad_height)
    min_img_dim = min(w, h)
    dimension_ratio = min_quad_dim / min_img_dim
    
    if dimension_ratio > 0.15:
        dimension_score = 1.0
    elif dimension_ratio > 0.08:
        dimension_score = 0.9
    else:
        dimension_score = max(0.4, dimension_ratio / 0.08)
    
    # Rectangularity Gate
    if rect_error > 35:
        rectangularity_gate = 0.7
    elif rect_error > 25:
        rectangularity_gate = 0.85
    elif rect_error > 15:
        rectangularity_gate = 0.95
    else:
        rectangularity_gate = 1.0
    
    # Combined Score
    base_score = (
        white_score * 0.35 +
        size_score * 0.25 +
        combined_rect_score * 0.20 +
        skin_penalty * 0.08 +
        aspect_score * 0.06 +
        dimension_score * 0.04 +
        brightness_score * 0.02
    )
    
    final_score = base_score * rectangularity_gate
    
    # Brightness boost
    if quad_brightness < 90:
        final_score = min(1.0, final_score * 1.3)
    elif quad_brightness < 130:
        final_score = min(1.0, final_score * 1.15)
    
    details = {
        'white_score': white_score,
        'size_score': size_score,
        'skin_penalty': skin_penalty,
        'rect_score': combined_rect_score,
        'rect_error': rect_error,
        'rectangularity_gate': rectangularity_gate,
        'aspect_score': aspect_score,
        'dimension_score': dimension_score,
        'brightness_score': brightness_score,
        'position_score': position_score,
        'white_ratio': white_ratio,
        'skin_ratio': skin_ratio,
        'area_ratio': area_ratio,
        'aspect_ratio': aspect,
        'quad_brightness': quad_brightness,
        'base_score': base_score
    }
    
    return final_score, details


def _filter_candidates(
    candidates: List[Tuple[np.ndarray, float, Dict]],
    debug: bool = False
) -> List[Tuple[np.ndarray, float, Dict]]:
    """Filter kandidat berdasarkan threshold"""
    filtered = []
    
    for quad, score, details in candidates:
        rect_error = details.get('rect_error', 100)
        
        # Strict filtering
        if rect_error < 45 and score > 0.2:
            filtered.append((quad, score, details))
        elif debug:
            print(f"[DEBUG] Rejected: rect_error={rect_error:.1f}, score={score:.3f}")
    
    # Ultra permissive fallback
    if not filtered and candidates:
        if debug:
            print("[DEBUG] Using ultra permissive fallback")
        for quad, score, details in candidates[:2]:
            if score > 0.15:
                filtered.append((quad, score, details))
    
    return filtered


def _fallback_detection(
    image: np.ndarray,
    white_mask: np.ndarray,
    skin_mask: np.ndarray,
    debug: bool = False
) -> Optional[Tuple[np.ndarray, float, Dict]]:
    """Fallback detection menggunakan largest white region"""
    h, w = image.shape[:2]
    
    if debug:
        print("[DEBUG] Using fallback detection")
    
    clean_white = cv2.bitwise_and(white_mask, cv2.bitwise_not(skin_mask))
    contours, _ = cv2.findContours(clean_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > w * h * 0.01:
            quad = _contour_to_quad(largest, image.shape)
            if quad is not None:
                score = 0.2
                details = {'fallback': 'white_region_fallback'}
                return (quad, score, details)
    
    # Ultimate fallback: adaptive margin
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    
    if brightness < 80:
        margin = min(w, h) // 25
    elif brightness < 120:
        margin = min(w, h) // 20
    else:
        margin = min(w, h) // 15
    
    quad = np.array([
        [margin, margin],
        [w - 1 - margin, margin],
        [w - 1 - margin, h - 1 - margin],
        [margin, h - 1 - margin]
    ], dtype=np.float32)
    
    score = 0.1
    details = {'fallback': 'adaptive_margin', 'brightness': brightness}
    
    return (quad, score, details)