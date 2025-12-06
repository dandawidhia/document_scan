"""
core/preprocessor.py
Image preprocessing: lighting enhancement, edge detection, masking
"""

import cv2
import numpy as np
from typing import Tuple


def preprocess_image(
    img: np.ndarray,
    debug: bool = False,
    fast_mode: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess image untuk edge detection.
    
    Args:
        img: Input image (BGR)
        debug: Enable debug output
        fast_mode: Skip heavy operations
    
    Returns:
        (edges, enhanced_gray)
    """
    # Lighting enhancement
    enhanced_gray = enhance_lighting(img, debug=debug, fast_mode=fast_mode)
    
    mean_brightness = np.mean(enhanced_gray)
    
    # Denoising
    if fast_mode:
        denoised = cv2.medianBlur(enhanced_gray, 3)
    else:
        if mean_brightness < 80:
            denoised = cv2.bilateralFilter(enhanced_gray, 13, 120, 120)
        elif mean_brightness < 120:
            denoised = cv2.bilateralFilter(enhanced_gray, 11, 100, 100)
        else:
            denoised = cv2.medianBlur(enhanced_gray, 5)
    
    # Edge detection
    if fast_mode:
        edges = cv2.Canny(denoised, 50, 150, apertureSize=3)
    else:
        edges = _adaptive_edge_detection(denoised, mean_brightness)
    
    # Morphology
    if mean_brightness < 110:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel_small, iterations=2)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.dilate(edges, kernel_small, iterations=1)
    
    return edges, enhanced_gray


def enhance_lighting(
    img: np.ndarray,
    debug: bool = False,
    fast_mode: bool = False
) -> np.ndarray:
    """
    Enhance pencahayaan untuk kondisi redup.
    
    Returns:
        Enhanced grayscale image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    if debug:
        print(f"[DEBUG] Original brightness: {mean_brightness:.1f}, std: {std_brightness:.1f}")
    
    # Fast mode: simple CLAHE
    if fast_mode:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        if debug:
            print(f"[DEBUG] Enhanced brightness: {np.mean(enhanced):.1f}")
        return enhanced
    
    # Adaptive CLAHE parameters
    if mean_brightness < 70:
        clip_limit, tile_size = 8.0, (3, 3)
    elif mean_brightness < 100:
        clip_limit, tile_size = 6.5, (4, 4)
    elif mean_brightness < 140:
        clip_limit, tile_size = 5.0, (6, 6)
    elif mean_brightness > 190:
        clip_limit, tile_size = 1.8, (10, 10)
    else:
        clip_limit, tile_size = 3.0, (8, 8)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced = clahe.apply(gray)
    
    # Gamma correction & brightening
    if mean_brightness < 60:
        gamma = 2.2
        enhanced = np.power(enhanced / 255.0, 1.0 / gamma) * 255
        enhanced = np.uint8(enhanced)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=35)
        enhanced = cv2.equalizeHist(enhanced)
    elif mean_brightness < 90:
        gamma = 1.9
        enhanced = np.power(enhanced / 255.0, 1.0 / gamma) * 255
        enhanced = np.uint8(enhanced)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=25)
    elif mean_brightness < 140:
        gamma = 1.6
        enhanced = np.power(enhanced / 255.0, 1.0 / gamma) * 255
        enhanced = np.uint8(enhanced)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.15, beta=15)
    elif mean_brightness > 200:
        gamma = 0.6
        enhanced = np.power(enhanced / 255.0, 1.0 / gamma) * 255
        enhanced = np.uint8(enhanced)
    
    # Sharpening untuk kontras rendah
    if std_brightness < 30:
        kernel = np.array([[-1, -1, -1], [-1, 12, -1], [-1, -1, -1]]) / 3.0
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        enhanced = cv2.addWeighted(enhanced, 0.5, sharpened, 0.5, 0)
    elif std_brightness < 40:
        kernel = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]]) / 2.5
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        enhanced = cv2.addWeighted(enhanced, 0.6, sharpened, 0.4, 0)
    
    final_brightness = np.mean(enhanced)
    if debug:
        print(f"[DEBUG] Enhanced brightness: {final_brightness:.1f}")
    
    return enhanced


def _adaptive_edge_detection(denoised: np.ndarray, mean_brightness: float) -> np.ndarray:
    """Edge detection adaptif berdasarkan brightness"""
    if mean_brightness < 70:
        edges1 = cv2.Canny(denoised, 15, 60, apertureSize=3)
        edges2 = cv2.Canny(denoised, 20, 80, apertureSize=3)
        edges3 = cv2.Canny(denoised, 10, 50, apertureSize=3)
        edges = cv2.bitwise_or(cv2.bitwise_or(edges1, edges2), edges3)
    elif mean_brightness < 100:
        edges1 = cv2.Canny(denoised, 18, 75, apertureSize=3)
        edges2 = cv2.Canny(denoised, 25, 95, apertureSize=3)
        edges3 = cv2.Canny(denoised, 12, 65, apertureSize=3)
        edges = cv2.bitwise_or(cv2.bitwise_or(edges1, edges2), edges3)
    elif mean_brightness < 140:
        edges1 = cv2.Canny(denoised, 25, 100, apertureSize=3)
        edges2 = cv2.Canny(denoised, 35, 120, apertureSize=3)
        edges = cv2.bitwise_or(edges1, edges2)
    else:
        edges = cv2.Canny(denoised, 50, 150, apertureSize=3)
    
    return edges


def detect_white_regions(img: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Deteksi area putih (kertas) adaptif.
    
    Returns:
        Binary mask of white regions
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    if debug:
        print(f"[DEBUG] White detection - brightness: {mean_brightness:.1f}")
    
    # LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    mean_l, std_l = np.mean(l_channel), np.std(l_channel)
    
    # Adaptive thresholds
    if mean_brightness < 70:
        high = min(180, mean_l + 0.8 * std_l)
        med = min(150, mean_l + 0.5 * std_l)
        low = min(120, mean_l + 0.2 * std_l)
    elif mean_brightness < 100:
        high = min(200, mean_l + 1.0 * std_l)
        med = min(170, mean_l + 0.7 * std_l)
        low = min(140, mean_l + 0.4 * std_l)
    elif mean_brightness < 140:
        high = min(210, mean_l + 1.2 * std_l)
        med = min(185, mean_l + 0.9 * std_l)
        low = min(160, mean_l + 0.6 * std_l)
    else:
        high = min(245, mean_l + 2.2 * std_l)
        med = min(225, mean_l + 1.7 * std_l)
        low = min(200, mean_l + 1.4 * std_l)
    
    _, white_lab1 = cv2.threshold(l_channel, high, 255, cv2.THRESH_BINARY)
    _, white_lab2 = cv2.threshold(l_channel, med, 255, cv2.THRESH_BINARY)
    _, white_lab3 = cv2.threshold(l_channel, low, 255, cv2.THRESH_BINARY)
    
    # HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    if mean_brightness < 70:
        sat_thresh, val_thresh = 80, 100
    elif mean_brightness < 100:
        sat_thresh, val_thresh = 70, 120
    elif mean_brightness < 140:
        sat_thresh, val_thresh = 60, 140
    else:
        sat_thresh, val_thresh = 30, 180
    
    sat_mask = hsv[:, :, 1] < sat_thresh
    val_mask = hsv[:, :, 2] > val_thresh
    white_hsv = (sat_mask & val_mask).astype(np.uint8) * 255
    
    # RGB mean/std
    bgr_mean = np.mean(img, axis=2)
    bgr_std = np.std(img, axis=2)
    
    if mean_brightness < 70:
        mean_thresh, std_thresh = 120, 45
    elif mean_brightness < 100:
        mean_thresh, std_thresh = 140, 40
    elif mean_brightness < 140:
        mean_thresh, std_thresh = 160, 35
    else:
        mean_thresh, std_thresh = 200, 15
    
    white_rgb = ((bgr_mean > mean_thresh) & (bgr_std < std_thresh)).astype(np.uint8) * 255
    
    # Combine masks
    if mean_brightness < 100:
        white_combined = cv2.bitwise_or(
            cv2.bitwise_or(white_lab3, white_lab2), white_lab1
        )
        white_combined = cv2.bitwise_or(
            cv2.bitwise_or(white_combined, white_hsv), white_rgb
        )
    else:
        white_combined = cv2.bitwise_or(white_lab2, white_lab1)
        white_combined = cv2.bitwise_or(
            cv2.bitwise_or(white_combined, white_hsv), white_rgb
        )
    
    # Morphology cleanup
    k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k_med = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    white_combined = cv2.morphologyEx(white_combined, cv2.MORPH_OPEN, k_small)
    white_combined = cv2.morphologyEx(white_combined, cv2.MORPH_CLOSE, k_med)
    
    white_ratio = np.sum(white_combined > 0) / (w * h)
    if debug:
        print(f"[DEBUG] White detection ratio: {white_ratio:.3f}")
    
    return white_combined


def detect_skin(img: np.ndarray) -> np.ndarray:
    """
    Deteksi area kulit (tangan) untuk filtering.
    
    Returns:
        Binary mask of skin regions
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    # HSV skin detection
    hsv_mask1 = cv2.inRange(hsv, np.array([0, 25, 50]), np.array([25, 255, 255]))
    hsv_mask2 = cv2.inRange(hsv, np.array([155, 25, 50]), np.array([179, 255, 255]))
    hsv_mask = cv2.bitwise_or(hsv_mask1, hsv_mask2)
    
    # YCrCb skin detection
    ycrcb_mask = cv2.inRange(ycrcb, np.array([0, 135, 85]), np.array([255, 170, 120]))
    
    # Combine
    skin_mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)
    
    # Morphology cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
    # Filter small regions
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = img.shape[0] * img.shape[1] * 0.008
    filtered = np.zeros_like(skin_mask)
    for c in contours:
        if cv2.contourArea(c) > min_area:
            cv2.fillPoly(filtered, [c], 255)
    
    return filtered