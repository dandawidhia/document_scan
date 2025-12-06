"""
utils/validation.py
Input validation dan error handling utilities
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple
import requests


def validate_image_input(src: Union[str, np.ndarray]) -> Tuple[bool, Optional[str]]:
    """
    Validate image input (URL, path, atau numpy array).
    
    Args:
        src: Image source (URL string, file path string, or numpy array)
    
    Returns:
        (is_valid, error_message)
    """
    # Numpy array
    if isinstance(src, np.ndarray):
        if src.size == 0:
            return False, "Empty numpy array"
        if len(src.shape) not in [2, 3]:
            return False, "Invalid image dimensions (must be 2D or 3D array)"
        if len(src.shape) == 3 and src.shape[2] not in [1, 3, 4]:
            return False, "Invalid number of channels (must be 1, 3, or 4)"
        return True, None
    
    # String (URL or path)
    if isinstance(src, str):
        src = src.strip()
        
        # URL
        if src.startswith('http://') or src.startswith('https://'):
            return True, None  # Will validate when downloading
        
        # File path
        path = Path(src)
        if not path.exists():
            return False, f"File not found: {src}"
        if path.is_dir():
            return False, f"Path is a directory, not a file: {src}"
        if path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            return False, f"Unsupported file format: {path.suffix}"
        return True, None
    
    return False, f"Invalid input type: {type(src)}"


def load_image_safe(src: Union[str, np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Safely load image dengan error handling.
    
    Args:
        src: Image source
    
    Returns:
        (image, error_message)
    """
    # Validate input
    is_valid, error = validate_image_input(src)
    if not is_valid:
        return None, error
    
    # Already numpy array
    if isinstance(src, np.ndarray):
        return src.copy(), None
    
    # String source
    src_str = src.strip()
    
    # URL
    if src_str.startswith('http://') or src_str.startswith('https://'):
        try:
            resp = requests.get(src_str, timeout=10)
            if resp.status_code != 200:
                return None, f"Failed to download: HTTP {resp.status_code}"
            
            img_array = np.frombuffer(resp.content, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return None, "Failed to decode image from URL"
            
            return image, None
        except requests.RequestException as e:
            return None, f"Network error: {str(e)}"
        except Exception as e:
            return None, f"Error loading from URL: {str(e)}"
    
    # File path
    try:
        path = Path(src_str)
        image = cv2.imread(str(path))
        
        if image is None:
            return None, f"Failed to read image file: {path}"
        
        return image, None
    except Exception as e:
        return None, f"Error loading image: {str(e)}"


def validate_quad_points(quad: np.ndarray) -> Tuple[bool, Optional[str]]:
    """
    Validate quad points array.
    
    Args:
        quad: Array of 4 points
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(quad, np.ndarray):
        return False, "Quad must be numpy array"
    
    if quad.shape != (4, 2):
        return False, f"Quad must have shape (4, 2), got {quad.shape}"
    
    # Check for NaN or Inf
    if np.any(np.isnan(quad)) or np.any(np.isinf(quad)):
        return False, "Quad contains NaN or Inf values"
    
    # Check if all points are different
    for i in range(4):
        for j in range(i + 1, 4):
            if np.allclose(quad[i], quad[j], atol=1e-3):
                return False, f"Points {i} and {j} are too close (duplicate)"
    
    return True, None


def validate_image_dimensions(
    image: np.ndarray,
    min_width: int = 50,
    min_height: int = 50,
    max_width: int = 10000,
    max_height: int = 10000
) -> Tuple[bool, Optional[str]]:
    """
    Validate image dimensions.
    
    Args:
        image: Input image
        min_width, min_height: Minimum dimensions
        max_width, max_height: Maximum dimensions
    
    Returns:
        (is_valid, error_message)
    """
    h, w = image.shape[:2]
    
    if w < min_width or h < min_height:
        return False, f"Image too small: {w}x{h} (minimum: {min_width}x{min_height})"
    
    if w > max_width or h > max_height:
        return False, f"Image too large: {w}x{h} (maximum: {max_width}x{max_height})"
    
    return True, None


def validate_enhancement_mode(mode: str) -> Tuple[bool, Optional[str]]:
    """
    Validate enhancement mode.
    
    Args:
        mode: Enhancement mode string
    
    Returns:
        (is_valid, error_message)
    """
    valid_modes = ['auto', 'text', 'photo', 'bw', 'color', 'grayscale']
    
    if mode not in valid_modes:
        return False, f"Invalid mode '{mode}'. Valid modes: {', '.join(valid_modes)}"
    
    return True, None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename untuk save.
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    # Ensure not empty
    if not filename:
        filename = "scanned_document"
    
    # Add extension if missing
    if not any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp']):
        filename += '.png'
    
    return filename


def check_image_quality(image: np.ndarray) -> dict:
    """
    Check basic image quality metrics.
    
    Args:
        image: Input image
    
    Returns:
        Dictionary with quality metrics
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Brightness
    brightness = np.mean(gray)
    
    # Contrast (standard deviation)
    contrast = np.std(gray)
    
    # Sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    
    # Darkness/lightness warnings
    warnings = []
    if brightness < 50:
        warnings.append("Image is very dark")
    elif brightness > 200:
        warnings.append("Image is very bright")
    
    if contrast < 30:
        warnings.append("Low contrast")
    
    if sharpness < 100:
        warnings.append("Image may be blurry")
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'sharpness': sharpness,
        'warnings': warnings,
        'quality_score': min(100, (brightness / 128) * (contrast / 50) * (sharpness / 500) * 100)
    }


def is_url(s: str) -> bool:
    """Check if string is URL"""
    s = s.strip()
    return s.startswith('http://') or s.startswith('https://')


def get_supported_formats() -> list:
    """Get list of supported image formats"""
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


def assert_valid_image(image: np.ndarray, name: str = "image"):
    """
    Assert image is valid, raise ValidationError if not.
    
    Args:
        image: Image to validate
        name: Name for error message
    
    Raises:
        ValidationError: If image is invalid
    """
    if image is None:
        raise ValidationError(f"{name} is None")
    
    if not isinstance(image, np.ndarray):
        raise ValidationError(f"{name} must be numpy array, got {type(image)}")
    
    if image.size == 0:
        raise ValidationError(f"{name} is empty")
    
    is_valid, error = validate_image_dimensions(image)
    if not is_valid:
        raise ValidationError(f"{name}: {error}")


def assert_valid_quad(quad: np.ndarray, name: str = "quad"):
    """
    Assert quad is valid, raise ValidationError if not.
    
    Args:
        quad: Quad to validate
        name: Name for error message
    
    Raises:
        ValidationError: If quad is invalid
    """
    is_valid, error = validate_quad_points(quad)
    if not is_valid:
        raise ValidationError(f"{name}: {error}")