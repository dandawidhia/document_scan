"""
core/__init__.py
Main exports dari core module
"""

from .detector import detect_receipt_contour
from .preprocessor import (
    preprocess_image,
    enhance_lighting,
    detect_white_regions,
    detect_skin
)
from .transformer import (
    order_points,
    four_point_transform,
    add_padding,
    enhance_scanned_image,
    rotate_image,
    resize_for_display,
    draw_detection_overlay
)
from .enhancer import (
    DocumentEnhancer,
    apply_enhancement_pipeline
)

from .realtime import VideoProcessor

__all__ = [
    # Detector
    'detect_receipt_contour',
    
    # Preprocessor
    'preprocess_image',
    'enhance_lighting',
    'detect_white_regions',
    'detect_skin',
    
    # Transformer
    'order_points',
    'four_point_transform',
    'add_padding',
    'enhance_scanned_image',
    'rotate_image',
    'resize_for_display',
    'draw_detection_overlay',
    
    # Enhancer
    'DocumentEnhancer',
    'apply_enhancement_pipeline',
    
    # Realtime
    'VideoProcessor',
]