"""
utils/__init__.py
Main exports dari utils module
"""

from .geometry import (
    calculate_angle,
    calculate_rectangularity_error,
    is_valid_receipt_quad,
    score_quad_geometry,
    calculate_aspect_ratio,
    calculate_quad_area,
    get_quad_dimensions,
    point_to_line_distance,
    snap_to_grid,
    expand_quad,
    contract_quad
)

from .validation import (
    validate_image_input,
    load_image_safe,
    validate_quad_points,
    validate_image_dimensions,
    validate_enhancement_mode,
    sanitize_filename,
    check_image_quality,
    is_url,
    get_supported_formats,
    ValidationError,
    assert_valid_image,
    assert_valid_quad
)

__all__ = [
    # Geometry
    'calculate_angle',
    'calculate_rectangularity_error',
    'is_valid_receipt_quad',
    'score_quad_geometry',
    'calculate_aspect_ratio',
    'calculate_quad_area',
    'get_quad_dimensions',
    'point_to_line_distance',
    'snap_to_grid',
    'expand_quad',
    'contract_quad',
    
    # Validation
    'validate_image_input',
    'load_image_safe',
    'validate_quad_points',
    'validate_image_dimensions',
    'validate_enhancement_mode',
    'sanitize_filename',
    'check_image_quality',
    'is_url',
    'get_supported_formats',
    'ValidationError',
    'assert_valid_image',
    'assert_valid_quad',
]