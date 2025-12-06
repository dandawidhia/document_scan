"""
config.py
Configuration settings untuk document scanner app
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DetectionConfig:
    """Configuration untuk detection"""
    # Processing
    max_processing_dim: int = 800
    fast_mode_dim: int = 640
    
    # Area thresholds (fraction of image area)
    min_area_ratio: float = 0.03
    max_area_ratio: float = 0.9
    
    # Dimension thresholds
    min_dimension_ratio: float = 0.10  # 10% of image
    max_aspect_ratio: float = 8.0
    min_aspect_ratio: float = 1.02
    
    # Scoring weights
    white_score_weight: float = 0.35
    size_score_weight: float = 0.25
    rect_score_weight: float = 0.20
    skin_penalty_weight: float = 0.08
    aspect_score_weight: float = 0.06
    dimension_score_weight: float = 0.04
    brightness_score_weight: float = 0.02
    
    # Thresholds
    min_confidence_score: float = 0.2
    max_rectangularity_error: float = 45.0


@dataclass
class PreprocessingConfig:
    """Configuration untuk preprocessing"""
    # CLAHE parameters
    clahe_clip_limit_low: float = 8.0
    clahe_clip_limit_normal: float = 3.0
    clahe_tile_size_low: Tuple[int, int] = (3, 3)
    clahe_tile_size_normal: Tuple[int, int] = (8, 8)
    
    # Brightness thresholds
    very_dark_threshold: float = 60.0
    dark_threshold: float = 90.0
    normal_threshold: float = 140.0
    bright_threshold: float = 200.0
    
    # Edge detection
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    
    # Morphology
    morph_kernel_size: int = 4


@dataclass
class EnhancementConfig:
    """Configuration untuk post-processing enhancement"""
    # Text enhancement
    adaptive_threshold_block_size: int = 15
    adaptive_threshold_c: int = 8
    
    # Photo enhancement
    clahe_clip_limit: float = 2.5
    clahe_tile_size: Tuple[int, int] = (8, 8)
    
    # Sharpening
    default_sharpen_strength: float = 1.0
    
    # Denoising
    default_denoise_strength: int = 10


@dataclass
class UIConfig:
    """Configuration untuk Streamlit UI"""
    # Display
    max_display_width: int = 800
    max_display_height: int = 600
    
    # Camera
    camera_resolution: Tuple[int, int] = (640, 480)
    frame_skip: int = 3  # Process every Nth frame
    
    # Colors (BGR)
    color_good: Tuple[int, int, int] = (0, 255, 0)  # Green
    color_adjust: Tuple[int, int, int] = (0, 255, 255)  # Yellow
    color_poor: Tuple[int, int, int] = (0, 0, 255)  # Red
    
    # Thresholds
    good_confidence: float = 0.5
    adjust_confidence: float = 0.3
    
    # Corner adjustment
    corner_nudge_amount: int = 10  # pixels


@dataclass
class AppConfig:
    """Main application configuration"""
    # App info
    app_name: str = "Document Scanner"
    app_icon: str = "📄"
    version: str = "1.0.0"
    
    # Paths
    temp_dir: str = "./temp"
    output_dir: str = "./scanned"
    
    # Image limits
    max_upload_size_mb: int = 10
    min_image_width: int = 50
    min_image_height: int = 50
    max_image_width: int = 10000
    max_image_height: int = 10000
    
    # Supported formats
    supported_formats: list = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'])
    
    # Sub-configs
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    enhancement: EnhancementConfig = field(default_factory=EnhancementConfig)
    ui: UIConfig = field(default_factory=UIConfig)


# Global config instance
CONFIG = AppConfig()


# Helper functions
def get_config() -> AppConfig:
    """Get global config instance"""
    return CONFIG


def update_config(**kwargs):
    """Update config values"""
    for key, value in kwargs.items():
        if hasattr(CONFIG, key):
            setattr(CONFIG, key, value)


def reset_config():
    """Reset config to defaults"""
    global CONFIG
    CONFIG = AppConfig()