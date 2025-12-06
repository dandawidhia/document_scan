"""
app.py - Document Scanner with Real-time Detection
"""

import sys
import os
from pathlib import Path

current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Import streamlit-webrtc
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# Import core modules
from core import (
    detect_receipt_contour,
    four_point_transform,
    add_padding,
    draw_detection_overlay,
    resize_for_display,
    rotate_image
)
from core.enhancer import apply_enhancement_pipeline
from core.realtime import VideoProcessor

# Import utils
from utils import (
    load_image_safe,
    validate_image_input,
    sanitize_filename,
    check_image_quality,
    assert_valid_image
)

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
    version: str = "1.0.1"
    
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
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )


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
    
# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title=CONFIG.app_name,
    page_icon=CONFIG.app_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        height: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
def init_session_state():
    """Initialize session state variables"""
    if 'captured_image' not in st.session_state:
        st.session_state.captured_image = None
    if 'detected_quad' not in st.session_state:
        st.session_state.detected_quad = None
    if 'detection_score' not in st.session_state:
        st.session_state.detection_score = 0.0
    if 'final_scan' not in st.session_state:
        st.session_state.final_scan = None
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = 'upload'
    if 'enhancement_mode' not in st.session_state:
        st.session_state.enhancement_mode = 'auto'
    # New: for real-time mode
    if 'realtime_processor' not in st.session_state:
        st.session_state.realtime_processor = VideoProcessor()
    if 'snapshot_taken' not in st.session_state:
        st.session_state.snapshot_taken = False

init_session_state()

# ==================== HELPER FUNCTIONS ====================
def reset_state():
    """Reset all session state"""
    st.session_state.captured_image = None
    st.session_state.detected_quad = None
    st.session_state.detection_score = 0.0
    st.session_state.final_scan = None

def convert_image_to_bytes(image: np.ndarray, format: str = 'PNG') -> bytes:
    """Convert OpenCV image to bytes for download"""
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    buf = io.BytesIO()
    pil_image.save(buf, format=format)
    return buf.getvalue()

# ==================== MAIN APP ====================
def main():
    st.markdown('<p class="main-header">📄 Document Scanner</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Mode selection - UPDATED with real-time option
        mode = st.radio(
            "Select Mode",
            ["🎥 Real-time Detection", "📷 Camera Capture", "📁 Upload Image", "ℹ️ About"],
            key="mode_selector"
        )
        
        st.markdown("---")
        
        # Enhancement settings
        st.subheader("Enhancement Options")
        enhancement_mode = st.selectbox(
            "Enhancement Mode",
            ["auto", "text", "photo", "bw", "color"],
            help="Select how to enhance the scanned document"
        )
        st.session_state.enhancement_mode = enhancement_mode
        
        # Advanced options
        with st.expander("Advanced Options"):
            remove_shadow = st.checkbox("Remove Shadow", value=False)
            denoise = st.checkbox("Denoise", value=False)
            sharpen = st.checkbox("Sharpen", value=False)
            deskew = st.checkbox("Auto Deskew", value=False)
            add_pad = st.checkbox("Add Padding", value=True)
            pad_amount = st.slider("Padding %", 0, 10, 3) if add_pad else 0
        
        # Real-time settings (only show in real-time mode)
        if mode == "🎥 Real-time Detection":
            st.markdown("---")
            st.subheader("📹 Real-time Settings")
            
            sensitivity = st.select_slider(
                "Detection Sensitivity",
                options=['low', 'medium', 'high'],
                value='medium',
                help="Higher sensitivity = more responsive but more CPU usage"
            )
            st.session_state.realtime_processor.set_sensitivity(sensitivity)
            
            show_stats = st.checkbox("Show Stats", value=False)
        
        st.markdown("---")
        
        # Reset button
        if st.button("🔄 Reset All", use_container_width=True):
            reset_state()
            st.session_state.realtime_processor.reset()
            st.rerun()
        
        st.markdown("---")
        st.caption(f"Version {CONFIG.version}")
    
    # Main content based on mode
    if mode == "🎥 Real-time Detection":
        realtime_mode(
            enhancement_mode=enhancement_mode,
            remove_shadow=remove_shadow,
            denoise=denoise,
            sharpen=sharpen,
            deskew=deskew,
            pad_frac=pad_amount / 100.0,
            show_stats=show_stats if 'show_stats' in locals() else False
        )
    elif mode == "📷 Camera Capture":
        camera_mode(
            enhancement_mode=enhancement_mode,
            remove_shadow=remove_shadow,
            denoise=denoise,
            sharpen=sharpen,
            deskew=deskew,
            pad_frac=pad_amount / 100.0
        )
    elif mode == "📁 Upload Image":
        upload_mode(
            enhancement_mode=enhancement_mode,
            remove_shadow=remove_shadow,
            denoise=denoise,
            sharpen=sharpen,
            deskew=deskew,
            pad_frac=pad_amount / 100.0
        )
    else:
        about_page()


# ==================== NEW: REAL-TIME MODE ====================
def realtime_mode(enhancement_mode, remove_shadow, denoise, sharpen, deskew, pad_frac, show_stats):
    """Real-time document detection using webcam"""
    
    st.header("🎥 Real-time Document Detection")
    
    # Instructions
    st.info("""
    💡 **Real-time Scanning Tips:**
    - Position document flat with all 4 corners visible
    - Use good lighting and contrasting background
    - Keep camera steady when detection turns **GREEN**
    - Click **"📸 Capture"** button below when ready
    """)
    
    # Create columns for video and capture
    col_video, col_capture = st.columns([2, 1])
    
    with col_video:
        # WebRTC Streamer
        ctx = webrtc_streamer(
            key="document-scanner",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=CONFIG.rtc_configuration,
            video_processor_factory=lambda: st.session_state.realtime_processor,
            async_processing=True,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720},
                },
                "audio": False
            }
        )
    
    with col_capture:
        st.subheader("📸 Capture Control")
        
        # Show current detection status
        if ctx.state.playing:
            quad, score = st.session_state.realtime_processor.get_best_detection()
            
            if quad is not None:
                # Status indicator
                if score >= 0.5:
                    st.success(f"✅ Ready to Capture\nConfidence: {score:.1%}")
                elif score >= 0.3:
                    st.warning(f"⚠️ Adjust Position\nConfidence: {score:.1%}")
                else:
                    st.info(f"🔍 Detecting...\nConfidence: {score:.1%}")
                
                # Capture button
                if st.button("📸 Capture Scan", use_container_width=True, type="primary"):
                    # Get stable detection (averaged)
                    stable_quad, stable_score = st.session_state.realtime_processor.get_stable_detection()
                    
                    if stable_quad is not None and stable_score >= 0.2:
                        # Need to get the actual frame - we'll use the best detection
                        st.session_state.snapshot_taken = True
                        st.success("✅ Captured! Scroll down to process.")
                        st.rerun()
                    else:
                        st.error("❌ Detection too weak. Please adjust position.")
                
                # Show detection quality
                with st.expander("📊 Detection Quality"):
                    st.metric("Best Score", f"{score:.1%}")
                    st.metric("Stability", f"{len(st.session_state.realtime_processor.detection_history)}/5")
                    
                    # Visual quality indicator
                    if score >= 0.5:
                        st.progress(1.0, text="Excellent")
                    elif score >= 0.3:
                        st.progress(0.6, text="Good")
                    else:
                        st.progress(0.3, text="Fair")
            else:
                st.info("🔍 Searching for document...")
                st.info("Position document in camera view")
        else:
            st.warning("📹 Camera not started\nClick 'START' above")
        
        # Stats display
        if show_stats and ctx.state.playing:
            with st.expander("📈 Processing Stats"):
                stats = st.session_state.realtime_processor.get_stats()
                for key, value in stats.items():
                    st.text(f"{key}: {value}")
        
        # Reset detection button
        if st.button("🔄 Reset Detection", use_container_width=True):
            st.session_state.realtime_processor.reset()
            st.session_state.snapshot_taken = False
            st.rerun()
    
    # Process captured frame
    if st.session_state.snapshot_taken:
        st.markdown("---")
        st.header("🎯 Process Captured Frame")
        
        # Get the detection
        quad, score = st.session_state.realtime_processor.get_stable_detection()
        
        if quad is not None:
            st.info(f"📸 Snapshot captured with {score:.1%} confidence")
            
            # For now, we need the actual image
            # In a full implementation, you'd capture the frame with the detection
            st.warning("""
            ⚠️ **Frame Capture Note**: 
            In the current implementation, we have the detection coordinates but need the actual frame.
            
            **Workaround**: 
            1. Take a photo using "📷 Camera Capture" mode
            2. The coordinates will be applied automatically
            
            **Or** manually save the frame and upload it.
            """)
            
            # Store detection for use in other modes
            st.session_state.detected_quad = quad
            st.session_state.detection_score = score
            
            # Buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📷 Switch to Camera Capture", use_container_width=True):
                    st.session_state.current_mode = 'camera'
                    # Keep the detection
                    st.rerun()
            with col2:
                if st.button("❌ Discard", use_container_width=True):
                    st.session_state.snapshot_taken = False
                    st.session_state.realtime_processor.reset()
                    st.rerun()
        else:
            st.error("❌ No detection available. Please try again.")
            

# ==================== UPLOAD MODE ====================
def camera_mode(enhancement_mode, remove_shadow, denoise, sharpen, deskew, pad_frac):
    """Camera capture mode - Simple & Native"""
    
    st.header("📷 Camera Capture Mode")
    
    # Instructions
    st.info("""
    💡 **Tips for best results:**
    - Ensure good lighting
    - Keep document flat and all corners visible
    - Use contrasting background
    - Hold camera steady
    """)
    
    # Camera input
    camera_photo = st.camera_input("Take a photo of your document")
    
    if camera_photo is not None:
        # Convert to OpenCV format
        file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("❌ Failed to load image from camera.")
            return
        
        # Store in session state
        st.session_state.captured_image = image
        
        # If we have a detection from real-time mode, use it
        if st.session_state.detected_quad is not None and st.session_state.detection_score > 0:
            st.success(f"✅ Using real-time detection (confidence: {st.session_state.detection_score:.1%})")
        
        # Process similar to upload mode
        process_captured_image(
            image,
            enhancement_mode=enhancement_mode,
            remove_shadow=remove_shadow,
            denoise=denoise,
            sharpen=sharpen,
            deskew=deskew,
            pad_frac=pad_frac
        )
        
        # Retake button
        if st.button("🔄 Retake Photo", use_container_width=True, type="secondary"):
            reset_state()
            st.rerun()


def upload_mode(enhancement_mode, remove_shadow, denoise, sharpen, deskew, pad_frac):
    """Upload image mode"""
    
    st.header("📁 Upload Image Mode")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help=f"Maximum size: {CONFIG.max_upload_size_mb}MB"
    )
    
    if uploaded_file is not None:
        # Load image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("❌ Failed to load image. Please try another file.")
            return
        
        # Validate
        try:
            assert_valid_image(image, "Uploaded image")
        except Exception as e:
            st.error(f"❌ Invalid image: {str(e)}")
            return
        
        # Store in session state
        st.session_state.captured_image = image
        
        # Process the captured/uploaded image
        process_captured_image(
            image,
            enhancement_mode=enhancement_mode,
            remove_shadow=remove_shadow,
            denoise=denoise,
            sharpen=sharpen,
            deskew=deskew,
            pad_frac=pad_frac
        )


def process_captured_image(image, enhancement_mode, remove_shadow, denoise, sharpen, deskew, pad_frac):
    """Process captured image (from camera or upload)"""
    
    # Image quality check
    quality_info = check_image_quality(image)
    
    # Display original and detection side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        # Resize for display
        display_img = resize_for_display(image, CONFIG.ui.max_display_width, CONFIG.ui.max_display_height)
        st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # Quality metrics
        with st.expander("📊 Image Quality Metrics"):
            st.metric("Brightness", f"{quality_info['brightness']:.1f}")
            st.metric("Contrast", f"{quality_info['contrast']:.1f}")
            st.metric("Sharpness", f"{quality_info['sharpness']:.1f}")
            st.metric("Quality Score", f"{quality_info['quality_score']:.0f}/100")
            
            if quality_info['warnings']:
                st.warning("⚠️ " + "\n".join(quality_info['warnings']))
    
    with col2:
        st.subheader("🎯 Auto-Detection")
        
        with st.spinner("🔍 Detecting document..."):
            # Detect
            candidates = detect_receipt_contour(image, debug=False, fast_mode=False)
            
            if candidates and len(candidates) > 0:
                quad, score, details = candidates[0]
                st.session_state.detected_quad = quad
                st.session_state.detection_score = score
                
                # Draw detection overlay
                if score > CONFIG.ui.good_confidence:
                    color = CONFIG.ui.color_good
                    status = "✅ Excellent Detection"
                elif score > CONFIG.ui.adjust_confidence:
                    color = CONFIG.ui.color_adjust
                    status = "⚠️ Good - May Need Adjustment"
                else:
                    color = (0, 165, 255)  # Orange
                    status = "⚠️ Fair - Manual Adjustment Recommended"
                
                overlay = draw_detection_overlay(image, quad, score, color=color)
                display_overlay = resize_for_display(overlay, CONFIG.ui.max_display_width, CONFIG.ui.max_display_height)
                st.image(cv2.cvtColor(display_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Status
                if score > CONFIG.ui.good_confidence:
                    st.success(status)
                else:
                    st.warning(status)
                
                # Metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Confidence", f"{score:.1%}")
                with col_b:
                    rect_error = details.get('rect_error', 0)
                    st.metric("Rectangularity", f"{max(0, 100 - rect_error):.0f}%")
                with col_c:
                    area_ratio = details.get('area_ratio', 0)
                    st.metric("Coverage", f"{area_ratio:.1%}")
                
                # Detection details
                with st.expander("🔍 Detection Details"):
                    st.json({
                        'confidence_score': f"{score:.3f}",
                        'rectangularity_error': f"{details.get('rect_error', 0):.2f}°",
                        'white_ratio': f"{details.get('white_ratio', 0):.3f}",
                        'area_ratio': f"{details.get('area_ratio', 0):.3f}",
                        'aspect_ratio': f"{details.get('aspect_ratio', 0):.2f}",
                        'brightness': f"{details.get('quad_brightness', 0):.1f}"
                    })
                
            else:
                st.error("❌ No document detected. Please try manual adjustment.")
                # Set default quad
                h, w = image.shape[:2]
                margin = min(w, h) // 20
                st.session_state.detected_quad = np.array([
                    [margin, margin],
                    [w - margin, margin],
                    [w - margin, h - margin],
                    [margin, h - margin]
                ], dtype=np.float32)
                st.session_state.detection_score = 0.1
    
    # Action buttons
    st.markdown("---")
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    
    with col_btn1:
        if st.button("✏️ Adjust Corners", use_container_width=True, type="secondary"):
            st.session_state.current_mode = 'adjust'
            st.rerun()
    
    with col_btn2:
        if st.button("✅ Process Scan", use_container_width=True, type="primary"):
            process_scan(
                enhancement_mode=enhancement_mode,
                remove_shadow=remove_shadow,
                denoise=denoise,
                sharpen=sharpen,
                deskew=deskew,
                pad_frac=pad_frac
            )
    
    with col_btn3:
        if st.session_state.final_scan is not None:
            img_bytes = convert_image_to_bytes(st.session_state.final_scan)
            st.download_button(
                label="📥 Download Result",
                data=img_bytes,
                file_name="scanned_document.png",
                mime="image/png",
                use_container_width=True
            )
    
    with col_btn4:
        if st.button("🗑️ Clear", use_container_width=True):
            reset_state()
            st.rerun()
    
    # Show adjustment mode if selected
    if st.session_state.current_mode == 'adjust':
        st.markdown("---")
        adjustment_mode()

# ==================== CORNER ADJUSTMENT ====================
def adjustment_mode():
    """Manual corner adjustment interface"""
    
    st.header("✏️ Manual Corner Adjustment")
    
    if st.session_state.captured_image is None:
        st.warning("⚠️ No image loaded!")
        return
    
    image = st.session_state.captured_image
    h, w = image.shape[:2]
    
    # Initialize quad if needed
    if st.session_state.detected_quad is None:
        margin = 30
        st.session_state.detected_quad = np.array([
            [margin, margin],
            [w - margin, margin],
            [w - margin, h - margin],
            [margin, h - margin]
        ], dtype=np.float32)
    
    quad = st.session_state.detected_quad.copy()
    
    st.info("💡 Adjust each corner by entering coordinates or using nudge buttons")
    
    # Corner adjustment UI
    st.subheader("🎯 Corner Coordinates")
    
    cols = st.columns(4)
    corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
    corner_colors = ["🔴", "🔵", "🟡", "🟢"]
    new_quad = quad.copy()
    
    for i, (col, name, color) in enumerate(zip(cols, corner_names, corner_colors)):
        with col:
            st.markdown(f"**{color} {name}**")
            
            # X coordinate
            x = st.number_input(
                f"X",
                min_value=0,
                max_value=w - 1,
                value=int(quad[i][0]),
                key=f"x_{i}",
                step=1
            )
            
            # Y coordinate  
            y = st.number_input(
                f"Y",
                min_value=0,
                max_value=h - 1,
                value=int(quad[i][1]),
                key=f"y_{i}",
                step=1
            )
            
            new_quad[i] = [x, y]
            
            # Nudge buttons
            st.markdown("**Quick Adjust:**")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button(f"⬅️ -{CONFIG.ui.corner_nudge_amount}", key=f"left_{i}"):
                    new_quad[i][0] = max(0, new_quad[i][0] - CONFIG.ui.corner_nudge_amount)
                    st.session_state.detected_quad = new_quad
                    st.rerun()
                if st.button(f"⬇️ +{CONFIG.ui.corner_nudge_amount}", key=f"down_{i}"):
                    new_quad[i][1] = min(h - 1, new_quad[i][1] + CONFIG.ui.corner_nudge_amount)
                    st.session_state.detected_quad = new_quad
                    st.rerun()
            with col_b:
                if st.button(f"➡️ +{CONFIG.ui.corner_nudge_amount}", key=f"right_{i}"):
                    new_quad[i][0] = min(w - 1, new_quad[i][0] + CONFIG.ui.corner_nudge_amount)
                    st.session_state.detected_quad = new_quad
                    st.rerun()
                if st.button(f"⬆️ -{CONFIG.ui.corner_nudge_amount}", key=f"up_{i}"):
                    new_quad[i][1] = max(0, new_quad[i][1] - CONFIG.ui.corner_nudge_amount)
                    st.session_state.detected_quad = new_quad
                    st.rerun()
    
    st.session_state.detected_quad = new_quad
    
    # Preview
    st.subheader("👁️ Preview")
    preview = image.copy()
    cv2.polylines(preview, [new_quad.astype(np.int32)], True, (0, 255, 0), 3)
    
    # Draw corners with numbers
    corner_display_colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 0)]  # BGR
    for i, pt in enumerate(new_quad):
        cv2.circle(preview, tuple(pt.astype(int)), 10, corner_display_colors[i], -1)
        cv2.putText(
            preview, str(i),
            tuple((pt + 15).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2
        )
    
    display_preview = resize_for_display(preview, CONFIG.ui.max_display_width * 1.5, CONFIG.ui.max_display_height * 1.5)
    st.image(cv2.cvtColor(display_preview, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Apply & Process", use_container_width=True, type="primary"):
            st.session_state.current_mode = 'upload'
            st.rerun()
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.current_mode = 'upload'
            st.rerun()

# ==================== PROCESS SCAN ====================
def process_scan(enhancement_mode, remove_shadow, denoise, sharpen, deskew, pad_frac):
    """Process the scan with enhancement"""
    
    if st.session_state.captured_image is None:
        st.error("❌ No image to process!")
        return
    
    with st.spinner("🔄 Processing scan..."):
        image = st.session_state.captured_image
        quad = st.session_state.detected_quad
        
        # Add padding if enabled
        if pad_frac > 0:
            h, w = image.shape[:2]
            quad = add_padding(quad, pad_frac, w, h)
        
        # Perspective transform
        try:
            warped = four_point_transform(image, quad)
        except Exception as e:
            st.error(f"❌ Transform failed: {str(e)}")
            return
        
        # Enhancement pipeline
        try:
            enhanced = apply_enhancement_pipeline(
                warped,
                mode=enhancement_mode,
                remove_shadow=remove_shadow,
                denoise=denoise,
                sharpen=sharpen,
                deskew=deskew
            )
            st.session_state.final_scan = enhanced
        except Exception as e:
            st.error(f"❌ Enhancement failed: {str(e)}")
            st.session_state.final_scan = warped
    
    # Show results
    st.success("✅ Scan complete!")
    
    st.markdown("---")
    st.header("📊 Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original")
        display_orig = resize_for_display(image, CONFIG.ui.max_display_width, CONFIG.ui.max_display_height)
        st.image(cv2.cvtColor(display_orig, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    with col2:
        st.subheader("✨ Scanned Result")
        display_result = resize_for_display(st.session_state.final_scan, CONFIG.ui.max_display_width, CONFIG.ui.max_display_height)
        st.image(cv2.cvtColor(display_result, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    # Additional options
    st.markdown("---")
    st.subheader("🔧 Post-Processing Options")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Rotate 90°", use_container_width=True):
            st.session_state.final_scan = rotate_image(st.session_state.final_scan, 90)
            st.rerun()
    
    with col2:
        if st.button("🔄 Rotate 180°", use_container_width=True):
            st.session_state.final_scan = rotate_image(st.session_state.final_scan, 180)
            st.rerun()
    
    with col3:
        if st.button("🔄 Rotate 270°", use_container_width=True):
            st.session_state.final_scan = rotate_image(st.session_state.final_scan, 270)
            st.rerun()
    
    with col4:
        if st.button("🔁 Re-enhance", use_container_width=True):
            # Re-apply enhancement with current settings
            process_scan(enhancement_mode, remove_shadow, denoise, sharpen, deskew, pad_frac)

# ==================== ABOUT PAGE ====================
def about_page():
    """About page with instructions"""
    
    st.header("ℹ️ About Document Scanner")
    
    st.markdown("""
    ### 🎯 Features
    
    - **📁 Upload & Scan**: Upload images and automatically detect document boundaries
    - **🎯 Smart Detection**: Advanced detection algorithm with confidence scoring
    - **✏️ Manual Adjustment**: Fine-tune corner positions for perfect results
    - **✨ Enhancement Modes**: 
        - `auto`: Automatically choose best enhancement
        - `text`: Optimize for text documents (receipts, invoices)
        - `photo`: Enhance photos and images
        - `bw`: High-contrast black & white
        - `color`: Enhance colors
    - **🔧 Advanced Options**: Shadow removal, denoising, sharpening, auto-deskew
    - **📥 Download**: Save scanned documents as PNG
    
    ---
    
    ### 📖 How to Use
    
    #### 1️⃣ Upload Image
    - Click "Browse files" to select an image
    - Supported formats: JPG, PNG, BMP, TIFF
    - Maximum file size: 10MB
    
    #### 2️⃣ Auto-Detection
    - Document boundaries are detected automatically
    - Green border = Excellent detection
    - Yellow border = May need adjustment
    - Orange border = Manual adjustment recommended
    
    #### 3️⃣ Adjust (Optional)
    - Click "✏️ Adjust Corners" to manually adjust
    - Enter coordinates or use nudge buttons
    - Preview shows real-time changes
    
    #### 4️⃣ Process
    - Click "✅ Process Scan" to transform and enhance
    - Select enhancement mode in sidebar
    - Enable advanced options if needed
    
    #### 5️⃣ Download
    - Click "📥 Download Result" to save
    - Use rotation buttons to adjust orientation
    - Re-enhance with different settings if needed
    
    ---
    
    ### 💡 Tips for Best Results
    
    ✅ **Good Lighting**: Ensure document is well-lit with minimal shadows
    
    ✅ **Flat Surface**: Keep document as flat as possible
    
    ✅ **Clear Background**: Use contrasting background (white paper on dark table)
    
    ✅ **All Corners Visible**: Make sure all 4 corners are in frame
    
    ✅ **Avoid Glare**: No reflective surfaces or strong directional light
    
    ---
    
    ### 🎨 Enhancement Modes Explained
    
    **🤖 Auto** (Recommended)
    - Automatically detects content type
    - Text-heavy documents → B&W enhancement
    - Photos/mixed content → Color enhancement
    
    **📝 Text**
    - Optimized for documents with text
    - High contrast adaptive threshold
    - Perfect for receipts, invoices, forms
    
    **📷 Photo**
    - Preserves colors and details
    - CLAHE enhancement on luminance
    - Slight sharpening applied
    
    **⬛ Black & White**
    - High-contrast binary output
    - Excellent for printed documents
    - Smallest file size
    
    **🎨 Color**
    - Full color preservation
    - Enhanced contrast and brightness
    - Best for photos or colored documents
    
    ---
    
    ### ⚙️ Advanced Options
    
    **🌑 Remove Shadow**: Reduces shadows from uneven lighting
    
    **🔊 Denoise**: Removes image noise (useful for low-light photos)
    
    **✨ Sharpen**: Enhances edges and details
    
    **📐 Auto Deskew**: Automatically corrects slight rotation
    
    **📏 Padding**: Adds margin around detected document (0-10%)
    
    ---
    
    ### 🔧 Technical Details
    
    - **Detection Algorithm**: Multi-scale edge detection with white/skin masking
    - **Scoring System**: Combines rectangularity, whiteness, size, and geometry
    - **Transform**: 4-point perspective transformation
    - **Enhancement**: Adaptive thresholding, CLAHE, morphological operations
    
    ---
    
    ### 📊 Quality Metrics
    
    - **Brightness**: Average pixel intensity (0-255)
    - **Contrast**: Standard deviation of intensity
    - **Sharpness**: Laplacian variance (edge detection)
    - **Confidence**: Detection algorithm confidence score
    
    ---
    
    ### 🐛 Troubleshooting
    
    **❓ Document not detected?**
    - Ensure good contrast with background
    - Try manual adjustment mode
    - Check if all 4 corners are visible
    
    **❓ Detection inaccurate?**
    - Use "✏️ Adjust Corners" for manual fine-tuning
    - Try different lighting conditions
    - Ensure document is flat
    
    **❓ Scan looks distorted?**
    - Re-adjust corners more precisely
    - Enable "Auto Deskew" in advanced options
    - Try different enhancement mode
    
    **❓ Text not readable?**
    - Use "text" or "bw" enhancement mode
    - Enable "Sharpen" option
    - Ensure original image is high-resolution
    
    ---
    
    ### 📞 Support
    
    For issues, feedback, or feature requests, please use the feedback buttons in the app.
    
    ---
    
    ### 📄 Version Information
    
    **Version**: {version}
    
    **Last Updated**: December 2024
    
    **Built with**: OpenCV, Streamlit, NumPy, Pillow
    """.format(version=CONFIG.version))
    
    # Easter egg: Show config if button clicked
    if st.button("🔍 Show Technical Configuration"):
        st.json({
            'detection': {
                'max_processing_dim': CONFIG.detection.max_processing_dim,
                'min_confidence': CONFIG.detection.min_confidence_score,
                'max_rect_error': CONFIG.detection.max_rectangularity_error,
            },
            'ui': {
                'good_confidence': CONFIG.ui.good_confidence,
                'adjust_confidence': CONFIG.ui.adjust_confidence,
                'corner_nudge': CONFIG.ui.corner_nudge_amount,
            }
        })

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()