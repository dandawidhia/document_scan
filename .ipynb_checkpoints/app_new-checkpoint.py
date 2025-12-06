"""
app.py - Document Scanner with Real-time Detection
Complete implementation
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
from dataclasses import dataclass, field
from typing import Tuple

# Import streamlit-webrtc
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

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
    check_image_quality,
    assert_valid_image
)

# ==================== CONFIGURATION ====================
@dataclass
class DetectionConfig:
    max_processing_dim: int = 800
    min_confidence_score: float = 0.2
    max_rectangularity_error: float = 45.0

@dataclass
class UIConfig:
    max_display_width: int = 800
    max_display_height: int = 600
    color_good: Tuple[int, int, int] = (0, 255, 0)
    color_adjust: Tuple[int, int, int] = (0, 255, 255)
    good_confidence: float = 0.5
    adjust_confidence: float = 0.3
    corner_nudge_amount: int = 10

@dataclass
class AppConfig:
    app_name: str = "Document Scanner"
    app_icon: str = "📄"
    version: str = "1.0.1"
    max_upload_size_mb: int = 10
    supported_formats: list = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'])
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

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
    defaults = {
        'captured_image': None,
        'detected_quad': None,
        'detection_score': 0.0,
        'final_scan': None,
        'current_mode': 'upload',
        'enhancement_mode': 'auto',
        'realtime_processor': None,
        'snapshot_taken': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==================== HELPER FUNCTIONS ====================
def reset_state():
    st.session_state.captured_image = None
    st.session_state.detected_quad = None
    st.session_state.detection_score = 0.0
    st.session_state.final_scan = None
    st.session_state.snapshot_taken = False

def convert_image_to_bytes(image: np.ndarray, format: str = 'PNG') -> bytes:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    pil_image.save(buf, format=format)
    return buf.getvalue()

# ==================== MAIN APP ====================
def main():
    st.markdown('<p class="main-header">📄 Document Scanner</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        mode = st.radio(
            "Select Mode",
            ["🎥 Real-time Detection", "📷 Camera Capture", "📁 Upload Image", "ℹ️ About"],
            key="mode_selector"
        )
        
        st.markdown("---")
        st.subheader("Enhancement Options")
        enhancement_mode = st.selectbox(
            "Enhancement Mode",
            ["auto", "text", "photo", "bw", "color"],
            help="Select how to enhance the scanned document"
        )
        st.session_state.enhancement_mode = enhancement_mode
        
        with st.expander("Advanced Options"):
            remove_shadow = st.checkbox("Remove Shadow", value=False)
            denoise = st.checkbox("Denoise", value=False)
            sharpen = st.checkbox("Sharpen", value=False)
            deskew = st.checkbox("Auto Deskew", value=False)
            add_pad = st.checkbox("Add Padding", value=True)
            pad_amount = st.slider("Padding %", 0, 10, 3) if add_pad else 0
        
        if mode == "🎥 Real-time Detection":
            st.markdown("---")
            st.subheader("📹 Real-time Settings")
            sensitivity = st.select_slider(
                "Detection Sensitivity",
                options=['low', 'medium', 'high'],
                value='medium'
            )
            show_stats = st.checkbox("Show Stats", value=False)
        
        st.markdown("---")
        if st.button("🔄 Reset All", use_container_width=True):
            reset_state()
            if st.session_state.get('realtime_processor'):
                st.session_state.realtime_processor.reset()
            st.rerun()
        
        st.markdown("---")
        st.caption(f"Version {CONFIG.version}")
    
    # Route to modes
    params = {
        'enhancement_mode': enhancement_mode,
        'remove_shadow': remove_shadow,
        'denoise': denoise,
        'sharpen': sharpen,
        'deskew': deskew,
        'pad_frac': pad_amount / 100.0
    }
    
    if mode == "🎥 Real-time Detection":
        params['show_stats'] = show_stats if 'show_stats' in locals() else False
        params['sensitivity'] = sensitivity if 'sensitivity' in locals() else 'medium'
        params['smoothing'] = smoothing if 'smoothing' in locals() else 0.3
        params['min_movement'] = min_movement if 'min_movement' in locals() else 3.0
        params['stable_frames'] = stable_frames if 'stable_frames' in locals() else 3
        realtime_mode(**params)
    elif mode == "📷 Camera Capture":
        camera_mode(**params)
    elif mode == "📁 Upload Image":
        upload_mode(**params)
    else:
        about_page()

# ==================== REAL-TIME MODE ====================
def realtime_mode(enhancement_mode, remove_shadow, denoise, sharpen, deskew, pad_frac, show_stats, sensitivity='medium', smoothing=0.3, min_movement=3.0, stable_frames=3):
    st.header("🎥 Real-time Document Detection")
    
    st.info("""
    💡 **Real-time Scanning Tips:**
    - Position document flat with all 4 corners visible
    - Use good lighting and contrasting background
    - Keep camera steady when detection turns **GREEN**
    - Click **"📸 Capture"** button when ready
    """)
    
    if 'realtime_processor' not in st.session_state or st.session_state.realtime_processor is None:
        st.session_state.realtime_processor = VideoProcessor()
    
    # Apply settings
    processor = st.session_state.realtime_processor
    processor.set_sensitivity(sensitivity)
    processor.smoothing_alpha = smoothing
    processor.min_movement_threshold = min_movement
    processor.min_stable_frames = stable_frames
    
    col_video, col_capture = st.columns([2, 1])
    
    with col_video:
        ctx = webrtc_streamer(
            key="document-scanner",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=CONFIG.rtc_configuration,
            video_processor_factory=VideoProcessor,
            async_processing=True,
            media_stream_constraints={
                "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
                "audio": False
            }
        )
        
        if ctx.video_processor:
            st.session_state.realtime_processor = ctx.video_processor
    
    with col_capture:
        st.subheader("📸 Capture Control")
        
        # Debug info
        st.caption(f"Camera status: {'🟢 Playing' if ctx.state.playing else '🔴 Stopped'}")
        
        if ctx.state.playing:
            processor = ctx.video_processor if ctx.video_processor else st.session_state.get('realtime_processor')
            
            if processor:
                quad, score = processor.get_best_detection()
                
                # Always show button, status depends on detection
                st.markdown("---")
                
                if quad is not None and score > 0:
                    # Show detection status
                    if score >= 0.5:
                        st.success(f"✅ Ready to Capture\nConfidence: {score:.1%}")
                    elif score >= 0.3:
                        st.warning(f"⚠️ Adjust Position\nConfidence: {score:.1%}")
                    else:
                        st.info(f"🔍 Detecting...\nConfidence: {score:.1%}")
                    
                    # Show detection quality
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Score", f"{score:.1%}")
                    with col_b:
                        st.metric("Frames", f"{len(processor.detection_history)}/5")
                    
                    # Progress bar
                    if score >= 0.5:
                        st.progress(1.0, text="Excellent ✨")
                    elif score >= 0.3:
                        st.progress(0.6, text="Good 👍")
                    else:
                        st.progress(0.3, text="Fair 📍")
                    
                else:
                    st.info("🔍 **Searching for document...**")
                    st.caption("Position document in camera view")
                
                # CAPTURE BUTTON - Always visible when camera is on
                st.markdown("---")
                if st.button("📸 Capture Scan", use_container_width=True, type="primary", key="capture_btn"):
                    stable_quad, stable_score = processor.get_stable_detection()
                    
                    if stable_quad is not None and stable_score >= 0.2:
                        captured_frame = processor.capture_frame()
                        
                        if captured_frame is not None:
                            st.session_state.captured_image = captured_frame
                            st.session_state.detected_quad = stable_quad
                            st.session_state.detection_score = stable_score
                            st.session_state.snapshot_taken = True
                            st.success("✅ Captured! Scroll down.")
                            st.rerun()
                        else:
                            st.error("❌ Failed to capture frame. Try again.")
                    else:
                        st.warning("⚠️ No detection found. Please position document properly.")
                
                # # Expandable details
                # with st.expander("📊 Detection Details"):
                #     if quad is not None:
                #         st.json({
                #             'confidence': f"{score:.3f}",
                #             'history_length': len(processor.detection_history),
                #             'frame_count': processor.frame_count,
                #             'corners_detected': 'Yes' if quad is not None else 'No',
                #             'smoothing_alpha': processor.smoothing_alpha,
                #             'stable_frames': processor.stable_detection_count
                #         })
                #     else:
                #         st.info("No detection available")
                
                # Visual settings info
                st.caption(f"⚙️ Smoothing: {smoothing:.2f} | Movement: {min_movement}px | Stability: {stable_frames} frames")
                
            else:
                st.warning("⚠️ Processor not ready. Please wait...")
                if st.button("🔄 Retry", key="retry_processor"):
                    st.rerun()
        else:
            st.warning("📹 **Camera not started**")
            st.info("👆 Click **START** button above to begin")
            st.caption("Note: Browser will ask for camera permission")
        
        if show_stats and ctx.state.playing:
            processor = ctx.video_processor if ctx.video_processor else st.session_state.get('realtime_processor')
            if processor:
                with st.expander("📈 Stats"):
                    stats = processor.get_stats()
                    for key, value in stats.items():
                        st.text(f"{key}: {value}")
        
        if st.button("🔄 Reset Detection", use_container_width=True):
            processor = ctx.video_processor if ctx.video_processor else st.session_state.get('realtime_processor')
            if processor:
                processor.reset()
            st.session_state.snapshot_taken = False
            st.rerun()
    
    if st.session_state.snapshot_taken:
        st.markdown("---")
        st.header("🎯 Process Captured Scan")
        
        if st.session_state.captured_image is not None:
            quad = st.session_state.detected_quad
            score = st.session_state.detection_score
            
            st.success(f"✅ Frame captured with {score:.1%} confidence")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Captured Frame")
                display_img = resize_for_display(
                    st.session_state.captured_image,
                    CONFIG.ui.max_display_width,
                    CONFIG.ui.max_display_height
                )
                st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            with col2:
                st.subheader("🎯 Detection Overlay")
                overlay = draw_detection_overlay(
                    st.session_state.captured_image,
                    quad,
                    score,
                    color=(0, 255, 0) if score > 0.5 else (0, 255, 255)
                )
                display_overlay = resize_for_display(overlay, CONFIG.ui.max_display_width, CONFIG.ui.max_display_height)
                st.image(cv2.cvtColor(display_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            st.markdown("---")
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                if st.button("✏️ Adjust Corners", use_container_width=True, type="secondary"):
                    st.session_state.current_mode = 'adjust'
                    st.rerun()
            
            with col_b:
                if st.button("✅ Process Scan", use_container_width=True, type="primary"):
                    process_scan(enhancement_mode, remove_shadow, denoise, sharpen, deskew, pad_frac)
            
            with col_c:
                if st.button("🔄 Capture Again", use_container_width=True):
                    st.session_state.snapshot_taken = False
                    st.session_state.captured_image = None
                    st.rerun()
            
            with col_d:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.snapshot_taken = False
                    st.session_state.captured_image = None
                    processor = ctx.video_processor if ctx.video_processor else st.session_state.get('realtime_processor')
                    if processor:
                        processor.reset()
                    st.rerun()
            
            if st.session_state.current_mode == 'adjust':
                st.markdown("---")
                adjustment_mode()
        else:
            st.error("❌ No frame captured.")
            if st.button("🔙 Back"):
                st.session_state.snapshot_taken = False
                st.rerun()

# ==================== CAMERA MODE ====================
def camera_mode(enhancement_mode, remove_shadow, denoise, sharpen, deskew, pad_frac):
    st.header("📷 Camera Capture Mode")
    
    st.info("💡 Ensure good lighting and keep document flat")
    
    camera_photo = st.camera_input("Take a photo")
    
    if camera_photo is not None:
        file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("❌ Failed to load image")
            return
        
        st.session_state.captured_image = image
        
        if st.session_state.detected_quad is not None and st.session_state.detection_score > 0:
            st.success(f"✅ Using real-time detection ({st.session_state.detection_score:.1%})")
        
        process_captured_image(image, enhancement_mode, remove_shadow, denoise, sharpen, deskew, pad_frac)
        
        if st.button("🔄 Retake", use_container_width=True):
            reset_state()
            st.rerun()

# ==================== UPLOAD MODE ====================
def upload_mode(enhancement_mode, remove_shadow, denoise, sharpen, deskew, pad_frac):
    st.header("📁 Upload Image Mode")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help=f"Max: {CONFIG.max_upload_size_mb}MB"
    )
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("❌ Failed to load image")
            return
        
        try:
            assert_valid_image(image, "Uploaded image")
        except Exception as e:
            st.error(f"❌ Invalid image: {str(e)}")
            return
        
        st.session_state.captured_image = image
        process_captured_image(image, enhancement_mode, remove_shadow, denoise, sharpen, deskew, pad_frac)

# ==================== PROCESS CAPTURED IMAGE ====================
def process_captured_image(image, enhancement_mode, remove_shadow, denoise, sharpen, deskew, pad_frac):
    quality_info = check_image_quality(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original")
        display_img = resize_for_display(image, CONFIG.ui.max_display_width, CONFIG.ui.max_display_height)
        st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        with st.expander("📊 Quality Metrics"):
            st.metric("Brightness", f"{quality_info['brightness']:.1f}")
            st.metric("Contrast", f"{quality_info['contrast']:.1f}")
            st.metric("Sharpness", f"{quality_info['sharpness']:.1f}")
            if quality_info['warnings']:
                st.warning("⚠️ " + "\n".join(quality_info['warnings']))
    
    with col2:
        st.subheader("🎯 Auto-Detection")
        
        with st.spinner("🔍 Detecting..."):
            candidates = detect_receipt_contour(image, debug=False, fast_mode=False)
            
            if candidates and len(candidates) > 0:
                quad, score, details = candidates[0]
                st.session_state.detected_quad = quad
                st.session_state.detection_score = score
                
                if score > CONFIG.ui.good_confidence:
                    color = CONFIG.ui.color_good
                    status = "✅ Excellent"
                elif score > CONFIG.ui.adjust_confidence:
                    color = CONFIG.ui.color_adjust
                    status = "⚠️ Good"
                else:
                    color = (0, 165, 255)
                    status = "⚠️ Fair"
                
                overlay = draw_detection_overlay(image, quad, score, color=color)
                display_overlay = resize_for_display(overlay, CONFIG.ui.max_display_width, CONFIG.ui.max_display_height)
                st.image(cv2.cvtColor(display_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                if score > CONFIG.ui.good_confidence:
                    st.success(status)
                else:
                    st.warning(status)
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Confidence", f"{score:.1%}")
                with col_b:
                    rect_error = details.get('rect_error', 0)
                    st.metric("Rectangularity", f"{max(0, 100 - rect_error):.0f}%")
                with col_c:
                    area_ratio = details.get('area_ratio', 0)
                    st.metric("Coverage", f"{area_ratio:.1%}")
            else:
                st.error("❌ No document detected")
                h, w = image.shape[:2]
                margin = min(w, h) // 20
                st.session_state.detected_quad = np.array([
                    [margin, margin],
                    [w - margin, margin],
                    [w - margin, h - margin],
                    [margin, h - margin]
                ], dtype=np.float32)
                st.session_state.detection_score = 0.1
    
    st.markdown("---")
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    
    with col_btn1:
        if st.button("✏️ Adjust Corners", use_container_width=True, type="secondary"):
            st.session_state.current_mode = 'adjust'
            st.rerun()
    
    with col_btn2:
        if st.button("✅ Process Scan", use_container_width=True, type="primary"):
            process_scan(enhancement_mode, remove_shadow, denoise, sharpen, deskew, pad_frac)
    
    with col_btn3:
        if st.session_state.final_scan is not None:
            img_bytes = convert_image_to_bytes(st.session_state.final_scan)
            st.download_button(
                label="📥 Download",
                data=img_bytes,
                file_name="scanned.png",
                mime="image/png",
                use_container_width=True
            )
    
    with col_btn4:
        if st.button("🗑️ Clear", use_container_width=True):
            reset_state()
            st.rerun()
    
    if st.session_state.current_mode == 'adjust':
        st.markdown("---")
        adjustment_mode()

# ==================== ADJUSTMENT MODE ====================
def adjustment_mode():
    st.header("✏️ Manual Corner Adjustment")
    
    if st.session_state.captured_image is None:
        st.warning("⚠️ No image loaded!")
        return
    
    image = st.session_state.captured_image
    h, w = image.shape[:2]
    
    if st.session_state.detected_quad is None:
        margin = 30
        st.session_state.detected_quad = np.array([
            [margin, margin],
            [w - margin, margin],
            [w - margin, h - margin],
            [margin, h - margin]
        ], dtype=np.float32)
    
    quad = st.session_state.detected_quad.copy()
    
    st.info("💡 Adjust corners using coordinates or nudge buttons")
    
    st.subheader("🎯 Corner Coordinates")
    
    cols = st.columns(4)
    corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
    corner_colors = ["🔴", "🔵", "🟡", "🟢"]
    new_quad = quad.copy()
    
    for i, (col, name, color) in enumerate(zip(cols, corner_names, corner_colors)):
        with col:
            st.markdown(f"**{color} {name}**")
            
            x = st.number_input(
                f"X",
                min_value=0,
                max_value=w - 1,
                value=int(quad[i][0]),
                key=f"x_{i}",
                step=1
            )
            
            y = st.number_input(
                f"Y",
                min_value=0,
                max_value=h - 1,
                value=int(quad[i][1]),
                key=f"y_{i}",
                step=1
            )
            
            new_quad[i] = [x, y]
            
            st.markdown("**Quick Adjust:**")
            col_a, col_b = st.columns(2)
            nudge = CONFIG.ui.corner_nudge_amount
            with col_a:
                if st.button(f"⬅️ -{nudge}", key=f"left_{i}"):
                    new_quad[i][0] = max(0, new_quad[i][0] - nudge)
                    st.session_state.detected_quad = new_quad
                    st.rerun()
                if st.button(f"⬇️ +{nudge}", key=f"down_{i}"):
                    new_quad[i][1] = min(h - 1, new_quad[i][1] + nudge)
                    st.session_state.detected_quad = new_quad
                    st.rerun()
            with col_b:
                if st.button(f"➡️ +{nudge}", key=f"right_{i}"):
                    new_quad[i][0] = min(w - 1, new_quad[i][0] + nudge)
                    st.session_state.detected_quad = new_quad
                    st.rerun()
                if st.button(f"⬆️ -{nudge}", key=f"up_{i}"):
                    new_quad[i][1] = max(0, new_quad[i][1] - nudge)
                    st.session_state.detected_quad = new_quad
                    st.rerun()
    
    st.session_state.detected_quad = new_quad
    
    st.subheader("👁️ Preview")
    preview = image.copy()
    cv2.polylines(preview, [new_quad.astype(np.int32)], True, (0, 255, 0), 3)
    
    corner_display_colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 0)]
    for i, pt in enumerate(new_quad):
        cv2.circle(preview, tuple(pt.astype(int)), 10, corner_display_colors[i], -1)
        cv2.putText(
            preview, str(i),
            tuple((pt + 15).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2
        )
    
    display_preview = resize_for_display(preview, int(CONFIG.ui.max_display_width * 1.5), int(CONFIG.ui.max_display_height * 1.5))
    st.image(cv2.cvtColor(display_preview, cv2.COLOR_BGR2RGB), use_container_width=True)
    
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
    if st.session_state.captured_image is None:
        st.error("❌ No image to process!")
        return
    
    with st.spinner("🔄 Processing..."):
        image = st.session_state.captured_image
        quad = st.session_state.detected_quad
        
        if pad_frac > 0:
            h, w = image.shape[:2]
            quad = add_padding(quad, pad_frac, w, h)
        
        try:
            warped = four_point_transform(image, quad)
        except Exception as e:
            st.error(f"❌ Transform failed: {str(e)}")
            return
        
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
    
    st.markdown("---")
    st.subheader("🔧 Post-Processing")
    
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
            process_scan(enhancement_mode, remove_shadow, denoise, sharpen, deskew, pad_frac)

# ==================== ABOUT PAGE ====================
def about_page():
    st.header("ℹ️ About Document Scanner")
    
    st.markdown(f"""
    ### 🎯 Features
    
    - **🎥 Real-time Detection**: Live webcam scanning with instant feedback
    - **📁 Upload & Scan**: Upload images and detect boundaries automatically
    - **✏️ Manual Adjustment**: Fine-tune corners for perfect results
    - **✨ Enhancement Modes**: Auto, Text, Photo, B&W, Color
    - **📥 Download**: Save scanned documents as PNG
    
    ---
    
    ### 💡 Real-time Mode Tips
    
    ✅ Position document flat with all 4 corners visible
    ✅ Use good lighting and contrasting background  
    ✅ Wait for **GREEN** indicator before capturing
    ✅ Keep camera steady during capture
    
    ---
    
    ### 📊 Detection Indicators
    
    🟢 **Green** = Excellent (>50% confidence) - Ready to capture!
    🟡 **Yellow** = Good (30-50% confidence) - Adjust slightly
    🟠 **Orange** = Fair (<30% confidence) - Reposition document
    
    ---
    
    ### ⚙️ Sensitivity Settings
    
    **Low**: Faster processing, less CPU usage (every 5 frames)
    **Medium**: Balanced (every 3 frames) - Recommended
    **High**: Most accurate, more CPU usage (every frame)
    
    ---
    
    ### 🔧 Technical Details
    
    - **WebRTC**: Real-time video streaming via browser
    - **Detection**: Multi-scale edge detection with scoring
    - **Averaging**: Stable detection from frame history
    - **Enhancement**: Adaptive processing pipeline
    
    ---
    
    ### 📞 Troubleshooting
    
    **Camera not starting?**
    - Allow camera permissions in browser
    - Check if another app is using camera
    - Try refreshing the page
    
    **Detection not working?**
    - Ensure good lighting
    - Use contrasting background
    - Make sure all corners visible
    
    **Capture button not responding?**
    - Wait for confidence score >20%
    - Adjust document position
    - Try different sensitivity setting
    
    ---
    
    **Version**: {CONFIG.version}
    
    **Built with**: OpenCV, Streamlit, streamlit-webrtc
    """)

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()