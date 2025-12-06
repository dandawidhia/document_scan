"""
core/realtime.py - Real-time video processing untuk document detection
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict
import av

from .detector import detect_receipt_contour
from .transformer import draw_detection_overlay


class VideoProcessor:
    """
    Video processor untuk real-time document detection
    Digunakan dengan streamlit-webrtc
    """
    
    def __init__(self):
        self.best_detection = None
        self.best_score = 0.0
        self.frame_count = 0
        self.process_every_n_frames = 3  # Process every 3rd frame for performance
        self.detection_history = []
        self.max_history = 5
        
        # Frame capture
        self.captured_frame = None
        self.last_processed_frame = None
        
        # Detection thresholds
        self.good_threshold = 0.5
        self.acceptable_threshold = 0.3
        
        # Visual settings
        self.color_excellent = (0, 255, 0)  # Green
        self.color_good = (0, 255, 255)     # Yellow
        self.color_fair = (0, 165, 255)     # Orange
        self.color_poor = (0, 0, 255)       # Red
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Process video frame - called by streamlit-webrtc
        
        Args:
            frame: Input video frame from camera
            
        Returns:
            Processed video frame with detection overlay
        """
        # Convert to numpy array (BGR)
        img = frame.to_ndarray(format="bgr24")
        
        # Store last frame for capture
        self.last_processed_frame = img.copy()
        
        # Process frame
        processed_img = self.process_frame(img)
        
        # Convert back to VideoFrame
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for document detection
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Frame with detection overlay
        """
        self.frame_count += 1
        
        # Skip frames for performance
        if self.frame_count % self.process_every_n_frames != 0:
            # Return last known detection if available
            if self.best_detection is not None:
                return self._draw_overlay(frame, self.best_detection, self.best_score)
            return frame
        
        # Run detection
        try:
            candidates = detect_receipt_contour(frame, debug=False, fast_mode=True)
            
            if candidates and len(candidates) > 0:
                quad, score, details = candidates[0]
                
                # Update history
                self.detection_history.append((quad.copy(), score))
                if len(self.detection_history) > self.max_history:
                    self.detection_history.pop(0)
                
                # Update best detection if this is better
                if score > self.best_score:
                    self.best_detection = quad.copy()
                    self.best_score = score
                
                # Draw overlay
                return self._draw_overlay(frame, quad, score)
            
        except Exception as e:
            # Silent fail - just return original frame
            pass
        
        # No detection - show instruction
        return self._draw_instructions(frame)
    
    def _draw_overlay(self, frame: np.ndarray, quad: np.ndarray, score: float) -> np.ndarray:
        """Draw detection overlay on frame"""
        
        # Choose color based on score
        if score >= self.good_threshold:
            color = self.color_excellent
            status = "EXCELLENT"
        elif score >= self.acceptable_threshold:
            color = self.color_good
            status = "GOOD"
        else:
            color = self.color_fair
            status = "ADJUST POSITION"
        
        # Draw detection overlay
        result = draw_detection_overlay(frame, quad, score, color=color)
        
#         # Add status text
#         h, w = frame.shape[:2]
        
#         # Background for text
#         cv2.rectangle(result, (10, 10), (w - 10, 80), (0, 0, 0), -1)
#         cv2.rectangle(result, (10, 10), (w - 10, 80), color, 2)
        
#         # Status text
#         cv2.putText(
#             result, 
#             f"Status: {status}", 
#             (20, 40),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             color,
#             2
#         )
        
#         # Confidence
#         cv2.putText(
#             result,
#             f"Confidence: {score:.1%}",
#             (20, 65),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             (255, 255, 255),
#             1
#         )
        
#         # Capture instruction
#         if score >= self.good_threshold:
#             cv2.putText(
#                 result,
#                 "Press CAPTURE to scan",
#                 (w - 300, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 (255, 255, 255),
#                 1
#             )
        
        return result
    
    def _draw_instructions(self, frame: np.ndarray) -> np.ndarray:
        """Draw instructions when no document detected"""
        
        result = frame.copy()
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = result.copy()
        cv2.rectangle(overlay, (w//4, h//3), (3*w//4, 2*h//3), (0, 0, 0), -1)
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
        
        # Instructions
        instructions = [
            "Position document in frame",
            "Ensure all 4 corners visible",
            "Use contrasting background",
            "Good lighting required"
        ]
        
        y_start = h//3 + 40
        for i, text in enumerate(instructions):
            cv2.putText(
                result,
                text,
                (w//4 + 20, y_start + i * 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
        
        # Icon/indicator
        cv2.rectangle(result, (w//2 - 50, h//2 - 50), (w//2 + 50, h//2 + 50), (0, 165, 255), 3)
        
        return result
    
    def get_best_detection(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Get the best detection from history
        
        Returns:
            (quad, score) tuple or (None, 0.0) if no detection
        """
        if self.best_detection is not None:
            return self.best_detection.copy(), self.best_score
        return None, 0.0
    
    def get_stable_detection(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Get stable detection by averaging recent detections
        More stable than single best detection
        
        Returns:
            (averaged_quad, avg_score) or (None, 0.0)
        """
        if not self.detection_history:
            return None, 0.0
        
        # Filter for good detections only
        good_detections = [
            (q, s) for q, s in self.detection_history 
            if s >= self.acceptable_threshold
        ]
        
        if not good_detections:
            return self.best_detection, self.best_score
        
        # Average the quads
        quads = np.array([q for q, s in good_detections])
        scores = [s for q, s in good_detections]
        
        avg_quad = np.mean(quads, axis=0)
        avg_score = np.mean(scores)
        
        return avg_quad, avg_score
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture the current frame
        
        Returns:
            Current frame or None
        """
        if self.last_processed_frame is not None:
            self.captured_frame = self.last_processed_frame.copy()
            return self.captured_frame
        return None
    
    def get_captured_frame(self) -> Optional[np.ndarray]:
        """Get the last captured frame"""
        return self.captured_frame
    
    def reset(self):
        """Reset processor state"""
        self.best_detection = None
        self.best_score = 0.0
        self.detection_history = []
        self.frame_count = 0
        self.captured_frame = None
        # Keep last_processed_frame for one more capture opportunity
    
    def set_sensitivity(self, sensitivity: str):
        """
        Adjust detection sensitivity
        
        Args:
            sensitivity: 'low', 'medium', or 'high'
        """
        if sensitivity == 'low':
            self.process_every_n_frames = 5
            self.acceptable_threshold = 0.4
        elif sensitivity == 'medium':
            self.process_every_n_frames = 3
            self.acceptable_threshold = 0.3
        elif sensitivity == 'high':
            self.process_every_n_frames = 1
            self.acceptable_threshold = 0.2
        else:
            # Default to medium
            self.process_every_n_frames = 3
            self.acceptable_threshold = 0.3
    
    def get_stats(self) -> Dict:
        """Get current processing stats"""
        return {
            'frames_processed': self.frame_count,
            'detections_found': len(self.detection_history),
            'best_score': self.best_score,
            'processing_rate': f"1/{self.process_every_n_frames} frames"
        }