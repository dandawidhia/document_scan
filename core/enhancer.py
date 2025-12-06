"""
core/enhancer.py
Post-processing enhancement untuk hasil scan
"""

import cv2
import numpy as np
from typing import Optional


class DocumentEnhancer:
    """Class untuk enhance hasil scan dokumen"""
    
    @staticmethod
    def auto_enhance(image: np.ndarray) -> np.ndarray:
        """
        Automatic enhancement berdasarkan konten.
        
        Args:
            image: Input scanned image
        
        Returns:
            Enhanced image
        """
        # Analyze content
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect document type
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Text-heavy document
        if edge_density > 0.05:
            return DocumentEnhancer.enhance_text(image)
        # Photo or mixed content
        else:
            return DocumentEnhancer.enhance_photo(image)
    
    @staticmethod
    def enhance_text(image: np.ndarray) -> np.ndarray:
        """
        Enhance untuk dokumen text (seperti receipt, invoice).
        
        Args:
            image: Input image
        
        Returns:
            Enhanced image for text
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15, 8
        )
        
        # Noise reduction
        binary = cv2.medianBlur(binary, 3)
        
        # Convert back to BGR if needed
        if len(image.shape) == 3:
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return binary
    
    @staticmethod
    def enhance_photo(image: np.ndarray) -> np.ndarray:
        """
        Enhance untuk foto atau dokumen dengan gambar.
        
        Args:
            image: Input image
        
        Returns:
            Enhanced photo
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE on L channel
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Slight sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 1.5
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    @staticmethod
    def enhance_bw(image: np.ndarray, threshold_mode: str = 'adaptive') -> np.ndarray:
        """
        Convert to high-contrast black & white.
        
        Args:
            image: Input image
            threshold_mode: 'adaptive', 'otsu', or 'fixed'
        
        Returns:
            Binary image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply threshold
        if threshold_mode == 'adaptive':
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                15, 8
            )
        elif threshold_mode == 'otsu':
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:  # fixed
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Noise reduction
        binary = cv2.medianBlur(binary, 3)
        
        # Convert to BGR if needed
        if len(image.shape) == 3:
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return binary
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
        """
        Adjust contrast and brightness.
        
        Args:
            image: Input image
            alpha: Contrast control (1.0-3.0)
            beta: Brightness control (-100 to 100)
        
        Returns:
            Adjusted image
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    @staticmethod
    def sharpen(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Sharpen image.
        
        Args:
            image: Input image
            strength: Sharpening strength (0.5-2.0)
        
        Returns:
            Sharpened image
        """
        # Sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * strength / 1.5
        
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened
    
    @staticmethod
    def denoise(image: np.ndarray, strength: int = 10) -> np.ndarray:
        """
        Remove noise from image.
        
        Args:
            image: Input image
            strength: Denoising strength (1-20)
        
        Returns:
            Denoised image
        """
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(
                image, None, strength, strength, 7, 21
            )
        else:
            denoised = cv2.fastNlMeansDenoising(
                image, None, strength, 7, 21
            )
        return denoised
    
    @staticmethod
    def adjust_white_balance(image: np.ndarray) -> np.ndarray:
        """
        Auto white balance correction.
        
        Args:
            image: Input image
        
        Returns:
            White-balanced image
        """
        result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result
    
    @staticmethod
    def remove_shadow(image: np.ndarray) -> np.ndarray:
        """
        Remove shadow dari dokumen.
        
        Args:
            image: Input image
        
        Returns:
            Shadow-removed image
        """
        rgb_planes = cv2.split(image)
        result_planes = []
        
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            result_planes.append(diff_img)
        
        result = cv2.merge(result_planes)
        return result
    
    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        """
        Correct skew/tilt dalam dokumen.
        
        Args:
            image: Input image
        
        Returns:
            Deskewed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Threshold
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Find coordinates of all non-zero points
        coords = np.column_stack(np.where(thresh > 0))
        
        # Calculate rotation angle
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Rotate image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated


def apply_enhancement_pipeline(
    image: np.ndarray,
    mode: str = 'auto',
    remove_shadow: bool = False,
    denoise: bool = False,
    sharpen: bool = False,
    deskew: bool = False
) -> np.ndarray:
    """
    Apply enhancement pipeline dengan berbagai options.
    
    Args:
        image: Input image
        mode: Enhancement mode ('auto', 'text', 'photo', 'bw')
        remove_shadow: Remove shadows
        denoise: Apply denoising
        sharpen: Apply sharpening
        deskew: Correct skew
    
    Returns:
        Enhanced image
    """
    enhancer = DocumentEnhancer()
    result = image.copy()
    
    # Shadow removal (before other enhancements)
    if remove_shadow:
        result = enhancer.remove_shadow(result)
    
    # Deskew
    if deskew:
        result = enhancer.deskew(result)
    
    # Main enhancement
    if mode == 'auto':
        result = enhancer.auto_enhance(result)
    elif mode == 'text':
        result = enhancer.enhance_text(result)
    elif mode == 'photo':
        result = enhancer.enhance_photo(result)
    elif mode == 'bw':
        result = enhancer.enhance_bw(result)
    
    # Denoising
    if denoise:
        result = enhancer.denoise(result, strength=10)
    
    # Sharpening (after main enhancement)
    if sharpen:
        result = enhancer.sharpen(result, strength=1.0)
    
    return result