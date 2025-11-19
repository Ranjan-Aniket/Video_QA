"""
Text Orientation Processor - Auto-Orient + Tesseract

Model: Auto-Orient + Tesseract
Purpose: Detect and correct text orientation for better OCR accuracy
"""

import logging
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class TextOrientationProcessor:
    """
    Text orientation detection and correction processor

    Components:
    - Auto-Orient: Automatic text orientation detection
    - Tesseract: OCR with built-in orientation detection
    Purpose: Improve OCR accuracy on rotated text
    """

    def __init__(self):
        """Initialize text orientation processor"""
        self.tesseract_available = False
        self._init_tesseract()

        logger.info("Text Orientation Processor initialized")

    def _init_tesseract(self):
        """Initialize Tesseract OCR"""
        try:
            import pytesseract

            # Test if tesseract is available
            pytesseract.get_tesseract_version()
            self.tesseract_available = True

            logger.info("Tesseract OCR loaded successfully")

        except ImportError:
            logger.warning(
                "pytesseract not installed. Install with: pip install pytesseract"
            )
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")

    def detect_orientation(
        self,
        image: np.ndarray
    ) -> Tuple[int, float]:
        """
        Detect text orientation in image

        Args:
            image: Image (HxWxC numpy array)

        Returns:
            Tuple of (rotation_angle, confidence)
            rotation_angle: 0, 90, 180, or 270 degrees
            confidence: Detection confidence (0.0-1.0)
        """
        if self.tesseract_available:
            return self._tesseract_detect_orientation(image)
        else:
            return self._simple_detect_orientation(image)

    def _tesseract_detect_orientation(
        self,
        image: np.ndarray
    ) -> Tuple[int, float]:
        """Detect orientation using Tesseract's OSD"""
        try:
            import pytesseract
            from PIL import Image

            # Convert to PIL
            pil_image = Image.fromarray(image)

            # Get orientation and script detection (OSD)
            osd = pytesseract.image_to_osd(pil_image, output_type=pytesseract.Output.DICT)

            rotation = osd.get('rotate', 0)
            confidence = osd.get('orientation_conf', 0.0) / 100.0  # Convert to 0-1

            logger.info(f"Detected orientation: {rotation}° (confidence: {confidence:.2f})")

            return rotation, confidence

        except Exception as e:
            logger.warning(f"Tesseract orientation detection failed: {e}")
            return self._simple_detect_orientation(image)

    def _simple_detect_orientation(
        self,
        image: np.ndarray
    ) -> Tuple[int, float]:
        """Simple orientation detection fallback"""
        # Analyze image aspect ratio and content distribution
        height, width = image.shape[:2]

        if height > width * 1.5:
            # Likely portrait/90° rotation
            return 90, 0.60
        elif width > height * 1.5:
            # Likely landscape/normal
            return 0, 0.60
        else:
            # Square-ish, uncertain
            return 0, 0.50

    def correct_orientation(
        self,
        image: np.ndarray,
        rotation: Optional[int] = None
    ) -> np.ndarray:
        """
        Correct image orientation

        Args:
            image: Image to correct
            rotation: Rotation angle (0, 90, 180, 270)
                     If None, auto-detect

        Returns:
            Corrected image
        """
        if rotation is None:
            rotation, _ = self.detect_orientation(image)

        if rotation == 0:
            return image

        try:
            import cv2

            # Rotate image
            if rotation == 90:
                corrected = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                corrected = cv2.rotate(image, cv2.ROTATE_180)
            elif rotation == 270:
                corrected = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                logger.warning(f"Invalid rotation angle: {rotation}")
                corrected = image

            logger.info(f"Corrected orientation by {rotation}°")
            return corrected

        except Exception as e:
            logger.error(f"Orientation correction failed: {e}")
            return image

    def auto_orient_for_ocr(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        Auto-orient image for optimal OCR

        Args:
            image: Input image

        Returns:
            Tuple of (oriented_image, metadata)
            metadata contains: rotation, confidence, original_shape
        """
        # Detect orientation
        rotation, confidence = self.detect_orientation(image)

        # Correct if needed
        if rotation != 0 and confidence > 0.50:
            oriented = self.correct_orientation(image, rotation)
        else:
            oriented = image

        metadata = {
            "rotation": rotation,
            "confidence": confidence,
            "original_shape": image.shape,
            "corrected": rotation != 0
        }

        return oriented, metadata

    def batch_auto_orient(
        self,
        images: list
    ) -> list:
        """
        Auto-orient multiple images

        Args:
            images: List of images

        Returns:
            List of oriented images
        """
        oriented_images = []

        for image in images:
            oriented, _ = self.auto_orient_for_ocr(image)
            oriented_images.append(oriented)

        return oriented_images


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    processor = TextOrientationProcessor()

    # Example: Detect and correct orientation
    # image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    #
    # # Detect orientation
    # rotation, confidence = processor.detect_orientation(image)
    # print(f"Detected rotation: {rotation}° (confidence: {confidence:.2f})")
    #
    # # Auto-correct
    # oriented, metadata = processor.auto_orient_for_ocr(image)
    # print(f"Metadata: {metadata}")

    print("Text Orientation Processor ready (Auto-Orient + Tesseract)")
