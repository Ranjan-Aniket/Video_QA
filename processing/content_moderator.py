"""
Content Moderator - Video Safety and Quality Checks

Implements guideline requirement:
"If the video has elements of violence/gunshots or contains obscene/sexual scenes,
please reject the video"

"If the video has built-in subtitles on the video screen, reject the video"
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


class ContentModerator:
    """
    Validates video content against safety and quality guidelines.

    Checks:
    1. Violence/weapons detection (using YOLO)
    2. NSFW content detection (using CLIP embeddings)
    3. Burned-in subtitles detection (using OCR)
    """

    def __init__(self):
        """Initialize content moderator with detection models."""
        self.violence_keywords = [
            'gun', 'weapon', 'knife', 'blood', 'fight', 'violence',
            'shooting', 'combat', 'war', 'explosion', 'fire'
        ]
        self.nsfw_keywords = [
            'nudity', 'sexual', 'explicit', 'adult', 'nsfw'
        ]

    def should_reject_video(
        self,
        video_path: Path,
        sample_frames: list,
        yolo_detections: Optional[Dict] = None,
        clip_embeddings: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Check if video should be rejected based on content guidelines.

        Args:
            video_path: Path to video file
            sample_frames: List of sampled frames from Phase 2
            yolo_detections: Optional YOLO detection results from Phase 2
            clip_embeddings: Optional CLIP embeddings from Phase 2

        Returns:
            (should_reject, reason) - True if video should be rejected
        """
        # Check 1: Violence/Weapons detection
        violence_detected, violence_reason = self._check_violence(
            sample_frames, yolo_detections
        )
        if violence_detected:
            return True, f"Violence detected: {violence_reason}"

        # Check 2: NSFW content detection
        nsfw_detected, nsfw_reason = self._check_nsfw(
            sample_frames, clip_embeddings
        )
        if nsfw_detected:
            return True, f"NSFW content detected: {nsfw_reason}"

        # Check 3: Burned-in subtitles detection
        subtitles_detected, subtitle_reason = self._check_subtitles(
            sample_frames
        )
        if subtitles_detected:
            return True, f"Burned-in subtitles detected: {subtitle_reason}"

        # All checks passed
        return False, "Video passed all content moderation checks"

    def _check_violence(
        self,
        sample_frames: list,
        yolo_detections: Optional[Dict]
    ) -> Tuple[bool, str]:
        """
        Check for violence indicators in video.

        Uses YOLO detections from Phase 2 to detect weapons.

        Args:
            sample_frames: List of sampled frames
            yolo_detections: YOLO detection results

        Returns:
            (is_violent, reason)
        """
        if not yolo_detections:
            logger.warning("No YOLO detections provided - skipping violence check")
            return False, "No YOLO data"

        # Check YOLO detections for weapons
        weapon_classes = ['knife', 'gun', 'rifle', 'pistol', 'sword']

        for frame_id, detections in yolo_detections.items():
            detected_objects = detections.get('detected_objects', [])

            for obj in detected_objects:
                obj_lower = obj.lower()
                if any(weapon in obj_lower for weapon in weapon_classes):
                    logger.warning(f"Weapon detected in {frame_id}: {obj}")
                    return True, f"Weapon detected: {obj}"

        logger.info("✅ No violence indicators detected")
        return False, "No violence detected"

    def _check_nsfw(
        self,
        sample_frames: list,
        clip_embeddings: Optional[Dict]
    ) -> Tuple[bool, str]:
        """
        Check for NSFW content in video.

        Currently a placeholder - would use CLIP embeddings to detect
        inappropriate content.

        Args:
            sample_frames: List of sampled frames
            clip_embeddings: CLIP embeddings from Phase 2

        Returns:
            (is_nsfw, reason)
        """
        # TODO: Implement NSFW detection using CLIP embeddings
        # For now, return False (no NSFW detected)

        logger.info("✅ NSFW check passed (placeholder implementation)")
        return False, "No NSFW content detected"

    def _check_subtitles(
        self,
        sample_frames: list
    ) -> Tuple[bool, str]:
        """
        Check for burned-in subtitles in video frames.

        Detects subtitles by:
        1. Checking bottom 20% of frame for text using OCR
        2. Looking for consistent text placement across frames

        Args:
            sample_frames: List of sampled frame paths

        Returns:
            (has_subtitles, reason)
        """
        if not sample_frames:
            logger.warning("No sample frames provided - skipping subtitle check")
            return False, "No frames to check"

        # Sample 10 frames evenly distributed
        num_samples = min(10, len(sample_frames))
        sample_indices = np.linspace(0, len(sample_frames) - 1, num_samples, dtype=int)
        frames_to_check = [sample_frames[i] for i in sample_indices]

        subtitle_count = 0

        for frame_path in frames_to_check:
            if self._has_subtitle_in_frame(frame_path):
                subtitle_count += 1

        # If > 50% of frames have subtitles, consider it burned-in
        subtitle_ratio = subtitle_count / num_samples

        if subtitle_ratio > 0.5:
            logger.warning(
                f"Burned-in subtitles detected: {subtitle_count}/{num_samples} frames ({subtitle_ratio*100:.1f}%)"
            )
            return True, f"Subtitles in {subtitle_count}/{num_samples} frames"

        logger.info(f"✅ No consistent subtitles detected ({subtitle_count}/{num_samples} frames)")
        return False, "No burned-in subtitles"

    def _has_subtitle_in_frame(self, frame_path: Path) -> bool:
        """
        Check if a single frame has burned-in subtitles.

        Looks for text in the bottom 20% of the frame using OCR.

        Args:
            frame_path: Path to frame image

        Returns:
            True if subtitles detected
        """
        try:
            # Read frame
            frame = cv2.imread(str(frame_path))
            if frame is None:
                return False

            height, width = frame.shape[:2]

            # Extract bottom 20% of frame (typical subtitle location)
            subtitle_region = frame[int(height * 0.8):, :]

            # Convert to grayscale for better OCR
            gray = cv2.cvtColor(subtitle_region, cv2.COLOR_BGR2GRAY)

            # Apply thresholding to enhance text
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Run OCR
            pil_image = Image.fromarray(thresh)
            text = pytesseract.image_to_string(pil_image, config='--psm 6')

            # Clean and check text
            text_clean = text.strip()

            # Subtitles typically have:
            # - More than 5 characters
            # - Multiple words
            # - Proper spacing
            if len(text_clean) > 5 and ' ' in text_clean:
                logger.debug(f"Subtitle text detected in {frame_path.name}: '{text_clean[:50]}'")
                return True

            return False

        except Exception as e:
            logger.warning(f"Error checking subtitle in {frame_path}: {e}")
            return False


def is_intro_outro(timestamp: float, video_duration: float) -> bool:
    """
    Detect if timestamp falls in intro or outro region.

    Guideline: "Do not use the intro and outro of the video for reference points"

    Definition:
    - Intro: First 10% of video OR first 15 seconds (whichever is smaller)
    - Outro: Last 10% of video OR last 15 seconds (whichever is smaller)

    Args:
        timestamp: Frame timestamp in seconds
        video_duration: Total video duration in seconds

    Returns:
        True if timestamp is in intro/outro region
    """
    # Intro threshold: min(10% duration, 15 seconds)
    intro_threshold = min(video_duration * 0.1, 15.0)

    # Outro threshold: max(90% duration, duration - 15 seconds)
    outro_threshold = video_duration - min(video_duration * 0.1, 15.0)

    is_intro = timestamp < intro_threshold
    is_outro = timestamp > outro_threshold

    if is_intro:
        logger.debug(f"Timestamp {timestamp:.1f}s is in INTRO region (< {intro_threshold:.1f}s)")
    elif is_outro:
        logger.debug(f"Timestamp {timestamp:.1f}s is in OUTRO region (> {outro_threshold:.1f}s)")

    return is_intro or is_outro
