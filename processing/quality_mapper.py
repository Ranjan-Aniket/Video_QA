"""
Quality Mapper - Phase 1: Map video quality (blur, brightness)

Samples every 1 second and checks:
- Blur score (Laplacian variance)
- Brightness (mean intensity)
- Quality score (0.0-1.0)
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class QualityMapper:
    """Map quality metrics across video timeline"""
    
    def __init__(self):
        self.min_blur_score = 100.0  # Laplacian variance threshold
        self.min_brightness = 0.1
        self.max_brightness = 0.9
    
    def map_quality(self, video_path: str) -> Dict:
        """
        Sample quality every 1 second.
        
        Returns:
            {
                'quality_scores': {timestamp: score},
                'average_quality': float
            }
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        quality_scores = {}
        frame_num = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample every 1 second
            if frame_num % int(fps) == 0:
                timestamp = frame_num / fps
                score = self._assess_quality(frame)
                quality_scores[timestamp] = score
            
            frame_num += 1
        
        cap.release()
        
        avg_quality = np.mean(list(quality_scores.values())) if quality_scores else 0.0
        
        logger.info(f"Quality map: {len(quality_scores)} samples, avg={avg_quality:.2f}")
        
        return {
            'quality_scores': quality_scores,
            'average_quality': avg_quality
        }
    
    def _assess_quality(self, frame):
        """Assess quality of single frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Blur score (Laplacian variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Brightness (mean intensity)
        brightness = np.mean(gray) / 255.0
        
        # Combined quality score
        blur_ok = blur_score > self.min_blur_score
        brightness_ok = self.min_brightness < brightness < self.max_brightness
        
        if blur_ok and brightness_ok:
            return 1.0
        elif blur_ok or brightness_ok:
            return 0.5
        else:
            return 0.0
