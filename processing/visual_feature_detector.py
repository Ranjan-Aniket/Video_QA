"""
Visual Feature Detector - Phase 3: Extract visual features for highlights

Detects:
- Motion intensity (frame difference)
- Color variance (visual excitement)
- Shot boundaries

NO SEMANTICS - uses computer vision only
"""

import cv2
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class VisualFeatureDetector:
    """Detect visual features for highlight detection"""
    
    def detect_visual_highlights(
        self,
        video_path: str
    ) -> List[Dict]:
        """
        Detect highlights from visual features.
        
        Returns:
            List of highlight dicts
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        highlights = []
        frame_num = 0
        prev_frame = None
        motion_history = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample at 1fps
            if frame_num % int(fps) == 0:
                timestamp = frame_num / fps
                
                # Motion intensity
                if prev_frame is not None:
                    motion = cv2.absdiff(frame, prev_frame)
                    motion_intensity = np.mean(motion) / 255.0
                    motion_history.append(motion_intensity)
                    
                    # Detect motion peaks
                    if len(motion_history) > 10:
                        mean_motion = np.mean(motion_history[-10:])
                        if motion_intensity > mean_motion * 1.8:
                            highlights.append({
                                'timestamp': timestamp,
                                'type': 'motion_peak',
                                'confidence': min(motion_intensity / mean_motion / 2.0, 1.0),
                                'signal': 'visual'
                            })
                
                # Color variance
                color_variance = np.var(frame)
                if color_variance > 2000:
                    highlights.append({
                        'timestamp': timestamp,
                        'type': 'color_variance',
                        'confidence': min(color_variance / 5000.0, 1.0),
                        'signal': 'visual'
                    })
                
                prev_frame = frame.copy()
            
            frame_num += 1
        
        cap.release()
        
        logger.info(f"Detected {len(highlights)} visual highlights")
        
        return highlights
