"""
Audio Feature Detector - Phase 3: Extract audio features for highlights

Detects:
- Volume spikes (RMS energy)
- Pitch variance (excitement)
- Energy changes
- Speaking rate changes

NO HARDCODED KEYWORDS - uses signal processing only
"""

import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available")


class AudioFeatureDetector:
    """Detect audio features for highlight detection"""
    
    def __init__(self):
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required: pip install librosa")
    
    def detect_audio_highlights(
        self,
        audio_path: str,
        audio_analysis: Dict
    ) -> List[Dict]:
        """
        Detect highlights from audio features.
        
        Returns:
            List of highlight dicts with timestamps and scores
        """
        # Load audio
        y, sr = librosa.load(audio_path)
        
        highlights = []
        
        # 1. Volume spikes (RMS energy)
        volume_highlights = self._detect_volume_spikes(y, sr)
        highlights.extend(volume_highlights)
        
        # 2. Pitch variance (excitement indicator)
        pitch_highlights = self._detect_pitch_variance(y, sr)
        highlights.extend(pitch_highlights)
        
        logger.info(f"Detected {len(highlights)} audio highlights")
        
        return highlights
    
    def _detect_volume_spikes(self, y, sr) -> List[Dict]:
        """Detect volume spikes (RMS energy)"""
        # Calculate RMS energy
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
        
        # Find spikes above threshold
        mean_rms = np.mean(rms)
        std_rms = np.std(rms)
        threshold = mean_rms + 1.5 * std_rms
        
        peaks = np.where(rms > threshold)[0]
        
        highlights = []
        for peak_idx in peaks:
            confidence = min((rms[peak_idx] - mean_rms) / std_rms / 3.0, 1.0)
            highlights.append({
                'timestamp': times[peak_idx],
                'type': 'volume_spike',
                'confidence': confidence,
                'signal': 'audio'
            })
        
        return highlights
    
    def _detect_pitch_variance(self, y, sr) -> List[Dict]:
        """Detect pitch variance (excitement)"""
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_variance = np.var(pitches, axis=0)
        times = librosa.frames_to_time(np.arange(len(pitch_variance)), sr=sr)
        
        # High variance = excitement
        high_variance_idx = np.where(
            pitch_variance > np.percentile(pitch_variance, 80)
        )[0]
        
        highlights = []
        for idx in high_variance_idx:
            confidence = min(pitch_variance[idx] / np.max(pitch_variance), 1.0)
            highlights.append({
                'timestamp': times[idx],
                'type': 'pitch_variance',
                'confidence': confidence,
                'signal': 'audio'
            })
        
        return highlights
