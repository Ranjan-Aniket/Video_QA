"""
Universal Highlight Detector - Phase 3: Multi-Signal Fusion

Orchestrates and fuses ALL highlight detection signals:
1. Audio features (volume spikes, pitch variance)
2. Visual features (motion peaks, color variance)
3. Semantic features (Claude LLM analysis)
4. FREE model features (BLIP-2, CLIP, YOLO, etc.)

Outputs: Ranked list of highlights with combined confidence scores
"""

import logging
from typing import List, Dict, Optional
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class UniversalHighlightDetector:
    """Orchestrate multi-signal highlight detection"""

    def __init__(
        self,
        audio_weight: float = 0.25,
        visual_weight: float = 0.25,
        semantic_weight: float = 0.35,
        free_models_weight: float = 0.15
    ):
        """
        Initialize with signal weights.

        Args:
            audio_weight: Weight for audio features
            visual_weight: Weight for visual features
            semantic_weight: Weight for LLM semantic analysis
            free_models_weight: Weight for FREE model features
        """
        self.audio_weight = audio_weight
        self.visual_weight = visual_weight
        self.semantic_weight = semantic_weight
        self.free_models_weight = free_models_weight

        # Validate weights sum to 1.0
        total = audio_weight + visual_weight + semantic_weight + free_models_weight
        if not np.isclose(total, 1.0):
            logger.warning(f"Weights sum to {total}, normalizing to 1.0")
            norm = 1.0 / total
            self.audio_weight *= norm
            self.visual_weight *= norm
            self.semantic_weight *= norm
            self.free_models_weight *= norm

    def detect_highlights(
        self,
        audio_highlights: List[Dict],
        visual_highlights: List[Dict],
        semantic_highlights: List[Dict],
        visual_samples: Optional[List[Dict]] = None,
        video_duration: Optional[float] = None
    ) -> Dict:
        """
        Fuse all signals into ranked highlights.

        Args:
            audio_highlights: From audio_feature_detector
            visual_highlights: From visual_feature_detector
            semantic_highlights: From llm_semantic_detector
            visual_samples: From quick_visual_sampler (optional)
            video_duration: Total video duration in seconds

        Returns:
            {
                'highlights': [ranked list of highlights],
                'total_highlights': int,
                'signal_breakdown': dict
            }
        """
        logger.info("Fusing multi-signal highlights...")
        logger.info(f"  Audio: {len(audio_highlights)} highlights")
        logger.info(f"  Visual: {len(visual_highlights)} highlights")
        logger.info(f"  Semantic: {len(semantic_highlights)} highlights")

        # Bin highlights into 1-second intervals
        time_bins = self._bin_highlights_by_time(
            audio_highlights,
            visual_highlights,
            semantic_highlights,
            visual_samples,
            video_duration
        )

        # Calculate combined scores for each bin
        ranked_highlights = self._rank_highlights(time_bins)

        # Extract signal breakdown
        signal_breakdown = self._calculate_signal_breakdown(
            audio_highlights,
            visual_highlights,
            semantic_highlights
        )

        logger.info(f"Generated {len(ranked_highlights)} fused highlights")

        return {
            'highlights': ranked_highlights,
            'total_highlights': len(ranked_highlights),
            'signal_breakdown': signal_breakdown
        }

    def _bin_highlights_by_time(
        self,
        audio_highlights: List[Dict],
        visual_highlights: List[Dict],
        semantic_highlights: List[Dict],
        visual_samples: Optional[List[Dict]],
        video_duration: Optional[float]
    ) -> Dict[int, Dict]:
        """Bin highlights into 1-second intervals"""

        # Determine max time
        all_times = []
        for h in audio_highlights + visual_highlights + semantic_highlights:
            all_times.append(h['timestamp'])

        if visual_samples:
            for sample in visual_samples:
                all_times.append(sample['timestamp'])

        if video_duration:
            max_time = video_duration
        elif all_times:
            max_time = max(all_times) + 1
        else:
            max_time = 0

        # Initialize bins
        time_bins = defaultdict(lambda: {
            'audio_score': 0.0,
            'visual_score': 0.0,
            'semantic_score': 0.0,
            'free_models_score': 0.0,
            'signals': [],
            'timestamp': 0.0
        })

        # Bin audio highlights
        for h in audio_highlights:
            bin_idx = int(h['timestamp'])
            time_bins[bin_idx]['audio_score'] = max(
                time_bins[bin_idx]['audio_score'],
                h['confidence']
            )
            time_bins[bin_idx]['signals'].append({
                'type': h['type'],
                'signal': 'audio',
                'confidence': h['confidence']
            })
            time_bins[bin_idx]['timestamp'] = h['timestamp']

        # Bin visual highlights
        for h in visual_highlights:
            bin_idx = int(h['timestamp'])
            time_bins[bin_idx]['visual_score'] = max(
                time_bins[bin_idx]['visual_score'],
                h['confidence']
            )
            time_bins[bin_idx]['signals'].append({
                'type': h['type'],
                'signal': 'visual',
                'confidence': h['confidence']
            })
            if time_bins[bin_idx]['timestamp'] == 0.0:
                time_bins[bin_idx]['timestamp'] = h['timestamp']

        # Bin semantic highlights
        for h in semantic_highlights:
            bin_idx = int(h['timestamp'])
            time_bins[bin_idx]['semantic_score'] = max(
                time_bins[bin_idx]['semantic_score'],
                h['confidence']
            )
            time_bins[bin_idx]['signals'].append({
                'type': h.get('type', 'semantic'),
                'signal': 'semantic',
                'confidence': h['confidence'],
                'reason': h.get('reason', '')
            })
            if time_bins[bin_idx]['timestamp'] == 0.0:
                time_bins[bin_idx]['timestamp'] = h['timestamp']

        # Bin FREE model samples (if provided)
        if visual_samples:
            for sample in visual_samples:
                bin_idx = int(sample['timestamp'])

                # Score based on quality and detected features
                quality = sample.get('quality', 0.0)
                has_objects = len(sample.get('objects', [])) > 0
                has_text = bool(sample.get('text_detected', ''))
                has_poses = len(sample.get('poses', [])) > 0
                has_emotions = len(sample.get('emotions', [])) > 0

                # Higher score if multiple features detected
                feature_score = (
                    (0.3 if has_objects else 0) +
                    (0.2 if has_text else 0) +
                    (0.3 if has_poses else 0) +
                    (0.2 if has_emotions else 0)
                )

                free_score = quality * 0.4 + feature_score * 0.6

                time_bins[bin_idx]['free_models_score'] = max(
                    time_bins[bin_idx]['free_models_score'],
                    free_score
                )
                if time_bins[bin_idx]['timestamp'] == 0.0:
                    time_bins[bin_idx]['timestamp'] = sample['timestamp']

        return time_bins

    def _rank_highlights(self, time_bins: Dict[int, Dict]) -> List[Dict]:
        """Calculate combined scores and rank highlights"""

        highlights = []

        for bin_idx, bin_data in time_bins.items():
            # Calculate weighted combined score
            combined_score = (
                bin_data['audio_score'] * self.audio_weight +
                bin_data['visual_score'] * self.visual_weight +
                bin_data['semantic_score'] * self.semantic_weight +
                bin_data['free_models_score'] * self.free_models_weight
            )

            # Only include if combined score > 0
            if combined_score > 0:
                highlights.append({
                    'timestamp': bin_data['timestamp'] if bin_data['timestamp'] > 0 else float(bin_idx),
                    'combined_score': combined_score,
                    'audio_score': bin_data['audio_score'],
                    'visual_score': bin_data['visual_score'],
                    'semantic_score': bin_data['semantic_score'],
                    'free_models_score': bin_data['free_models_score'],
                    'signals': bin_data['signals'],
                    'signal_count': len(bin_data['signals'])
                })

        # Sort by combined score (descending)
        highlights.sort(key=lambda x: x['combined_score'], reverse=True)

        return highlights

    def _calculate_signal_breakdown(
        self,
        audio_highlights: List[Dict],
        visual_highlights: List[Dict],
        semantic_highlights: List[Dict]
    ) -> Dict:
        """Calculate breakdown of signal contributions"""

        return {
            'audio': {
                'count': len(audio_highlights),
                'weight': self.audio_weight
            },
            'visual': {
                'count': len(visual_highlights),
                'weight': self.visual_weight
            },
            'semantic': {
                'count': len(semantic_highlights),
                'weight': self.semantic_weight
            },
            'free_models': {
                'weight': self.free_models_weight
            }
        }

    def get_top_highlights(
        self,
        highlights: List[Dict],
        top_n: int = 50,
        min_score: float = 0.3
    ) -> List[Dict]:
        """
        Get top N highlights above minimum score.

        Args:
            highlights: Ranked highlights from detect_highlights()
            top_n: Maximum number of highlights to return
            min_score: Minimum combined score threshold

        Returns:
            List of top highlights
        """
        # Filter by minimum score
        filtered = [h for h in highlights if h['combined_score'] >= min_score]

        # Return top N
        return filtered[:top_n]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with mock data
    detector = UniversalHighlightDetector()

    mock_audio = [
        {'timestamp': 10.5, 'type': 'volume_spike', 'confidence': 0.8, 'signal': 'audio'},
        {'timestamp': 25.3, 'type': 'pitch_variance', 'confidence': 0.7, 'signal': 'audio'}
    ]

    mock_visual = [
        {'timestamp': 10.2, 'type': 'motion_peak', 'confidence': 0.9, 'signal': 'visual'},
        {'timestamp': 45.8, 'type': 'color_variance', 'confidence': 0.6, 'signal': 'visual'}
    ]

    mock_semantic = [
        {'timestamp': 10.0, 'type': 'emphasis', 'confidence': 0.95, 'signal': 'semantic', 'reason': 'Strong emphasis'},
        {'timestamp': 60.0, 'type': 'topic_shift', 'confidence': 0.85, 'signal': 'semantic', 'reason': 'New topic'}
    ]

    result = detector.detect_highlights(
        mock_audio,
        mock_visual,
        mock_semantic,
        video_duration=120.0
    )

    print(f"\nDetected {result['total_highlights']} fused highlights")
    print("\nTop 5 highlights:")
    top = detector.get_top_highlights(result['highlights'], top_n=5)
    for i, h in enumerate(top, 1):
        print(f"{i}. {h['timestamp']:.1f}s - score: {h['combined_score']:.3f}")
        print(f"   Audio: {h['audio_score']:.2f} | Visual: {h['visual_score']:.2f} | Semantic: {h['semantic_score']:.2f}")
        print(f"   Signals: {h['signal_count']}")
