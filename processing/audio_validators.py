"""
Audio Timestamp Validators

Two validation methods:
1. Librosa Feature Validation - Verify timestamps point to valid audio
2. Audio Perceptual Hashing - Detect duplicate audio segments
"""

import logging
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import hashlib

logger = logging.getLogger(__name__)


class AudioTimestampValidator:
    """
    Validate audio timestamps using librosa feature analysis

    Checks:
    - Segment is not silent
    - No drastic energy changes (scene cuts)
    - Reasonable audio characteristics
    """

    # Thresholds
    MIN_RMS_ENERGY = 0.01      # Minimum energy (below = silence)
    MIN_SEGMENT_DURATION = 0.1 # Minimum segment length (seconds)
    MAX_ENERGY_CHANGE = 3.0    # Max energy change ratio (scene cut detection)

    def __init__(self, audio_path: Path):
        """
        Initialize validator

        Args:
            audio_path: Path to audio file (WAV format)
        """
        self.audio_path = audio_path
        self.y = None
        self.sr = None

        # Lazy load audio (only when needed)
        self._audio_loaded = False

    def _load_audio(self):
        """Load audio file using librosa (lazy loading)"""
        if not self._audio_loaded:
            try:
                import librosa
                logger.debug(f"Loading audio: {self.audio_path}")
                self.y, self.sr = librosa.load(str(self.audio_path), sr=16000)
                self._audio_loaded = True
                logger.debug(f"  ✓ Audio loaded: duration={len(self.y)/self.sr:.1f}s")
            except ImportError:
                logger.error("librosa not installed. Run: pip install librosa")
                raise
            except Exception as e:
                logger.error(f"Failed to load audio: {e}")
                raise

    def validate_timestamp(
        self,
        start_time: float,
        end_time: float,
        tolerance: float = 0.5
    ) -> Tuple[bool, float, str]:
        """
        Validate that audio segment at given timestamps is valid

        Args:
            start_time: Start timestamp (seconds)
            end_time: End timestamp (seconds)
            tolerance: Time tolerance for checking neighbors (seconds)

        Returns:
            Tuple of (is_valid: bool, confidence: float, reason: str)
        """
        import librosa

        self._load_audio()

        # Extract segment
        start_sample = int(start_time * self.sr)
        end_sample = int(end_time * self.sr)

        # Boundary checks
        if start_sample < 0:
            return False, 0.0, "Start time before audio begins"

        if end_sample > len(self.y):
            return False, 0.0, "End time after audio ends"

        segment = self.y[start_sample:end_sample]
        duration = (end_sample - start_sample) / self.sr

        # Check 1: Segment duration
        if duration < self.MIN_SEGMENT_DURATION:
            return False, 0.0, f"Segment too short ({duration:.2f}s < {self.MIN_SEGMENT_DURATION}s)"

        # Check 2: Energy (not silent)
        rms = librosa.feature.rms(y=segment)[0]
        avg_rms = np.mean(rms)

        if avg_rms < self.MIN_RMS_ENERGY:
            return False, 0.0, f"Silent segment (RMS={avg_rms:.4f} < {self.MIN_RMS_ENERGY})"

        # Check 3: Zero crossing rate (reasonable audio characteristics)
        zcr = librosa.feature.zero_crossing_rate(segment)[0]
        avg_zcr = np.mean(zcr)

        confidence = 1.0
        reasons = []

        # Low energy warning
        if avg_rms < 0.05:
            confidence *= 0.7
            reasons.append(f"Low energy (RMS={avg_rms:.3f})")

        # Unusual ZCR warning
        if avg_zcr < 0.01 or avg_zcr > 0.5:
            confidence *= 0.8
            reasons.append(f"Unusual zero-crossing rate ({avg_zcr:.3f})")

        # Check 4: Continuity (no drastic changes = scene cuts)
        # Split segment into windows and check energy profile
        window_size = int(self.sr * 1.0)  # 1-second windows
        if len(segment) >= window_size * 2:
            energy_profile = []

            for i in range(0, len(segment) - window_size, window_size):
                window = segment[i:i+window_size]
                window_rms = np.mean(librosa.feature.rms(y=window)[0])
                energy_profile.append(window_rms)

            if len(energy_profile) > 1:
                # Check for drastic changes
                max_change_ratio = 0.0
                for i in range(1, len(energy_profile)):
                    ratio = abs(energy_profile[i] - energy_profile[i-1]) / (energy_profile[i-1] + 1e-10)
                    max_change_ratio = max(max_change_ratio, ratio)

                if max_change_ratio > self.MAX_ENERGY_CHANGE:
                    confidence *= 0.5
                    reasons.append(f"Drastic energy change (ratio={max_change_ratio:.1f})")

                    # If too drastic, mark as invalid (likely scene cut)
                    if max_change_ratio > self.MAX_ENERGY_CHANGE * 2:
                        return False, confidence, f"Scene cut detected (energy change ratio={max_change_ratio:.1f})"

        # Check 5: Neighbor continuity (segment fits with surrounding audio)
        if tolerance > 0:
            # Check before segment
            before_start = max(0, start_time - tolerance)
            before_end = start_time

            if before_end > before_start:
                before_segment = self.y[int(before_start*self.sr):int(before_end*self.sr)]

                if len(before_segment) > self.sr * 0.1:
                    before_rms = np.mean(librosa.feature.rms(y=before_segment)[0])
                    rms_change = abs(avg_rms - before_rms) / (avg_rms + 1e-10)

                    if rms_change > 5.0:
                        confidence *= 0.8
                        reasons.append(f"Large energy jump from previous segment (ratio={rms_change:.1f})")

            # Check after segment
            after_start = end_time
            after_end = min(len(self.y)/self.sr, end_time + tolerance)

            if after_end > after_start:
                after_segment = self.y[int(after_start*self.sr):int(after_end*self.sr)]

                if len(after_segment) > self.sr * 0.1:
                    after_rms = np.mean(librosa.feature.rms(y=after_segment)[0])
                    rms_change = abs(avg_rms - after_rms) / (avg_rms + 1e-10)

                    if rms_change > 5.0:
                        confidence *= 0.8
                        reasons.append(f"Large energy jump to next segment (ratio={rms_change:.1f})")

        # Final verdict
        is_valid = confidence >= 0.6
        reason = "; ".join(reasons) if reasons else "Valid audio segment"

        return is_valid, confidence, reason


class AudioPerceptualHash:
    """
    Audio perceptual hashing for duplicate detection

    Similar to imagehash - creates fingerprint of audio segment
    that's similar for similar audio.
    """

    def __init__(self, audio_path: Path, hash_size: int = 8):
        """
        Initialize audio hasher

        Args:
            audio_path: Path to audio file
            hash_size: Hash matrix size (default: 8x8 = 64 bits)
        """
        self.audio_path = audio_path
        self.hash_size = hash_size
        self.y = None
        self.sr = None
        self._audio_loaded = False

    def _load_audio(self):
        """Load audio file (lazy loading)"""
        if not self._audio_loaded:
            try:
                import librosa
                logger.debug(f"Loading audio for hashing: {self.audio_path}")
                self.y, self.sr = librosa.load(str(self.audio_path), sr=16000)
                self._audio_loaded = True
            except Exception as e:
                logger.error(f"Failed to load audio: {e}")
                raise

    def compute_hash(self, start_time: float, end_time: float) -> str:
        """
        Compute perceptual hash for audio segment

        Args:
            start_time: Start timestamp (seconds)
            end_time: End timestamp (seconds)

        Returns:
            Hex string hash (like imagehash)
        """
        import librosa

        self._load_audio()

        # Extract segment
        start_sample = int(start_time * self.sr)
        end_sample = int(end_time * self.sr)
        segment = self.y[start_sample:end_sample]

        if len(segment) < self.sr * 0.1:
            # Too short - return zero hash
            return "0" * (self.hash_size * self.hash_size // 4)

        # Compute spectrogram
        S = np.abs(librosa.stft(segment, n_fft=512, hop_length=256))

        # Reduce to hash_size x hash_size using bilinear interpolation
        # This makes hash robust to small timing variations
        from scipy.ndimage import zoom

        scale_factors = (self.hash_size / S.shape[0], self.hash_size / S.shape[1])
        S_small = zoom(S, scale_factors, order=1)  # Bilinear interpolation

        # Threshold by median (perceptual hashing core idea)
        median = np.median(S_small)
        hash_bits = (S_small > median).flatten()

        # Convert to hex string
        hash_bytes = np.packbits(hash_bits).tobytes()
        hash_hex = hashlib.md5(hash_bytes).hexdigest()

        return hash_hex

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """
        Compute Hamming distance between two hashes

        Args:
            hash1: First hash (hex string)
            hash2: Second hash (hex string)

        Returns:
            Number of differing bits
        """
        # Convert hex to binary
        bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
        bin2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)

        # Count differing bits
        distance = sum(b1 != b2 for b1, b2 in zip(bin1, bin2))

        return distance

    @staticmethod
    def similarity(hash1: str, hash2: str, max_bits: int = 128) -> float:
        """
        Compute similarity score between two hashes

        Args:
            hash1: First hash
            hash2: Second hash
            max_bits: Maximum possible distance (default: 128 for MD5)

        Returns:
            Similarity score (0.0 = completely different, 1.0 = identical)
        """
        distance = AudioPerceptualHash.hamming_distance(hash1, hash2)
        return 1.0 - (distance / max_bits)


def validate_question_audio(
    audio_path: Path,
    start_time: float,
    end_time: float
) -> Tuple[bool, float, str]:
    """
    Convenience function: Validate single question's audio timestamps

    Args:
        audio_path: Path to audio file
        start_time: Question start timestamp
        end_time: Question end timestamp

    Returns:
        (is_valid, confidence, reason)
    """
    validator = AudioTimestampValidator(audio_path)
    return validator.validate_timestamp(start_time, end_time)


def detect_duplicate_audio_segments(
    audio_path: Path,
    questions: List[dict],
    similarity_threshold: float = 0.85
) -> List[Tuple[int, int, float]]:
    """
    Detect questions using very similar audio segments

    Args:
        audio_path: Path to audio file
        questions: List of question dicts with start_timestamp, end_timestamp
        similarity_threshold: Similarity above which segments are considered duplicates

    Returns:
        List of (question_idx1, question_idx2, similarity) tuples
    """
    hasher = AudioPerceptualHash(audio_path)

    # Compute hashes for all questions
    hashes = []
    logger.info(f"Computing audio hashes for {len(questions)} questions...")

    for i, q in enumerate(questions):
        start = q.get('start_timestamp', q.get('timestamp', 0))
        end = q.get('end_timestamp', start + 10.0)

        try:
            audio_hash = hasher.compute_hash(start, end)
            hashes.append(audio_hash)
        except Exception as e:
            logger.warning(f"Failed to hash question {i}: {e}")
            hashes.append("0" * 32)  # Zero hash for failed segments

    # Compare all pairs
    duplicates = []
    logger.info("Comparing audio hashes for duplicates...")

    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            similarity = AudioPerceptualHash.similarity(hashes[i], hashes[j])

            if similarity > similarity_threshold:
                duplicates.append((i, j, similarity))
                logger.debug(f"  Duplicate audio: Q{i} ↔ Q{j} (similarity={similarity:.2f})")

    return duplicates


def validate_audio_cue_content(audio_cue: str) -> Tuple[bool, str]:
    """
    Critical Gap #4: Validate that audio cue is meaningful, not placeholder or vague.

    Checks for:
    - Placeholder text like "{audio}", "{AUDIO}", "XXX"
    - Vague descriptions like "Audio occurs at this moment"
    - Empty or whitespace-only content

    Args:
        audio_cue: The audio_cue field to validate

    Returns:
        (is_valid, rejection_reason)
    """
    if not audio_cue or not audio_cue.strip():
        return False, "Audio cue is empty"

    audio_cue_lower = audio_cue.lower().strip()

    # Check for placeholder patterns
    PLACEHOLDER_PATTERNS = [
        '{audio}', '{AUDIO}', '<audio>', '[audio]',
        'xxx', 'tbd', 'todo', 'placeholder',
        '{cue}', '{audio_cue}', 'n/a', 'none'
    ]

    for pattern in PLACEHOLDER_PATTERNS:
        if pattern.lower() in audio_cue_lower:
            return False, f"Audio cue contains placeholder: '{pattern}'"

    # Check for vague descriptions (exact matches and substrings)
    VAGUE_DESCRIPTIONS = [
        'audio occurs at this moment',
        'audio occurs here',
        'sound occurs at this moment',
        'audio happens at this time',
        'audio plays at this time',
        'the audio',  # Too vague if this is the entire cue
        'some audio',
        'background audio',
        'audio is heard',
        'audio can be heard',
        'at this moment',
        'at this time',
        'during this time',
    ]

    # Check exact match or if cue is just the vague phrase
    if audio_cue_lower in VAGUE_DESCRIPTIONS:
        return False, f"Audio cue is too vague: '{audio_cue}'"

    # Check for extremely short cues (likely placeholder)
    if len(audio_cue.strip()) < 5:
        return False, f"Audio cue too short to be meaningful: '{audio_cue}'"

    # Check for minimum word count (should describe actual audio)
    words = audio_cue.split()
    if len(words) < 2:
        return False, f"Audio cue too brief (only {len(words)} word): '{audio_cue}'"

    # All checks passed
    return True, ""


def check_audio_cue_quality(questions: List[dict]) -> Tuple[bool, List[str]]:
    """
    Check all questions for audio cue quality issues.

    Args:
        questions: List of question dicts with audio_cue fields

    Returns:
        (has_issues, warning_messages)
    """
    warnings = []
    has_issues = False

    for i, q in enumerate(questions):
        question_id = q.get('question_id', f'Q{i+1}')
        audio_cue = q.get('audio_cue', '')

        is_valid, reason = validate_audio_cue_content(audio_cue)

        if not is_valid:
            has_issues = True
            warnings.append(f"   {question_id}: {reason}")

    return has_issues, warnings


def validate_audio_modality_diversity(
    questions: List[dict],
    min_modalities: int = 2
) -> Tuple[bool, List[str]]:
    """
    Validate questions use diverse audio modalities per Guidelines requirements.

    Guidelines doc lines 19-22: "For audio cues, do not just take words being said
    but also background sounds like environmental sounds (bird chirping), background
    music (piano playing), sudden change in tone/pitch of music, audience clapping etc."

    Audio modalities:
    - SPEECH: dialogue, narration, words, phrases
    - MUSIC: background music, tempo changes, tone shifts
    - SOUND_EFFECTS: environmental sounds, impacts, mechanical sounds
    - SILENCE: pauses, gaps, quiet moments
    - CROWD: applause, cheering, audience reactions

    Args:
        questions: List of question dicts with audio_cue fields
        min_modalities: Minimum number of different modalities required (default 2)

    Returns:
        (is_valid, warnings) - True if diversity requirements met
    """
    MODALITY_PATTERNS = {
        'speech': [
            'says', 'said', 'narrator', 'narration', 'dialogue', 'speaks',
            'voice', 'words', 'phrase', 'tells', 'asks', 'responds',
            'announces', 'mentions', 'states', 'exclaims', 'whispers', 'shouts'
        ],
        'music': [
            'tempo', 'music', 'bpm', 'melody', 'beat', 'rhythm', 'song',
            'tune', 'soundtrack', 'bass', 'drums', 'guitar', 'piano',
            'instrumental', 'pitch', 'tone shift', 'musical'
        ],
        'sound_effect': [
            'sound', 'crash', 'click', 'whoosh', 'bang', 'impact', 'noise',
            'thud', 'splash', 'rustle', 'creak', 'slam', 'beep', 'buzz',
            'swoosh', 'clang', 'chirping', 'environmental sound', 'mechanical'
        ],
        'silence': [
            'silence', 'silent', 'pause', 'quiet', 'gap', 'still', 'no sound',
            'absence of', 'stops playing', 'fades out'
        ],
        'crowd': [
            'applause', 'clapping', 'cheer', 'cheering', 'crowd', 'audience',
            'roar', 'booing', 'ovation', 'fans', 'spectators'
        ],
    }

    modalities_used = set()
    modality_counts = {m: 0 for m in MODALITY_PATTERNS}
    question_modalities = {}  # Track which modality each question uses

    for i, q in enumerate(questions):
        question_id = q.get('question_id', f'Q{i+1}')
        audio_cue = q.get('audio_cue', '').lower()

        # Detect modality for this question
        detected_modality = None
        for modality, patterns in MODALITY_PATTERNS.items():
            if any(p in audio_cue for p in patterns):
                modalities_used.add(modality)
                modality_counts[modality] += 1
                detected_modality = modality
                break

        question_modalities[question_id] = detected_modality

    warnings = []

    # Check minimum diversity
    if len(modalities_used) < min_modalities:
        warnings.append(
            f"⚠️ AUDIO DIVERSITY: Only {len(modalities_used)} audio modality(ies) used: {modalities_used}. "
            f"Need at least {min_modalities} different modalities."
        )
        warnings.append(
            f"   Guidelines requirement: 'Do not just take words being said but also "
            f"background sounds, music, sound effects, silence, etc.'"
        )

    # Check for all-speech (common anti-pattern)
    if modalities_used == {'speech'} or (len(modalities_used) == 1 and 'speech' in modalities_used):
        warnings.append(
            "⚠️ SPEECH-ONLY: All questions use speech-only audio cues. "
            "Add at least 1 non-speech question (music/sound/silence/crowd)."
        )

    # Provide modality breakdown
    if len(modalities_used) > 0:
        breakdown = ", ".join([f"{m}: {modality_counts[m]}Q" for m in sorted(modalities_used)])
        warnings.append(f"   Audio modality distribution: {breakdown}")

    is_valid = len(modalities_used) >= min_modalities
    return is_valid, warnings
