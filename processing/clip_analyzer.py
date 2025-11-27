"""
CLIP/SigLIP Analyzer for Audio-Visual Correspondence Detection

This module uses CLIP and SigLIP models to:
1. Generate visual embeddings for all frames
2. Generate text embeddings for transcript segments
3. Compute text-image similarity scores
4. Flag spurious candidates (semantic mismatches)
5. Detect visual anomalies (outlier frames)
6. Score frames for ontology-specific potential
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import cv2
from PIL import Image
from dataclasses import dataclass
import json
from loguru import logger

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    logger.warning("OpenCLIP not available. Install with: pip install open_clip_torch")


@dataclass
class FrameEmbedding:
    """Store frame embedding and metadata"""
    frame_id: int
    timestamp: float
    embedding: np.ndarray
    scene_type: Optional[str] = None
    has_ocr: bool = False
    object_count: int = 0


@dataclass
class TextEmbedding:
    """Store text embedding and metadata"""
    text: str
    start_time: float
    end_time: float
    embedding: np.ndarray
    speaker: Optional[str] = None


@dataclass
class SpuriousCandidate:
    """Spurious candidate detected by CLIP"""
    frame_id: int
    timestamp: float
    reason: str
    mismatch_score: float  # Higher = more mismatch
    text_segment: str
    expected_description: str
    actual_description: str
    confidence: float


class CLIPAnalyzer:
    """
    CLIP/SigLIP analyzer for semantic alignment detection
    """

    def __init__(
        self,
        model_name: str = "ViT-L/14",
        use_siglip: bool = False,
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Initialize CLIP analyzer

        Args:
            model_name: CLIP model name (ViT-B/32, ViT-L/14, etc.)
            use_siglip: Use SigLIP instead of CLIP (better text-image alignment)
            device: torch device (cpu, cuda, mps)
            batch_size: Batch size for processing (default: 32)
        """
        # Auto-detect best device: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("Using Apple Silicon MPS")
        else:
            self.device = "cpu"
            logger.warning("No GPU available, using CPU (will be slow)")

        self.model_name = model_name
        self.use_siglip = use_siglip
        self.batch_size = batch_size

        # Load model
        if use_siglip and OPEN_CLIP_AVAILABLE:
            logger.info(f"Loading SigLIP model: {model_name}")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-SO400M-14-SigLIP-384',
                pretrained='webli'
            )
            self.tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP-384')
            self.model = self.model.to(self.device)
        elif CLIP_AVAILABLE:
            logger.info(f"Loading CLIP model: {model_name}")
            self.model, self.preprocess = clip.load(model_name, device=self.device)
        else:
            raise RuntimeError("Neither CLIP nor OpenCLIP is available. Install dependencies first.")

        self.model.eval()

        # Ontology-specific prompts
        self.ontology_prompts = {
            "needle": [
                "a photo with small text visible",
                "a photo with specific numbers or labels",
                "a photo with a sign or badge",
                "a photo with fine details"
            ],
            "counting": [
                "a photo with multiple identical objects",
                "a photo with repeated elements",
                "a photo showing several items of the same type"
            ],
            "temporal": [
                "a photo showing a clear state change",
                "a photo showing before and after",
                "a photo showing a transformation"
            ],
            "inference": [
                "a photo showing cause and effect",
                "a photo where someone's intent is visible",
                "a photo showing reasoning or purpose"
            ],
            "comparative": [
                "a photo showing two different states",
                "a photo showing contrast or difference",
                "a photo comparing two elements"
            ],
            "object_interaction": [
                "a photo showing someone manipulating an object",
                "a photo showing object transformation",
                "a photo showing hands interacting with something"
            ]
        }

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """
        Encode image to CLIP embedding

        Args:
            image: OpenCV image (BGR)

        Returns:
            Normalized embedding vector
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Preprocess and encode
        image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.use_siglip and OPEN_CLIP_AVAILABLE:
                image_features = self.model.encode_image(image_tensor)
            else:
                image_features = self.model.encode_image(image_tensor)

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy()[0]

    def encode_images_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Encode multiple images to CLIP embeddings (BATCH PROCESSING)

        Args:
            images: List of OpenCV images (BGR)

        Returns:
            Array of normalized embedding vectors (shape: [N, embedding_dim])
        """
        all_embeddings = []

        # Process in batches
        for batch_start in range(0, len(images), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(images))
            batch_images = images[batch_start:batch_end]

            # Convert and preprocess batch
            batch_tensors = []
            for image in batch_images:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)

                # Preprocess
                image_tensor = self.preprocess(pil_image)
                batch_tensors.append(image_tensor)

            # Stack into batch
            batch_tensor = torch.stack(batch_tensors).to(self.device)

            # Encode batch
            with torch.no_grad():
                if self.use_siglip and OPEN_CLIP_AVAILABLE:
                    batch_features = self.model.encode_image(batch_tensor)
                else:
                    batch_features = self.model.encode_image(batch_tensor)

            # Normalize
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)

            # Add to results
            all_embeddings.append(batch_features.cpu().numpy())

        # Concatenate all batches
        return np.concatenate(all_embeddings, axis=0)

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to CLIP embedding

        Args:
            text: Text string

        Returns:
            Normalized embedding vector
        """
        if self.use_siglip and OPEN_CLIP_AVAILABLE:
            text_tensor = self.tokenizer([text]).to(self.device)
        else:
            text_tensor = clip.tokenize([text], truncate=True).to(self.device)

        with torch.no_grad():
            if self.use_siglip and OPEN_CLIP_AVAILABLE:
                text_features = self.model.encode_text(text_tensor)
            else:
                text_features = self.model.encode_text(text_tensor)

        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy()[0]

    def compute_similarity(self, image_emb: np.ndarray, text_emb: np.ndarray) -> float:
        """
        Compute cosine similarity between image and text embeddings

        Returns:
            Similarity score (0-1, higher = more similar)
        """
        similarity = np.dot(image_emb, text_emb)
        return float(similarity)

    def analyze_frames(
        self,
        frames: List[Dict],
        video_path: str,
        batch_size: int = 32
    ) -> List[FrameEmbedding]:
        """
        Generate embeddings for all frames with batching

        Args:
            frames: List of frame metadata dicts
            video_path: Path to video file
            batch_size: Number of frames to process in each batch

        Returns:
            List of FrameEmbedding objects
        """
        logger.info(f"Generating CLIP embeddings for {len(frames)} frames (batch_size={batch_size})...")

        frame_embeddings = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            return frame_embeddings

        # Process in batches
        for batch_idx in range(0, len(frames), batch_size):
            batch = frames[batch_idx:batch_idx + batch_size]
            batch_images = []
            batch_metadata = []

            # Read frames in batch
            for frame_meta in batch:
                frame_id = frame_meta.get('frame_id', frame_meta.get('frame_number', 0))
                timestamp = frame_meta.get('timestamp', 0)

                try:
                    # Seek to frame
                    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                    ret, frame = cap.read()

                    if not ret:
                        logger.warning(f"Failed to read frame at {timestamp}s")
                        continue

                    # Validate frame is not empty
                    if frame is None or frame.size == 0:
                        logger.warning(f"Empty frame at {timestamp}s")
                        continue

                    batch_images.append(frame)
                    batch_metadata.append({
                        'frame_id': frame_id,
                        'timestamp': timestamp,
                        'scene_type': frame_meta.get('scene_type'),
                        'has_ocr': bool(frame_meta.get('ocr_text')),
                        'object_count': len(frame_meta.get('objects', []))
                    })
                except Exception as e:
                    logger.warning(f"Error reading frame {frame_id} at {timestamp}s: {e}")
                    continue

            # Batch encode images
            if batch_images:
                try:
                    embeddings = self._encode_image_batch(batch_images)

                    # Create FrameEmbedding objects
                    for meta, embedding in zip(batch_metadata, embeddings):
                        frame_emb = FrameEmbedding(
                            frame_id=meta['frame_id'],
                            timestamp=meta['timestamp'],
                            embedding=embedding,
                            scene_type=meta['scene_type'],
                            has_ocr=meta['has_ocr'],
                            object_count=meta['object_count']
                        )
                        frame_embeddings.append(frame_emb)
                except Exception as e:
                    logger.error(f"Error encoding batch at index {batch_idx}: {e}")
                    # Continue to next batch instead of failing completely
                    continue

            # Progress logging
            progress = min(batch_idx + batch_size, len(frames))
            logger.info(f"Progress: {progress}/{len(frames)} frames processed ({progress*100//len(frames)}%)")

            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cap.release()
        logger.info(f"Generated {len(frame_embeddings)} frame embeddings")

        return frame_embeddings

    def _encode_image_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Encode multiple images in a batch

        Args:
            images: List of OpenCV images (BGR)

        Returns:
            List of normalized embedding vectors
        """
        # Preprocess all images
        image_tensors = []
        for i, image in enumerate(images):
            try:
                # Validate image
                if image is None or image.size == 0:
                    logger.warning(f"Skipping invalid image at index {i}")
                    continue

                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                image_tensor = self.preprocess(pil_image)
                image_tensors.append(image_tensor)
            except Exception as e:
                logger.warning(f"Error preprocessing image {i}: {e}")
                continue

        if not image_tensors:
            logger.error("No valid images to encode in batch")
            return []

        # Stack into batch
        batch_tensor = torch.stack(image_tensors).to(self.device)

        # Encode batch
        with torch.no_grad():
            if self.use_siglip and OPEN_CLIP_AVAILABLE:
                image_features = self.model.encode_image(batch_tensor)
            else:
                image_features = self.model.encode_image(batch_tensor)

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Convert to list of numpy arrays
        embeddings = image_features.cpu().numpy()

        return [emb for emb in embeddings]

    def analyze_transcript(
        self,
        transcript_segments: List[Dict]
    ) -> List[TextEmbedding]:
        """
        Generate embeddings for transcript segments

        Args:
            transcript_segments: List of transcript segment dicts

        Returns:
            List of TextEmbedding objects
        """
        logger.info(f"Generating text embeddings for {len(transcript_segments)} segments...")

        text_embeddings = []

        for segment in transcript_segments:
            text = segment.get('text', '').strip()
            if not text:
                continue

            embedding = self.encode_text(text)

            text_emb = TextEmbedding(
                text=text,
                start_time=segment.get('start', 0),
                end_time=segment.get('end', 0),
                embedding=embedding,
                speaker=segment.get('speaker')
            )

            text_embeddings.append(text_emb)

        logger.info(f"Generated {len(text_embeddings)} text embeddings")

        return text_embeddings

    def detect_spurious_candidates(
        self,
        frame_embeddings: List[FrameEmbedding],
        text_embeddings: List[TextEmbedding],
        threshold: float = 0.15
    ) -> List[SpuriousCandidate]:
        """
        Detect frames where audio and visual semantics mismatch

        Args:
            frame_embeddings: Frame embeddings
            text_embeddings: Text embeddings
            threshold: Similarity threshold (lower = more mismatch)

        Returns:
            List of spurious candidates
        """
        logger.info("Detecting spurious candidates (audio-visual mismatches)...")

        spurious_candidates = []

        for frame_emb in frame_embeddings:
            # Find overlapping text segments
            overlapping_text = [
                text_emb for text_emb in text_embeddings
                if text_emb.start_time <= frame_emb.timestamp <= text_emb.end_time
            ]

            if not overlapping_text:
                continue

            # Compute similarity with each text segment
            for text_emb in overlapping_text:
                similarity = self.compute_similarity(
                    frame_emb.embedding,
                    text_emb.embedding
                )

                # Low similarity = potential spurious correlation
                if similarity < threshold:
                    mismatch_score = 1.0 - similarity

                    spurious = SpuriousCandidate(
                        frame_id=frame_emb.frame_id,
                        timestamp=frame_emb.timestamp,
                        reason="Low text-image similarity suggests semantic mismatch",
                        mismatch_score=mismatch_score,
                        text_segment=text_emb.text,
                        expected_description="(Audio suggests one thing)",
                        actual_description="(Visual shows something different)",
                        confidence=mismatch_score
                    )

                    spurious_candidates.append(spurious)

        # Sort by mismatch score (highest first)
        spurious_candidates.sort(key=lambda x: x.mismatch_score, reverse=True)

        logger.info(f"Detected {len(spurious_candidates)} spurious candidates")

        return spurious_candidates

    def detect_visual_anomalies(
        self,
        frame_embeddings: List[FrameEmbedding],
        window_size: int = 10,
        threshold: float = 0.6
    ) -> List[int]:
        """
        Detect frames that look different from their neighbors

        Args:
            frame_embeddings: Frame embeddings
            window_size: Number of neighbors to compare
            threshold: Similarity threshold (lower = more different)

        Returns:
            List of anomaly frame IDs
        """
        logger.info("Detecting visual anomalies (outlier frames)...")

        anomalies = []

        for i, frame_emb in enumerate(frame_embeddings):
            # Get neighbors
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(frame_embeddings), i + window_size // 2 + 1)
            neighbors = frame_embeddings[start_idx:end_idx]

            # Compute average similarity to neighbors
            similarities = []
            for neighbor in neighbors:
                if neighbor.frame_id == frame_emb.frame_id:
                    continue
                sim = self.compute_similarity(
                    frame_emb.embedding,
                    neighbor.embedding
                )
                similarities.append(sim)

            if similarities:
                avg_similarity = np.mean(similarities)

                # Low similarity = anomaly
                if avg_similarity < threshold:
                    anomalies.append(frame_emb.frame_id)

        logger.info(f"Detected {len(anomalies)} visual anomalies")

        return anomalies

    def score_ontology_potential(
        self,
        frame_embeddings: List[FrameEmbedding]
    ) -> Dict[int, Dict[str, float]]:
        """
        Score each frame's potential for different ontology types

        Args:
            frame_embeddings: Frame embeddings

        Returns:
            Dict mapping frame_id to {ontology_type: score}
        """
        logger.info("Scoring frames for ontology potential...")

        scores = {}

        # Encode all ontology prompts once
        ontology_embeddings = {}
        for ontology, prompts in self.ontology_prompts.items():
            prompt_embeddings = [self.encode_text(prompt) for prompt in prompts]
            ontology_embeddings[ontology] = prompt_embeddings

        # Score each frame
        for frame_emb in frame_embeddings:
            frame_scores = {}

            for ontology, prompt_embeddings in ontology_embeddings.items():
                # Compute max similarity across all prompts for this ontology
                similarities = [
                    self.compute_similarity(frame_emb.embedding, prompt_emb)
                    for prompt_emb in prompt_embeddings
                ]
                frame_scores[ontology] = float(np.max(similarities))

            scores[frame_emb.frame_id] = frame_scores

        logger.info(f"Scored {len(scores)} frames for ontology potential")

        return scores

    def save_analysis(
        self,
        output_path: str,
        frame_embeddings: List[FrameEmbedding],
        text_embeddings: List[TextEmbedding],
        spurious_candidates: List[SpuriousCandidate],
        visual_anomalies: List[int],
        ontology_scores: Dict[int, Dict[str, float]]
    ):
        """
        Save analysis results to JSON

        Args:
            output_path: Path to save JSON
            frame_embeddings: Frame embeddings (embeddings will be omitted from JSON)
            text_embeddings: Text embeddings (embeddings will be omitted from JSON)
            spurious_candidates: Spurious candidates
            visual_anomalies: Anomaly frame IDs
            ontology_scores: Ontology scores
        """
        results = {
            "frame_count": len(frame_embeddings),
            "text_segment_count": len(text_embeddings),
            "spurious_candidates": [
                {
                    "frame_id": sc.frame_id,
                    "timestamp": sc.timestamp,
                    "reason": sc.reason,
                    "mismatch_score": sc.mismatch_score,
                    "text_segment": sc.text_segment,
                    "confidence": sc.confidence
                }
                for sc in spurious_candidates
            ],
            "visual_anomalies": visual_anomalies,
            "ontology_scores": ontology_scores
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved CLIP analysis to {output_path}")


def run_clip_analysis(
    video_path: str,
    frames_metadata: List[Dict],
    transcript_segments: List[Dict],
    output_path: str,
    use_siglip: bool = True
) -> Dict:
    """
    Run full CLIP analysis pipeline

    Args:
        video_path: Path to video file
        frames_metadata: List of frame metadata dicts
        transcript_segments: List of transcript segment dicts
        output_path: Path to save results
        use_siglip: Use SigLIP instead of CLIP

    Returns:
        Analysis results dict
    """
    analyzer = CLIPAnalyzer(use_siglip=use_siglip)

    # Generate embeddings
    frame_embeddings = analyzer.analyze_frames(frames_metadata, video_path)
    text_embeddings = analyzer.analyze_transcript(transcript_segments)

    # Detect spurious candidates
    spurious_candidates = analyzer.detect_spurious_candidates(
        frame_embeddings,
        text_embeddings,
        threshold=0.15
    )

    # Detect visual anomalies
    visual_anomalies = analyzer.detect_visual_anomalies(
        frame_embeddings,
        window_size=10,
        threshold=0.6
    )

    # Score ontology potential
    ontology_scores = analyzer.score_ontology_potential(frame_embeddings)

    # Save results
    analyzer.save_analysis(
        output_path,
        frame_embeddings,
        text_embeddings,
        spurious_candidates,
        visual_anomalies,
        ontology_scores
    )

    return {
        "frame_embeddings": frame_embeddings,
        "text_embeddings": text_embeddings,
        "spurious_candidates": spurious_candidates,
        "visual_anomalies": visual_anomalies,
        "ontology_scores": ontology_scores
    }
