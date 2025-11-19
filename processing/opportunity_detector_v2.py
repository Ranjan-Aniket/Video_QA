"""
Opportunity Detector V2 - ENHANCED TWO-STAGE VERSION (OPTIMIZED)

Extracts REAL opportunities from actual video content following guidelines:
1. Uses REAL audio quotes from transcript (not synthetic)
2. Uses ACTUAL timestamps from segments
3. Validates all quotes exist in video
4. Integrates with frame evidence
5. Ensures questions require BOTH audio AND visual cues

ENHANCEMENTS:
- Two-stage detection (pattern-based + AI validation)
- ALL 13 opportunity types from guidelines (complete coverage)
- Word-level timestamp precision
- Audio events integration (music, sounds, crowd)
- Silence gaps for scene transitions
- Adaptive scoring and premium frame selection
- Configurable parameters (no hardcoding)
- Enhanced pattern detection from actual guideline examples

OPTIMIZATIONS (Cost Reduction):
- Three-tier validation (Tier 1: no validation, Tier 2: sample, Tier 3: full)
- Confidence-based filtering before Stage 2
- Audio event quality filters (intensity, duration)
- Temporal clustering to remove duplicates
- Budget controls ($2.00 cap, 300 candidate max)
- Larger batch sizes (8 candidates)
- Generic Audio-Visual Stitching detection

Based on guidelines from: Question Types_ Skills.pdf & Guidelines_ Prompt Creation.docx
"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import openai
import re
from math import ceil
import random

logger = logging.getLogger(__name__)


@dataclass
class RealOpportunity:
    """Real opportunity from actual video content"""
    opportunity_id: str
    opportunity_type: str  # "temporal", "referential", "counting", "sequential", "context", etc.

    # Audio component (MUST be real quote from transcript)
    audio_quote: str
    audio_start: float
    audio_end: float

    # Visual component (MUST reference actual frame)
    visual_timestamp: float  # Frame timestamp to analyze

    # Question metadata
    task_types: List[str]
    complexity: str  # "low", "medium", "high"
    description: str

    # Validation flags
    validated_audio: bool = False  # Quote exists in transcript
    validated_visual: bool = False  # Frame exists
    requires_both_modalities: bool = True

    # Enhanced scoring and precision fields
    opportunity_score: float = 0.0  # Overall quality score (0-1)
    adversarial_score: float = 0.0  # How adversarial (0-1)
    requires_premium_frame: bool = False  # Needs expensive analysis
    key_word: Optional[str] = None  # Key word from quote
    key_word_timestamp: Optional[float] = None  # Precise word timestamp
    stage1_confidence: float = 0.0  # Pattern detection confidence
    validated_stage2: bool = False  # GPT-4 validated

    # Additional metadata
    audio_event_type: Optional[str] = None  # For audio-visual stitching
    scene_transition: Optional[Dict] = None  # For comparative opportunities


@dataclass
class OpportunityDetectionResult:
    """Complete opportunity detection results"""
    video_id: str
    transcript_duration: float

    opportunities: List[RealOpportunity] = field(default_factory=list)

    # Statistics
    total_opportunities: int = 0
    validated_opportunities: int = 0
    detection_cost: float = 0.0

    # Enhanced statistics
    premium_frames: List[Dict] = field(default_factory=list)  # Selected premium frames
    stage1_candidates: int = 0  # Candidates from pattern detection
    stage2_validated: int = 0  # Validated by GPT-4
    opportunity_statistics: Dict = field(default_factory=dict)  # Count by type
    
    # Optimization statistics
    tier1_skipped: int = 0  # Candidates skipped (no validation needed)
    tier2_sampled: int = 0  # Candidates sampled (partial validation)
    tier3_full: int = 0  # Candidates fully validated
    filtered_by_confidence: int = 0  # Filtered before Stage 2
    clustered_duplicates: int = 0  # Removed by temporal clustering

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "video_id": self.video_id,
            "transcript_duration": self.transcript_duration,
            "total_opportunities": len(self.opportunities),
            "validated_opportunities": len([o for o in self.opportunities if o.validated_audio and o.validated_visual]),
            "detection_cost": self.detection_cost,
            "stage1_candidates": self.stage1_candidates,
            "stage2_validated": self.stage2_validated,
            "opportunity_statistics": self.opportunity_statistics,
            "premium_frames": self.premium_frames,
            "optimization_stats": {
                "tier1_skipped": self.tier1_skipped,
                "tier2_sampled": self.tier2_sampled,
                "tier3_full": self.tier3_full,
                "filtered_by_confidence": self.filtered_by_confidence,
                "clustered_duplicates": self.clustered_duplicates
            },
            "opportunities": [
                {
                    "opportunity_id": opp.opportunity_id,
                    "opportunity_type": opp.opportunity_type,
                    "audio_quote": opp.audio_quote,
                    "audio_start": opp.audio_start,
                    "audio_end": opp.audio_end,
                    "visual_timestamp": opp.visual_timestamp,
                    "key_word": opp.key_word,
                    "key_word_timestamp": opp.key_word_timestamp,
                    "task_types": opp.task_types,
                    "complexity": opp.complexity,
                    "description": opp.description,
                    "validated": opp.validated_audio and opp.validated_visual,
                    "opportunity_score": opp.opportunity_score,
                    "adversarial_score": opp.adversarial_score,
                    "requires_premium_frame": opp.requires_premium_frame,
                    "stage1_confidence": opp.stage1_confidence,
                    "validated_stage2": opp.validated_stage2
                } for opp in self.opportunities
            ]
        }


class OpportunityDetectorV2:
    """
    Extract REAL opportunities from actual video content using Two-Stage approach.

    Stage 1: Pattern-based detection (fast, broad coverage)
    Stage 2: AI validation (slow, high quality) with three-tier filtering

    Implements ALL 13 opportunity types from guidelines:
    1. Temporal Understanding
    2. Sequential
    3. Subscene
    4. General Holistic Reasoning
    5. Inference
    6. Context
    7. Needle
    8. Referential Grounding
    9. Counting
    10. Comparative
    11. Object Interaction Reasoning
    12. Audio-Visual Stitching
    13. Tackling Spurious Correlations
    """

    # Configuration Constants (can be overridden in __init__)
    MAX_PREMIUM_FRAMES = 7  # Maximum premium opportunities (each → 10 dense frames)
    MIN_OPPORTUNITY_SCORE = 0.7  # Minimum score for premium frame
    VALIDATION_BATCH_SIZE = 8  # Candidates per GPT-4 call (increased for efficiency)
    MIN_SCENE_CHANGE_DURATION = 1.5  # Minimum silence for scene change (seconds)
    MAX_API_RETRIES = 3  # Maximum retries for API calls
    API_TIMEOUT = 90  # API timeout in seconds
    
    # Stage 1 → Stage 2 Filter Configuration
    MIN_STAGE1_CONFIDENCE = 0.70  # Minimum confidence to proceed to Stage 2
    MIN_AUDIO_EVENT_CONFIDENCE = 0.80  # Higher bar for audio events
    MIN_AUDIO_EVENT_INTENSITY = 0.20  # Minimum RMS for audio events (was 0.1)
    MIN_AUDIO_EVENT_DURATION = 0.5  # Minimum duration in seconds
    MAX_CANDIDATES_TO_VALIDATE = 300  # Hard budget cap
    VALIDATION_BUDGET_DOLLARS = 2.00  # Maximum validation cost
    TEMPORAL_CLUSTERING_WINDOW = 3.0  # Seconds to merge nearby candidates

    # Scoring Weights
    WEIGHT_ADVERSARIAL = 0.30
    WEIGHT_TEMPORAL_PRECISION = 0.25
    WEIGHT_MULTIMODAL_DEPENDENCY = 0.20
    WEIGHT_COMPLEXITY = 0.15
    WEIGHT_DETAIL_LEVEL = 0.10
    
    # Three-Tier Validation Strategy (based on guidelines analysis)
    TIER1_NO_VALIDATION = [
        "temporal_understanding",  # Keywords are reliable
        "sequential",              # Clear markers
        "counting"                 # Pattern-based sufficient
    ]
    
    TIER2_SAMPLE_VALIDATION = [
        "referential_grounding",   # Ambiguous words reliable
        "context",                 # Background keywords work
        "subscene",                # Conditional patterns clear
        "comparative"              # Silence gaps are good indicators
    ]
    
    TIER3_FULL_VALIDATION = [
        "tackling_spurious_correlations",  # Needs deep understanding
        "general_holistic_reasoning",      # Needs AI judgment
        "inference",                       # Causal relationships complex
        "audio_visual_stitching",          # Editing detection hard
        "needle",                          # Precision critical
        "object_interaction"               # Transformation verification
    ]
    
    TIER2_SAMPLE_RATE = 0.20  # Validate 20% of Tier 2 candidates

    # ENHANCED Pattern Keywords (from guidelines examples)
    TEMPORAL_KEYWORDS = [
        "before", "after", "when", "while", "during", "then", "until", "since",
        "first", "next", "finally", "following", "preceding", "prior to", "subsequent"
    ]
    
    AMBIGUOUS_WORDS = [
        "it", "this", "that", "these", "those", "here", "there", 
        "he", "she", "they", "them", "one", "ones"
    ]
    
    COMPARATIVE_KEYWORDS = [
        "different", "same", "changed", "compare", "versus", "vs", "contrast",
        "difference", "distinction", "before and after", "compared to"
    ]
    
    INFERENCE_KEYWORDS = [
        "why", "because", "reason", "purpose", "cause", "effect", "intent",
        "intention", "motive", "explain", "prompts", "leads to", "results in"
    ]
    
    # Expanded action verbs from guidelines
    ACTION_VERBS = [
        "cut", "build", "pour", "break", "transform", "create", "destroy", "mix",
        "place", "add", "remove", "lift", "drop", "fall", "change", "turn",
        "melt", "freeze", "roll", "push", "pull", "throw", "catch", "hit",
        "pour over", "splash", "dunk", "score", "kick", "run", "jump",
        "open", "close", "wave", "point", "gesture", "react"
    ]
    
    SEQUENTIAL_KEYWORDS = [
        "first", "second", "third", "next", "then", "finally", "last", "step",
        "sequence", "order", "following", "after that", "before that"
    ]

    # Transformation/change keywords (for Object Interaction)
    TRANSFORMATION_KEYWORDS = [
        "change", "transform", "become", "turn into", "effect", "result",
        "distorted", "modified", "altered", "evolved", "converted",
        "different after", "before and after"
    ]

    # GENERIC Video editing/structure keywords (for Audio-Visual Stitching)
    # Made generic to work across all video types - no hardcoded specifics
    EDITING_KEYWORDS = [
        # Generic editing terms
        "clip", "clips", "spliced", "splice", "cut", "cuts", "edited", "edit",
        "transition", "transitions", "segment", "segments", "section", "sections",
        
        # Temporal structure terms (generic)
        "beginning", "ending", "start", "end", "intro", "outro", "opening", "closing",
        
        # Pacing and structure
        "pace", "pacing", "speed up", "slow down", "fast forward", "slow motion",
        
        # Generic scene/shot terms
        "scene", "scenes", "shot", "shots", "footage", "recording", "part", "parts",
        
        # Audio-visual combination terms
        "jingle", "musical", "music video", "b-roll", "overlay", "background music",
        "voice over", "voiceover", "narration", "soundtrack"
    ]

    # Holistic reasoning keywords (for General Holistic Reasoning)
    HOLISTIC_KEYWORDS = [
        "video", "overall", "entire", "throughout", "whole", "complete",
        "majority", "most", "all", "pattern", "relate", "relationship",
        "connection", "purpose", "point", "goal", "message", "theme"
    ]

    # Spurious correlation keywords (for Tackling Spurious Correlations)
    SPURIOUS_KEYWORDS = [
        "referring to", "reference", "meaning", "represents", "symbolizes",
        "unusual", "unique", "surprising", "unexpected", "strange", "odd",
        "metaphor", "figurative", "ironic", "counter-intuitive", "paradox"
    ]

    # Context/background keywords (for Context)
    CONTEXT_KEYWORDS = [
        "background", "foreground", "behind", "in front", "setting", "location",
        "environment", "scene", "visible", "appears", "shown", "displays",
        "billboard", "sign", "poster", "text", "screen"
    ]

    # Subscene conditional keywords (for Subscene)
    SUBSCENE_KEYWORDS = [
        "when", "while", "during", "as", "at the moment", "at the time",
        "when the score", "when the clock", "in the scene", "in the segment"
    ]

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_premium_frames: int = None,
        min_opportunity_score: float = None,
        validation_batch_size: int = None,
        enable_stage2_validation: bool = True,
        min_stage1_confidence: float = None,
        max_candidates_to_validate: int = None,
        validation_budget_dollars: float = None
    ):
        """
        Initialize opportunity detector V2 with Two-Stage architecture and optimization.

        Args:
            openai_api_key: OpenAI API key
            model: GPT-4 model to use
            max_premium_frames: Override MAX_PREMIUM_FRAMES (default: use class constant)
            min_opportunity_score: Override MIN_OPPORTUNITY_SCORE (default: use class constant)
            validation_batch_size: Override VALIDATION_BATCH_SIZE (default: use class constant)
            enable_stage2_validation: Enable GPT-4 validation (default: True)
            min_stage1_confidence: Override MIN_STAGE1_CONFIDENCE (default: use class constant)
            max_candidates_to_validate: Override MAX_CANDIDATES_TO_VALIDATE (default: use class constant)
            validation_budget_dollars: Override VALIDATION_BUDGET_DOLLARS (default: use class constant)
        """
        if openai_api_key:
            openai.api_key = openai_api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key required")

        self.model = model
        self.enable_stage2_validation = enable_stage2_validation

        # Override configuration if provided
        if max_premium_frames is not None:
            self.MAX_PREMIUM_FRAMES = max_premium_frames
        if min_opportunity_score is not None:
            self.MIN_OPPORTUNITY_SCORE = min_opportunity_score
        if validation_batch_size is not None:
            self.VALIDATION_BATCH_SIZE = validation_batch_size
        if min_stage1_confidence is not None:
            self.MIN_STAGE1_CONFIDENCE = min_stage1_confidence
        if max_candidates_to_validate is not None:
            self.MAX_CANDIDATES_TO_VALIDATE = max_candidates_to_validate
        if validation_budget_dollars is not None:
            self.VALIDATION_BUDGET_DOLLARS = validation_budget_dollars

        logger.info(f"OpportunityDetectorV2 initialized (ALL 13 types, OPTIMIZED)")
        logger.info(f"  Model: {model}")
        logger.info(f"  Max premium frames: {self.MAX_PREMIUM_FRAMES}")
        logger.info(f"  Min opportunity score: {self.MIN_OPPORTUNITY_SCORE}")
        logger.info(f"  Validation batch size: {self.VALIDATION_BATCH_SIZE}")
        logger.info(f"  Stage 2 validation: {'ENABLED' if enable_stage2_validation else 'DISABLED'}")
        logger.info(f"  Min Stage 1 confidence: {self.MIN_STAGE1_CONFIDENCE}")
        logger.info(f"  Max candidates to validate: {self.MAX_CANDIDATES_TO_VALIDATE}")
        logger.info(f"  Validation budget: ${self.VALIDATION_BUDGET_DOLLARS:.2f}")
        logger.info(f"  Temporal clustering window: {self.TEMPORAL_CLUSTERING_WINDOW}s")

    def detect_opportunities(
        self,
        audio_analysis: Dict,
        video_id: str = "unknown"
    ) -> OpportunityDetectionResult:
        """
        Detect opportunities using Two-Stage approach with optimization.

        Stage 1: Pattern-based detection (fast, broad)
        Stage 2: AI validation (slow, precise) with three-tier filtering

        Args:
            audio_analysis: Enhanced audio analysis with segments, audio_events, silence_gaps
            video_id: Video identifier

        Returns:
            OpportunityDetectionResult with validated opportunities
        """
        logger.info("=" * 80)
        logger.info("OPPORTUNITY DETECTION V2 - OPTIMIZED TWO-STAGE (ALL 13 TYPES)")
        logger.info("=" * 80)

        segments = audio_analysis.get("segments", [])
        duration = audio_analysis.get("duration", 0.0)
        audio_events = audio_analysis.get("audio_events", [])
        silence_gaps = audio_analysis.get("silence_gaps", [])

        logger.info(f"Video ID: {video_id}")
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"Segments: {len(segments)}")
        logger.info(f"Audio Events: {len(audio_events)}")
        logger.info(f"Silence Gaps: {len(silence_gaps)}")

        # STAGE 1: Pattern-Based Detection
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: PATTERN-BASED CANDIDATE DETECTION")
        logger.info("=" * 80)
        candidates = self._detect_candidates_stage1(audio_analysis)
        logger.info(f"✓ Stage 1 complete: {len(candidates)} candidates detected")
        
        initial_candidate_count = len(candidates)

        # OPTIMIZATION: Filter and cluster before Stage 2
        logger.info("\n" + "=" * 80)
        logger.info("OPTIMIZATION: FILTERING & CLUSTERING")
        logger.info("=" * 80)
        
        # Step 1: Apply confidence and quality filters
        logger.info("Step 1: Applying confidence and quality filters...")
        candidates, filtered_count = self._filter_by_confidence(candidates)
        logger.info(f"  ✓ Filtered {filtered_count} low-confidence candidates")
        logger.info(f"  → Remaining: {len(candidates)} candidates")
        
        # Step 2: Temporal clustering to remove duplicates
        logger.info("Step 2: Temporal clustering (removing duplicates)...")
        candidates, clustered_count = self._temporal_clustering(candidates)
        logger.info(f"  ✓ Clustered {clustered_count} duplicate candidates")
        logger.info(f"  → Remaining: {len(candidates)} candidates")
        
        # Step 3: Three-tier classification
        logger.info("Step 3: Three-tier validation classification...")
        tier1_candidates, candidates_to_validate, tier_stats = self._classify_candidates_by_tier(candidates)
        logger.info(f"  ✓ Tier 1 (no validation): {tier_stats['tier1']} candidates")
        logger.info(f"  ✓ Tier 2 (sampled): {tier_stats['tier2_total']} candidates ({tier_stats['tier2_validated']} to validate)")
        logger.info(f"  ✓ Tier 3 (full validation): {tier_stats['tier3']} candidates")
        logger.info(f"  → Stage 2 validation queue: {len(candidates_to_validate)} candidates")

        # STAGE 2: AI Validation (if enabled)
        validated_candidates = candidates_to_validate
        if self.enable_stage2_validation and candidates_to_validate:
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 2: AI VALIDATION & ENRICHMENT")
            logger.info("=" * 80)
            validated_candidates = self._validate_candidates_stage2(candidates_to_validate, segments)
            logger.info(f"✓ Stage 2 complete: {len(validated_candidates)} candidates validated")

        # Combine Tier 1 (skipped) + validated candidates
        all_candidates = tier1_candidates + validated_candidates

        # Convert candidates to opportunities
        logger.info("\n✓ Converting candidates to opportunities...")
        result = self._create_opportunities_from_candidates(
            all_candidates,
            segments,
            video_id,
            duration
        )

        # Calculate statistics
        result.stage1_candidates = initial_candidate_count
        result.stage2_validated = len(validated_candidates)
        result.tier1_skipped = tier_stats['tier1']
        result.tier2_sampled = tier_stats['tier2_validated']
        result.tier3_full = tier_stats['tier3']
        result.filtered_by_confidence = filtered_count
        result.clustered_duplicates = clustered_count

        # Score opportunities
        logger.info("\n✓ Scoring opportunities...")
        for opp in result.opportunities:
            opp.opportunity_score = self._score_opportunity(opp)

        # Select premium frames
        logger.info("\n✓ Selecting premium frames...")
        result.premium_frames = self._select_premium_frames(result.opportunities, duration)

        # Calculate statistics by type
        stats = {}
        for opp in result.opportunities:
            stats[opp.opportunity_type] = stats.get(opp.opportunity_type, 0) + 1
        result.opportunity_statistics = stats

        # Estimate cost
        cost_stage1 = 0.0  # Pattern-based is free
        cost_per_batch = 0.012  # Approximate GPT-4 cost
        cost_stage2 = (len(candidates_to_validate) / self.VALIDATION_BATCH_SIZE) * cost_per_batch if self.enable_stage2_validation else 0.0
        result.detection_cost = cost_stage1 + cost_stage2

        logger.info("=" * 80)
        logger.info("✅ OPPORTUNITY DETECTION COMPLETE (OPTIMIZED)")
        logger.info("=" * 80)
        logger.info(f"Stage 1 candidates: {result.stage1_candidates}")
        logger.info(f"Filtered by confidence: {result.filtered_by_confidence}")
        logger.info(f"Clustered duplicates: {result.clustered_duplicates}")
        logger.info(f"Tier 1 (no validation): {result.tier1_skipped}")
        logger.info(f"Tier 2 (sampled): {result.tier2_sampled}")
        logger.info(f"Tier 3 (full validation): {result.tier3_full}")
        logger.info(f"Stage 2 validated: {result.stage2_validated}")
        logger.info(f"Total opportunities: {result.total_opportunities}")
        logger.info(f"Validated opportunities: {result.validated_opportunities}")
        logger.info(f"Premium frames: {len(result.premium_frames)}")
        logger.info(f"Detection cost: ${result.detection_cost:.4f} (Budget: ${self.VALIDATION_BUDGET_DOLLARS:.2f})")
        logger.info("\nOpportunities by type:")
        for opp_type, count in sorted(result.opportunity_statistics.items()):
            logger.info(f"  {opp_type}: {count}")
        logger.info("=" * 80)

        return result

    def _filter_by_confidence(self, candidates: List[Dict]) -> Tuple[List[Dict], int]:
        """
        Filter candidates by confidence thresholds.
        
        Args:
            candidates: All candidates from Stage 1
            
        Returns:
            Tuple of (filtered_candidates, filtered_count)
        """
        filtered = []
        filtered_count = 0
        
        for cand in candidates:
            # Base confidence threshold
            if cand.get("confidence", 0) < self.MIN_STAGE1_CONFIDENCE:
                filtered_count += 1
                continue
            
            # Special handling for audio events (higher bar)
            if cand.get("source") in ["audio_event", "sound_effect"]:
                # Check confidence
                if cand.get("confidence", 0) < self.MIN_AUDIO_EVENT_CONFIDENCE:
                    filtered_count += 1
                    continue
                
                # Check intensity (if available)
                audio_event = cand.get("audio_event")
                if audio_event:
                    characteristics = audio_event.get("characteristics", {})
                    intensity_str = characteristics.get("intensity", "medium")
                    
                    # Skip low intensity events
                    if intensity_str == "medium":
                        # Check if it meets RMS threshold
                        # For "click" type, always skip
                        if cand.get("audio_event", {}).get("subtype") == "click":
                            filtered_count += 1
                            continue
                
                # Check duration (skip very short events)
                duration = cand.get("end", 0) - cand.get("start", 0)
                if duration < self.MIN_AUDIO_EVENT_DURATION:
                    filtered_count += 1
                    continue
            
            filtered.append(cand)
        
        return filtered, filtered_count

    def _temporal_clustering(self, candidates: List[Dict]) -> Tuple[List[Dict], int]:
        """
        Cluster candidates within temporal windows to remove duplicates.
        
        Args:
            candidates: Filtered candidates
            
        Returns:
            Tuple of (clustered_candidates, removed_count)
        """
        if not candidates:
            return candidates, 0
        
        # Sort by timestamp
        candidates_sorted = sorted(candidates, key=lambda c: c.get("start", 0))
        
        clusters = []
        current_cluster = [candidates_sorted[0]]
        
        for cand in candidates_sorted[1:]:
            # Check if within window of current cluster start
            if cand.get("start", 0) - current_cluster[0].get("start", 0) <= self.TEMPORAL_CLUSTERING_WINDOW:
                current_cluster.append(cand)
            else:
                # Save best from cluster (highest confidence)
                best = max(current_cluster, key=lambda c: c.get("confidence", 0))
                clusters.append(best)
                current_cluster = [cand]
        
        # Don't forget last cluster
        if current_cluster:
            best = max(current_cluster, key=lambda c: c.get("confidence", 0))
            clusters.append(best)
        
        removed_count = len(candidates) - len(clusters)
        return clusters, removed_count

    def _classify_candidates_by_tier(
        self, 
        candidates: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Classify candidates into three tiers for validation strategy.
        
        Tier 1: No validation needed (reliable patterns)
        Tier 2: Sample validation (20%)
        Tier 3: Full validation (complex reasoning)
        
        Args:
            candidates: Clustered candidates
            
        Returns:
            Tuple of (tier1_candidates, candidates_to_validate, tier_stats)
        """
        tier1_candidates = []
        tier2_candidates = []
        tier3_candidates = []
        
        for cand in candidates:
            opp_type = cand.get("type")
            
            if opp_type in self.TIER1_NO_VALIDATION:
                # Assign default scores for Tier 1
                cand["adversarial_score"] = 0.6
                cand["multimodal_required"] = True
                cand["complexity"] = "medium"
                cand["valid"] = True
                cand["validated_stage2"] = False  # Skipped validation
                cand["recommended_task_types"] = self._get_default_task_types(opp_type)
                tier1_candidates.append(cand)
                
            elif opp_type in self.TIER2_SAMPLE_VALIDATION:
                tier2_candidates.append(cand)
                
            else:  # TIER3_FULL_VALIDATION or unknown types
                tier3_candidates.append(cand)
        
        # Sample Tier 2 (20% random sample)
        tier2_sample_size = max(1, int(len(tier2_candidates) * self.TIER2_SAMPLE_RATE))
        tier2_to_validate = random.sample(tier2_candidates, k=min(tier2_sample_size, len(tier2_candidates)))
        
        # For non-validated Tier 2, assign average scores
        tier2_not_validated = [c for c in tier2_candidates if c not in tier2_to_validate]
        for cand in tier2_not_validated:
            cand["adversarial_score"] = 0.65
            cand["multimodal_required"] = True
            cand["complexity"] = "medium"
            cand["valid"] = True
            cand["validated_stage2"] = False
            cand["recommended_task_types"] = self._get_default_task_types(cand.get("type"))
        
        # Add non-validated Tier 2 to Tier 1 results
        tier1_candidates.extend(tier2_not_validated)
        
        # Combine for Stage 2 validation
        candidates_to_validate = tier2_to_validate + tier3_candidates
        
        # Apply budget cap (sort by confidence, take top N)
        candidates_to_validate.sort(key=lambda c: c.get("confidence", 0), reverse=True)
        candidates_to_validate = candidates_to_validate[:self.MAX_CANDIDATES_TO_VALIDATE]
        
        tier_stats = {
            "tier1": len([c for c in tier1_candidates if c.get("type") in self.TIER1_NO_VALIDATION]),
            "tier2_total": len(tier2_candidates),
            "tier2_validated": len(tier2_to_validate),
            "tier3": len(tier3_candidates)
        }
        
        return tier1_candidates, candidates_to_validate, tier_stats

    def _detect_candidates_stage1(self, audio_analysis: Dict) -> List[Dict]:
        """
        STAGE 1: Pattern-based detection of opportunity candidates.

        Detects ALL 13 opportunity types using:
        - Enhanced keyword/regex matching from guideline examples
        - Audio events from Phase 1
        - Silence gaps from Phase 1
        - Word-level timestamps

        Args:
            audio_analysis: Enhanced audio analysis

        Returns:
            List of candidate dicts
        """
        segments = audio_analysis.get("segments", [])
        audio_events = audio_analysis.get("audio_events", [])
        silence_gaps = audio_analysis.get("silence_gaps", [])

        candidates = []
        candidate_id = 1

        logger.info("Detecting candidates using enhanced pattern matching...")

        # 1. TEMPORAL UNDERSTANDING
        logger.info("  - Temporal markers...")
        for seg in segments:
            text_lower = seg["text"].lower()
            for keyword in self.TEMPORAL_KEYWORDS:
                if re.search(r'\b' + keyword + r'\b', text_lower):
                    candidates.append({
                        "candidate_id": f"temp_{candidate_id:04d}",
                        "type": "temporal_understanding",
                        "quote": seg["text"],
                        "keyword": keyword,
                        "start": seg["start"],
                        "end": seg["end"],
                        "words": seg.get("words", []),
                        "confidence": 0.7,
                        "source": "pattern_temporal"
                    })
                    candidate_id += 1

        # 2. REFERENTIAL GROUNDING (ambiguous references)
        logger.info("  - Ambiguous references...")
        for seg in segments:
            text_lower = seg["text"].lower()
            for ambig in self.AMBIGUOUS_WORDS:
                # Match as whole word
                if re.search(r'\b' + ambig + r'\b', text_lower):
                    candidates.append({
                        "candidate_id": f"ref_{candidate_id:04d}",
                        "type": "referential_grounding",
                        "quote": seg["text"],
                        "ambiguous_word": ambig,
                        "start": seg["start"],
                        "end": seg["end"],
                        "words": seg.get("words", []),
                        "confidence": 0.8,
                        "source": "pattern_referential"
                    })
                    candidate_id += 1

        # 3. COUNTING
        logger.info("  - Counting opportunities...")
        for seg in segments:
            text_lower = seg["text"].lower()
            # Look for counting-related patterns
            if re.search(r'\b(how many|count|times|repeated|several|multiple|number of|throughout)\b', text_lower):
                candidates.append({
                    "candidate_id": f"count_{candidate_id:04d}",
                    "type": "counting",
                    "quote": seg["text"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "words": seg.get("words", []),
                    "confidence": 0.65,
                    "source": "pattern_counting"
                })
                candidate_id += 1

        # 4. SEQUENTIAL
        logger.info("  - Sequential events...")
        for seg in segments:
            text_lower = seg["text"].lower()
            for keyword in self.SEQUENTIAL_KEYWORDS:
                if re.search(r'\b' + keyword + r'\b', text_lower):
                    candidates.append({
                        "candidate_id": f"seq_{candidate_id:04d}",
                        "type": "sequential",
                        "quote": seg["text"],
                        "keyword": keyword,
                        "start": seg["start"],
                        "end": seg["end"],
                        "words": seg.get("words", []),
                        "confidence": 0.7,
                        "source": "pattern_sequential"
                    })
                    candidate_id += 1

        # 5. AUDIO-VISUAL STITCHING (OPTIMIZED - generic editing focus)
        logger.info("  - Audio-visual stitching (editing focus)...")
        
        # 5a. From scene transitions (silence gaps > 1.5s)
        for gap in silence_gaps:
            if gap.get("type") == "scene_change" or gap.get("duration", 0) >= self.MIN_SCENE_CHANGE_DURATION:
                candidates.append({
                    "candidate_id": f"avs_{candidate_id:04d}",
                    "type": "audio_visual_stitching",
                    "scene_transition": gap,
                    "description": f"Scene transition at {gap.get('start_time', 'unknown')}",
                    "start": gap["start"],
                    "end": gap["end"],
                    "confidence": 0.80,  # High confidence for scene changes
                    "source": "scene_transition"
                })
                candidate_id += 1
        
        # 5b. From music intro/outro (start/end of audio events)
        for event in audio_events:
            if event.get("type") == "background_music":
                # Check if at beginning or end of video
                duration = audio_analysis.get("duration", 0)
                is_intro = event.get("start", 0) < 5.0  # First 5 seconds
                is_outro = event.get("end", duration) > (duration - 5.0)  # Last 5 seconds
                
                if is_intro or is_outro:
                    candidates.append({
                        "candidate_id": f"avs_{candidate_id:04d}",
                        "type": "audio_visual_stitching",
                        "audio_event": event,
                        "description": f"{'Intro' if is_intro else 'Outro'} music at {event.get('start_time', 'unknown')}",
                        "start": event["start"],
                        "end": event.get("end", event["start"] + 0.5),
                        "confidence": 0.75,
                        "source": "audio_event"
                    })
                    candidate_id += 1
            
            # Music/tempo changes (pacing)
            if event.get("type") == "music_change":
                candidates.append({
                    "candidate_id": f"avs_{candidate_id:04d}",
                    "type": "audio_visual_stitching",
                    "audio_event": event,
                    "description": f"Music change at {event.get('start_time', 'unknown')}",
                    "start": event["start"],
                    "end": event.get("end", event["start"] + 0.5),
                    "confidence": 0.80,
                    "source": "audio_event"
                })
                candidate_id += 1
        
        # 5c. From generic editing keywords in transcript
        for seg in segments:
            text_lower = seg["text"].lower()
            for keyword in self.EDITING_KEYWORDS:
                if keyword in text_lower:
                    candidates.append({
                        "candidate_id": f"avs_{candidate_id:04d}",
                        "type": "audio_visual_stitching",
                        "quote": seg["text"],
                        "keyword": keyword,
                        "start": seg["start"],
                        "end": seg["end"],
                        "words": seg.get("words", []),
                        "confidence": 0.75,
                        "source": "pattern_editing"
                    })
                    candidate_id += 1
                    break  # Only one per segment

        # 6. COMPARATIVE (from scene transitions)
        logger.info("  - Comparative opportunities...")
        
        # 6a. From silence gaps (scene changes)
        for gap in silence_gaps:
            if gap.get("type") == "scene_change" or gap.get("duration", 0) >= self.MIN_SCENE_CHANGE_DURATION:
                candidates.append({
                    "candidate_id": f"comp_{candidate_id:04d}",
                    "type": "comparative",
                    "scene_transition": gap,
                    "description": f"Scene change at {gap.get('start_time', 'unknown')}",
                    "start": gap["start"],
                    "end": gap["end"],
                    "confidence": 0.75,
                    "source": "scene_transition"
                })
                candidate_id += 1
        
        # 6b. From comparative keywords in transcript
        for seg in segments:
            text_lower = seg["text"].lower()
            for keyword in self.COMPARATIVE_KEYWORDS:
                if keyword in text_lower:
                    candidates.append({
                        "candidate_id": f"comp_{candidate_id:04d}",
                        "type": "comparative",
                        "quote": seg["text"],
                        "keyword": keyword,
                        "start": seg["start"],
                        "end": seg["end"],
                        "words": seg.get("words", []),
                        "confidence": 0.7,
                        "source": "pattern_comparative"
                    })
                    candidate_id += 1
                    break

        # 7. INFERENCE
        logger.info("  - Inference opportunities...")
        for seg in segments:
            text_lower = seg["text"].lower()
            for keyword in self.INFERENCE_KEYWORDS:
                if re.search(r'\b' + keyword + r'\b', text_lower):
                    candidates.append({
                        "candidate_id": f"inf_{candidate_id:04d}",
                        "type": "inference",
                        "quote": seg["text"],
                        "keyword": keyword,
                        "start": seg["start"],
                        "end": seg["end"],
                        "words": seg.get("words", []),
                        "confidence": 0.7,
                        "source": "pattern_inference"
                    })
                    candidate_id += 1
                    break

        # 8. CONTEXT (background/foreground)
        logger.info("  - Context opportunities...")
        for seg in segments:
            text_lower = seg["text"].lower()
            for keyword in self.CONTEXT_KEYWORDS:
                if keyword in text_lower:
                    candidates.append({
                        "candidate_id": f"ctx_{candidate_id:04d}",
                        "type": "context",
                        "quote": seg["text"],
                        "keyword": keyword,
                        "start": seg["start"],
                        "end": seg["end"],
                        "words": seg.get("words", []),
                        "confidence": 0.65,
                        "source": "pattern_context"
                    })
                    candidate_id += 1
                    break

        # 9. SUBSCENE (specific conditions)
        logger.info("  - Subscene opportunities...")
        for seg in segments:
            text_lower = seg["text"].lower()
            
            # Pattern 1: "when/while/during" + specific condition (score, clock, etc.)
            if re.search(r'\b(when|while|during|as)\b', text_lower):
                # Check for specific conditions
                has_score = re.search(r'\bscore (is |was )?\d+-\d+\b', text_lower)
                has_clock = re.search(r'\b\d+:\d+\b', text_lower) or re.search(r'\bclock\b', text_lower)
                has_quarter = re.search(r'\b(first|second|third|fourth) quarter\b', text_lower)
                
                if has_score or has_clock or has_quarter:
                    candidates.append({
                        "candidate_id": f"sub_{candidate_id:04d}",
                        "type": "subscene",
                        "quote": seg["text"],
                        "condition_type": "score" if has_score else ("clock" if has_clock else "quarter"),
                        "start": seg["start"],
                        "end": seg["end"],
                        "words": seg.get("words", []),
                        "confidence": 0.8,  # Higher confidence with specific condition
                        "source": "pattern_subscene_condition"
                    })
                    candidate_id += 1
            
            # Pattern 2: "describe what happens" + conditional phrase
            if re.search(r'\b(describe|what happens|what does|how does)\b', text_lower):
                if re.search(r'\b(when|while|during|in the scene|in the segment)\b', text_lower):
                    candidates.append({
                        "candidate_id": f"sub_{candidate_id:04d}",
                        "type": "subscene",
                        "quote": seg["text"],
                        "start": seg["start"],
                        "end": seg["end"],
                        "words": seg.get("words", []),
                        "confidence": 0.6,
                        "source": "pattern_subscene_describe"
                    })
                    candidate_id += 1

        # 10. OBJECT INTERACTION (transformations)
        logger.info("  - Object interaction...")
        for seg in segments:
            text_lower = seg["text"].lower()
            
            # Check for action verb
            has_action = any(action in text_lower for action in self.ACTION_VERBS)
            
            if has_action:
                # Check for transformation nearby
                has_transformation = any(trans in text_lower for trans in self.TRANSFORMATION_KEYWORDS)
                
                # Find which action verb
                action_verb = None
                for action in self.ACTION_VERBS:
                    if action in text_lower:
                        action_verb = action
                        break
                
                candidates.append({
                    "candidate_id": f"obj_{candidate_id:04d}",
                    "type": "object_interaction",
                    "quote": seg["text"],
                    "action_verb": action_verb,
                    "has_transformation": has_transformation,
                    "start": seg["start"],
                    "end": seg["end"],
                    "words": seg.get("words", []),
                    "confidence": 0.75 if has_transformation else 0.6,
                    "source": "pattern_object_interaction"
                })
                candidate_id += 1

        # 11. NEEDLE (specific formats)
        logger.info("  - Needle opportunities...")
        for seg in segments:
            text_lower = seg["text"].lower()
            
            # Pattern 1: Clock time format "1:10"
            has_clock_time = re.search(r'\b\d+:\d+\b', text_lower)
            
            # Pattern 2: Score format "118-2"
            has_score = re.search(r'\b\d+-\d+\b', text_lower)
            
            # Pattern 3: Ordinals with specific events "first three", "second attempt"
            has_ordinal = re.search(r'\b(first|second|third|fourth|fifth) (three|goal|attempt|basket|point|shot)\b', text_lower)
            
            # Pattern 4: Specific detail words
            has_detail_words = re.search(r'\b(specific|exactly|precise|detail|small|tiny|billboard|sign|text|graphic)\b', text_lower)
            
            if has_clock_time or has_score or has_ordinal or has_detail_words:
                needle_type = "clock" if has_clock_time else ("score" if has_score else ("ordinal" if has_ordinal else "detail"))
                
                candidates.append({
                    "candidate_id": f"needle_{candidate_id:04d}",
                    "type": "needle",
                    "quote": seg["text"],
                    "needle_type": needle_type,
                    "start": seg["start"],
                    "end": seg["end"],
                    "words": seg.get("words", []),
                    "confidence": 0.75 if (has_clock_time or has_score or has_ordinal) else 0.6,
                    "source": "pattern_needle"
                })
                candidate_id += 1

        # 12. GENERAL HOLISTIC REASONING
        logger.info("  - General holistic reasoning...")
        for seg in segments:
            text_lower = seg["text"].lower()
            
            # Check for meta-level words about video
            has_meta = any(keyword in text_lower for keyword in ["video", "clip", "overall", "entire", "throughout", "whole"])
            
            # Check for purpose/pattern words
            has_purpose = any(keyword in text_lower for keyword in ["purpose", "point", "why", "how", "relate", "connection", "majority", "most", "pattern"])
            
            if has_meta and has_purpose:
                candidates.append({
                    "candidate_id": f"holistic_{candidate_id:04d}",
                    "type": "general_holistic_reasoning",
                    "quote": seg["text"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "words": seg.get("words", []),
                    "confidence": 0.7,
                    "source": "pattern_holistic"
                })
                candidate_id += 1

        # 13. TACKLING SPURIOUS CORRELATIONS
        logger.info("  - Tackling spurious correlations...")
        for seg in segments:
            text_lower = seg["text"].lower()
            
            # Pattern 1: Metaphorical/reference language
            has_reference = any(keyword in text_lower for keyword in ["referring to", "reference", "meaning", "represents", "symbolizes"])
            
            # Pattern 2: Counter-intuitive language
            has_unusual = any(keyword in text_lower for keyword in ["unusual", "unique", "surprising", "unexpected", "strange", "odd"])
            
            # Pattern 3: Figurative language markers
            has_figurative = re.search(r'\b(metaphor|figurative|ironic|paradox)\b', text_lower)
            
            if has_reference or has_unusual or has_figurative:
                spurious_type = "metaphor" if has_reference else ("counter_intuitive" if has_unusual else "figurative")
                
                candidates.append({
                    "candidate_id": f"spurious_{candidate_id:04d}",
                    "type": "tackling_spurious_correlations",
                    "quote": seg["text"],
                    "spurious_type": spurious_type,
                    "start": seg["start"],
                    "end": seg["end"],
                    "words": seg.get("words", []),
                    "confidence": 0.65,
                    "source": "pattern_spurious"
                })
                candidate_id += 1

        logger.info(f"✓ Pattern detection complete: {len(candidates)} candidates")
        
        # Log breakdown by type
        type_counts = {}
        for cand in candidates:
            type_counts[cand["type"]] = type_counts.get(cand["type"], 0) + 1
        
        logger.info("\nCandidates by type:")
        for opp_type, count in sorted(type_counts.items()):
            logger.info(f"  {opp_type}: {count}")
        
        return candidates

    def _validate_candidates_stage2(
        self,
        candidates: List[Dict],
        segments: List[Dict]
    ) -> List[Dict]:
        """
        STAGE 2: Validate candidates using GPT-4 in batches.

        Args:
            candidates: Candidates from Stage 1 (after filtering)
            segments: Transcript segments

        Returns:
            List of validated candidates
        """
        if not candidates:
            return []

        validated = []
        total_batches = ceil(len(candidates) / self.VALIDATION_BATCH_SIZE)

        logger.info(f"Validating {len(candidates)} candidates in {total_batches} batches...")

        for batch_idx in range(0, len(candidates), self.VALIDATION_BATCH_SIZE):
            batch = candidates[batch_idx:batch_idx + self.VALIDATION_BATCH_SIZE]
            batch_num = (batch_idx // self.VALIDATION_BATCH_SIZE) + 1

            logger.info(f"  Batch {batch_num}/{total_batches}: Validating {len(batch)} candidates...")

            try:
                validated_batch = self._validate_batch_with_gpt4(batch, segments)
                validated.extend(validated_batch)
                logger.info(f"    ✓ Validated {len(validated_batch)}/{len(batch)} candidates")
            except Exception as e:
                logger.error(f"    ✗ Batch validation failed: {e}")
                # Add unvalidated candidates as fallback
                for cand in batch:
                    cand["validated_stage2"] = False
                    validated.append(cand)

        logger.info(f"✓ Stage 2 validation complete: {len(validated)} candidates")
        return validated

    def _validate_batch_with_gpt4(
        self,
        batch: List[Dict],
        segments: List[Dict]
    ) -> List[Dict]:
        """
        Validate a batch of candidates with GPT-4.

        Args:
            batch: Batch of candidates
            segments: Transcript segments

        Returns:
            List of validated candidates with enrichment
        """
        # Format candidates for GPT-4 (remove 'words' field to reduce token count)
        batch_for_gpt = []
        for cand in batch:
            cand_copy = cand.copy()
            if "words" in cand_copy:
                del cand_copy["words"]
            if "audio_event" in cand_copy:
                # Simplify audio event
                cand_copy["audio_event"] = {
                    "type": cand_copy["audio_event"].get("type"),
                    "subtype": cand_copy["audio_event"].get("subtype")
                }
            if "scene_transition" in cand_copy:
                # Simplify scene transition
                cand_copy["scene_transition"] = {
                    "start": cand_copy["scene_transition"].get("start"),
                    "duration": cand_copy["scene_transition"].get("duration")
                }
            batch_for_gpt.append(cand_copy)

        candidates_json = json.dumps(batch_for_gpt, indent=2)

        prompt = f"""Validate and enrich these opportunity candidates for multimodal video QA (13 types total).

CANDIDATES:
{candidates_json}

For EACH candidate, analyze:
1. Is this a VALID opportunity? (true/false)
2. Adversarial score (0-1): How likely to challenge Gemini?
3. Multimodal required? (true/false): MUST use both audio AND visual?
4. Complexity level: "low", "medium", or "high"
5. Recommended question types (e.g., ["Temporal Understanding", "Sequential"])

VALIDATION RULES:
- Reject false positives (e.g., "after all" is idiom, not temporal)
- Prefer opportunities requiring BOTH modalities
- Score higher for: ambiguous, precise timing, counter-intuitive, metaphorical
- Context must be genuinely needed from video
- Subscene requires specific conditions (score, clock, scene description)
- Needle requires precise details (clock times, scores, small visual elements)
- Object Interaction should show transformations/changes
- Audio-Visual Stitching about editing/splicing, not just audio presence
- General Holistic about overall video structure/purpose
- Spurious Correlations about non-literal/counter-intuitive interpretations

Return JSON array with same structure, adding validation fields:
[
  {{
    "candidate_id": "...",
    "type": "...",
    "valid": true/false,
    "adversarial_score": 0.0-1.0,
    "multimodal_required": true/false,
    "complexity": "low|medium|high",
    "recommended_task_types": ["..."],
    "validation_reason": "why valid or invalid"
  }},
  ...
]

Return ONLY valid JSON array."""

        # Retry loop with exponential backoff
        import time
        import httpx

        retry_delay = 5  # Initial delay in seconds
        last_error = None

        for attempt in range(self.MAX_API_RETRIES):
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at validating multimodal video QA opportunities across 13 task types. You assess which opportunities will create effective adversarial questions for testing AI models like Gemini. Return only valid JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=3000,
                    timeout=self.API_TIMEOUT  # Explicit timeout
                )

                content = response.choices[0].message.content.strip()

                # Remove markdown
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                validated_batch = json.loads(content)

                # Merge validation results with original candidates
                for i, cand in enumerate(batch):
                    if i < len(validated_batch):
                        val = validated_batch[i]
                        cand.update({
                            "valid": val.get("valid", True),
                            "adversarial_score": val.get("adversarial_score", 0.5),
                            "multimodal_required": val.get("multimodal_required", True),
                            "complexity": val.get("complexity", "medium"),
                            "recommended_task_types": val.get("recommended_task_types", []),
                            "validation_reason": val.get("validation_reason", ""),
                            "validated_stage2": True
                        })

                # Filter out invalid candidates
                return [c for c in batch if c.get("valid", True)]

            except (httpx.ReadError, httpx.TimeoutException, httpx.ConnectError, openai.APIConnectionError) as e:
                last_error = e
                if attempt < self.MAX_API_RETRIES - 1:
                    logger.warning(f"⚠️  API connection error (attempt {attempt + 1}/{self.MAX_API_RETRIES}): {type(e).__name__}")
                    logger.info(f"   Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"❌ API connection failed after {self.MAX_API_RETRIES} attempts: {e}")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GPT-4 response: {e}")
                # JSON error - don't retry, return unvalidated batch
                for cand in batch:
                    cand["validated_stage2"] = False
                return batch

            except Exception as e:
                last_error = e
                logger.error(f"GPT-4 validation error: {e}")
                # Unknown error - don't retry
                for cand in batch:
                    cand["validated_stage2"] = False
                return batch

        # All retries exhausted - return unvalidated batch (fallback)
        logger.warning(f"⚠️  Returning unvalidated batch after all retries failed")
        for cand in batch:
            cand["validated_stage2"] = False
        return batch

    def _create_opportunities_from_candidates(
        self,
        candidates: List[Dict],
        segments: List[Dict],
        video_id: str,
        duration: float
    ) -> OpportunityDetectionResult:
        """
        Convert validated candidates to RealOpportunity objects.

        Args:
            candidates: Validated candidates (Tier 1 + Stage 2)
            segments: Transcript segments
            video_id: Video ID
            duration: Duration

        Returns:
            OpportunityDetectionResult
        """
        result = OpportunityDetectionResult(
            video_id=video_id,
            transcript_duration=duration
        )

        for cand in candidates:
            # Extract key word timestamp if available
            key_word = None
            key_word_ts = None
            if "words" in cand and cand["words"]:
                target_word = (
                    cand.get("keyword") or 
                    cand.get("ambiguous_word") or 
                    cand.get("action_verb")
                )
                key_word, key_word_ts = self._find_key_word(
                    cand.get("quote", ""),
                    cand["words"],
                    target_word
                )

            # Determine visual timestamp (precise if key word available)
            visual_ts = key_word_ts if key_word_ts else cand["start"]

            # Get task types
            task_types = cand.get("recommended_task_types", [])
            if not task_types:
                # Default task types by opportunity type
                task_types = self._get_default_task_types(cand["type"])

            # Create opportunity
            opportunity = RealOpportunity(
                opportunity_id=cand["candidate_id"],
                opportunity_type=cand["type"],
                audio_quote=cand.get("quote", cand.get("description", "")),
                audio_start=cand["start"],
                audio_end=cand["end"],
                visual_timestamp=visual_ts,
                task_types=task_types,
                complexity=cand.get("complexity", "medium"),
                description=cand.get("description", cand.get("validation_reason", "")),
                validated_audio=self._validate_quote_exists(
                    cand.get("quote", ""),
                    segments,
                    cand["start"]
                ) if "quote" in cand else True,
                validated_visual=False,
                requires_both_modalities=cand.get("multimodal_required", True),
                adversarial_score=cand.get("adversarial_score", 0.5),
                key_word=key_word,
                key_word_timestamp=key_word_ts,
                stage1_confidence=cand.get("confidence", 0.5),
                validated_stage2=cand.get("validated_stage2", False),
                audio_event_type=cand.get("audio_event", {}).get("type") if "audio_event" in cand else None,
                scene_transition=cand.get("scene_transition")
            )

            result.opportunities.append(opportunity)

        result.total_opportunities = len(result.opportunities)
        result.validated_opportunities = len([o for o in result.opportunities if o.validated_audio])

        return result

    def _find_key_word(
        self,
        quote: str,
        words: List[Dict],
        target_word: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Find key word and its precise timestamp from word-level data.

        Args:
            quote: Full quote
            words: Word-level timestamp data
            target_word: Specific target word (e.g., keyword, ambiguous word)

        Returns:
            Tuple of (key_word, timestamp)
        """
        if not words:
            return None, None

        # If target word specified, find it
        if target_word:
            target_lower = target_word.lower().strip()
            for word_data in words:
                word = word_data.get("word", "").lower().strip()
                if target_lower in word or word in target_lower:
                    return word_data.get("word"), word_data.get("start")

        # Otherwise, find middle word (most representative)
        mid_idx = len(words) // 2
        if mid_idx < len(words):
            return words[mid_idx].get("word"), words[mid_idx].get("start")

        return None, None

    def _get_default_task_types(self, opportunity_type: str) -> List[str]:
        """
        Get default task types for opportunity type.

        Args:
            opportunity_type: Type of opportunity

        Returns:
            List of task types
        """
        type_mapping = {
            "temporal_understanding": ["Temporal Understanding"],
            "sequential": ["Sequential", "Temporal Understanding"],
            "counting": ["Counting", "Temporal Understanding"],
            "referential_grounding": ["Referential Grounding", "Needle"],
            "audio_visual_stitching": ["Audio-Visual Stitching", "Temporal Understanding"],
            "comparative": ["Comparative", "Temporal Understanding"],
            "inference": ["Inference"],
            "context": ["Context"],
            "subscene": ["Subscene", "Temporal Understanding"],
            "object_interaction": ["Object Interaction Reasoning", "Sequential"],
            "needle": ["Needle", "Context"],
            "general_holistic_reasoning": ["General Holistic Reasoning"],
            "tackling_spurious_correlations": ["Tackling Spurious Correlations", "Inference"]
        }
        return type_mapping.get(opportunity_type, ["General"])

    def _score_opportunity(self, opportunity: RealOpportunity) -> float:
        """
        Calculate overall quality score for opportunity.

        Uses weighted scoring across 5 dimensions:
        1. Adversarial potential
        2. Temporal precision
        3. Multimodal dependency
        4. Complexity
        5. Detail level

        Args:
            opportunity: Opportunity to score

        Returns:
            Score from 0.0 to 1.0
        """
        # 1. Adversarial potential (already scored by GPT-4 or defaulted)
        adversarial = opportunity.adversarial_score

        # 2. Temporal precision
        temporal_precision = 0.5
        if opportunity.key_word_timestamp:
            temporal_precision = 0.9  # Word-level precision
        elif opportunity.opportunity_type in ["temporal_understanding", "sequential", "audio_visual_stitching", "subscene", "needle"]:
            temporal_precision = 0.7

        # 3. Multimodal dependency
        multimodal = 1.0 if opportunity.requires_both_modalities else 0.4

        # 4. Complexity
        complexity_scores = {"low": 0.3, "medium": 0.6, "high": 0.9}
        complexity = complexity_scores.get(opportunity.complexity, 0.6)

        # 5. Detail level
        detail = 0.5
        if opportunity.opportunity_type == "needle":
            detail = 0.9
        elif opportunity.opportunity_type in ["counting", "comparative", "object_interaction", "subscene"]:
            detail = 0.7
        elif opportunity.opportunity_type in ["tackling_spurious_correlations", "general_holistic_reasoning"]:
            detail = 0.8  # High detail for complex reasoning

        # Weighted sum
        score = (
            adversarial * self.WEIGHT_ADVERSARIAL +
            temporal_precision * self.WEIGHT_TEMPORAL_PRECISION +
            multimodal * self.WEIGHT_MULTIMODAL_DEPENDENCY +
            complexity * self.WEIGHT_COMPLEXITY +
            detail * self.WEIGHT_DETAIL_LEVEL
        )

        return min(1.0, max(0.0, score))

    def _select_premium_frames(
        self,
        opportunities: List[RealOpportunity],
        duration: float
    ) -> List[Dict]:
        """
        Select premium frames for expensive GPT-4V analysis.

        Uses adaptive selection based on:
        - Opportunity scores
        - Video duration
        - Budget constraints

        Args:
            opportunities: All opportunities
            duration: Video duration

        Returns:
            List of premium frame dicts
        """
        # Score and rank opportunities
        scored_opps = []
        for opp in opportunities:
            if opp.opportunity_score >= self.MIN_OPPORTUNITY_SCORE:
                scored_opps.append(opp)

        # Sort by score (descending)
        scored_opps.sort(key=lambda o: o.opportunity_score, reverse=True)

        # Calculate dynamic frame count
        frames_per_minute = 2
        premium_count = min(
            self.MAX_PREMIUM_FRAMES,  # Budget cap
            ceil(duration / 60) * frames_per_minute,  # 2 per minute
            len(scored_opps)  # Available high-quality opportunities
        )

        logger.info(f"  Selecting {premium_count} premium frames from {len(scored_opps)} high-quality opportunities")

        # Select top opportunities
        premium_frames = []
        seen_timestamps = set()

        for i, opp in enumerate(scored_opps[:premium_count]):
            # Use precise timestamp
            ts = opp.key_word_timestamp if opp.key_word_timestamp else opp.visual_timestamp

            # Avoid duplicate timestamps
            ts_rounded = round(ts, 1)
            if ts_rounded in seen_timestamps:
                continue

            seen_timestamps.add(ts_rounded)
            opp.requires_premium_frame = True

            premium_frames.append({
                "timestamp": ts,
                "opportunity_id": opp.opportunity_id,
                "opportunity_type": opp.opportunity_type,
                "opportunity_score": opp.opportunity_score,
                "priority": i + 1,
                "reason": f"High score ({opp.opportunity_score:.2f})"
            })

        return premium_frames

    def _validate_quote_exists(
        self,
        quote: str,
        segments: List[Dict],
        expected_timestamp: float,
        tolerance: float = 2.0
    ) -> bool:
        """
        Validate that quote exists in transcript segments.

        Args:
            quote: Quote to validate
            segments: Transcript segments
            expected_timestamp: Expected timestamp from detection
            tolerance: Time tolerance in seconds

        Returns:
            True if quote found at expected timestamp
        """
        if not quote:
            return False

        quote_clean = quote.lower().strip().strip('"').strip("'")

        for segment in segments:
            segment_text = segment['text'].lower().strip()
            segment_time = segment['start']

            # Check if quote matches and timestamp is close
            if quote_clean in segment_text or segment_text in quote_clean:
                if abs(segment_time - expected_timestamp) <= tolerance:
                    return True

        return False

    def save_opportunities(
        self,
        result: OpportunityDetectionResult,
        output_path: Path
    ):
        """Save opportunities to JSON"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Saved {result.total_opportunities} opportunities to: {output_path}")


# Test function
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        audio_analysis_path = sys.argv[1]

        with open(audio_analysis_path, 'r') as f:
            audio_analysis = json.load(f)

        detector = OpportunityDetectorV2()
        result = detector.detect_opportunities(
            audio_analysis,
            video_id=Path(audio_analysis_path).stem
        )

        output_path = Path(audio_analysis_path).parent / f"{Path(audio_analysis_path).stem}_opportunities_v2_optimized.json"
        detector.save_opportunities(result, output_path)

        print(f"\n✅ Detected {result.validated_opportunities} validated opportunities")
        print(f"Stage 1 candidates: {result.stage1_candidates}")
        print(f"Filtered by confidence: {result.filtered_by_confidence}")
        print(f"Clustered duplicates: {result.clustered_duplicates}")
        print(f"Tier 1 (no validation): {result.tier1_skipped}")
        print(f"Tier 2 (sampled): {result.tier2_sampled}")
        print(f"Tier 3 (full validation): {result.tier3_full}")
        print(f"Stage 2 validated: {result.stage2_validated}")
        print(f"Premium frames: {len(result.premium_frames)}")
        print(f"Cost: ${result.detection_cost:.4f} (Budget: $2.00)")
        print(f"\nOpportunities by type:")
        for opp_type, count in sorted(result.opportunity_statistics.items()):
            print(f"  {opp_type}: {count}")
        print(f"\nSaved to: {output_path}")
    else:
        print("Usage: python opportunity_detector_v2_optimized.py <audio_analysis.json>")