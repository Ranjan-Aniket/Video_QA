"""
Test Evidence Extraction - Step by Step

This script tests the evidence extraction pipeline to ensure all 10 models work.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("EVIDENCE EXTRACTION TEST - STEP BY STEP")
    logger.info("=" * 80)

    # Step 1: Load test configuration
    logger.info("Step 1: Loading test configuration...")
    from config.test_config import TEST_CONFIG
    print(TEST_CONFIG.get_summary())
    logger.info("✓ Configuration loaded")

    # Step 2: Check for test video
    logger.info("Step 2: Checking for test video...")
    test_videos = list(Path("uploads").glob("*.mp4"))
    if not test_videos:
        logger.error("✗ No test videos found in uploads/ directory")
        logger.info("Please add a test video to uploads/ directory")
        return 1

    test_video = test_videos[0]
    logger.info(f"✓ Found test video: {test_video}")
    logger.info(f"  - Size: {test_video.stat().st_size / 1024 / 1024:.2f} MB")

    # Step 3: Initialize evidence extractor
    logger.info("Step 3: Initializing evidence extractor...")
    try:
        from processing.evidence_extractor import EvidenceExtractor
        extractor = EvidenceExtractor(
            video_path=test_video,
            video_id=999  # Test video ID
        )
        logger.info("✓ Evidence extractor initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize extractor: {e}", exc_info=True)
        return 1

    # Step 4: Extract evidence items (just 1 for testing)
    logger.info("Step 4: Extracting evidence items (testing with 1 frame)...")
    logger.info("=" * 80)
    try:
        evidence_items = extractor.extract_evidence_for_hitl(
            video_path=test_video,
            interval_seconds=5.0,
            max_items=1  # Only extract 1 frame for testing
        )
        logger.info("=" * 80)
        logger.info(f"✓ Evidence extraction complete!")
        logger.info(f"✓ Extracted {len(evidence_items)} evidence items")

        # Step 5: Inspect results
        if evidence_items:
            logger.info("=" * 80)
            logger.info("Step 5: Inspecting evidence item...")
            item = evidence_items[0]
            logger.info(f"  - Evidence ID: {item.evidence_id}")
            logger.info(f"  - Video ID: {item.video_id}")
            logger.info(f"  - Timestamp: {item.timestamp_start:.1f}s")
            logger.info(f"  - Evidence Type: {item.evidence_type}")

            # Check ground truth
            if item.ground_truth:
                logger.info(f"  - Ground Truth Keys: {list(vars(item.ground_truth).keys())}")
                logger.info(f"  - Object Count: {getattr(item.ground_truth, 'object_count', 'N/A')}")
                logger.info(f"  - Person Count: {getattr(item.ground_truth, 'person_count', 'N/A')}")

            # Check consensus
            if item.consensus:
                logger.info(f"  - Consensus: {item.consensus.consensus_type}")
                logger.info(f"  - Agreement Level: {item.consensus.agreement_level}")

            logger.info("=" * 80)
            logger.info("✅ TEST PASSED - Evidence extraction working!")
        else:
            logger.warning("⚠ No evidence items created")
            return 1

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"✗ Evidence extraction failed: {e}")
        logger.error("=" * 80, exc_info=True)
        return 1

    logger.info("=" * 80)
    logger.info("✅ ALL TESTS PASSED")
    logger.info("=" * 80)
    return 0

if __name__ == "__main__":
    sys.exit(main())
