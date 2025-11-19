"""
Evidence Review Stage - Pipeline stage for human-in-the-loop evidence review

Manages the evidence review workflow:
1. Waits for evidence extraction to complete
2. Checks if items need human review
3. Pauses pipeline until review is complete
4. Resumes question generation with reviewed evidence
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import time

from database.evidence_operations import EvidenceOperations
from database.operations import VideoOperations

logger = logging.getLogger(__name__)


class EvidenceReviewStage:
    """
    Pipeline stage that manages evidence review workflow

    This stage sits between evidence extraction and question generation,
    ensuring that human review is complete before proceeding.
    """

    def __init__(self):
        """Initialize evidence review stage"""
        self.video_ops = VideoOperations
        self.evidence_ops = EvidenceOperations

    def should_run(self, video_id: str) -> bool:
        """
        Check if evidence review stage should run for this video

        Args:
            video_id: Video ID

        Returns:
            True if stage should run, False otherwise
        """
        video = self.video_ops.get_video_by_id(video_id)

        if not video:
            logger.error(f"Video {video_id} not found")
            return False

        # Check if evidence extraction is complete
        evidence_status = video.get('evidence_extraction_status', 'pending')

        if evidence_status == 'pending':
            logger.info(f"Evidence extraction not started for {video_id}")
            return False

        if evidence_status == 'extracting':
            logger.info(f"Evidence extraction in progress for {video_id}")
            return False

        if evidence_status in ['ai_complete', 'awaiting_review']:
            return True

        if evidence_status == 'review_complete':
            logger.info(f"Evidence review already complete for {video_id}")
            return False

        return False

    def run(self, video_id: str) -> Dict[str, Any]:
        """
        Run evidence review stage

        Args:
            video_id: Video ID

        Returns:
            Result dictionary with status and stats
        """
        logger.info("=" * 80)
        logger.info(f"EVIDENCE REVIEW STAGE - Video {video_id}")
        logger.info("=" * 80)

        try:
            # Check if review is needed
            progress = self.evidence_ops.get_review_progress(video_id)

            total_items = progress.get('total', 0)
            needs_review = progress.get('needs_review', 0)
            pending = progress.get('pending', 0)

            logger.info(f"Evidence items: {total_items}")
            logger.info(f"Needs review: {needs_review}")
            logger.info(f"Pending review: {pending}")

            if total_items == 0:
                logger.warning(f"No evidence items found for {video_id}")
                self._update_video_status(video_id, 'review_complete')
                return {
                    'status': 'complete',
                    'message': 'No evidence items to review',
                    'items_reviewed': 0
                }

            if pending > 0:
                # Review is pending - update status and wait
                logger.info(f"Waiting for {pending} items to be reviewed")
                self._update_video_status(video_id, 'awaiting_review')

                return {
                    'status': 'waiting',
                    'message': f'Waiting for {pending} items to be reviewed',
                    'pending': pending,
                    'total': total_items,
                    'percent_complete': ((total_items - pending) / total_items * 100) if total_items > 0 else 0
                }

            # All reviews complete!
            logger.info("✅ All evidence items reviewed!")

            # Update video statistics
            self.evidence_ops.update_video_evidence_stats(video_id)

            # Mark review complete
            self._update_video_status(video_id, 'review_complete')

            # Get final stats
            approved = progress.get('approved', 0)
            corrected = progress.get('corrected', 0)
            rejected = progress.get('rejected', 0)
            skipped = progress.get('skipped', 0)

            logger.info(f"Review complete:")
            logger.info(f"  Approved: {approved}")
            logger.info(f"  Corrected: {corrected}")
            logger.info(f"  Rejected: {rejected}")
            logger.info(f"  Skipped: {skipped}")

            return {
                'status': 'complete',
                'message': 'Evidence review complete',
                'total_items': total_items,
                'approved': approved,
                'corrected': corrected,
                'rejected': rejected,
                'skipped': skipped,
                'accuracy_estimate': self._estimate_accuracy(progress)
            }

        except Exception as e:
            logger.error(f"Evidence review stage failed: {e}", exc_info=True)
            self._update_video_status(video_id, 'review_failed', str(e))
            raise

    def _update_video_status(
        self,
        video_id: str,
        status: str,
        error_message: Optional[str] = None
    ):
        """Update video evidence review status"""
        try:
            update_data = {'evidence_extraction_status': status}
            if error_message:
                update_data['error_message'] = error_message

            self.video_ops.update_video(video_id, **update_data)
            logger.info(f"Updated video {video_id} status to: {status}")
        except Exception as e:
            logger.error(f"Failed to update video status: {e}")

    def _estimate_accuracy(self, progress: Dict) -> float:
        """
        Estimate overall accuracy based on review results

        Args:
            progress: Review progress dict

        Returns:
            Estimated accuracy (0.0 to 1.0)
        """
        total = progress.get('total', 0)
        if total == 0:
            return 0.0

        approved = progress.get('approved', 0)
        corrected = progress.get('corrected', 0)

        # Approved items: AI was correct
        # Corrected items: AI was wrong
        # (Rejected and skipped don't count toward accuracy)

        items_with_decision = approved + corrected
        if items_with_decision == 0:
            return 0.0

        accuracy = approved / items_with_decision
        return round(accuracy, 3)

    def wait_for_review(
        self,
        video_id: str,
        timeout_seconds: int = 3600,
        check_interval: int = 10
    ) -> bool:
        """
        Wait for review to complete (blocking)

        Args:
            video_id: Video ID
            timeout_seconds: Maximum time to wait
            check_interval: How often to check (seconds)

        Returns:
            True if review completed, False if timeout
        """
        logger.info(f"Waiting for evidence review to complete (timeout: {timeout_seconds}s)")

        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            if elapsed > timeout_seconds:
                logger.warning(f"Review timeout after {timeout_seconds}s")
                return False

            progress = self.evidence_ops.get_review_progress(video_id)
            pending = progress.get('pending', 0)

            if pending == 0:
                logger.info("✅ Review complete!")
                return True

            logger.info(f"Waiting... {pending} items pending ({elapsed:.0f}s elapsed)")
            time.sleep(check_interval)

    def get_review_url(self, video_id: str, base_url: str = "http://localhost:3000") -> str:
        """
        Get URL for reviewing evidence for this video

        Args:
            video_id: Video ID
            base_url: Frontend base URL

        Returns:
            URL to review page
        """
        return f"{base_url}/reviews/evidence?video_id={video_id}"

    def send_review_notification(self, video_id: str) -> bool:
        """
        Send notification that evidence is ready for review

        Could trigger:
        - Email to reviewers
        - Slack message
        - WebSocket notification to frontend

        Args:
            video_id: Video ID

        Returns:
            True if notification sent successfully
        """
        try:
            # Get video info
            video = self.video_ops.get_video_by_id(video_id)
            if not video:
                return False

            # Get review stats
            progress = self.evidence_ops.get_review_progress(video_id)
            needs_review = progress.get('needs_review', 0)

            # Log notification (in production, send actual notification)
            logger.info("=" * 80)
            logger.info("📧 REVIEW NOTIFICATION")
            logger.info(f"Video: {video.get('original_filename', video_id)}")
            logger.info(f"Items needing review: {needs_review}")
            logger.info(f"Review URL: {self.get_review_url(video_id)}")
            logger.info("=" * 80)

            # TODO: Implement actual notification system
            # - Email via SendGrid/AWS SES
            # - Slack webhook
            # - WebSocket push to connected clients

            return True

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False


class EvidenceReviewOrchestrator:
    """
    Orchestrates evidence review workflow across multiple videos

    Useful for batch processing or managing review queue.
    """

    def __init__(self):
        """Initialize orchestrator"""
        self.stage = EvidenceReviewStage()
        self.evidence_ops = EvidenceOperations

    def get_videos_needing_review(self) -> List[Dict]:
        """
        Get all videos that need evidence review

        Returns:
            List of video dicts
        """
        # Query videos with status 'awaiting_review'
        videos = VideoOperations.get_videos_by_pipeline_stage('awaiting_review')

        return videos

    def get_review_queue_summary(self) -> Dict[str, Any]:
        """
        Get summary of all pending reviews across all videos

        Returns:
            Summary dict with stats
        """
        # Get all evidence items needing review
        high_priority = self.evidence_ops.get_review_queue(priority='high', limit=1000)
        medium_priority = self.evidence_ops.get_review_queue(priority='medium', limit=1000)
        low_priority = self.evidence_ops.get_review_queue(priority='low', limit=1000)

        return {
            'total_pending': len(high_priority) + len(medium_priority) + len(low_priority),
            'high_priority': len(high_priority),
            'medium_priority': len(medium_priority),
            'low_priority': len(low_priority),
            'videos_waiting': len(self.get_videos_needing_review())
        }

    def process_completed_reviews(self) -> List[str]:
        """
        Check all videos and advance those with completed reviews

        Returns:
            List of video IDs that were advanced
        """
        videos = self.get_videos_needing_review()
        advanced = []

        for video in videos:
            video_id = video['video_id']

            try:
                result = self.stage.run(video_id)

                if result['status'] == 'complete':
                    advanced.append(video_id)
                    logger.info(f"Advanced {video_id} to question generation")

            except Exception as e:
                logger.error(f"Failed to process {video_id}: {e}")

        return advanced
