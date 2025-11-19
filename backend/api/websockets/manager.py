"""
WebSocket Connection Manager

Manages WebSocket connections for real-time review updates.
Supports multiple concurrent connections per video_id.
"""
from fastapi import WebSocket
from typing import Dict, List, Set
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections for real-time notifications.

    Features:
    - Multiple connections per video_id
    - Broadcast messages to all connections for a video
    - Connection lifecycle management
    - Automatic cleanup on disconnect
    """

    def __init__(self):
        # Maps video_id -> list of active WebSocket connections
        self.active_connections: Dict[str, List[WebSocket]] = {}

        # Track all connection metadata
        self.connection_metadata: Dict[WebSocket, dict] = {}

    async def connect(self, websocket: WebSocket, video_id: str):
        """
        Accept and register a new WebSocket connection.

        Args:
            websocket: The WebSocket connection
            video_id: Video ID this connection is monitoring
        """
        await websocket.accept()

        # Initialize connection list for this video if needed
        if video_id not in self.active_connections:
            self.active_connections[video_id] = []

        # Add connection to video's list
        self.active_connections[video_id].append(websocket)

        # Store metadata
        self.connection_metadata[websocket] = {
            "video_id": video_id,
            "connected_at": datetime.now().isoformat(),
            "messages_sent": 0
        }

        logger.info(
            f"WebSocket connected: video_id={video_id}, "
            f"total_connections={len(self.active_connections[video_id])}"
        )

        # Send welcome message
        await self._send_to_connection(websocket, {
            "type": "connected",
            "video_id": video_id,
            "timestamp": datetime.now().isoformat(),
            "message": "WebSocket connection established"
        })

    def disconnect(self, websocket: WebSocket):
        """
        Remove and cleanup a WebSocket connection.

        Args:
            websocket: The WebSocket connection to remove
        """
        # Get metadata before removing
        metadata = self.connection_metadata.get(websocket, {})
        video_id = metadata.get("video_id", "unknown")

        # Remove from active connections
        if video_id in self.active_connections:
            if websocket in self.active_connections[video_id]:
                self.active_connections[video_id].remove(websocket)

            # Clean up empty lists
            if not self.active_connections[video_id]:
                del self.active_connections[video_id]

        # Remove metadata
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]

        logger.info(
            f"WebSocket disconnected: video_id={video_id}, "
            f"messages_sent={metadata.get('messages_sent', 0)}"
        )

    async def _send_to_connection(self, websocket: WebSocket, message: dict):
        """
        Send message to a single WebSocket connection.

        Args:
            websocket: Target connection
            message: Message dictionary to send
        """
        try:
            await websocket.send_json(message)

            # Update message counter
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["messages_sent"] += 1

        except Exception as e:
            logger.error(f"Failed to send message to WebSocket: {e}")
            self.disconnect(websocket)

    async def broadcast_to_video(self, video_id: str, message: dict):
        """
        Broadcast message to all connections monitoring a video.

        Args:
            video_id: Target video ID
            message: Message dictionary to broadcast
        """
        if video_id not in self.active_connections:
            logger.debug(f"No active connections for video_id={video_id}")
            return

        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()

        # Send to all connections for this video
        connections = self.active_connections[video_id].copy()  # Copy to avoid modification during iteration

        for websocket in connections:
            await self._send_to_connection(websocket, message)

        logger.debug(
            f"Broadcasted message to {len(connections)} connections: "
            f"video_id={video_id}, type={message.get('type', 'unknown')}"
        )

    # ==================== NOTIFICATION METHODS ====================

    async def notify_stage1_ready(self, video_id: str, question_count: int):
        """
        Notify that Stage 1 review is ready.

        Args:
            video_id: Video ID
            question_count: Number of questions generated
        """
        await self.broadcast_to_video(video_id, {
            "type": "stage1_ready",
            "video_id": video_id,
            "stage": 1,
            "question_count": question_count,
            "message": f"Stage 1 ready: {question_count} questions generated",
            "action_required": "Review and approve/reject questions"
        })

    async def notify_stage2_ready(self, video_id: str, failed_count: int):
        """
        Notify that Stage 2 review is ready.

        Args:
            video_id: Video ID
            failed_count: Number of failed questions awaiting selection
        """
        await self.broadcast_to_video(video_id, {
            "type": "stage2_ready",
            "video_id": video_id,
            "stage": 2,
            "failed_count": failed_count,
            "message": f"Stage 2 ready: Select final 4 from {failed_count} failed questions",
            "action_required": "Select 4 questions for manual labeling"
        })

    async def notify_pipeline_complete(self, video_id: str, total_questions: int):
        """
        Notify that entire pipeline is complete.

        Args:
            video_id: Video ID
            total_questions: Total number of approved questions
        """
        await self.broadcast_to_video(video_id, {
            "type": "pipeline_complete",
            "video_id": video_id,
            "status": "completed",
            "total_questions": total_questions,
            "message": f"Pipeline complete: {total_questions} questions finalized"
        })

    async def notify_error(self, video_id: str, error_message: str, stage: str = None):
        """
        Notify about an error in the pipeline.

        Args:
            video_id: Video ID
            error_message: Error description
            stage: Stage where error occurred (optional)
        """
        await self.broadcast_to_video(video_id, {
            "type": "error",
            "video_id": video_id,
            "stage": stage,
            "error": error_message,
            "message": f"Error: {error_message}"
        })

    async def notify_question_approved(self, video_id: str, question_id: int, reviewer: str):
        """
        Notify that a question was approved.

        Args:
            video_id: Video ID
            question_id: Question ID
            reviewer: Reviewer who approved it
        """
        await self.broadcast_to_video(video_id, {
            "type": "question_approved",
            "video_id": video_id,
            "question_id": question_id,
            "reviewer": reviewer,
            "message": f"Question {question_id} approved by {reviewer}"
        })

    async def notify_question_rejected(
        self,
        video_id: str,
        question_id: int,
        reviewer: str,
        feedback: str
    ):
        """
        Notify that a question was rejected.

        Args:
            video_id: Video ID
            question_id: Question ID
            reviewer: Reviewer who rejected it
            feedback: Rejection feedback
        """
        await self.broadcast_to_video(video_id, {
            "type": "question_rejected",
            "video_id": video_id,
            "question_id": question_id,
            "reviewer": reviewer,
            "feedback": feedback,
            "message": f"Question {question_id} rejected by {reviewer}"
        })

    async def notify_evidence_updated(
        self,
        video_id: str,
        evidence_id: int,
        action: str,
        reviewer: str
    ):
        """
        Notify that evidence was updated.

        Args:
            video_id: Video ID
            evidence_id: Evidence ID
            action: Action performed (approved/rejected/skipped)
            reviewer: Reviewer who performed the action
        """
        await self.broadcast_to_video(video_id, {
            "type": "evidence_updated",
            "video_id": video_id,
            "evidence_id": evidence_id,
            "action": action,
            "reviewer": reviewer,
            "message": f"Evidence {evidence_id} {action} by {reviewer}"
        })

    async def notify_progress(
        self,
        video_id: str,
        stage: str,
        progress_percent: float,
        current_step: str
    ):
        """
        Notify about processing progress.

        Args:
            video_id: Video ID
            stage: Current stage name
            progress_percent: Progress percentage (0-100)
            current_step: Description of current step
        """
        await self.broadcast_to_video(video_id, {
            "type": "progress_update",
            "video_id": video_id,
            "stage": stage,
            "progress_percent": progress_percent,
            "current_step": current_step,
            "message": f"{stage}: {current_step} ({progress_percent:.1f}%)"
        })

    # ==================== UTILITY METHODS ====================

    def get_connection_count(self, video_id: str = None) -> int:
        """
        Get number of active connections.

        Args:
            video_id: If provided, count for specific video. Otherwise, total count.

        Returns:
            Number of active connections
        """
        if video_id:
            return len(self.active_connections.get(video_id, []))
        else:
            return sum(len(conns) for conns in self.active_connections.values())

    def get_connected_videos(self) -> List[str]:
        """
        Get list of video IDs with active connections.

        Returns:
            List of video IDs
        """
        return list(self.active_connections.keys())

    def get_stats(self) -> dict:
        """
        Get WebSocket manager statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_connections": self.get_connection_count(),
            "active_videos": len(self.active_connections),
            "videos": {
                video_id: len(connections)
                for video_id, connections in self.active_connections.items()
            }
        }


# Global manager instance
manager = WebSocketManager()
