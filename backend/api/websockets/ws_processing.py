"""WebSocket handler for processing updates"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

@router.websocket("/processing/{batch_id}")
async def websocket_processing(websocket: WebSocket, batch_id: int):
    """WebSocket endpoint for real-time processing updates"""
    await websocket.accept()
    
    try:
        while True:
            # Listen for messages from client
            data = await websocket.receive_text()
            
            # Send updates (would be triggered by Celery in production)
            await websocket.send_json({
                'type': 'progress_update',
                'batch_id': batch_id,
                'video_id': 123,
                'progress_percent': 65,
                'current_stage': 'Q&A Generation'
            })
            
    except WebSocketDisconnect:
        print(f"Client disconnected from batch {batch_id}")