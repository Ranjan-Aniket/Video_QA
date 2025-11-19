"""WebSocket handler for log streaming"""
from fastapi import APIRouter, WebSocket

router = APIRouter()

@router.websocket("/logs")
async def websocket_logs(websocket: WebSocket):
    """WebSocket endpoint for real-time log streaming"""
    await websocket.accept()
    
    try:
        while True:
            # Stream logs
            await websocket.send_json({
                'timestamp': '2024-11-16 12:30:45',
                'level': 'info',
                'message': 'Processing video 123...'
            })
            
    except:
        print("Logs client disconnected")