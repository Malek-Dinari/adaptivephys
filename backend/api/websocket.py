import uuid
import base64
import numpy as np
import cv2
import traceback
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.config import PipelineConfig
from backend.pipeline.orchestrator import RPPGPipeline

router = APIRouter()

@router.websocket("/ws/vitals")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    config = PipelineConfig()
    pipeline = RPPGPipeline(config)
    session_id = str(uuid.uuid4())
    print(f"[{session_id}] New WebSocket connection initialized")

    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "frame":
                b64_data = data.get("data")
                timestamp = data.get("timestamp", 0)
                
                # Base64 decode to numpy array
                if b64_data.startswith("data:image"):
                    b64_data = b64_data.split(",")[1]
                
                img_bytes = base64.b64decode(b64_data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    try:
                        result = await pipeline.process_frame(frame, timestamp)
                        if result:
                            if result.status == "success" and result.hr is not None:
                                sig = result.signal
                                await websocket.send_json({
                                    "type": "result",
                                    "hr_bpm": float(result.hr.hr_bpm),
                                    "confidence": float(result.hr.confidence),
                                    "model": result.model_name,
                                    "signals": {
                                        "filtered": sig.filtered.tolist()[-min(60, len(sig.filtered)):], # send tail to save bw
                                        "fft_freqs": sig.freqs.tolist(),
                                        "fft_magnitude": sig.spectrum.tolist()
                                    },
                                    "frame_count": result.frame_count
                                })
                            elif result.status == "warmup":
                                await websocket.send_json({
                                    "type": "warmup",
                                    "progress": result.progress
                                })
                            elif result.status == "warning":
                                await websocket.send_json({
                                    "type": "quality_warning",
                                    "message": result.message
                                })
                    except Exception as e:
                        print(f"Pipeline error: {e}")
                        traceback.print_exc()

    except WebSocketDisconnect:
        print(f"[{session_id}] Client disconnected")
