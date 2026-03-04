import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from core.camera import CameraStream
from core.detector import EPIDetector
import asyncio
import json
import os
import uvicorn 
from fastapi.staticfiles import StaticFiles 
from contextlib import asynccontextmanager 
from fastapi.responses import HTMLResponse, FileResponse

ALERTS_DIR = os.path.join("data", "alerts")
os.makedirs(ALERTS_DIR, exist_ok=True)
app = FastAPI()

# Initialisation (A adapter avec tes IPs)
detector = EPIDetector(model_path="../runs\detect\model_gilet2\weights\\best.pt")
cameras = {
    "cam_1": CameraStream("http://192.168.1.157:4747/video") 
}

# Gestion des clients WebSockets (pour les alertes)
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

# --- LOGIQUE DE DÉTECTION EN ARRIÈRE-PLAN ---
async def background_monitoring():
    while True:
        for cam_id, cam in cameras.items():
            frame = cam.get_frame()
            if frame is not None:
                _, detections = detector.process_frame(frame)
                
                for d in detections:
                    if d['label'] == "no_safety_vest" and d['conf'] > 0.6:
                        filename = f"data/alerts/alert_{cam_id}.jpg"
                        cv2.imwrite(filename, frame)
                        
                        await manager.broadcast({
                            "type": "ALERTE",
                            "camera": cam_id,
                            "msg": "EPI MANQUANT !",
                            "img_url": f"/static/alerts/alert_{cam_id}.jpg"
                        })
        await asyncio.sleep(0.5) 

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Démarrage du monitoring IA...")
    task = asyncio.create_task(background_monitoring())
    
    yield 
    
    print("Arrêt du monitoring...")
    task.cancel()
app.mount("/static", StaticFiles(directory=ALERTS_DIR), name="static")
app = FastAPI(lifespan=lifespan)

def generate_frames(cam_id):
    print(f"DEBUG: Requête de flux pour {cam_id}")
    while True:
        frame = cameras[cam_id].get_frame()

        if frame is not None:
            annotated_frame, _ = detector.process_frame(frame)
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
@app.get("/")
async def get_index():
    return FileResponse("web/index.html")
@app.get("/video/{cam_id}")
async def video_feed(cam_id: str):
    return StreamingResponse(generate_frames(cam_id), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)