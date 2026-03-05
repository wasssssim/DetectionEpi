import cv2
import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from datetime import datetime
from picamera2 import Picamera2


from core.camera import CameraStream
from core.detector import EPIDetector

# --- CONFIGURATION ---
ALERTS_DIR = os.path.join("data", "alerts")
os.makedirs(ALERTS_DIR, exist_ok=True)
infractions_history = [] 

# Correction du chemin (utilisation de / pour éviter les soucis Windows/Linux)
MODEL_PATH = "runs/detect/model_gilet2/weights/best.pt"

# Initialisation des composants


class PiCamWrapper:
    def __init__(self):
        self.picam = Picamera2()
        # Configuration standard pour le Pi 5
        config = self.picam.create_preview_configuration(main={"format": "BGR24", "size": (640, 480)})
        self.picam.configure(config)
        self.picam.start()

    def get_frame(self):
        # On capture une image directement en format OpenCV (BGR)
        return self.picam.capture_array()

# --- Initialisation ---
picam2_wrapped = PiCamWrapper()
detector = EPIDetector(model_path=MODEL_PATH)

cameras = {
    "cam_1": CameraStream("http://192.168.1.157:4747/video"),
    "cam_2": picam2_wrapped  # Utilise le wrapper ici !
}



class ConnectionManager:
    def __init__(self):
        self.active_connections = []
    async def connect(self, ws):
        await ws.accept()
        self.active_connections.append(ws)
    def disconnect(self, ws):
        if ws in self.active_connections:
            self.active_connections.remove(ws)
    async def broadcast(self, data):
        for conn in self.active_connections:
            try:
                await conn.send_json(data)
            except:
                continue # Évite de bloquer si une connexion est morte

manager = ConnectionManager()

# --- LOGIQUE IA ---
async def monitor_loop():
    while True:
        for cid, cam in cameras.items(): # cid est la clé (ex: "cam_1")
            img = cam.get_frame()
            if img is not None:
                # On analyse l'image
                annotated_img, detections = detector.process_frame(img)
                
                for d in detections:
                    if d['label'] == "NO_VEST":
                        # On enregistre l'image ANNOTÉE (avec les dessins) pour la preuve
                        fname = f"alert_{cid}_{datetime.now().strftime('%H%M%S')}.jpg"
                        save_path = os.path.join(ALERTS_DIR, fname)
                        cv2.imwrite(save_path, annotated_img) 
                        
                        new_alert = {
                            "id": len(infractions_history) + 1,
                            "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                            "epi": "Gilet de sécurité",
                            "camera": cid, # <-- FIX : on utilise 'cid', pas 'cam_id'
                            "photo": f"/static/{fname}"
                        }
                        infractions_history.append(new_alert)
                        # Alerte en temps réel via WebSocket
                        await manager.broadcast({"type": "ALERTE", **new_alert})
        
        # Pause de 0.5s pour ne pas brûler le CPU de la RPi5 inutilement
        await asyncio.sleep(0.5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 SafeGate : Démarrage du moteur IA...")
    task = asyncio.create_task(monitor_loop())
    yield
    print("🛑 SafeGate : Arrêt du système...")
    task.cancel()

app = FastAPI(lifespan=lifespan)

# --- ROUTES ---
app.mount("/static", StaticFiles(directory=ALERTS_DIR), name="static")
app.mount("/web", StaticFiles(directory="web"), name="web")

@app.get("/")
async def get_index():
    return FileResponse("web/index.html")

@app.get("/video/{cam_id}")
async def video_feed(cam_id: str):
    if cam_id not in cameras:
        return {"error": "Camera non trouvée"}
        
    def frame_gen():
        while True:
            frame = cameras[cam_id].get_frame()
            if frame is not None:
                # On dessine les box sur le flux live
                annotated_frame, _ = detector.process_frame(frame)
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    return StreamingResponse(frame_gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stats")
async def get_stats():
    return {"today": len(infractions_history), "yesterday": 0}

@app.get("/history")
async def get_history():
    return infractions_history[::-1][:10] # 10 derniers

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    # Important : host 0.0.0.0 pour être accessible depuis l'IP de la RPi
    uvicorn.run(app, host="0.0.0.0", port=8000)