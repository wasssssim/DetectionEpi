import cv2
import threading
import time

class CameraStream:
    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.frame = None
        self.status = False
        self.is_running = True
        
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        print(f"DEBUG: Tentative de connexion à {self.source}")
        while self.is_running:
            if self.cap.isOpened():
                (self.status, frame) = self.cap.read()
                if self.status:
                    self.frame = frame
            else:
                print(f"ERREUR: Impossible de se connecter à {self.source}")
            time.sleep(0.01) 

    def get_frame(self):
        return self.frame

    def stop(self):
        self.is_running = False
        self.cap.release()