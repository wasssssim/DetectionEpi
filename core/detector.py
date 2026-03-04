import cv2

class EPIDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf_threshold

    def process_frame(self, frame):
        if frame is None: return None, []
        
        results = self.model(frame, stream=False, conf=self.conf, verbose=False)
        detections = []
        annotated_frame = frame.copy()

        for r in results:
            for box in r.boxes:
                # Récupération des coordonnées et infos
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                # Supposons : 0 = no_safety_vest, 1 = safety_vest
                label = "NO_VEST" if cls == 0 else "OK_VEST"
                
                # --- LE DESSIN ---
                # Rouge pour danger, Vert pour conforme
                color = (0, 0, 255) if cls == 0 else (0, 255, 0) 
                
                # Rectangle autour de la personne
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                
                # Label avec fond coloré pour la lisibilité
                cv2.rectangle(annotated_frame, (x1, y1-25), (x1+100, y1), color, -1)
                cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1+5, y1-8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                detections.append({"label": label, "conf": conf, "box": [x1, y1, x2, y2]})

        return annotated_frame, detections