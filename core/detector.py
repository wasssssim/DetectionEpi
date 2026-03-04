from ultralytics import YOLO
import cv2

class EPIDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.class_names = ["no_safety_vest", "safety_vest"]

    def process_frame(self, frame):
        if frame is None:
            return None, []

        # Inférence
        results = self.model(frame, stream=False, conf=self.conf, verbose=False)
        detections = []
        annotated_frame = frame.copy()

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.class_names[cls] if cls < len(self.class_names) else "Inconnu"

                # Stockage pour la logique d'alerte
                detections.append({"label": label, "conf": conf, "box": [x1, y1, x2, y2]})

                # Dessin (Optionnel : on peut le faire uniquement si un client regarde)
                color = (0, 255, 0) if label == "safety_vest" else (0, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return annotated_frame, detections