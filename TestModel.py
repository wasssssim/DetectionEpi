import cv2
from ultralytics import YOLO
import math
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--droidcam', action='store_true', help='Utiliser DroidCam au lieu de la webcam')
parser.add_argument('--ip', type=str, help="Adresse IP de DroidCam (ex: 192.168.1.28)")
parser.add_argument('--port', type=str, default="4747", help="Port de DroidCam (défaut: 4747)")
parser.add_argument('--conf', type=float, default=0.5, help="Seuil de confiance (0.1 à 1.0)")
parser.add_argument('--model', type=str, default="casque_v1.pt", help="Chemin vers le modèle")

args = parser.parse_args()

if args.droidcam:
    if not args.ip:
        print("ERREUR : Vous devez spécifier l'IP pour DroidCam !")
        print("Exemple : python TestVideo.py --droidcam --ip 192.168.1.25")
        exit()
    
    source_video = f"http://{args.ip}:{args.port}/video"
    print(f"Connexion à DroidCam sur : {source_video}")
else:
    source_video = 0
    print(" Utilisation de la Webcam par défaut ")




model = YOLO('runs\detect\model_casque8\weights\\best.pt')

        
classNames = ["Casque"]

cap = cv2.VideoCapture(source_video)

cap.set(3, 640)
cap.set(4, 480)

print(" Flux vidéo démarré. Appuyez sur 'q' pour quitter.")

while True:
    success, img = cap.read()
    if not success:
        print(" Flux vidéo perdu.")
        break

    # Détection
    results = model(img, stream=True, conf=args.conf)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            # Gestion des classes
            cls = int(box.cls[0])
            label = classNames[cls] if cls < len(classNames) else "Inconnu"

            # Design
            color = (0, 255, 0) # Vert
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, f'{label} {conf}', (max(0, x1), max(35, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Detection EPI (q pour quitter)", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()