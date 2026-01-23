from ultralytics import YOLO

if __name__ == "__main__":
    # 1. On charge un modèle "squelette" pré-entraîné
    # 'yolov8n.pt' = nano (Le plus rapide, moins précis) -> Recommandé pour votre vidéo
    # 'yolov8s.pt' = small (Un peu plus lent, plus précis) -> Si le nano rate des casques
    model = YOLO('yolov8n.pt') 

    # 2. On lance l'entraînement
    results = model.train(
        data='Datasets/data.yaml',  
        epochs=50,       # Combien de fois il voit tout le dataset (50 est un bon début)
        imgsz=640,       
        batch=16,        # Combien d'images il traite en parallèle
        name='model_casque' # Le nom du dossier de sauvegarde
)