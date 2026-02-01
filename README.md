# DetectionEpi 

Système de détection de port de casque de sécurité (EPI) en temps réel basé sur **YOLOv8**.
Ce projet utilise l'apprentissage profond (Deep Learning) pour vérifier via une webcam ou un flux vidéo si une personne porte son équipement de protection.

---

##  CONFIGURATION  

**AVANT DE LANCER L'ENTRAÎNEMENT, VOUS DEVEZ MODIFIER LE FICHIER DE CONFIGURATION.**

YOLO nécessite un **chemin absolu** (chemin complet sur votre disque) pour localiser les images. Ce chemin change d'un ordinateur à l'autre.

1. Ouvrez le fichier : `Datasets/data.yaml`
2. Modifiez la ligne `path:` avec **VOTRE** chemin local complet.


**Exemple de modification à faire :**

```yaml
#  NE LAISSEZ PAS ÇA :
# path: C:\Users\raouf\Cours\5A\DetectionEpi\Datasets

# REMPLACEZ PAR VOTRE CHEMIN (Anass si tu me dis que ça marche pas à cause de ça de je t'encule) :
path: C:\Users\Clemeeeeento\Documents\Projets\DetectionEpi\Datasets

train: train/images
val: valid/images
test: test/images 
```
##  Installation
Si vous récupérez le projet pour la première fois :
```bash
 pip install -r requirements.txt
```

##  Entraînement

Pour lancer un nouvel entraînement du modèle sur votre GPU :

Récupérer un Datasets 
```bash
 python Train.py
```

##  Utilisation / Exécution

Le script principal `TestVideo.py` permet de lancer la détection en temps réel. Il est compatible avec une webcam classique ou une caméra IP (DroidCam).

### 1. Webcam Classique (USB)
Pour lancer la détection avec la webcam par défaut de l'ordinateur.

```bash
python TestModel.py
```
### 2. DroidCam (Téléphone)

```bash
python TestModel.py --droidcam --ip 192.168.1.25 

```

### 3. Options avancées 

Vous pouvez personnaliser l'exécution avec des arguments :

| Argument | Description | Exemple |
| :--- | :--- | :--- |
| `--model` | Choisir un fichier modèle spécifique | `--model best.pt` |
| `--conf` | Régler la sensibilité (0.1 à 1.0) | `--conf 0.7` |
| `--port` | Changer le port DroidCam (défaut 4747) | `--port 4747` |

```bash

python TestVideo.py --droidcam --ip 192.168.1.25 --conf 0.6 --model runs/detect/train/weights/best.pt
```
