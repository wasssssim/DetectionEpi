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

# REMPLACEZ PAR VOTRE CHEMIN (Exemple pour un collègue) :
path: C:\Users\JEAN\Documents\Projets\DetectionEpi\Datasets

train: train/images
val: valid/images
test: test/images