Classification d'images sur Raspberry Pi
Ce dépôt contient le code d'un projet de classification d'images déployé sur un Raspberry Pi en utilisant des modèles TensorFlow Lite. Le projet consiste à exécuter des modèles d'apprentissage automatique (ML) pour classifier des images capturées via une caméra Pi ou téléchargées par l'utilisateur, avec un accent sur des modèles efficaces en énergie adaptés aux systèmes embarqués.

Table des matières
Présentation du projet
Modèles utilisés
Instructions d'installation
Exécution du projet
Temps d'inférence et consommation d'énergie
Technologies utilisées
Contributions

Présentation du projet
Ce projet se concentre sur la classification d'images à faible consommation d'énergie en utilisant différents modèles de deep learning optimisés pour les appareils embarqués comme le Raspberry Pi 4. Le système permet une classification en temps réel à partir d'une caméra Pi ou via l'upload manuel d'images. L'objectif est de trouver un équilibre entre la précision, la vitesse d'inférence et la consommation d'énergie pour une solution pratique d'IA embarquée.

Objectifs clés :
Classification d'images écoénergétique sur un Raspberry Pi 4.
Faible latence d'inférence avec des modèles TensorFlow Lite.
Comparaison de différents modèles pour évaluer les compromis entre performance et consommation énergétique.
Modèles utilisés
Trois modèles TensorFlow Lite sont implémentés et comparés :

Binary Weight Network (BWN) - Modèle optimisé pour l'efficacité énergétique avec des poids binaires.
Convolutional Neural Network (CNN) - Modèle classique de deep learning pour la classification d'images.
MobileNet - Réseau léger optimisé pour les systèmes mobiles et embarqués.
Ces modèles sont entraînés et ajustés pour classer un ensemble de catégories prédéfinies.

Instructions d'installation
Matériel requis :
Raspberry Pi 4
PiCamera (ou une caméra externe)
Carte microSD (16 Go ou plus)
Connexion Internet
Logiciels requis :
Raspberry Pi OS (Buster ou version plus récente)
Python 3.x
TensorFlow Lite
Flask (pour l'interface web)
Git
Installation :
Cloner ce dépôt :
git clone https://github.com/HANINasandratra/Image_classification.git

Installer les dépendances :
pip install -r requirements.txt

Activer la caméra Pi:
sudo raspi-config

Se rendre dans le répertoire du projet :
cd ~/image_classification

Exécution du projet
Lancer le serveur web Flask :
python app.py

Accéder à l'interface web : Ouvrir un navigateur et accéder à l'adresse http://<ip_raspberry_pi>:5000. Vous pourrez capturer une image à l'aide de la PiCamera ou télécharger une image pour la classifier.

Affichage des résultats : Les résultats de la classification (étiquette prédite, score de confiance) seront affichés sur la page web, avec l'image correspondante.
![image](https://github.com/user-attachments/assets/b22a1d08-fa42-4be6-b108-1d4e6a43a46a)
![image](https://github.com/user-attachments/assets/b357b5e5-9d42-4040-9362-2cfe060b4d1f)
![image](https://github.com/user-attachments/assets/ba31e48c-8c0c-419d-8227-0eb529f1ca5f)




Temps d'inférence et consommation d'énergie
Pour chaque modèle, le temps d'inférence et la consommation d'énergie sont mesurés directement sur le Raspberry Pi en utilisant des outils comme time pour la durée d'exécution et du matériel externe pour le profilage énergétique. Les résultats sont présentés dans le rapport inclus dans ce dépôt.

Technologies utilisées
TensorFlow Lite - pour exécuter des modèles de deep learning sur le Raspberry Pi.
Flask - pour créer une interface web simple permettant de capturer et classifier des images.
PiCamera - pour capturer des images directement à partir du module caméra du Raspberry Pi.
Python - pour le scripting et l'exécution de l'application.
Contributions
Les contributions sont les bienvenues ! N'hésitez pas à forker ce dépôt, à apporter des modifications et à soumettre des pull requests. Toute amélioration des performances des modèles, de l'efficacité énergétique ou de l'interface web est appréciée.
