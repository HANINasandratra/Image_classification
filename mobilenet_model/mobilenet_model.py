from flask import Flask, render_template, request
from picamera import PiCamera
import time
import numpy as np
import os
from tflite_runtime.interpreter import Interpreter
from PIL import Image

app = Flask(__name__)
camera = PiCamera()

# Charger le modèle TFLite pour MobileNet
model_path = "mobilenet_model_quantized.tflite"  
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Définir le seuil de confiance (70% par exemple)
threshold = 0.7

@app.route('/')
def index():
    # Préparation du preview pour le display
    camera.start_preview(fullscreen=False, window=(100, 20, 640, 480))  # Ajuster la taille du preview si nécessaire
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    try:
        camera.start_preview()
        time.sleep(2)  # Délai pour ajuster l'image de la caméra

        # Capture temporaire de l'image
        temp_image_path = os.path.join('static', 'capture_temp.jpg')
        camera.capture(temp_image_path)
        camera.stop_preview()

        # Préparation de l'image pour le modèle
        img = Image.open(temp_image_path)
        img = img.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        # Exécution de la prédiction
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data)

        # Liste des labels (à vérifier selon ton modèle MobileNet)
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'frog', 'horse', 'ship', 'truck']
        accuracy = np.max(output_data) * 100

        # Vérification du seuil de confiance
        if accuracy >= threshold:
            predicted_label = labels[prediction]
            image_path = os.path.join('static', f'{predicted_label}.jpg')
        else:
            predicted_label = "unknown"
            accuracy = 0.0
            image_path = os.path.join('static', 'unknown.jpg')

        # Renommer l'image capturée avec le label prédit
        os.rename(temp_image_path, image_path)

    except Exception as e:
        # Gestion d'erreur
        camera.close()
        return render_template('index.html', result="Error capturing image", accuracy=0, image_path=""), 500

    finally:
        camera.close()  # Fermer la caméra après usage

    return render_template('index.html', result=predicted_label, accuracy=accuracy, image_path=image_path)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Vérification si un fichier est bien uploadé
        if 'image' not in request.files:
            return render_template('index.html', result="No file uploaded", accuracy=0, image_path="")

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', result="No selected file", accuracy=0, image_path="")

        # Sauvegarder l'image uploadée
        upload_path = os.path.join('static', 'uploaded_image.jpg')
        file.save(upload_path)

        # Préparer l'image pour le modèle
        img = Image.open(upload_path)
        img = img.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        # Exécuter le modèle pour prédiction
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data)

        # Liste des labels
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'frog', 'horse', 'ship', 'truck']
        accuracy = np.max(output_data) * 100

        # Vérification du seuil de confiance
        predicted_label = labels[prediction] if accuracy >= threshold else "unknown"

    except Exception as e:
        # Gestion des erreurs lors de l'upload
        return render_template('index.html', result=f"Error processing image: {str(e)}", accuracy=0, image_path=""), 500

    return render_template('index.html', result=predicted_label, accuracy=accuracy, image_path=upload_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

