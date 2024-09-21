from flask import Flask, render_template, request
from picamera import PiCamera
import time
import numpy as np
import os
from tflite_runtime.interpreter import Interpreter
from PIL import Image

app = Flask(__name__)
camera = PiCamera()

# Charger le modèle TFLite
model_path = "bwn_model_quantized.tflite"
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Définir le seuil de confiance
threshold = 0.7  # 70% confidence threshold

@app.route('/')
def index():
    # Démarrer le preview sur le display du Raspberry Pi
    camera.start_preview(fullscreen=False, window=(100, 20, 640, 480))  # Ajuster la position et la taille du preview
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    try:
        camera.start_preview()
        time.sleep(2)  # Temps pour ajuster la caméra

        predicted_label = "unknown"
        image_path = os.path.join('static', f'{predicted_label}.jpg')

        # Capturer l'image
        temp_image_path = os.path.join('static', 'capture_temp.jpg')
        camera.capture(temp_image_path)
        camera.stop_preview()

        # Préparer l'image pour le modèle
        img = Image.open(temp_image_path)
        img = img.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        # Exécuter le modèle pour faire la prédiction
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data)

        # Liste des labels
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'frog', 'horse', 'ship', 'truck']
        accuracy = np.max(output_data) * 100

        # Vérifier et nommer l'image capturée
        if accuracy >= threshold:
            predicted_label = labels[prediction]
            image_path = os.path.join('static', f'{predicted_label}.jpg')
        else:
            accuracy = 0.0
            predicted_label = "unknown"

        os.rename(temp_image_path, image_path)
    finally:
        camera.close()  # Ferme la caméra pour libérer les ressources.

    return render_template('index.html', result=predicted_label, accuracy=accuracy, image_path=image_path)

@app.route('/upload', methods=['POST'])
def upload():
    # Gérer le téléchargement d'image
    if 'image' not in request.files:
        return render_template('index.html', result="No file uploaded.", accuracy=0, image_path="")
    
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', result="No selected file.", accuracy=0, image_path="")
    
    # Sauvegarder l'image téléchargée
    upload_path = os.path.join('static', 'uploaded_image.jpg')
    file.save(upload_path)

    # Préparer l'image pour le modèle
    img = Image.open(upload_path)
    img = img.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    # Exécuter le modèle pour faire la prédiction
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)

    # Liste des labels
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'frog', 'horse', 'ship', 'truck']
    accuracy = np.max(output_data) * 100

    # Vérifier et nommer l'image uploadée
    predicted_label = labels[prediction] if accuracy >= threshold else "unknown"
    
    return render_template('index.html', result=predicted_label, accuracy=accuracy, image_path=upload_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


