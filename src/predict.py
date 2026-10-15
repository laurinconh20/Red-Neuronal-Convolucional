import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

CLASS_NAMES = [
    "freshapples", "freshbanana", "freshoranges",
    "rottenapples", "rottenbanana", "rottenoranges"
]

parser = argparse.ArgumentParser(description="Predice el estado de una fruta")
parser.add_argument("imagen", help="Ruta de la imagen")
parser.add_argument("--modelo", default="models/modelo_frutas_final.keras")
args = parser.parse_args()

model = tf.keras.models.load_model(args.modelo)
image = Image.open(args.imagen).convert("RGB").resize((224, 224))
array = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)
probabilities = model.predict(array, verbose=0)[0]
index = int(np.argmax(probabilities))

print(f"Clase predicha: {CLASS_NAMES[index]}")
print(f"Confianza: {probabilities[index] * 100:.2f}%")
