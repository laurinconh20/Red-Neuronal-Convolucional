import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Clasificador de frutas", page_icon="🍎", layout="centered")

CLASS_NAMES = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges",
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "modelo_frutas_final.keras")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def readable_name(label):
    names = {
        "freshapples": "Manzana fresca",
        "freshbanana": "Banano fresco",
        "freshoranges": "Naranja fresca",
        "rottenapples": "Manzana podrida",
        "rottenbanana": "Banano podrido",
        "rottenoranges": "Naranja podrida",
    }
    return names.get(label, label)

st.title("🍎 Clasificador de frutas")
st.write("Carga una imagen para identificar la fruta y determinar si está fresca o podrida.")

uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_container_width=True)

    try:
        model = load_model()
        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        probabilities = model.predict(image_array, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(probabilities[predicted_index]) * 100

        st.success(f"Predicción: {readable_name(predicted_class)}")
        st.metric("Confianza", f"{confidence:.2f}%")

        st.subheader("Probabilidades por clase")
        for label, probability in zip(CLASS_NAMES, probabilities):
            st.write(f"{readable_name(label)}: {probability * 100:.2f}%")
            st.progress(float(probability))
    except Exception as error:
        st.error(f"No fue posible cargar el modelo o realizar la predicción: {error}")
