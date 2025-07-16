
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("rice_model.h5")
    return model

# Class labels
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Streamlit UI
st.title("🍚 Rice Classification using CNN")
st.write("Upload an image of rice grain to predict its type.")

uploaded_file = st.file_uploader("Choose a rice grain image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = image.resize((100, 100))
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    model = load_model()
    prediction = model.predict(image_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Predicted Rice Type: **{predicted_class}**")
