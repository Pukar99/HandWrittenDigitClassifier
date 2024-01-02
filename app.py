import streamlit as st
import os
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image, ImageOps

# Load your trained model
model = tf.keras.models.load_model('notebook\my_mnist_model')

def import_and_predict(image, model):
    # Preprocess the image for digit recognition
    image = ImageOps.grayscale(image)
    image = np.array(image)

    # Resize to expected input size for the model
    image_resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    image_resized = image_resized.reshape(1, 28, 28, 1)
    image_resized = image_resized.astype('float32') / 255

    # Predict the digit
    prediction = model.predict(image_resized)
    return np.argmax(prediction)

def process_uploaded_file(uploaded_file):
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Error: {e}")
        return None
    return image

st.title("MNIST Digit Classifier")
st.header("Upload an image of a handwritten digit")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = process_uploaded_file(uploaded_file)
    if image is not None:
        # Adjust the size of the uploaded image displayed
        st.image(image, caption='Uploaded Image', width=250)  # Width set to 250 pixels
        st.write("Classifying...")
        label = import_and_predict(image, model)
        st.write(f'The handwritten digit is likely a {label}.')
