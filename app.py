import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('/content/drive/MyDrive/datasets/human_detection_model_dataset.h5')

# Function to preprocess the input image
def preprocess_image(img, target_size=(128, 128)):
    img = img.resize(target_size)  # Resize the image
    img_array = np.array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch processing
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Streamlit app code
st.title("Human Detection Application")
st.write("Upload an image to check if it contains a human.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    
    if prediction[0] > 0.5:
        st.write("Prediction: Human Detected")
    else:
        st.write("Prediction: No Human Detected")
