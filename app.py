import os
import gdown
import tensorflow as tf
import streamlit as st

@st.cache_resource()
def load_model():
    model_path = 'human_detection_model_dataset.h5'
    
    # Check if the model file already exists
    if not os.path.exists(model_path):
        # Replace 'your_model_id' with the actual file ID from Google Drive
        url = 'https://drive.google.com/uc?id=1abcd12345efghijklmnop'  # Update with your Google Drive file ID
        gdown.download(url, model_path, quiet=False)
    
    # Check if the file was downloaded successfully
    if os.path.exists(model_path):
        st.write("Model downloaded successfully.")
    else:
        st.error("Failed to download the model.")
        return None

    # Load the model after downloading
    try:
        model = tf.keras.models.load_model(model_path)
        st.write("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    
    return model
