import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Path to the model file
MODEL_PATH = 'anchal_model.h5'

# Load your model
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    st.error(f"Model file '{MODEL_PATH}' not found. Please ensure the file is in the correct location.")
    st.stop()

# Preprocess the uploaded image
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Make a prediction on the image
def predict_image(model, img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    return predictions

# Streamlit app
def main():
    st.title("Bike Prediction Model")
    
    # File uploader for user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        # Save the uploaded image to a temporary file
        with open("temp_image", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Make predictions
        predictions = predict_image(model, "temp_image")
        
        # Display top-5 predictions
        st.write("Predictions:")
        for i, prediction in enumerate(predictions[0]):
            st.write(f"{i + 1}: {prediction * 100:.2f}%")
        
        # Plot the predictions
        labels = [f"Class {i}" for i in range(len(predictions[0]))]
        scores = [score * 100 for score in predictions[0]]
        
        fig, ax = plt.subplots()
        ax.barh(labels, scores, color='skyblue')
        ax.set_xlabel('Confidence (%)')
        ax.set_title('Predictions')
        plt.gca().invert_yaxis()
        st.pyplot(fig)
        
        # Allow user to download the uploaded image
        st.download_button(
            label="Download Image",
            data=uploaded_file,
            file_name=uploaded_file.name,
            mime="image/jpeg"
        )

if __name__ == "__main__":
    main()
