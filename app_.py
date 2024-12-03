import os
import time
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load your pre-trained model (ensure the model is in the same directory or provide the correct path)
MODEL_PATH = "thyroid_segmentation_model.h5"  # Replace with your model's path
model = load_model(MODEL_PATH)

# Function to preprocess the uploaded image
def preprocess_image(image):
    """
    Preprocess the uploaded image to match the input shape required by the model.
    """
    # Resize the image to the target dimensions (e.g., 224x224 or as per your model)
    target_size = (224, 224)
    image = image.resize(target_size)
    image_array = img_to_array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to predict using the model
def predict_segmentation(image):
    """
    Perform segmentation using the loaded model.
    """
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return prediction[0]  # Assuming the model outputs a single prediction

# Measure performance
def measure_performance(start_time, operation_name):
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    st.write(f"Performance of {operation_name}: {duration} seconds")

# Main function
def main():
    st.set_page_config(page_title="THYRO-ViT: Thyroid Segmentation", layout="wide")

    # Project Overview
    st.markdown("""
        <h1 style="text-align:center; color: #6a5acd;">THYRO-ViT: Thyroid Cancer Segmentation</h1>
        <p style="font-size:18px; color: #4682b4;">
            Welcome to THYRO-ViT, an advanced tool that uses **Vision Transformer (ViT)** technology to segment thyroid cancer from medical images like ultrasound and CT scans.
            This tool is designed to provide early insights into thyroid cancer detection, empowering healthcare providers with AI-driven assistance.
        </p>
    """, unsafe_allow_html=True)

    # Instructions for the user
    st.markdown("""
        <h2 style="color:#4682b4;">How to Use This Tool:</h2>
        <ol style="font-size:16px; color:#4682b4;">
            <li>Click on the <strong>"Upload Image"</strong> button below to upload your ultrasound or CT scan image.</li>
            <li>Once uploaded, the tool will process your image and display the corresponding segmented result.</li>
            <li>Use the segmented result to visualize potential areas of concern for thyroid cancer.</li>
        </ol>
    """, unsafe_allow_html=True)

    # Upload section
    st.markdown("<h2 style='color:#6a5acd;'>Upload Your Image</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="upload")
    
    if uploaded_file is not None:
        upload_start_time = time.time()

        # Display the uploaded image
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        measure_performance(upload_start_time, "Image Upload")

        # Predict segmentation
        st.markdown("<h3 style='color:#6a5acd;'>Segmented Result</h3>", unsafe_allow_html=True)
        predict_start_time = time.time()

        try:
            segmentation_result = predict_segmentation(uploaded_image)
            segmentation_image = Image.fromarray((segmentation_result * 255).astype(np.uint8))
            st.image(segmentation_image, caption="Segmented Image", use_column_width=True)
            measure_performance(predict_start_time, "Prediction")
        except Exception as e:
            st.error(f"Error during segmentation: {e}")

    # Feedback button
    if st.button("Feedback", key="feedback"):
        st.markdown("""
            <h3 style='color:#6a5acd;'>We Value Your Feedback!</h3>
            <p style="color:#4682b4;">
                Please help us improve by providing your feedback.
                <a href="https://forms.gle/mAyKPqAq9xaoz4WV9" target="_blank" style="text-decoration: none; color: #1f77b4;">
                Click here to fill out the feedback form</a>.
            </p>
        """, unsafe_allow_html=True)

# Run the Streamlit application
if __name__ == "__main__":
    main()
