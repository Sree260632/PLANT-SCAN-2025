import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

# Load and preprocess the image
def model_predict(image_path):
    # Correct model path
    model = tf.keras.models.load_model(r"C:\Users\sreeram\Downloads\trained_model.keras")
    
    # Read and process the image
    img = cv2.imread(image_path)
    H, W, C = 224, 224, 3
    img = cv2.resize(img, (H, W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0  # Rescale
    img = img.reshape(1, H, W, C)  # Reshape
    
    # Predict and return results
    prediction = model.predict(img)
    result_index = np.argmax(prediction)
    confidence = np.max(prediction)
    
    return result_index, confidence

# Sidebar for navigation
st.sidebar.title("🌱 Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Displaying home image
img = Image.open(r"C:\Users\sreeram\Downloads\logo app.png")
st.image(img)

# Main Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

elif app_mode == "DISEASE RECOGNITION":
    st.header("🌿 Plant Disease Recognition")
    test_image = st.file_uploader("Upload an Image:", type=["jpg", "png", "jpeg"])
    
    if test_image is not None:
        save_path = os.path.join(os.getcwd(), test_image.name)
        with open(save_path, "wb") as f:
            f.write(test_image.getbuffer())
        
        if st.button("Show Image"):
            st.image(test_image, width=400, use_column_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("🔍 **Model is analyzing...**")
            
            result_index, confidence = model_predict(save_path)

            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                          'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                          'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                          'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                          'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                          'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                          'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                          'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                          'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                          'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                          'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                          'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                          'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                          'Tomato___healthy']
            
            st.success(f"✅ Model predicts: **{class_name[result_index]}** (Confidence: **{confidence:.2f}**)")

            # Clean up temporary image
            os.remove(save_path)


