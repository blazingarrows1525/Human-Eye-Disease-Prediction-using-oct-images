import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
from recommendation import cnv, dme, drusen, normal
import tempfile
import os

# Set Streamlit config for file uploads
st.set_page_config(page_title="Retinal OCT Analysis", layout="wide")

# Tensorflow Model Prediction
def model_prediction(test_image_path):
    try:
        model = tf.keras.models.load_model("Trained_Model.keras")
        img = tf.keras.utils.load_img(test_image_path, target_size=(224, 224))
        x = tf.keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        predictions = model.predict(x)
        return np.argmax(predictions)
    except Exception as e:
        st.error(f"Error in model prediction: {str(e)}")
        return None

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Identification"])

# Home Page
if app_mode == "Home":
    st.markdown("""
    ## **OCT Retinal Analysis Platform**
    
    #### **Welcome to the Retinal OCT Analysis Platform**
    
    Optical Coherence Tomography (OCT) is a powerful imaging technique that provides high-resolution cross-sectional images of the retina.
    
    **This platform uses AI to automatically classify retinal diseases from OCT scans.**
    
    ### Disease Categories:
    - **Normal**: Healthy retina with no abnormalities
    - **CNV**: Choroidal Neovascularization - abnormal blood vessel growth
    - **DME**: Diabetic Macular Edema - fluid accumulation in the macula
    - **Drusen**: Age-related macular degeneration (Early AMD)
    """)

# About Page
elif app_mode == "About":
    st.markdown("""
    ## About Retinal Diseases
    
    ### Choroidal Neovascularization (CNV)
    - Abnormal blood vessel growth beneath the retina
    - Can lead to vision loss if untreated
    
    ### Diabetic Macular Edema (DME)
    - Swelling in the macula caused by diabetes
    - Results from fluid accumulation
    
    ### Drusen (Early AMD)
    - Yellow deposits beneath the retina
    - Sign of age-related macular degeneration
    
    ### Normal Retina
    - No visible abnormalities
    - Healthy retinal structure
    """)

# Disease Identification Page
elif app_mode == "Disease Identification":
    st.header("OCT Image Analysis")
    st.write("Upload an OCT retinal image for AI-powered disease classification")
    
    uploaded_file = st.file_uploader(
        "Choose an OCT image (JPG, PNG):",
        type=["jpg", "jpeg", "png"],
        help="Upload a retinal OCT scan image"
    )
    
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        
        # Display uploaded image
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded OCT Image", use_column_width=True)
        
        with col2:
            st.info("File Details:")
            st.write(f"**Filename**: {uploaded_file.name}")
            st.write(f"**File size**: {uploaded_file.size / 1024:.2f} KB")
        
        # Predict button
        if st.button("🔍 Analyze Image", key="predict_btn"):
            try:
                with st.spinner("Analyzing image with AI model..."):
                    # Save uploaded file to temporary location
                    temp_dir = tempfile.gettempdir()
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Make prediction
                    result_index = model_prediction(temp_file_path)
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_file_path)
                    except:
                        pass
                
                if result_index is not None:
                    class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
                    predicted_class = class_names[result_index]
                    
                    st.success(f"### 🎯 Prediction Result")
                    st.write(f"**Disease Classification: {predicted_class}**")
                    
                    # Show recommendations
                    with st.expander("📋 Learn More About This Condition"):
                        if result_index == 0:
                            st.markdown(cnv)
                        elif result_index == 1:
                            st.markdown(dme)
                        elif result_index == 2:
                            st.markdown(drusen)
                        else:
                            st.markdown(normal)
                else:
                    st.error("Failed to make prediction. Please try again.")
                    
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.info("Please ensure the file is a valid image and try again.")
    else:
        st.info("👆 Please upload an OCT retinal image to begin analysis")
