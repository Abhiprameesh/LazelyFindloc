import streamlit as st
import os
import sys
import tempfile
from PIL import Image

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.inference import LandmarkClassifier

# Set paths
MODEL_PATH = os.path.abspath(r"C:\ALLMLPROJ\lazyfinder\model\lazy_landmark_resnet18.pth")

# Class names (fixed order)
class_names = [
    "Big Ben",
    "Burj Khalifa", 
    "Eiffel Tower",
    "Statue of Liberty",
    "Taj Mahal"
]

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = LandmarkClassifier(MODEL_PATH, class_names)

# App title and description
st.title("Lazy Landmark Finder")
st.write("Upload an image of a famous landmark and let the model guess which one it is!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        # Convert to RGB if needed before saving as JPEG
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(tmp_file.name, format='JPEG')
        
        # Make prediction
        prediction = st.session_state.classifier.predict(tmp_file.name)
    
    # Display prediction
    st.success(f"Predicted Landmark: {prediction}")
    
    # Clean up
    try:
        os.unlink(tmp_file.name)
    except:
        pass