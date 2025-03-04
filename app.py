# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf

# Loading the Model
model = load_model('/content/drive/MyDrive/mymodel.h5')
                    
# Name of Classes
CLASS_NAMES = ("Black_rot", "Esca_(Black_Measles)", "Healthy", "Leaf_blight_(Isariopsis_Leaf_Spot)")

# Setting Title of App
st.title("Grape Plant Disease Detection")
st.markdown("Upload an image of the plant leaf")

# Uploading the dog image
plant_image = st.file_uploader("Choose an image...", type = "jpg")
submit = st.button('predict Disease')

# On predict button click
if submit:
    if plant_image is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray (bytearray(plant_image.read()), dtype = np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        
        # Resizing the image
        opencv_image = cv2.resize(opencv_image, (224, 224))
        
        # Convert image to 4 Dimension
        opencv_image.shape = (1, 224, 224, 3)
        
        #Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        # Safely split class label and display a meaningful message
        if '-' in result:
          disease_info = result.split('-')
          leaf_type = disease_info[0]
          disease_name = disease_info[1]
          st.title(f"This is a {leaf_type} leaf with {disease_name}")
        else:
          st.title(f"This is a {result} leaf")
