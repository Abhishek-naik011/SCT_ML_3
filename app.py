# app.py
import streamlit as st
from PIL import Image
import numpy as np
import pickle

# ------------------------------
# 1️⃣ Load pre-trained model
# ------------------------------
model = pickle.load(open("svm_model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

image_size = 64

# ------------------------------
# 2️⃣ Streamlit UI
# ------------------------------
st.title("🐱🐶 Cat vs Dog Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    img_resized = img.resize((image_size, image_size))
    arr = np.array(img_resized).reshape(1, -1)
    
    prediction = model.predict(arr)
    label_pred = le.inverse_transform(prediction)[0]
    
    st.write(f"### Predicted Label: {label_pred.upper()}")
