import streamlit as st
from scripts.predict import predict
from PIL import Image

st.title("PCOS Predictor")

st.write("Upload an ultrasound image to predict PCOS.")

uploaded_file = st.file_uploader("Upload Ultrasound Image", type=["png","jpg","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    result = predict(uploaded_file)

    st.write("### Prediction:", result)
