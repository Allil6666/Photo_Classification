import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

model = load_model('ecommerce_model.h5')  # 确保模型文件在同一目录

class_labels = {0: 't-shirt', 1: 'pants', 2: 'shoes'}

def preprocess_and_predict(image_path):
    image = load_img(image_path, target_size=(150, 150))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]
    return predicted_label

st.title('E-commerce Product Classifier')

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image_path = f"static/uploads/{uploaded_file.name}"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(image_path, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = preprocess_and_predict(image_path)
    st.write(f'Prediction: {label}')
