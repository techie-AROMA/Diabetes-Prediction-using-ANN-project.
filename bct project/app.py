import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

model = tf.keras.models.load_model("diabetes_model.h5")
scaler = pickle.load(open("scaler.pkl","rb"))

st.title("Diabetes Prediction System")

preg = st.number_input("Pregnancies")
glucose = st.number_input("Glucose")
bp = st.number_input("Blood Pressure")
skin = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

if st.button("Predict"):

    input_data = np.array([[preg,glucose,bp,skin,insulin,bmi,dpf,age]])
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    if prediction > 0.5:
        st.error("High chance of Diabetes")
    else:
        st.success("Low chance of Diabetes")