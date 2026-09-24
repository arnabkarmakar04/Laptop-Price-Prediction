import streamlit as st
import pickle
import numpy as np
import pandas as pd

@st.cache_resource
def load_artifacts():
    with open("pipe.pkl", "rb") as f:
        pipe = pickle.load(f)
    with open("df.pkl", "rb") as f:
        df = pickle.load(f)
    return pipe, df

pipe, df = load_artifacts()

st.title("Laptop Price Predictor")

company = st.selectbox("Brand", sorted(df["Company"].unique()))
laptop_type = st.selectbox("Type", sorted(df["TypeName"].unique()))
inches = st.number_input("Display Size (inches)", min_value=10.0, max_value=18.0, value=13.3, step=0.1)
screen_type = st.selectbox("Screen Type", sorted(df["ScreenType"].unique()))
resolution = st.selectbox("Screen Resolution", sorted(df["Resolution"].unique()))
cpu = st.selectbox("CPU", sorted(df["Cpu"].unique()))
ram = st.selectbox("RAM (GB)", sorted(df["Ram"].unique()))
gpu = st.selectbox("GPU", sorted(df["Gpu"].unique()))
os = st.selectbox("Operating System", sorted(df["OpSys"].unique()))
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=6.0, value=1.5, step=0.01)
hdd = st.selectbox("HDD (GB)", sorted(df["HDD"].unique()))
ssd = st.selectbox("SSD (GB)", sorted(df["SSD"].unique()))

if st.button("Predict Price"):
    x_res, y_res = map(int, resolution.split("x"))
    ppi = np.sqrt(x_res ** 2 + y_res ** 2) / inches

    query = pd.DataFrame([{
        "Company": company,
        "TypeName": laptop_type,
        "Inches": inches,
        "Cpu": cpu,
        "Ram": ram,
        "Gpu": gpu,
        "OpSys": os,
        "Weight": weight,
        "ScreenType": screen_type,
        "Resolution": resolution,
        "ppi": ppi,
        "HDD": hdd,
        "SSD": ssd
    }])

    prediction = np.exp(pipe.predict(query)[0])

    st.success(f"Predicted Price: ₹{prediction:,.2f}")