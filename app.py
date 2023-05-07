import streamlit as st
import numpy as np
from pickle import load
from streamlit_extras.colored_header import colored_header

st.set_page_config(page_title="Breast Cancer Diagnosis", page_icon=":guardsman:", layout="wide")
st.title("Breast Cancer Diagnosis")

scaler = load(open('models/standard_scaler.pkl', 'rb'))
lr_model = load(open('models/lr_model.pkl', 'rb'))

# HEADER
colored_header(
    label=":syringe: :green[Welcome to Open Pubs!] :syringe:",
    description="This app predicts weather the mass is simply benign or malignant",
    color_name="blue-70",
)

rm = st.text_input("Radius Mean", placeholder="Enter value :")
tm = st.text_input("Texture Mean", placeholder="Enter value :")
sm = st.text_input("Smoothness Mean", placeholder="Enter value :")
com_m = st.text_input("Compactness Mean", placeholder="Enter value :")
con_m = st.text_input("Concavity Mean", placeholder="Enter value :")

buttonn = st.button("PREDICT")

if buttonn == True:
    if rm and tm and sm and com_m and con_m :
        query_point = np.array([float(rm), float(tm), float(sm), float(com_m),float(con_m)]).reshape(1, -1)
        query_point_transformed = scaler.transform(query_point)
        pred = lr_model.predict(query_point_transformed)
        st.success(pred)
    else:
        st.error("The values arent entered properly.🚨")