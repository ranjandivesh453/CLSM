import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, r2_score

# === Load trained model ===
with open("XGBoost_SSO_UCS.pkl", "rb") as file:
    model_loaded = pickle.load(file)

# === Streamlit UI ===
st.set_page_config(page_title="UCS Prediction (XGBoost + SSO)", layout="centered")
st.title("UCS Prediction using XGBoost Optimized with SSO")

st.markdown("""
**Developed by:**
- Dr. Divesh Ranjan Kumar, Chulalongkorn University  
- Dr. Shashikant Kumar, Muzaffarpur Institute of Technology  
- Dr. Teerapong Senjuntichai, Chulalongkorn University  
- Dr. Sakdirat Kaewunruen, University of Birmingham  
📧 Contact: ranjandivesh453@gmail.com  
""")

st.subheader("Enter Input Parameters")

# === Input Form ===
with st.form("input_form"):
    pond_ash = st.number_input("Pond Ash (gms)", min_value=0.0, step=0.1)
    cement = st.number_input("Cement (gms)", min_value=0.0, step=0.1)
    sp = st.number_input("SP (gms)", min_value=0.0, step=0.01)
    lime = st.number_input("Lime Content (gms)", min_value=0.0, step=0.1)
    curing_days = st.number_input("Curing Period (days)", min_value=1.0, step=1.0)
    
    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame([[
        pond_ash, cement, sp, lime, curing_days
    ]], columns=['Pond Ash', 'Cement', 'SP', 'Lime Content', 'Curing period'])

    prediction = model_loaded.predict(input_data)[0]
    prediction = round(prediction, 2)
    st.success(f"Predicted UCS: **{prediction} MPa**")

    # === Export Section ===
    if st.button("Export Prediction to CSV"):
        result_df = input_data.copy()
        result_df["UCS (MPa)"] = prediction
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "ucs_prediction.csv", "text/csv")

# === Footer ===
st.markdown("---")
st.markdown("© 2025 | UCS Prediction Tool (XGBoost + SSO)")
