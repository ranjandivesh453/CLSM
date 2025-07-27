import streamlit as st
import pandas as pd
import pickle

# === Load trained model ===
with open("XGBoost_SSO_UCS.pkl", "rb") as file:
    model_loaded = pickle.load(file)

# === Set up Streamlit page ===
st.set_page_config(page_title="UCS Prediction (XGBoost + SSO)", layout="centered")
st.title("🧱 UCS Prediction using Hybrid XGBoost + SSO")

# === Developer Info ===
with st.expander("ℹ️ About"):
    st.markdown("""
    **Developed by:**
    - Dr. Divesh Ranjan Kumar, Chulalongkorn University  
    - Dr. Shashikant Kumar, Muzaffarpur Institute of Technology  
    - Dr. Teerapong Senjuntichai, Chulalongkorn University  
    - Dr. Sakdirat Kaewunruen, University of Birmingham  

    📧 Contact: ranjandivesh453@gmail.com  
    """)

# === Input Form ===
st.subheader("📝 Enter Input Parameters")

with st.form("prediction_form"):
    pond_ash = st.number_input("Pond Ash (gms)", min_value=0.0, step=0.1)
    cement = st.number_input("Cement (gms)", min_value=0.0, step=0.1)
    sp = st.number_input("SP (gms)", min_value=0.0, step=0.01)
    lime = st.number_input("Lime Content (gms)", min_value=0.0, step=0.1)
    curing_days = st.number_input("Curing Period (days)", min_value=1.0, step=1.0)

    col1, col2 = st.columns(2)
    predict_btn = col1.form_submit_button("🔍 Predict")
    clear_btn = col2.form_submit_button("🧹 Clear")

# === Handle Prediction ===
if predict_btn:
    input_df = pd.DataFrame([[
        pond_ash, cement, sp, lime, curing_days
    ]], columns=['Pond Ash', 'Cement', 'SP', 'Lime Content', 'Curing period'])

    prediction = model_loaded.predict(input_df)[0]
    prediction = round(prediction, 2)

    st.success(f"✅ Predicted UCS: **{prediction} MPa**")

    # Export CSV
    st.markdown("### 📤 Export Prediction to CSV")
    export_df = input_df.copy()
    export_df["UCS (MPa)"] = prediction
    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("📄 Download CSV", csv_data, "ucs_prediction.csv", "text/csv")

# === Clear inputs by rerunning app ===
if clear_btn:
    st.experimental_rerun()

# === Footer ===
st.markdown("---")
st.markdown("© 2025 | UCS Prediction Tool (XGBoost + SSO)")
