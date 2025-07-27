{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5368424f-9ae6-4cd0-a147-c6d17afdb317",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import pickle\n",
    "from sklearn.metrics import mean_squared_error, r2_score\n",
    "\n",
    "# === Load trained model ===\n",
    "with open(\"XGBoost_SSO_UCS.pkl\", \"rb\") as file:\n",
    "    model_loaded = pickle.load(file)\n",
    "\n",
    "# === Streamlit UI ===\n",
    "st.set_page_config(page_title=\"UCS Prediction (XGBoost + SSO)\", layout=\"centered\")\n",
    "st.title(\"UCS Prediction using XGBoost Optimized with SSO\")\n",
    "\n",
    "st.markdown(\"\"\"\n",
    "**Developed by:**\n",
    "- Dr. Divesh Ranjan Kumar, Chulalongkorn University  \n",
    "- Dr. Shashikant Kumar, Muzaffarpur Institute of Technology  \n",
    "- Dr. Teerapong Senjuntichai, Chulalongkorn University  \n",
    "- Dr. Sakdirat Kaewunruen, University of Birmingham  \n",
    "📧 Contact: ranjandivesh453@gmail.com  \n",
    "\"\"\")\n",
    "\n",
    "st.subheader(\"Enter Input Parameters\")\n",
    "\n",
    "# === Input Form ===\n",
    "with st.form(\"input_form\"):\n",
    "    pond_ash = st.number_input(\"Pond Ash (gms)\", min_value=0.0, step=0.1)\n",
    "    cement = st.number_input(\"Cement (gms)\", min_value=0.0, step=0.1)\n",
    "    sp = st.number_input(\"SP (gms)\", min_value=0.0, step=0.01)\n",
    "    lime = st.number_input(\"Lime Content (gms)\", min_value=0.0, step=0.1)\n",
    "    curing_days = st.number_input(\"Curing Period (days)\", min_value=1.0, step=1.0)\n",
    "    \n",
    "    submitted = st.form_submit_button(\"Predict\")\n",
    "\n",
    "if submitted:\n",
    "    input_data = pd.DataFrame([[\n",
    "        pond_ash, cement, sp, lime, curing_days\n",
    "    ]], columns=['Pond Ash', 'Cement', 'SP', 'Lime Content', 'Curing period'])\n",
    "\n",
    "    prediction = model_loaded.predict(input_data)[0]\n",
    "    prediction = round(prediction, 2)\n",
    "    st.success(f\"Predicted UCS: **{prediction} MPa**\")\n",
    "\n",
    "    # === Export Section ===\n",
    "    if st.button(\"Export Prediction to CSV\"):\n",
    "        result_df = input_data.copy()\n",
    "        result_df[\"UCS (MPa)\"] = prediction\n",
    "        csv = result_df.to_csv(index=False).encode(\"utf-8\")\n",
    "        st.download_button(\"Download CSV\", csv, \"ucs_prediction.csv\", \"text/csv\")\n",
    "\n",
    "# === Footer ===\n",
    "st.markdown(\"---\")\n",
    "st.markdown(\"© 2025 | UCS Prediction Tool (XGBoost + SSO)\")\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
