import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Liver Disease Stage Prediction", layout="wide")

DATA_PATH = "liver_c.csv"

# Sample fallback data in case the real file is missing
sample_csv_data = """
Age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alanine_Aminotransferase,Aspartate_Aminotransferase,Total_Proteins,Albumin,Albumin_and_Globulin_Ratio,Gender,Stage
65,0.7,0.1,187,16,18,6.8,3.3,1.0,Male,1
62,10.9,5.5,699,64,100,7.5,3.2,0.74,Female,2
46,7.3,4.1,490,60,68,7.0,3.3,0.89,Female,2
"""

@st.cache_data
def load_and_train_model():
    # Load real or fallback sample dataset
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, encoding='latin1')
    else:
        st.warning("⚠️ 'liver_c.csv' not found. Using sample data.")
        df = pd.read_csv(io.StringIO(sample_csv_data.strip()))

    df.drop_duplicates(inplace=True)

    # Label encode categorical features
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop("Stage", axis=1)
    y = df["Stage"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)

    return df, model, scaler, label_encoders, X_train, X_test, y_train, y_test

df, model, scaler, label_encoders, X_train, X_test, y_train, y_test = load_and_train_model()

st.title("🩺 Liver Disease Stage Prediction App")

menu = st.sidebar.radio("Select Option", ["Show Dataset", "Predict on New Data", "Model Evaluation"])

if menu == "Show Dataset":
    st.subheader("📊 Dataset Preview")
    st.dataframe(df)

elif menu == "Predict on New Data":
    st.subheader("🔍 Predict Liver Disease Stage")

    with st.form("prediction_form"):
        input_data = {}

        for col in df.drop("Stage", axis=1).columns:
            if col in label_encoders:
                input_data[col] = st.selectbox(col, label_encoders[col].classes_)
            else:
                input_data[col] = st.number_input(col, value=float(df[col].mean()))

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([input_data])

        for col in input_df.columns:
            if col in label_encoders:
                le = label_encoders[col]
                input_df[col] = le.transform([input_df[col]])

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        st.success(f"🎯 Predicted Stage: **{prediction}**")

elif menu == "Model Evaluation":
    st.subheader("📈 Model Performance")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    st.write(f"✅ **Accuracy:** {acc:.2f}")
    st.text("Classification Report:")
    st.text(report)
