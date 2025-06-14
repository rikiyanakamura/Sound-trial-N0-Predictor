
import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt

st.set_page_config(page_title="Lymph Node Metastasis Prediction", layout="wide")

st.title("Prediction Tool for Lymph Node Metastasis (pN0)")
st.markdown("Enter patient information to predict probability of node-negative disease.")

# Load data and train model
@st.cache_data
def load_data():
    df = pd.read_csv("Train_data.csv")
    return df

df = load_data()

# Preprocess
target = "metastasis"
features = ['Age', 'Height', 'Weight', 'Axillary Diagnosis', 'Menopause', 'Clinical T stage',
            'CNB Histology', 'Clinical Histologic Grade', 'ER(%)', 'PgR(%)', 'HER2',
            'HER2 protein', 'Tumor size (US)']

df = df.dropna(subset=features + [target])
X = df[features]
y = df[target]

X_encoded = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Input form
st.header("Patient Input")
input_data = {}
for col in features:
    if df[col].dtype == 'object':
        options = sorted(df[col].dropna().unique())
        input_data[col] = st.selectbox(col, options)
    else:
        input_data[col] = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

input_df = pd.DataFrame([input_data])
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Predict
if st.button("Predict"):
    prob = model.predict_proba(input_encoded)[0][1]
    st.success(f"Predicted probability of lymph node metastasis: **{prob:.2%}**")

    # ROC Curve
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    st.subheader("ROC Curve")
    fig, ax = plt.subplots()
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=ax)
    st.pyplot(fig)
