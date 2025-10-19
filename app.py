import os
import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline (preprocessor + classifier)
model_path = os.path.join(os.path.dirname(__file__), "logistic_model.pkl")
model = joblib.load(model_path)

st.title("🚢 Logistic Regression Survival Predictor")
st.write("This app predicts survival probability based on passenger information.")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)
embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

# Collect inputs into dataframe
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'Fare': [fare],
    'Embarked': [embarked]
})

# Predict using the full pipeline (preprocessing + model)
if st.button("Predict Survival"):
    try:
        pred = model.predict(input_data)[0]
        pred_prob = model.predict_proba(input_data)[0][1]

        st.write(f"**Prediction:** {'✅ Survived' if pred == 1 else '❌ Did not survive'}")
        st.write(f"**Survival Probability:** {pred_prob:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Make sure the input features match what the model expects.")
