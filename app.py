import streamlit as st
import joblib
import numpy as np

# Load models at the beginning to avoid reloading
sc = joblib.load("sc.pkl")
model = joblib.load("model.pkl")

st.title("Churn Prediction App")
st.divider()
st.write("Please Enter the values and hit the prediction button for getting prediction")
st.divider()

# Input fields
age = st.number_input("Enter Age", min_value=10, max_value=100, value=30)
nps = st.number_input("Enter NPS", min_value=0, max_value=130, value=10)
gender_encoded = st.selectbox("Enter Gender", ["Male", "Female"])
segment_encoded = st.number_input("Enter Segment", min_value=30, max_value=150)

st.divider()

predictbutton = st.button("Predict!")

st.divider()

if predictbutton:
    # Gender encoding
    gender_encoded_selected = 1 if gender_encoded == "Female" else 0
    
    # Prepare input features
    X = [age, nps, gender_encoded_selected, segment_encoded]
    X_array = np.array(X).reshape(1, -1)  # Reshape for single prediction
    
    # Scale features and make prediction
    X_scaled = sc.transform(X_array)
    prediction = model.predict(X_scaled)[0]
    
    # Display result
    result = "Churn" if prediction == 1 else "Not Churn"

    st.balloons()
    st.write(f"Predicted: {result}")

else:
    st.write("Please enter the values and use predict button")