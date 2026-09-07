import streamlit as st
import joblib as jb
import pandas as pd

# Load the trained Linear Regression model
loaded_lr_model = joblib.load('linear_regression_model.sav')

st.title('Sales Prediction using Linear Regression')

st.write(
    'Enter the advertising spending for TV, Radio, and Newspaper '
    'to predict sales.'
)

# Input features from the user
tv = st.slider(
    'TV Advertising Spending ($)',
    0.0, 300.0, 150.0
)

radio = st.slider(
    'Radio Advertising Spending ($)',
    0.0, 50.0, 25.0
)

newspaper = st.slider(
    'Newspaper Advertising Spending ($)',
    0.0, 120.0, 30.0
)

# Create DataFrame for input
input_data = pd.DataFrame([{
    'TV': tv,
    'Radio': radio,
    'Newspaper': newspaper
}])

# Make prediction
if st.button('Predict Sales'):
    prediction = loaded_lr_model.predict(input_data)

    st.success(
        f'Predicted Sales: {prediction[0]:.2f} units'
    )
