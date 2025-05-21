import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from datetime import datetime



# Load the pre-trained model
model = joblib.load('models/xgboost_sales_model.pkl')


#UI Title
st.title("Walmart Weekly Sales Predictor")


#Sidebar inputs
st.sidebar.header("Enter Store Details")

store = st.sidebar.number_input("Store ID", min_value=1, max_value=45, step=1)
department = st.sidebar.number_input("Department ID", min_value=1, max_value=99, step=1)
store_type = st.sidebar.selectbox("Store Type", ["A", "B", "C"])
size = st.sidebar.number_input("Store Size (in square feet)", min_value=1000, max_value=250000, step=1000)
temperature = st.sidebar.slider("Temperature (in Fahrenheit)", min_value=-10.0, max_value=120.0, step=0.1,value=70.0)
fuel_price = st.sidebar.slider("Fuel Price (in USD)", min_value=0.0, max_value=5.0, step=0.01,value=3.25)
is_holiday = st.sidebar.selectbox("Is Holiday week?", ["Yes", "No"])
date = st.sidebar.date_input("Date", value=datetime.today())

#Feature Engineering

date = pd.to_datetime(date)
year = date.year
month = date.month
week = date.isocalendar().week

# Markdowns: optional

st.sidebar.header("Optional: Promotion Data")

markdowns = {}

for i in range(1, 6):
    markdowns[f"MarkDown{i}"] = st.sidebar.number_input(f"MarkDown {i}", min_value=0.0, step=10.0, value=0.0)


# Label Encoding
type_map = {'A': 0, 'B': 1, 'C': 2} 
holiday_map = {'Yes': 1, 'No': 0}   

# Create input DataFrame

input_data = pd.DataFrame([{
    'Store': store,
    'Dept': department,
    'Type': type_map[store_type],
    'Size': size,
    'Temperature': temperature,
    'Fuel_Price': fuel_price,
    'IsHoliday_x': holiday_map[is_holiday],
    'Year': year,
    'Month': month,
    'Week': week,
    **markdowns
}])

#Predict

log_prediction = model.predict(input_data)
prediction = np.expm1(log_prediction[0])

#Display the prediction
st.subheader("Predicted Weekly Sales:")
st.success(f"${prediction:,.2f}")
