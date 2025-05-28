import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from datetime import date , datetime
from io import BytesIO
import shap




# Load the pre-trained model
model = joblib.load('models/xgboost_sales_model.pkl')

# Load full historical data for charting
@st.cache_data
def load_sales_data():
    df=pd.read_csv('data/cleaned_sales.csv', parse_dates=['Date'])
    return df
sales_data = load_sales_data()


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

min_date = date(2010, 1, 1)
max_date = date(2012, 12, 31)
date = st.sidebar.date_input(
    "Week Date",
    value=datetime(2011, 6, 1),
    min_value=min_date,
    max_value=max_date,
)

#Feature Engineering

date = pd.to_datetime(date)
year = date.year
month = date.month
week = date.isocalendar().week

# Markdowns: optional

st.sidebar.header("Optional: Promotion Data")

markdowns = {}
markdown_tooltips = {
    "MarkDown1": "Holiday-related promotions (e.g., Christmas, Back-to-School)",
    "MarkDown2": "In-store or seasonal sales markdowns",
    "MarkDown3": "Clearance discounts or deep price cuts",
    "MarkDown4": "Category-specific markdowns (e.g., electronics, apparel)",
    "MarkDown5": "Miscellaneous or store-wide promotions"
}

for i in range(1, 6):
    key = f"MarkDown{i}"
    markdowns[key] = st.sidebar.number_input(
        f"{key}",
        min_value=0.0,
        step=10.0,
        value=0.0,
        help=markdown_tooltips[key]
    )

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

st.markdown("---")
st.subheader("Feature Sensitivity Test")

# User picks one feature to tweak

sensitivity_feature = st.selectbox(
    "Select a feature to test sensitivity:",
    ["Size", "Temperature", "Fuel_Price"]  + [f"MarkDown{i}" for i in range(1, 6)]
)

# Get current value of the selected feature
current_value = input_data[sensitivity_feature].values[0]

if sensitivity_feature == "Size":
    min_val, max_val, step_val = 5000.0, 250000.0, 1000.0
elif current_value == 0:
    min_val, max_val, step_val = 0.0, 1000.0, 10.0
else:
    min_val = float(current_value * 0.5)
    max_val = float(current_value * 1.5)
    step_val = 1.0

# Create a slider to adjust the feature value
new_value = st.slider(
    f"Adjust {sensitivity_feature}",
    min_value=min_val,
    max_value=max_val,
    step=step_val,
    value=float(current_value)
)

# Clone input data and update the selected feature

modified_input = input_data.copy()
modified_input[sensitivity_feature] = new_value

# Predict with modified input
new_prediction_log = model.predict(modified_input)[0]
new_prediction = np.expm1(new_prediction_log)

# Show Difference
delta = new_prediction - prediction
direction = "increase" if delta > 0 else "decrease"

st.success(
    f"Changing **{sensitivity_feature}** from {current_value} → {new_value} "
    f"would result in a **{direction}** of ${abs(delta):,.2f} in predicted weekly sales."
)

with st.expander("📊 View Historical Weekly Sales for This Store & Department"):
    history = sales_data[
        (sales_data["Store"] == store) & (sales_data["Dept"] == department)
    ].sort_values("Date")

    if history.empty:
        st.warning("No historical data available for this Store & Department.")
    else:
        import matplotlib.pyplot as plt
     

        # Set up the plot
        fig, ax = plt.subplots(figsize=(10, 4))

        # Plot historical sales line
        ax.plot(history["Date"], history["Weekly_Sales"], label="Historical Sales", color="skyblue")

        # Plot historical average line
        avg_sales = history["Weekly_Sales"].mean()
        ax.axhline(avg_sales, color="gray", linestyle="--", label=f"Historical Avg: ${avg_sales:,.0f}")

        # Position forecast dot one week after last historical date
        forecast_date = history["Date"].max() + pd.Timedelta(days=7)
        ax.scatter([forecast_date], [prediction], color="orange", s=100, label="Predicted Sales (Forecast)", zorder=5)

        # Vertical line to mark forecast boundary
        ax.axvline(forecast_date, color="orange", linestyle="--", alpha=0.5)

        # Title and labels
        ax.set_title(f"Weekly Sales History & Forecast for Store {store} - Dept {department}")
        ax.set_ylabel("Weekly Sales (USD)")
        ax.set_xlabel("Date")
        ax.legend()

        # Display the plot 
        st.pyplot(fig)

        st.caption("🟠 The orange dot represents the predicted sales for your selected week, placed just beyond historical data to reflect forecasting.")

        # Download button for the chart
        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches='tight')
        buffer.seek(0)

        st.download_button(
            label="📥 Download Chart as PNG",
            data=buffer,
            file_name=f"sales_forecast_store_{store}_dept_{department}.png",
            mime="image/png",
        )

        prediction_data = input_data.copy()
        prediction_data["Predicted_Weekly_Sales"] = prediction
        prediction_data["Forecast_Week"]= date

        cols = ["Forecast_Week", "Store", "Dept", "Type", "Size", "Temperature",
                "Fuel_Price", "IsHoliday_x", "Year", "Month", "Week"] + \
               [f"MarkDown{i}" for i in range(1, 6)] + ["Predicted_Weekly_Sales"]

        cols = [col for col in cols if col in prediction_data.columns]
        prediction_data = prediction_data[cols]

        csv_buffer = prediction_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Prediction Data as CSV",
            data=csv_buffer,
            file_name=f"sales_forecast_store_{store}_dept_{department}.csv",
            mime="text/csv",
        )


@st.cache_resource
def get_explainer(_model):
    explainer = shap.Explainer(_model)
    return explainer

explainer = get_explainer(model)

shap_values = explainer(input_data)

st.subheader("🔍 Why Did the Model Predict This Sales Value?")
st.markdown(
    "This chart explains which features pushed the predicted weekly sales **up or down** compared to the average.\n"
    "- 📈 Positive bars **increase** the prediction\n"
    "- 📉 Negative bars **decrease** the prediction\n\n"
    "The model adds up all of these effects to arrive at the final predicted sales value you saw above."
)
shap_df = pd.DataFrame({
    "Feature": input_data.columns,
    "SHAP Value": shap_values.values[0]
}).sort_values(by="SHAP Value", key=abs, ascending=False) 
st.bar_chart(shap_df.set_index("Feature"))