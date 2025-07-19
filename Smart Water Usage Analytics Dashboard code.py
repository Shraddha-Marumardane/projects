import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px

# Title
st.markdown("💧 **Smart Water Usage Analytics Dashboard**")

# Load Data
@st.cache_data
def load_data():
    file_path = "water_usage.csv"
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("🚫 Dataset file 'water_usage.csv' not found in the folder.")
    st.info("Please place the file in the same folder as this Python script.")
    st.stop()

# Show Raw Data
if st.checkbox("Show raw data"):
    st.write(df.head())

# Summary
st.subheader("📊 Water Usage Summary")
total_consumption = df['consumption_liters'].sum()
average_consumption = df['consumption_liters'].mean()
st.metric("Total Consumption (Liters)", f"{total_consumption:,.2f}")
st.metric("Average Daily Consumption", f"{average_consumption:,.2f}")

# Time Series Visualization
st.subheader("📈 Water Consumption Over Time")
fig = px.line(df, x='date', y='consumption_liters', title='Daily Water Consumption')
st.plotly_chart(fig, use_container_width=True)

# Region-wise Distribution
st.subheader("🌍 Region-wise Usage")
region_fig = px.box(df, x='region', y='consumption_liters', color='region', title="Distribution by Region")
st.plotly_chart(region_fig, use_container_width=True)

# Forecasting
st.subheader("🔮 Forecast Water Usage (Prophet)")

# Prepare data for Prophet
forecast_df = df[['date', 'consumption_liters']].rename(columns={"date": "ds", "consumption_liters": "y"})

# Fit Prophet model
model = Prophet()
model.fit(forecast_df)

# Forecast future
future = model.make_future_dataframe(periods=30)  # forecast 30 days ahead
forecast = model.predict(future)

# Plot forecast
fig_forecast = plot_plotly(model, forecast)
st.plotly_chart(fig_forecast, use_container_width=True)

# Optional: Show forecast data
if st.checkbox("Show forecasted data"):
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))
