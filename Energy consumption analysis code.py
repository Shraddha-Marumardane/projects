import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


USER_CREDENTIALS = {"admin": "password123", "user": "userpass"}
RESET_TOKENS = {}

def check_login(username, password):
    return USER_CREDENTIALS.get(username) == password

def send_password_reset_email(username):
    reset_token = f"reset_token_for_{username}"
    RESET_TOKENS[username] = reset_token
    return reset_token

def reset_user_password(username, new_password):
    if username in USER_CREDENTIALS:
        USER_CREDENTIALS[username] = new_password
        return True
    return False

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔒 Login to Energy Dashboard")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if check_login(username, password):
                st.session_state.logged_in = True
                st.success("Login successful.")
            else:
                st.error("Invalid username or password.")

    if st.button("Forgot Password?"):
        reset_user = st.text_input("Enter username for reset")
        if st.button("Send Reset Link"):
            if reset_user in USER_CREDENTIALS:
                token = send_password_reset_email(reset_user)
                st.success(f"Reset link sent. Token: {token}")
                new_pass = st.text_input("Enter new password", type="password")
                if st.button("Reset Password"):
                    if reset_user_password(reset_user, new_pass):
                        st.success("Password reset successful.")
                    else:
                        st.error("Reset failed.")
            else:
                st.error("User not found.")
    st.stop()


st.sidebar.title("🔧 Settings")
st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"logged_in": False}))
forecast_days = st.sidebar.slider("Forecast days", 1, 90, 30)

data = pd.read_csv("C:/Users/shrad/OneDrive/Desktop/netleap/AEP_hourly.csv")
data['Datetime'] = pd.to_datetime(data['Datetime'])
data.set_index('Datetime', inplace=True)

daily_data = data['AEP_MW'].resample('D').mean()
hourly_data = data.groupby(data.index.hour).mean()
monthly_data = data['AEP_MW'].resample('M').mean()


st.title("⚡ Smart Energy Consumption Dashboard")
tab1, tab2 = st.tabs(["📊 Visualization", "🔮 Forecasting"])


with tab1:
    st.subheader("Raw Data")
    st.dataframe(data.tail(100))

    st.subheader("Hourly Energy Usage")
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(data['AEP_MW'], color='dodgerblue')
    ax1.set_title("Hourly Energy Usage")
    ax1.set_ylabel("MW")
    st.pyplot(fig1)

    st.subheader("Daily Energy Consumption")
    st.line_chart(daily_data)

    st.subheader("Monthly Trend")
    fig2, ax2 = plt.subplots()
    ax2.plot(monthly_data, marker='o', linestyle='-', color='seagreen')
    ax2.set_title("Monthly Energy Usage")
    st.pyplot(fig2)

    st.subheader("Hourly Usage Distribution")
    fig3 = plt.figure()
    sns.barplot(x=hourly_data.index, y=hourly_data['AEP_MW'], palette='coolwarm')
    plt.title("Average Usage by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("MW")
    st.pyplot(fig3)

    st.subheader("Correlation Heatmap")
    corr = data.resample('H').mean().corr()
    fig4, ax4 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax4)
    st.pyplot(fig4)


with tab2:
    st.subheader("Forecasting with Prophet")

    prophet_data = daily_data.reset_index().rename(columns={'Datetime': 'ds', 'AEP_MW': 'y'})
    train = prophet_data[:-forecast_days]
    test = prophet_data[-forecast_days:]

    model = Prophet()
    model.fit(train)

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    test['yhat'] = forecast.iloc[-forecast_days:]['yhat'].values

    
    mae = np.mean(np.abs(test['y'] - test['yhat']))
    mape = np.mean(np.abs((test['y'] - test['yhat']) / test['y'])) * 100
    rmse = np.sqrt(mean_squared_error(test['y'], test['yhat']))
    r2 = r2_score(test['y'], test['yhat'])

    st.metric("MAE", f"{mae:.2f}")
    st.metric("MAPE", f"{mape:.2f}%")
    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("R²", f"{r2:.2f}")

    st.write("Forecasted Values (last 10 days):")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

    st.subheader("Forecast Plot")
    fig5 = model.plot(forecast)
    st.pyplot(fig5)

    st.subheader("Forecast Components")
    fig6 = model.plot_components(forecast)
    st.pyplot(fig6)
