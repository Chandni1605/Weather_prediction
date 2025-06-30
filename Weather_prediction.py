import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Streamlit App
st.title("🌤️ Seattle Weather Forecast App")
st.markdown("Predict future max temperatures using historical weather data.")

# Load the CSV file
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/chandnisingh/Downloads/seattle-weather.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df[['date', 'temp_max']].dropna()
    df.rename(columns={'temp_max': 'temp'}, inplace=True)
    df.sort_values('date', inplace=True)
    df['days'] = (df['date'] - df['date'].min()).dt.days
    return df

df = load_data()

# Show basic info
if st.checkbox("Show raw data"):
    st.write(df.head())

# Plot historical temperature trend
st.subheader("📈 Historical Max Temperature")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df['date'], df['temp'], label='Max Temp')
ax1.set_xlabel("Date")
ax1.set_ylabel("Temperature (°F)")
ax1.set_title("Seattle Daily Max Temperature Trend")
ax1.grid(True)
st.pyplot(fig1)

# Train/test split
X = df[['days']]
y = df['temp']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

st.subheader("📊 Model Evaluation")
st.write(f"**RMSE**: {rmse:.2f} °F")
st.write(f"**MAE**: {mae:.2f} °F")

# Actual vs Predicted
st.subheader("🔍 Actual vs Predicted Temperatures")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df['date'].iloc[len(X_train):], y_test.values, label='Actual')
ax2.plot(df['date'].iloc[len(X_train):], y_pred, label='Predicted', linestyle='--')
ax2.set_xlabel("Date")
ax2.set_ylabel("Temperature (°F)")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# Forecast future temperatures
st.subheader("🔮 30-Day Future Temperature Forecast")
future_days = 30
last_day = df['days'].max()
future_X = pd.DataFrame({'days': range(last_day + 1, last_day + future_days + 1)})
future_pred = model.predict(future_X)
future_dates = pd.date_range(start=df['date'].max() + pd.Timedelta(days=1), periods=future_days)

fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(df['date'], df['temp'], label='Historical')
ax3.plot(future_dates, future_pred, label='Forecast', linestyle='--', color='orange')
ax3.set_xlabel("Date")
ax3.set_ylabel("Temperature (°F)")
ax3.set_title("Forecast for Next 30 Days")
ax3.legend()
ax3.grid(True)
st.pyplot(fig3)
