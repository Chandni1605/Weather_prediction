# Weather_prediction


This project analyzes historical weather data and predicts future maximum temperatures in Seattle using a simple linear regression model. Built using Streamlit, it provides interactive visualizations for:

📈 Historical temperature trends
🔍 Actual vs predicted temperature comparison
🔮 30-day future temperature forecast
🚀 Features
Loads and processes seattle-weather.csv dataset
Visualizes historical max temperature patterns
Trains a Linear Regression model to predict temperature
Calculates RMSE and MAE for model performance
Forecasts next 30 days of temperature
Interactive charts using Matplotlib inside Streamlit
🛠️ Tech Stack
Python
Pandas – Data handling
Matplotlib – Visualizations
Scikit-learn – Machine Learning (Linear Regression)
Streamlit – Web app interface
📂 Dataset
The app uses the publicly available dataset seattle-weather.csv, which includes:

date
temp_max (daily maximum temperature)
temp_min, precipitation, weather, etc.
You can replace this dataset with any other city’s historical temperature data by adjusting column names accordingly.
