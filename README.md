# Stock_Trend_Prediction_YBI

📈 Stock Trend Prediction Using LSTM
This project focuses on building a stock price trend prediction model using Long Short-Term Memory (LSTM) neural networks. The model has been trained on Apple Inc. (AAPL) stock data for the past 20 years using data from Yahoo Finance.

🔍 Objective
To fetch and visualize historical stock data

To apply rolling averages and percent change analysis

To train a deep learning model (LSTM) that predicts future stock prices based on historical patterns

📂 Files Included
stock_trend_prediction.ipynb — Jupyter notebook containing the full project code (data loading, preprocessing, visualization, model training, and evaluation)

Latest_stock_price_model.keras — Trained LSTM model

README.md — Project description and usage instructions

🧠 Technologies Used
Python

NumPy, Pandas, Matplotlib

Scikit-learn (MinMaxScaler)

Keras (LSTM, Dense)

Yahoo Finance (yfinance)

🔄 Generalization
This model was trained on Apple's stock data, but you can analyze and predict stock trends for any other publicly traded company. To do this:

Change the stock ticker symbol in the code (stock = "AAPL") to the desired company’s symbol (e.g., "MSFT" for Microsoft, "GOOGL" for Alphabet).

Re-run the notebook cells to fetch new data, train the model, and make predictions.

📊 Visualizations
Rolling Mean (100 and 250 days)

Adjusted Close Price Trends

Percentage Daily Change

Actual vs Predicted Price Comparisons

✅ Model Performance
Root Mean Squared Error (RMSE) is calculated to evaluate model accuracy.

Visual comparison of actual vs predicted stock price trends is shown.
