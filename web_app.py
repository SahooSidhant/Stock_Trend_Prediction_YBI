import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime


st.title("📈 Stock Price Predictor App")

stock = st.text_input("Enter the Stock ID (e.g. AAPL, MSFT, TSLA)", "AAPL")


end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)
apple_data = yf.download(stock, start, end)

model = load_model("Latest_stock_price_model.keras")


st.subheader("📊 Stock Data")
st.write(apple_data)

def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'orange', label='Moving Average')
    plt.plot(full_data.Close, 'b', label='Close Price')
    if extra_data:
        plt.plot(extra_dataset, 'green', label='Extra MA')
    plt.legend()
    return fig


apple_data['MA_for_250_days'] = apple_data.Close.rolling(250).mean()
st.subheader('📈 Close Price vs 250-Day Moving Average')
st.pyplot(plot_graph((15, 8), apple_data['MA_for_250_days'], apple_data))

apple_data['MA_for_200_days'] = apple_data.Close.rolling(200).mean()
st.subheader('📈 Close Price vs 200-Day Moving Average')
st.pyplot(plot_graph((15, 8), apple_data['MA_for_200_days'], apple_data))

apple_data['MA_for_100_days'] = apple_data.Close.rolling(100).mean()
st.subheader('📈 Close Price vs 100-Day Moving Average')
st.pyplot(plot_graph((15, 8), apple_data['MA_for_100_days'], apple_data))

st.subheader('📈 Close Price vs 100-Day and 250-Day Moving Averages')
st.pyplot(plot_graph((15, 8), apple_data['MA_for_100_days'], apple_data, 1, apple_data['MA_for_250_days']))

splitting_len = int(len(apple_data) * 0.7)
x_test = apple_data[['Close']].iloc[splitting_len:]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test)


x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)


predictions = model.predict(x_data)


inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)


ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    },
    index=apple_data.index[splitting_len+100:]
)

 
st.subheader("📋 Original vs Predicted Close Prices")
st.write(ploting_data)


st.subheader("🔍 Predicted vs Original (Test Data Only)")
fig2 = plt.figure(figsize=(15, 8))
plt.plot(ploting_data['original_test_data'], label='Original', color='blue')
plt.plot(ploting_data['predictions'], label='Predicted', color='red')
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title(f"{stock} Stock Price: Predicted vs Original")
plt.legend()
st.pyplot(fig2)
