import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import streamlit as st
from keras.models import load_model
import yfinance as yf

st.title('Stock Trend Prediction')
user_input=st.text_input('Enter stock ticker','AAPL')
df = yf.download(user_input,start='2016-01-01')
df = df.reset_index()
st.subheader('Data after 2016')
df1=pd.DataFrame(df['Close'])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(df1)

def split(dataset):
	time_step=120
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

X_train,y_train=split(df1)
X_train=np.array(X_train)
y_train=np.array(y_train)
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)

model = load_model('/workspaces/projects/Stock_predict/lmodel.h5')

y_pred= model.predict(X_train)

y_pred=scaler.inverse_transform(y_pred)

look_back=120
trainPredictPlot = np.empty_like(df[['Close']])
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(y_pred)+look_back, :] = y_pred

st.subheader('Original vs Predicted Price')
fig1= plt.figure(figsize=(12,8))
plt.plot(trainPredictPlot,label='Predicted Price')
plt.plot(df[['Close']],label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig1)