import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import streamlit as st
from keras.models import load_model
import yfinance as yf

st.title('Stock Trend Prediction')
user_input=st.text_input('Enter stock ticker')
df = yf.download(user_input,start='2010-01-01')
st.subheader('Data after 2010')
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(df1)
df1.shape
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)

model = load_model('/workspaces/projects/Stock_predict/lmodel.h5')

past_100 = train_data.tail(100)
final_df= past_100.append(test_data,ignore_index=True)

X_test, y_test = create_dataset(final_df, time_step)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
y_pred= model.predict()
y_test=scaler.inverse_transform(y_test)
y_pred=scaler.inverse_transform(y_pred)

st.subheader('Original vs Predicted Price')
fig1= plt.figure(figsize=(12,8))
plt.plot(y_test,label='Original Price')
plt.plot(y_pred,label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig1)