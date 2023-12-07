import streamlit as st
import joblib
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from datetime import datetime
from geopy.distance import geodesic


file_path='/workspaces/projects/Uber_Fare/pipeline.joblib'
with open(file_path, 'rb') as file:
    pipeline = joblib.load(file)

st.title('Ride Fare Estimation')

pickup_latitude = st.number_input('Enter Pickup Latitude:')
pickup_longitude = st.number_input('Enter Pickup Longitude:')
dropoff_latitude = st.number_input('Enter Drop Off Latitude:')
dropoff_longitude = st.number_input('Enter Drop Off Longitude:')
pickup_coordinates =(pickup_latitude,pickup_longitude)
dropoff_coordinates=(dropoff_latitude,dropoff_longitude)
distance = geodesic(pickup_coordinates, dropoff_coordinates).kilometers
passenger_count = st.number_input('Enter Passenger Count:', min_value=1, value=1)

if st.button('Predict Fare'):
    if pickup_coordinates and dropoff_coordinates:
        now = datetime.now()
        day= now.day
        hour= now.hour
        weekday= 'Tue'
        nweekday= now.weekday()
        year= now.year
        month= now.month
        
        input_data = pd.DataFrame({
            'passenger_count': [passenger_count],
            'Year': [year],
            'Month': [month],
            'Day of Week': [weekday],
            'Day':[day],
            'Day of Week_num':[nweekday],
            'Hour': [hour],
            'distance': [distance],
            'pickup_latitude': [pickup_coordinates[0]],
            'pickup_longitude': [pickup_coordinates[1]],
            'dropoff_latitude': [dropoff_coordinates[0]],
            'dropoff_longitude': [dropoff_coordinates[1]]
        })

        fare_prediction = pipeline.predict(input_data)

        st.subheader('Estimated Fare Amount:')
        st.write(f"${fare_prediction[0]:.2f}")
    else:
        st.warning("Please enter valid pickup and dropoff location names.")
