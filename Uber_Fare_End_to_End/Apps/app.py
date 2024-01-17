import streamlit as st
import joblib
import os
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from datetime import datetime
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
current_path = os.getcwd()
path = os.path.join(current_path, 'Uber_Fare_End_to_End/Apps/pipeline.joblib')
pipeline = joblib.load(file)

st.title('Ride Fare Estimation')


# Function to convert location to coordinates
def get_coordinates(location):
    geolocator = Nominatim(user_agent="location_converter")
    location_data = geolocator.geocode(location)
    if location_data:
        return location_data.latitude, location_data.longitude
    else:
        return None

pickup_location = st.text_input('Enter Pickup Location')
dropoff_location = st.text_input('Enter Drop Off Location')
pickup_coordinates =get_coordinates(pickup_location)
dropoff_coordinates=get_coordinates(dropoff_location)
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
