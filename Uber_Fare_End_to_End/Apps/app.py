import streamlit as st
import joblib
import os
import pandas as pd
from geopy.exc import GeocoderTimedOut
from datetime import datetime
from geopy.distance import geodesic
from geopy.geocoders import OpenCage
import plotly.express as px
import sklearn
current_path = os.getcwd()
path = os.path.join(current_path, 'Uber_Fare_End_to_End/Apps/pipeline.joblib')
pipeline = joblib.load(path)

st.title('Ride Fare Estimation')

key ='23db542af7ac47b7a62c8f0f91fbfc4a'
from geopy.exc import GeocoderTimedOut, GeocoderServiceError, GeocoderQueryError

def get_coordinates(location, api_key):
    geolocator = OpenCage(api_key)
    
    try:
        location_data = geolocator.geocode(location)
        if location_data:
            return location_data.latitude, location_data.longitude
        else:
            print(f"Location data not found for {location}")
            return None
    except (GeocoderTimedOut, GeocoderServiceError, GeocoderQueryError) as e:
        print(f"Geocoding error: {e}")
        return None

import streamlit as st
from geopy.distance import geodesic

def parse_coordinates(coord_str):
    try:
        # Split by lines and parse each line
        coords = [tuple(map(float, line.strip('()').split(','))) for line in coord_str.strip().splitlines()]
        return coords
    except ValueError:
        raise ValueError("Invalid format. Please use (latitude, longitude) for each location.")

# Get multiple pickup and dropoff coordinates from user input
pickup_coordinates = st.text_area('Enter Pickup Locations (format: (lat, lon), one per line)')
dropoff_coordinates = st.text_area('Enter Drop Off Locations (format: (lat, lon), one per line)')

if pickup_coordinates and dropoff_coordinates:
    try:
        # Parse input strings into lists of tuples
        pickup_coords = parse_coordinates(pickup_coordinates)
        dropoff_coords = parse_coordinates(dropoff_coordinates)

        # Calculate distances for each pair of pickup and dropoff locations
        distances = []
        for pickup in pickup_coords:
            for dropoff in dropoff_coords:
                distance = geodesic(pickup, dropoff).kilometers
                distances.append(distance)

        st.write("Distances (in km):")
        for distance in distances:
            st.write(f'{distance:.2f} km')

    except ValueError as e:
        st.error(str(e))
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

current_path = os.getcwd()
path = os.path.join(current_path, 'Uber_Fare_End_to_End/Data/uber.csv')
df = pd.read_csv(path)

# Calculate the average fare for each pickup location
avg_fare_pickup = df.groupby(['pickup_latitude', 'pickup_longitude'])['fare_amount'].mean().reset_index()

# Create a Streamlit app
st.title('Average Fare Price Heatmaps')

# Display pickup location heatmap
st.subheader('Pickup Location Heatmap')
fig_pickup = px.density_mapbox(avg_fare_pickup,
                               lat='pickup_latitude',
                               lon='pickup_longitude',
                               z='fare_amount',
                               radius=10,
                               center=dict(lat=40.7128, lon=-74.0060),  # Center around New York
                               zoom=10,
                               mapbox_style='carto-positron',
                               opacity=0.6,
                               title='Average Fare Price Heatmap - Pickup Locations')

# Update the layout to add mapbox access token (replace 'your_mapbox_token' with your actual Mapbox token)
fig_pickup.update_layout(mapbox=dict(accesstoken='your_mapbox_token'))

# Display both plots in two columns
st.plotly_chart(fig_pickup, use_container_width=True)
