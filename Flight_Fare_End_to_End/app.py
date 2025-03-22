import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

import os
import streamlit as st
import pandas as pd
import pickle

# Set page configuration
st.set_page_config(page_title="Fair Fare", layout='wide', initial_sidebar_state='expanded')

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load data
data_path = os.path.join(current_dir, "Data_Train.xlsx")
if os.path.exists(data_path):
    df = pd.read_excel(data_path)
else:
    st.error(f"Data file not found: {data_path}")
    st.stop()

# Load model with error handling
model_path = os.path.join(current_dir, "Flight.pkl")
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as model_file:
            pipe = pickle.load(model_file)
    except (ValueError, pickle.UnpicklingError) as e:
        st.error(f"Error loading the model: {e}")
        st.stop()
else:
    st.error(f"Model file not found: {model_path}")
    st.stop()


# Title and description
st.title("Fair Fare")
st.write("Find the best flight options tailored to your needs.")

# Layout columns
col1, col2, col3 = st.columns(3)

# User inputs
with col1:
    Airline = st.selectbox("Choose an Airline", ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet',
                                                 'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia',
                                                 'Vistara Premium economy', 'Jet Airways Business',
                                                 'Multiple carriers Premium economy', 'Trujet'])

with col2:
    Source = st.selectbox("From", ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai'])

with col3:
    Destination = st.selectbox("Destination", ['New Delhi', 'Banglore', 'Cochin', 'Kolkata', 'Delhi', 'Hyderabad'])

# Additional inputs
col4, col5 = st.columns(2)
with col4:
    stops = st.number_input("Number of stops", min_value=0, max_value=5, value=0)

with col5:
    Date = st.date_input("When", min_value=datetime.today())

# Define the exchange rate (1 USD = 83 INR)
EXCHANGE_RATE = 83

# Predict button
if st.button("Estimate Price and Duration"):
    # Create input data with all required columns
    input_data = pd.DataFrame({
        "Airline": [Airline],
        "Source": [Source],
        "Destination": [Destination],
        "Total_Stops": [stops],
        "Month": [Date.month],
        "Day": [Date.day],
        "Duration_Hour": [0]  # Add the missing column with a default value
    })
    
    # Predict the result using the model
    result = pipe.predict(input_data)
    
    # Check the shape of the result
    if isinstance(result, (list, np.ndarray)) and len(result) > 0:
        # If result is a 1D array or list
        price_inr = result[0]  # Access the first prediction directly
        
        # Convert price to USD
        price_usd = price_inr / EXCHANGE_RATE
        
        # Display the results
        st.write(f"### Estimated Price (INR): ₹{price_inr:,.0f}")
        st.write(f"### Estimated Price (USD): ${price_usd:,.2f}")
    else:
        st.error("Invalid prediction result. Please check the model output.")