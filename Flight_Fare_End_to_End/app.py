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

# Visualizations
st.title("Flight Data Visualizations")


# Function to convert duration to minutes
def convert_duration_to_minutes(duration):
    if 'h' in duration and 'm' in duration:
        # Case: "2h 50m"
        hours = int(duration.split('h')[0])
        minutes = int(duration.split('m')[0].split()[-1])
        return hours * 60 + minutes
    elif 'h' in duration:
        # Case: "19h"
        hours = int(duration.split('h')[0])
        return hours * 60
    elif 'm' in duration:
        # Case: "50m"
        minutes = int(duration.split('m')[0])
        return minutes
    else:
        # Handle unexpected cases
        return 0

# Apply the function to the Duration column
df['Duration_Minutes'] = df['Duration'].apply(convert_duration_to_minutes)

# Set up the layout for visualizations
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
col5, col6 = st.columns(2)

# Visualization 1: Average Price by Airline
with col1:
    st.subheader("Average Price by Airline")
    avg_price_by_airline = df.groupby('Airline')['Price'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=avg_price_by_airline.index, y=avg_price_by_airline.values, palette="viridis", ax=ax)
    plt.xticks(rotation=45)
    plt.xlabel("Airline")
    plt.ylabel("Average Price (INR)")
    st.pyplot(fig)

# Visualization 2: Price Distribution by Number of Stops
with col2:
    st.subheader("Price Distribution by Number of Stops")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Total_Stops', y='Price', data=df, palette="magma", ax=ax)
    plt.xlabel("Number of Stops")
    plt.ylabel("Price (INR)")
    st.pyplot(fig)

# Visualization 3: Flight Duration Distribution
with col3:
    st.subheader("Flight Duration Distribution")
    df['Duration_Minutes'] = df['Duration'].apply(convert_duration_to_minutes)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Duration_Minutes'], bins=30, kde=True, color='skyblue', ax=ax)
    plt.xlabel("Duration (Minutes)")
    plt.ylabel("Frequency")
    st.pyplot(fig)
    
# Visualization 4: Price vs. Duration
with col4:
    st.subheader("Price vs. Duration")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Duration_Minutes', y='Price', data=df, hue='Total_Stops', palette="coolwarm", ax=ax)
    plt.xlabel("Duration (Minutes)")
    plt.ylabel("Price (INR)")
    st.pyplot(fig)

# Visualization 5: Top 10 Routes by Average Price
with col5:
    st.subheader("Top 10 Routes by Average Price")
    df['Route'] = df['Route'].str.replace(' → ', '-')
    avg_price_by_route = df.groupby('Route')['Price'].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=avg_price_by_route.values, y=avg_price_by_route.index, palette="plasma", ax=ax)
    plt.xlabel("Average Price (INR)")
    plt.ylabel("Route")
    st.pyplot(fig)

# Visualization 6: Price Distribution by Source City
with col6:
    st.subheader("Price Distribution by Source City")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Source', y='Price', data=df, palette="Set2", ax=ax)
    plt.xlabel("Source City")
    plt.ylabel("Price (INR)")
    st.pyplot(fig)