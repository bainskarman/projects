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
from streamlit.components.v1 import html
# Set page configuration
st.set_page_config(page_title="Fair Fare", layout='wide', initial_sidebar_state='expanded')

# Title and description with links
st.title("Fair Fare")
st.write("Find the best flight options tailored to your needs.")

# Add project links
st.markdown(
    '<a href="https://github.com/bainskarman/projects/tree/main/Flight_Fare_End_to_End" target="_blank">Detailed Project</a> | '
    '<a href="https://public.tableau.com/app/profile/karman.bains8888/viz/FlightFares/Dashboard1?publish=yes" target="_blank">Dashboard</a>',
    unsafe_allow_html=True
)

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
    # Assuming df is already loaded
    st.subheader("Average Price by Airline")

    # Compute the average price per airline
    avg_price_by_airline = df.groupby('Airline')['Price'].mean().sort_values(ascending=False)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))  # Increase figure size for better readability
    sns.barplot(x=avg_price_by_airline.index, y=avg_price_by_airline.values, palette="viridis", ax=ax)

    # Rotate and align x-axis labels for better visibility
    plt.xticks(rotation=45, ha='right')

    # Label axes
    plt.xlabel("Airline")
    plt.ylabel("Average Price (INR)")

    # Display the plot
    st.pyplot(fig)
    st.subheader("Insights")
    st.write("- **Jet Airways Business** has the highest fares by a large margin.")
    st.write("- **Jet Airways and premium economy fares** are also on the higher side.")
    st.write("- **Air India & Vistara** fall in the mid-range pricing.")
    st.write("- **GoAir, IndiGo, Air Asia, SpiceJet, and Trujet** are the most budget-friendly.")
    st.write("- There's a **big gap between business and economy fares**, showing clear pricing segmentation.")

# Visualization 2: Price Distribution by Number of Stops
with col2:
    st.subheader("Price Distribution by Number of Stops")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Total_Stops', y='Price', data=df, palette="magma", ax=ax)
    plt.xlabel("Number of Stops")
    plt.ylabel("Price (INR)")
    st.pyplot(fig)
    st.subheader("")
    st.subheader("Insights")
    st.write("- *Non-stop** majority flights have lower prices compared to flights with a stop.")
    st.write("- **Step growth** in price can be seen with increasing number of stops.")
    st.write("- **Maximum varience** if for flights with single stop.")
    st.write("- **4 Stop Flights** are rare as all of these are domestic flights.")
    st.write("- Most Outliers are for **1 Stop Flights**.")

# Visualization 3: Flight Duration Distribution
with col3:
    st.subheader("Flight Duration Distribution")
    df['Duration_Minutes'] = df['Duration'].apply(convert_duration_to_minutes)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Duration_Minutes'], bins=30, kde=True, color='skyblue', ax=ax)
    plt.xlabel("Duration (Minutes)")
    plt.ylabel("Frequency")
    st.pyplot(fig)
    st.subheader("Insights")
    st.write("- Its a **right skewed** distribution.")
    st.write("- **Maximum flights** are nearly 3 hours as major cities are nearly 3hrs away from each other.")
    
# Visualization 4: Price vs. Duration
with col4:
    st.subheader("Price vs. Duration")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Duration_Minutes', y='Price', data=df, hue='Total_Stops', palette="coolwarm", ax=ax)
    plt.xlabel("Duration (Minutes)")
    plt.ylabel("Price (INR)")
    st.pyplot(fig)
    st.subheader("Insights")
    st.write("- **Clusters** clarify that non-stop flights are always cheaper while for others price varies alot.")
    st.write("- **Maximum flights** have a single stop.")

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


def tableau_dashboard():
    tableau_html = """
    <div class='tableauPlaceholder' id='viz1743203317234' style='position: relative'><noscript><a href='#'><img alt='Flight Prices ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fl&#47;FlightFares&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='FlightFares&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Fl&#47;FlightFares&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1743203317234');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1520px';vizElement.style.minHeight='587px';vizElement.style.maxHeight='787px';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1520px';vizElement.style.minHeight='587px';vizElement.style.maxHeight='787px';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='1127px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
    """
    return html(tableau_html, width=3200, height=1300)

# Use in your app
st.title("Flight Fares Dashboard")
st.subheader("Key Findings from the Flight Dashboard")

# Findings
st.markdown("""
### 1. Flight Prices & Duration Trends
- The average flight prices across different city pairs vary, with some routes being more expensive than others.
- The average duration of flights remains relatively consistent, ranging between 10 to 14 hours.

### 2. Economy Class Insights
- Kolkata routes tend to have higher fares compared to other cities. For example, flights from Bangalore and Chennai to Kolkata show some of the highest Economy fares.
- Hyderabad and Chennai routes appear to be more cost-effective in Economy class.
- Mumbai consistently has lower Economy class fares across multiple routes.

### 3. Business Class Insights
- Kolkata again emerges as the most expensive destination in Business class, with fares significantly higher than other routes.
- Flights from Bangalore and Delhi to Kolkata have some of the highest Business class fares.
- Business class fares are relatively higher across all routes, with significant price differences compared to Economy class.

### 4. Source-Destination Trends
- Bangalore and Delhi serve as key hubs, with frequent flights to major cities.
- Flights between Chennai and Hyderabad show lower fares compared to routes involving Kolkata or Mumbai.
- Mumbai has some of the most affordable Business and Economy class flights when compared to other metro cities.

### Conclusion
The dashboard highlights pricing variations across different routes, showing that Kolkata-bound flights are generally expensive in both classes. Hyderabad and Chennai routes tend to be more budget-friendly. These insights can help travelers choose cost-effective options, while businesses and airlines can use the data to optimize pricing strategies.
""")
tableau_dashboard()
