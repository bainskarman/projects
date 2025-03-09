import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from prophet import Prophet
import os
from streamlit.components.v1 import html
from ta.trend import ADXIndicator


# Load data
current_path = os.getcwd()
path = os.path.join(current_path, 'TTC_Delay_Analysis/data.csv')
df = pd.read_csv(path)


# Set page config
st.set_page_config(layout='wide', page_title="TTC Delays Analysis Report", page_icon="🚌")

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    .stHeader {
        color: #2c3e50;
    }
    .stSubheader {
        color: #34495e;
    }
    .stMarkdown {
        font-size: 16px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Introduction
st.title("🚌 TTC Delays Analysis Report")
st.markdown("""
    This report provides a comprehensive analysis of delays in the Toronto Transit Commission (TTC) system. 
    The analysis includes visualizations, insights, and forecasting to help understand and mitigate delays.
    """)


# Section 4: Rolling Average and ADX Analysis
st.header("📈 Rolling Average and ADX Analysis")
df['Date'] = pd.to_datetime(df['Date'])  # Convert to datetime
df.set_index('Date', inplace=True) 

# Calculate rolling average and ADX
avg_delay_per_7_days = df['Min Delay'].resample('7D').mean()  # Resample by 7 days
rolling_avg_30_days = avg_delay_per_7_days.rolling(window=4).mean()  # 30-day rolling average

# Calculate ADX
adx_indicator = ADXIndicator(
    high=avg_delay_per_7_days,  # Use the same series for high, low, and close
    low=avg_delay_per_7_days,
    close=avg_delay_per_7_days,
    window=14
)
adx_values = adx_indicator.adx()

# Plotting
st.subheader("Rolling Average and ADX of Delays")
fig, ax = plt.subplots(figsize=(18, 6))
ax.plot(rolling_avg_30_days.index, rolling_avg_30_days.values, label='30-Day Rolling Avg Min Delay')
ax.plot(avg_delay_per_7_days.index, avg_delay_per_7_days.values, label='Average Min Delay per 7 Days', color='orange')
ax.plot(adx_values.index, adx_values.values, label='ADX', color='red')
ax.set_title('Average Min Delay per 7 Days with 30-Day Rolling Avg and ADX')
ax.set_xlabel('Date')
ax.set_ylabel('Average Min Delay / ADX')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Insights from Rolling Average and ADX
st.header("🔍 Insights from Rolling Average and ADX")
st.markdown("""
    - **30-Day Rolling Average:** The rolling average shows fluctuations in delays over time, indicating variability in delay patterns.
    - **ADX (Average Directional Index):** Despite the fluctuations in the rolling average, the ADX values are trending downward, suggesting a weakening trend in delay patterns.
    - **Trend Analysis:** The combination of the rolling average and ADX indicates that while delays may fluctuate, the overall trend strength is decreasing, which could signal improving conditions or reduced variability in delays.
    """)