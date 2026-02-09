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
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    layout='wide',
    page_title="TTC Delays Dashboard",
    page_icon="🚌",
    initial_sidebar_state="expanded"
)

# ==================== DARK MODE ONLY ====================
# Set dark mode as default (no toggle)
st.session_state.dark_mode = True

# ==================== DARK MODE CSS STYLING ====================
st.markdown("""
<style>
    /* Dark mode styling only */
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(90deg, #8B0000 0%, #660000 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .section-header {
        color: #ffffff !important;
        padding: 1rem 0;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid #8B0000;
        font-weight: 700;
        font-size: 1.5rem;
    }
    
    .chart-container {
        background: #2d2d2d;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        margin-bottom: 1.5rem;
        border: 1px solid #404040;
    }
    
    .metric-card {
        background: #2d2d2d;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
        text-align: center;
        border-left: 4px solid #8B0000;
        border: 1px solid #404040;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #cccccc;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .info-box {
        background: linear-gradient(135deg, #333333 0%, #2a2a2a 100%);
        border-left: 4px solid #8B0000;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        font-size: 0.95rem;
        line-height: 1.5;
        color: #ffffff;
        border: 1px solid #404040;
    }
    
    /* Chart background for dark mode */
    .js-plotly-plot, .plotly, .modebar {
        background: #2d2d2d !important;
    }
    
    .stSelectbox, .stMultiselect, .stSlider {
        background-color: #2d2d2d;
        color: white;
    }
    
    .st-bb, .st-at, .st-ae {
        background-color: #2d2d2d !important;
    }
    
    /* Dark mode headings */
    h2, h3, h4 {
        color: #ffffff !important;
    }
    
    /* Common styling */
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #ffffff !important;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #CC0000 0%, #990000 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(204, 0, 0, 0.2);
    }
    
    /* Compact sidebar styling */
    .sidebar .stSelectbox, .sidebar .stMultiselect {
        margin-bottom: 0.3rem;
    }
    
    /* Tab styling for dark mode */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #404040;
        color: white;
        border-radius: 5px 5px 0 0;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #8B0000;
        color: white;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
        
        .section-header {
            font-size: 1.3rem;
        }
    }
    
    /* Table styling */
    .stDataFrame {
        background-color: #2d2d2d;
        color: white;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #2d2d2d;
        color: white;
        border-color: #404040;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #2d2d2d;
        color: white;
    }
    
    /* Multi-select */
    .stMultiSelect > div > div {
        background-color: #2d2d2d;
        color: white;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background-color: #404040;
    }
</style>
""", unsafe_allow_html=True)

# ==================== DATA LOADING ====================
@st.cache_data
def load_data():
    """Load and preprocess TTC delay data"""
    try:
        possible_paths = [
            'TTC_Delay_Analysis/data.csv',
            'data.csv',
            './data.csv',
            os.path.join(os.getcwd(), 'TTC_Delay_Analysis', 'data.csv')
        ]
        
        df = None
        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                break
            except:
                continue
        
        if df is None:
            st.error("Could not find data.csv file. Please ensure it's in the correct location.")
            st.stop()
        
        # Basic data preprocessing
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Create day parts if not exists
        if 'Hour' in df.columns:
            def get_day_part(hour):
                if 5 <= hour < 12:
                    return 'Morning'
                elif 12 <= hour < 17:
                    return 'Afternoon'
                elif 17 <= hour < 22:
                    return 'Evening'
                else:
                    return 'Night'
            
            df['day_part'] = df['Hour'].apply(get_day_part)
        
        # Ensure Month is categorical with proper ordering
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']
        if 'Month' in df.columns:
            df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Load data
df = load_data()

# ==================== COMPACT SIDEBAR ====================
with st.sidebar:
    # Dashboard title (removed dark mode toggle)
    st.markdown("""
    <div style="text-align: left; margin-bottom: 1rem;">
        <h2 style="color: #CC0000; font-size: 1.3rem;">🚌 TTC Dashboard</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Compact navigation
    st.markdown("### 📋 Navigation")
    sections = {
        "Overview": "📊",
        "Time Analysis": "⏰", 
        "Route Analysis": "🛣️",
        "Forecasting": "🔮",
        "Tableau Dash": "📈",
        "Power BI Dash": "📊"
    }
    
    # Using buttons for navigation instead of radio
    for section_name, emoji in sections.items():
        if st.button(f"{emoji} {section_name}", 
                    key=f"nav_{section_name}",
                    use_container_width=True):
            st.session_state.active_section = section_name
    
    st.markdown("---")
    
    # Compact filters with icons
    st.markdown("### 🔍 Filters")
    
    # Year filter with selectbox for single/multi selection
    if 'Year' in df.columns:
        years = sorted(df['Year'].unique())
        selected_years = st.multiselect(
            "📅 Years",
            options=years,
            default=years[-1:] if len(years) >= 1 else years,
            key="year_filter"
        )
    
    # Month filter with select slider for range
    if 'Month' in df.columns:
        months = ['January', 'February', 'March', 'April', 'May', 'June',
                 'July', 'August', 'September', 'October', 'November', 'December']
        selected_months = st.multiselect(
            "📆 Months",
            options=months,
            default=months,
            key="month_filter"
        )
    
    # Time of day filter
    if 'day_part' in df.columns:
        day_parts = ['Morning', 'Afternoon', 'Evening', 'Night']
        selected_day_parts = st.multiselect(
            "⏰ Time of Day",
            options=day_parts,
            default=day_parts,
            key="daypart_filter"
        )
    
    # Route filter for route analysis
    if 'Route Name' in df.columns:
        top_routes = df['Route Name'].value_counts().head(20).index.tolist()
        selected_routes = st.multiselect(
            "🛣️ Routes (Top 20)",
            options=top_routes,
            default=top_routes[:5] if len(top_routes) >= 5 else top_routes,
            key="route_filter"
        )
    
    st.markdown("---")
    
    # Quick stats in sidebar
    st.markdown("### ⚡ Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        if 'Min Delay' in df.columns:
            st.metric("Avg Delay", f"{df['Min Delay'].mean():.1f}m")
    with col2:
        st.metric("Incidents", f"{len(df):,}")

# Initialize active section
if 'active_section' not in st.session_state:
    st.session_state.active_section = "Overview"

# Apply filters
df_filtered = df.copy()

if 'selected_years' in locals() and selected_years:
    df_filtered = df_filtered[df_filtered['Year'].isin(selected_years)]

if 'selected_months' in locals() and selected_months:
    df_filtered = df_filtered[df_filtered['Month'].isin(selected_months)]

if 'selected_day_parts' in locals() and selected_day_parts:
    df_filtered = df_filtered[df_filtered['day_part'].isin(selected_day_parts)]

if 'selected_routes' in locals() and selected_routes and st.session_state.active_section == "Route Analysis":
    df_filtered = df_filtered[df_filtered['Route Name'].isin(selected_routes)]

# ==================== DASHBOARD CONTENT ====================
# Main header
st.markdown("""
<div class="main-header">
    <h1>🚌 TTC Transit Delays Analysis Dashboard</h1>
    <p>Comprehensive analysis of Toronto Transit Commission delays with interactive visualizations and forecasting</p>
</div>
""", unsafe_allow_html=True)

# ==================== OVERVIEW TAB ====================
if st.session_state.active_section == "Overview":
    st.markdown('<h2 class="section-header">📊 Executive Dashboard</h2>', unsafe_allow_html=True)
    
    # Extended KPI Section - 2 rows of 4 metrics
    # Row 1
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_delay = df_filtered['Min Delay'].mean() if 'Min Delay' in df_filtered.columns else 0
        st.markdown(f'<div class="metric-value">{avg_delay:.1f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Avg Delay (min)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_incidents = len(df_filtered)
        st.markdown(f'<div class="metric-value">{total_incidents:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Incidents</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'Min Delay' in df_filtered.columns:
            max_delay = df_filtered['Min Delay'].max()
            st.markdown(f'<div class="metric-value">{max_delay:.0f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Max Delay (min)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'Route Name' in df_filtered.columns:
            unique_routes = df_filtered['Route Name'].nunique()
            st.markdown(f'<div class="metric-value">{unique_routes}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Routes Affected</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Row 2
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'Min Delay' in df_filtered.columns:
            median_delay = df_filtered['Min Delay'].median()
            st.markdown(f'<div class="metric-value">{median_delay:.1f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Median Delay</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'Route Name' in df_filtered.columns:
            top_route = df_filtered['Route Name'].mode()[0] if len(df_filtered) > 0 else "N/A"
            st.markdown(f'<div class="metric-value" style="font-size: 1.3rem;">{top_route}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Most Delayed Route</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'Hour' in df_filtered.columns:
            peak_hour = df_filtered['Hour'].mode()[0] if len(df_filtered) > 0 else "N/A"
            st.markdown(f'<div class="metric-value">{peak_hour}:00</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Peak Delay Hour</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'Day' in df_filtered.columns:
            weekday_delays = df_filtered[df_filtered['Day'].isin(['Monday','Tuesday','Wednesday','Thursday','Friday'])]['Min Delay'].mean() if 'Min Delay' in df_filtered.columns else 0
            st.markdown(f'<div class="metric-value">{weekday_delays:.1f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Avg Weekday Delay</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Data Overview and Methodology
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 📋 Data Overview")
        st.dataframe(df_filtered.head(10), height=300)
        st.caption(f"Showing 10 of {len(df_filtered):,} records")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 📊 Data Summary")
        
        if 'Min Delay' in df_filtered.columns:
            delay_stats = df_filtered['Min Delay'].describe()
            stats_df = pd.DataFrame({
                'Statistic': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
                'Value': [
                    f"{delay_stats['count']:,}",
                    f"{delay_stats['mean']:.1f}",
                    f"{delay_stats['std']:.1f}",
                    f"{delay_stats['min']:.1f}",
                    f"{delay_stats['25%']:.1f}",
                    f"{delay_stats['50%']:.1f}",
                    f"{delay_stats['75%']:.1f}",
                    f"{delay_stats['max']:.1f}"
                ]
            })
            st.dataframe(stats_df, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Methodology and Tools
    st.markdown('<h2 class="section-header">📋 Methodology & Tools</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### Data Sources")
        st.markdown("""
        - Toronto Open Data Portal
        - TTC Public Datasets  
        - Historical Delay Records
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### Methodology")
        st.markdown("""
        - Data Cleaning & Merging
        - Time Series Analysis
        - Prophet Forecasting
        - Statistical Visualization
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### Tools Used")
        st.markdown("""
        - Streamlit for Dashboard
        - Pandas for Data Processing
        - Plotly & Seaborn for Visualization
        - Facebook Prophet for Forecasting
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== TIME ANALYSIS TAB ====================
elif st.session_state.active_section == "Time Analysis":
    st.markdown('<h2 class="section-header">⏰ Time-Based Analysis</h2>', unsafe_allow_html=True)
    
    # First row with two charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Delay Patterns by Time of Day")
        
        if 'day_part' in df_filtered.columns and 'Month' in df_filtered.columns and 'Min Delay' in df_filtered.columns:
            time_data = df_filtered.groupby(['Month', 'day_part'])['Min Delay'].mean().reset_index()
            
            fig = px.line(
                time_data,
                x='Month',
                y='Min Delay',
                color='day_part',
                markers=True,
                line_shape='linear',
                color_discrete_sequence=['#CC0000', '#FF6B6B', '#4ECDC4', '#45B7D1']
            )
            
            fig.update_layout(
                height=400,
                plot_bgcolor='#2d2d2d',
                paper_bgcolor='#2d2d2d',
                font_color='white',
                showlegend=True,
                legend=dict(
                    title="Time of Day",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.02
                ),
                margin=dict(r=100),
                xaxis_title="Month",
                yaxis_title="Average Delay (minutes)"
            )
            
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<div class="info-box">Night-time buses show maximum delays, while afternoon buses experience the least delays.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Hourly Incident Analysis")
        
        if 'Hour' in df_filtered.columns and 'Min Delay' in df_filtered.columns:
            hour_counts = df_filtered.groupby('Hour').size().reset_index(name='count')
            median_delays = df_filtered.groupby('Hour')['Min Delay'].median().reset_index()
            
            # Create subplots with adjusted spacing
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('<b>Incident Frequency by Hour</b>', '<b>Median Delay by Hour</b>'),
                vertical_spacing=0.2,
                row_heights=[0.5, 0.5]
            )
            
            # Incident frequency
            fig.add_trace(
                go.Bar(
                    x=hour_counts['Hour'],
                    y=hour_counts['count'],
                    name='Incident Count',
                    marker_color='#CC0000',
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # Median delay
            fig.add_trace(
                go.Scatter(
                    x=median_delays['Hour'],
                    y=median_delays['Min Delay'],
                    mode='lines+markers',
                    name='Median Delay',
                    line=dict(color='#2E86C1', width=3),
                    marker=dict(size=8)
                ),
                row=2, col=1
            )
            
            # Update axes with better spacing
            fig.update_xaxes(
                title_text="Hour of Day", 
                row=1, col=1, 
                tickmode='linear', 
                dtick=2,
                range=[-0.5, 23.5]
            )
            fig.update_xaxes(
                title_text="Hour of Day", 
                row=2, col=1, 
                tickmode='linear', 
                dtick=2,
                range=[-0.5, 23.5]
            )
            fig.update_yaxes(title_text="Number of Incidents", row=1, col=1)
            fig.update_yaxes(title_text="Median Delay (minutes)", row=2, col=1)
            
            fig.update_layout(
                height=500,
                showlegend=False,
                plot_bgcolor='#2d2d2d',
                paper_bgcolor='#2d2d2d',
                font_color='white',
                margin=dict(t=50, b=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<div class="info-box">Maximum delays occur between 3 PM and 8 PM, with longer delays observed during night hours.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Second row - Monthly Trend Analysis
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Monthly Delay Trends")
        
        if 'Year' in df_filtered.columns and 'Month' in df_filtered.columns and 'Min Delay' in df_filtered.columns:
            monthly_trend = df_filtered.groupby(['Year', 'Month'])['Min Delay'].mean().reset_index()
            
            fig = px.line(
                monthly_trend,
                x='Month',
                y='Min Delay',
                color='Year',
                markers=True,
                line_shape='linear'
            )
            
            fig.update_layout(
                height=400,
                plot_bgcolor='#2d2d2d',
                paper_bgcolor='#2d2d2d',
                font_color='white',
                xaxis_title="Month",
                yaxis_title="Average Delay (minutes)",
                legend_title="Year"
            )
            
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Peak Hours Analysis")
        
        if 'Hour' in df_filtered.columns:
            # Get top 5 peak hours
            hourly_counts = df_filtered['Hour'].value_counts().head(5).reset_index()
            hourly_counts.columns = ['Hour', 'Count']
            hourly_counts = hourly_counts.sort_values('Hour')
            
            fig = px.bar(
                hourly_counts,
                x='Hour',
                y='Count',
                color='Count',
                color_continuous_scale='Reds',
                text='Count'
            )
            
            fig.update_layout(
                height=350,
                plot_bgcolor='#2d2d2d',
                paper_bgcolor='#2d2d2d',
                font_color='white',
                xaxis_title="Hour",
                yaxis_title="Incident Count",
                showlegend=False
            )
            
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            peak_hours = ', '.join([f"{h}:00" for h in hourly_counts['Hour'].tolist()])
            st.markdown(f'**Top Peak Hours:** {peak_hours}')
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== ROUTE ANALYSIS TAB ====================
elif st.session_state.active_section == "Route Analysis":
    st.markdown('<h2 class="section-header">🛣️ Route Performance Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Top 10 Routes by Delay Frequency")
        
        if 'Route Name' in df_filtered.columns:
            route_counts = df_filtered['Route Name'].value_counts().head(10).reset_index()
            route_counts.columns = ['Route Name', 'Delay Count']
            
            fig = px.bar(
                route_counts,
                x='Delay Count',
                y='Route Name',
                orientation='h',
                color='Delay Count',
                color_continuous_scale='Reds',
                text='Delay Count'
            )
            
            fig.update_layout(
                height=500,
                plot_bgcolor='#2d2d2d',
                paper_bgcolor='#2d2d2d',
                font_color='white',
                xaxis_title="Number of Delays",
                yaxis_title="Route Name",
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False
            )
            
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<div class="info-box">Routes Eglinton, Finch, and Lawrence experience highest delay frequencies due to higher traffic volumes.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Top 10 Routes by Delay Severity")
        
        if 'Route Name' in df_filtered.columns and 'Min Delay' in df_filtered.columns:
            # Get top 10 routes by frequency
            top_routes = df_filtered['Route Name'].value_counts().head(10).index.tolist()
            route_severity = df_filtered[df_filtered['Route Name'].isin(top_routes)]
            
            # Calculate average delay per route for top 10
            avg_delay_per_route = route_severity.groupby('Route Name')['Min Delay'].mean().sort_values(ascending=False).reset_index()
            
            fig = px.bar(
                avg_delay_per_route.head(10),  # Ensure we only show top 10
                x='Route Name',
                y='Min Delay',
                color='Min Delay',
                color_continuous_scale='Viridis',
                text_auto='.1f'
            )
            
            fig.update_layout(
                height=500,
                plot_bgcolor='#2d2d2d',
                paper_bgcolor='#2d2d2d',
                font_color='white',
                xaxis_title="Route Name",
                yaxis_title="Average Delay (minutes)",
                xaxis_tickangle=45,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<div class="info-box">Analysis shows correlation between route length and average delay duration for top 10 delayed routes.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Route Comparison Over Time
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("#### Top 10 Routes Performance Over Time")
    
    if 'Route Name' in df_filtered.columns and 'Month' in df_filtered.columns and 'Min Delay' in df_filtered.columns:
        # Select top 10 routes for comparison (changed from top 5 to top 10)
        top_10_routes = df_filtered['Route Name'].value_counts().head(10).index.tolist()
        route_comparison = df_filtered[df_filtered['Route Name'].isin(top_10_routes)]
        
        monthly_route_avg = route_comparison.groupby(['Month', 'Route Name'])['Min Delay'].mean().reset_index()
        
        fig = px.line(
            monthly_route_avg,
            x='Month',
            y='Min Delay',
            color='Route Name',
            markers=True,
            line_shape='linear'
        )
        
        fig.update_layout(
            height=450,
            plot_bgcolor='#2d2d2d',
            paper_bgcolor='#2d2d2d',
            font_color='white',
            xaxis_title="Month",
            yaxis_title="Average Delay (minutes)",
            xaxis_tickangle=45,
            legend_title="Route Name",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                bgcolor='rgba(45,45,45,0.8)'
            ),
            margin=dict(r=150)  # Add right margin for legend
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="info-box">Monthly performance trends for the top 10 most delayed routes. Patterns show seasonal variations and peak periods.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional Route Insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Route Delay Distribution")
        
        if 'Route Name' in df_filtered.columns and 'Min Delay' in df_filtered.columns:
            # Get delay statistics for top 10 routes
            top_routes_stats = df_filtered[df_filtered['Route Name'].isin(top_10_routes)].groupby('Route Name').agg({
                'Min Delay': ['mean', 'median', 'std', 'count']
            }).round(2)
            
            top_routes_stats.columns = ['Avg Delay', 'Median Delay', 'Std Deviation', 'Incident Count']
            top_routes_stats = top_routes_stats.sort_values('Incident Count', ascending=False)
            
            # Display as a table
            st.dataframe(
                top_routes_stats,
                column_config={
                    "Avg Delay": st.column_config.NumberColumn(format="%.1f min"),
                    "Median Delay": st.column_config.NumberColumn(format="%.1f min"),
                    "Std Deviation": st.column_config.NumberColumn(format="%.1f"),
                    "Incident Count": st.column_config.NumberColumn(format="%d")
                }
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Top 10 Routes Peak Hours")
        
        if 'Route Name' in df_filtered.columns and 'Hour' in df_filtered.columns:
            # Analyze peak hours for top 10 routes
            peak_hours_data = []
            for route in top_10_routes:
                route_data = df_filtered[df_filtered['Route Name'] == route]
                if len(route_data) > 0:
                    peak_hour = route_data['Hour'].mode()[0] if len(route_data['Hour'].mode()) > 0 else 'N/A'
                    avg_delay = route_data['Min Delay'].mean() if 'Min Delay' in route_data.columns else 0
                    peak_hours_data.append({
                        'Route': route,
                        'Peak Hour': f"{peak_hour}:00",
                        'Avg Delay': avg_delay
                    })
            
            peak_hours_df = pd.DataFrame(peak_hours_data)
            
            # Create a bar chart for peak hours
            fig = px.bar(
                peak_hours_df,
                x='Route',
                y='Avg Delay',
                color='Peak Hour',
                text='Peak Hour',
                title="Peak Delay Hours by Route"
            )
            
            fig.update_layout(
                height=400,
                plot_bgcolor='#2d2d2d',
                paper_bgcolor='#2d2d2d',
                font_color='white',
                xaxis_title="Route",
                yaxis_title="Average Delay (minutes)",
                xaxis_tickangle=45,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== FORECASTING TAB ====================
elif st.session_state.active_section == "Forecasting":
    st.markdown('<h2 class="section-header">🔮 Delay Forecasting with Prophet</h2>', unsafe_allow_html=True)
    
    # Prepare data for Prophet
    if 'Date' in df_filtered.columns and 'Min Delay' in df_filtered.columns:
        prophet_df = df_filtered[['Date', 'Min Delay']].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        prophet_df = prophet_df.groupby('ds')['y'].mean().reset_index()
        
        # Use a single column layout for configuration to avoid extra space
        col_config, col_plot = st.columns([1, 2])
        
        with col_config:
            # Remove the fixed height and extra div styling that creates space
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### Forecast Configuration")
            
            # Model settings - compact layout

            forecast_period = st.slider(
                "Forecast Period (days)",
                min_value=30,
                max_value=365,
                value=180,
                step=30,
                key="forecast_days"
            )
            
            confidence_level = st.slider(
                "Confidence Interval",
                min_value=0.80,
                max_value=0.99,
                value=0.95,
                step=0.01,
                key="confidence"
            )
            
            seasonality_mode = st.selectbox(
                "Seasonality Mode",
                options=['multiplicative', 'additive'],
                index=0,
                key="seasonality"
            )
            
            # Generate forecast button with compact spacing
            generate_col1, generate_col2 = st.columns([2, 1])
            with generate_col1:
                if st.button("Generate Forecast", type="primary", use_container_width=True):
                    with st.spinner("Training Prophet model..."):
                        m = Prophet(
                            seasonality_mode=seasonality_mode,
                            yearly_seasonality=True,
                            weekly_seasonality=True,
                            interval_width=confidence_level
                        )
                        
                        m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                        m.fit(prophet_df)
                        
                        future = m.make_future_dataframe(periods=forecast_period)
                        forecast = m.predict(future)
                        
                        st.session_state.forecast = forecast
                        st.session_state.prophet_model = m
                        st.session_state.forecast_generated = True
                        
                        st.success("Forecast generated successfully!")
            
            # Forecast metrics - only show if forecast exists
            if 'forecast' in st.session_state:
                st.markdown("---")
                st.markdown("#### Forecast Summary")
                
                last_forecast = st.session_state.forecast.tail(forecast_period)
                avg_forecast = last_forecast['yhat'].mean()
                max_forecast = last_forecast['yhat'].max()
                min_forecast = last_forecast['yhat'].min()
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Average", f"{avg_forecast:.1f} min", delta=None)
                    st.metric("Minimum", f"{min_forecast:.1f} min", delta=None)
                with metric_col2:
                    st.metric("Maximum", f"{max_forecast:.1f} min", delta=None)
                    st.metric("Horizon", f"{forecast_period} days", delta=None)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_plot:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### Delay Forecast Visualization")
            
            if 'forecast' in st.session_state:
                # Plot forecast
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=prophet_df['ds'],
                    y=prophet_df['y'],
                    mode='markers',
                    name='Historical Data',
                    marker=dict(color='#2E86C1', size=6, opacity=0.6)
                ))
                
                # Forecast
                forecast_data = st.session_state.forecast
                fig.add_trace(go.Scatter(
                    x=forecast_data['ds'],
                    y=forecast_data['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#CC0000', width=3)
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast_data['ds'].tolist() + forecast_data['ds'].tolist()[::-1],
                    y=forecast_data['yhat_upper'].tolist() + forecast_data['yhat_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(204, 0, 0, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{int(confidence_level*100)}% Confidence Interval',
                    showlegend=True
                ))
                
                fig.update_layout(
                    height=500,
                    plot_bgcolor='#2d2d2d',
                    paper_bgcolor='#2d2d2d',
                    font_color='white',
                    xaxis_title="Date",
                    yaxis_title="Delay (minutes)",
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(45,45,45,0.8)'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Configure your forecast settings and click 'Generate Forecast' to see predictions.")
                # Show sample data
                fig = px.line(
                    prophet_df.tail(100),
                    x='ds',
                    y='y',
                    title="Recent Historical Delay Data"
                )
                fig.update_layout(height=450, plot_bgcolor='#2d2d2d', paper_bgcolor='#2d2d2d', font_color='white')
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Second row: Forecast components - only show if forecast exists
        if 'forecast' in st.session_state:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### Forecast Components Analysis")
            
            try:
                # Create components visualization
                fig_components = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('<b>Trend Component</b>', '<b>Weekly Seasonality</b>', '<b>Yearly Seasonality</b>'),
                    vertical_spacing=0.17,
                    row_heights=[0.4, 0.3, 0.3]
                )
                
                # Get components from forecast
                forecast_data = st.session_state.forecast
                
                # Trend
                fig_components.add_trace(
                    go.Scatter(
                        x=forecast_data['ds'],
                        y=forecast_data['trend'],
                        mode='lines',
                        name='Trend',
                        line=dict(color='#CC0000', width=2)
                    ),
                    row=1, col=1
                )
                
                # Weekly seasonality
                weekly_data = forecast_data[['ds', 'weekly']].tail(7*4)  # Last 4 weeks
                fig_components.add_trace(
                    go.Scatter(
                        x=weekly_data['ds'],
                        y=weekly_data['weekly'],
                        mode='lines',
                        name='Weekly',
                        line=dict(color='#2E86C1', width=2)
                    ),
                    row=2, col=1
                )
                
                # Yearly seasonality
                yearly_data = forecast_data[['ds', 'yearly']].tail(365)  # Last year
                fig_components.add_trace(
                    go.Scatter(
                        x=yearly_data['ds'],
                        y=yearly_data['yearly'],
                        mode='lines',
                        name='Yearly',
                        line=dict(color='#4CAF50', width=2)
                    ),
                    row=3, col=1
                )
                
                fig_components.update_layout(
                    height=600,
                    plot_bgcolor='#2d2d2d',
                    paper_bgcolor='#2d2d2d',
                    font_color='white',
                    showlegend=False,
                    margin=dict(t=50, b=40)  # Reduced top and bottom margins
                )
                
                fig_components.update_xaxes(title_text="Date", row=1, col=1)
                fig_components.update_xaxes(title_text="Date", row=2, col=1)
                fig_components.update_xaxes(title_text="Date", row=3, col=1)
                
                fig_components.update_yaxes(title_text="Trend", row=1, col=1)
                fig_components.update_yaxes(title_text="Weekly Effect", row=2, col=1)
                fig_components.update_yaxes(title_text="Yearly Effect", row=3, col=1)
                
                st.plotly_chart(fig_components, use_container_width=True)
                
            except Exception as e:
                st.warning("Could not generate detailed component analysis.")
                try:
                    components_fig = st.session_state.prophet_model.plot_components(
                        st.session_state.forecast,
                        figsize=(10, 9)  # Reduced figsize
                    )
                    st.pyplot(components_fig)
                except:
                    st.info("Component plot not available.")
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Date or Min Delay columns not found in data. Forecasting requires temporal data.")

# ==================== TABLEAU DASHBOARD TAB ====================
elif st.session_state.active_section == "Tableau Dash":
    st.markdown('<h2 class="section-header">📈 Tableau Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">Interactive Tableau dashboard showing comprehensive analysis of TTC delays. This visualization provides insights into delay patterns, frequency, and operational metrics.</div>', unsafe_allow_html=True)
    
    # Remove the max-width constraint and use full width
    def tableau_dashboard_component():
        tableau_html_code = """
        <div class='tableauPlaceholder' id='viz1705632432757' style='position: relative; width: 100%;'>
        <noscript>
            <a href='#'>
                <img alt='TTC Bus Delay' src='https://public.tableau.com/static/images/TT/TTCDelayDash/Dashboard1/1_rss.png' style='border: none; width: 100%;' />
            </a>
        </noscript>
        <object class='tableauViz' style='display:none;'>
            <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
            <param name='embed_code_version' value='3' />
            <param name='site_root' value='' />
            <param name='name' value='TTCDelayDash/Dashboard1' />
            <param name='tabs' value='no' />
            <param name='toolbar' value='yes' />
            <param name='static_image' value='https://public.tableau.com/static/images/TT/TTCDelayDash/Dashboard1/1.png' />
            <param name='animate_transition' value='yes' />
            <param name='display_static_image' value='yes' />
            <param name='display_spinner' value='yes' />
            <param name='display_overlay' value='yes' />
            <param name='display_count' value='yes' />
            <param name='language' value='en-US' />
            <param name='filter' value='publish=yes' />
        </object>
        </div>
        <script type='text/javascript'>
            var divElement = document.getElementById('viz1705632432757');
            var vizElement = divElement.getElementsByTagName('object')[0];
            
            // Set full width and proportional height
            function setTableauDimensions() {
                var containerWidth = window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
                
                // Use a fixed height for the Tableau dashboard (adjust as needed)
                var dashboardHeight = 900;
                
                // Set width to 100% of container
                vizElement.style.width = '100%';
                vizElement.style.height = dashboardHeight + 'px';
            }
            
            // Set initial dimensions
            setTableauDimensions();
            
            // Update on window resize
            window.addEventListener('resize', setTableauDimensions);
            
            var scriptElement = document.createElement('script');
            scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
            vizElement.parentNode.insertBefore(scriptElement, vizElement);
        </script>
        """
        return html(tableau_html_code, height=950, scrolling=False)
    
    # Display the Tableau dashboard with full width
    tableau_dashboard_component()
    
    # Key Insights
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 📊 Key Insights from Tableau Analysis")
    
    insight_cols = st.columns(2)
    
    with insight_cols[0]:
        st.markdown("""
        **Temporal Patterns:**
        - Higher delays during midnight hours on weekdays
        - Peak delays from 6 AM - 8 AM on weekends
        - Evening rush hour (5 PM - 7 PM) shows consistent delay patterns
        
        **Incident Analysis:**
        - Diversion incidents cause longest delays
        - Mechanical issues are most frequent
        - Investigation delays account for significant portion
        """)
    
    with insight_cols[1]:
        st.markdown("""
        **Operational Insights:**
        - Downtown core shows highest frequency
        - Suburban terminals experience longer individual delays
        - Transfer stations are delay hotspots
        
        **Comparative Analysis:**
        - Early 2023 saw higher delays than 2022
        - Improvements noted from mid-September 2023
        - Weekend patterns differ significantly from weekdays
        """)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== POWER BI DASHBOARD TAB ====================
elif st.session_state.active_section == "Power BI Dash":
    st.markdown('<h2 class="section-header">📊 Power BI Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">Power BI dashboard providing advanced analytics and geographical insights into TTC delay patterns. Interactive visualizations help identify trends and operational challenges.</div>', unsafe_allow_html=True)
    
    # Power BI Link in a prominent box
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 🔗 Access Power BI Dashboard")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #333333 0%, #2a2a2a 100%); 
                border-left: 4px solid #0078d4; 
                padding: 1.5rem; 
                border-radius: 8px; 
                margin: 1.5rem 0;
                text-align: center;">
        <h4 style="color: #0078d4; margin-bottom: 1rem;">Interactive Power BI Dashboard</h4>
        <p style="margin-bottom: 1.5rem; color: #cccccc;">For the complete interactive experience with all visualizations and filters, click the link below:</p>
        <a href="https://app.powerbi.com/reportEmbed?reportId=3aa46b8e-572e-44fa-80cb-ba8226145eed" 
           target="_blank" 
           style="background: linear-gradient(90deg, #0078d4 0%, #005a9e 100%); 
                  color: white; 
                  padding: 0.75rem 1.5rem; 
                  border-radius: 5px; 
                  text-decoration: none;
                  font-weight: 600;
                  display: inline-block;
                  transition: all 0.3s ease;">
           📊 Open Power BI Dashboard
        </a>
        <p style="margin-top: 1rem; font-size: 0.9rem; color: #999;">
            <i>Note: You'll need appropriate permissions to access the Power BI report</i>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Alternative images if Power BI doesn't load
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 📷 Power BI Report Previews")
    
    # Initialize session state for image pagination
    if 'powerbi_page' not in st.session_state:
        st.session_state.powerbi_page = 0
    
    # Get image paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_files = ["img2.png", "img3.png", "img4.png", "img5.png"]
    image_paths = [os.path.join(current_dir, img) for img in image_files]
    
    # Filter existing images
    existing_paths = [img for img in image_paths if os.path.exists(img)]
    
    if existing_paths:
        total_images = len(existing_paths)
        
        # Display two images per row
        cols = st.columns(2)
        
        for i in range(2):
            idx = st.session_state.powerbi_page * 2 + i
            if idx < total_images:
                with cols[i]:
                    st.image(
                        existing_paths[idx],
                        caption=f"Power BI Report Preview {idx + 1}"
                    )
        
        # Navigation
        if total_images > 2:
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                if st.button("◀ Previous", disabled=st.session_state.powerbi_page == 0):
                    st.session_state.powerbi_page -= 1
                    st.rerun()
            
            with col3:
                if st.button("Next ▶", disabled=(st.session_state.powerbi_page + 1) * 2 >= total_images):
                    st.session_state.powerbi_page += 1
                    st.rerun()
            
            with col2:
                current_page = st.session_state.powerbi_page + 1
                total_pages = (total_images + 1) // 2
                st.markdown(f"**Page {current_page} of {total_pages}**", unsafe_allow_html=True)
    else:
        st.info("Power BI preview images not found. Ensure image files are in the correct directory.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Insights from Power BI
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 🔍 Key Insights from Power BI Analysis")
    
    insight_cols = st.columns(2)
    
    with insight_cols[0]:
        st.markdown("""
        **Geographic Patterns:**
        - Highest delay concentrations in downtown core
        - Suburban routes experience fewer but longer delays
        - Transfer points are critical delay hotspots
        
        **Temporal Analysis:**
        - Morning peak shows highest frequency
        - Evening peak shows longest durations
        - Weekends have different delay patterns
        """)
    
    with insight_cols[1]:
        st.markdown("""
        **Operational Metrics:**
        - Route efficiency varies by time of day
        - Certain vehicle types more prone to delays
        - Weather impacts delay patterns significantly
        
        **Improvement Areas:**
        - Focus on top 5 delay-prone routes
        - Optimize scheduling during peak hours
        - Improve maintenance scheduling
        """)
    st.markdown('</div>', unsafe_allow_html=True)