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

# Section 1: Matplotlib and Seaborn Visualizations
st.header("📊 Matplotlib and Seaborn Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Delay During Different Parts of the Day")
    custom_palette = sns.color_palette("Set1", n_colors=len(df['day_part'].unique()))
    sns.set_style('darkgrid')
    sns.set_palette(custom_palette)

    g = sns.relplot(data=df, kind='line', y='Min Delay', x='Month', hue='day_part', style='day_part',
                    aspect=2, markers=True, ci=None, dashes=False)
    g.fig.suptitle("Delay During Different Parts of the Day in all Months", y=1.03)
    g.set(xlabel='Months Name', ylabel='Total Minimum Delay')
    plt.xticks(rotation=75)
    st.pyplot(plt)

with col2:
    st.subheader("Incident Frequency and Median Delay Analysis")
    df_count = df.groupby('Hour')['Min Delay'].count().reset_index()
    df_median = df.groupby('Hour')['Min Delay'].median().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.lineplot(data=df_count, x='Hour', y='Min Delay', label='Freq. of incidents hours by hours', color='red', ax=axes[0])
    axes[0].set_xticks(np.arange(0, 24))
    axes[0].set_xlabel("Hour of the Day")
    axes[0].set_ylabel('Num of Incidents')
    axes[0].set_title("Incident Frequency in TTC")

    sns.lineplot(data=df_median, x='Hour', y='Min Delay', label='Median Delay', color='green', ax=axes[1])
    axes[1].set_xticks(np.arange(0, 24))
    axes[1].set_xlabel('Hour of the Day')
    axes[1].set_ylabel('Median Delay')
    axes[1].set_title('Median Delay of Incidents')

    plt.tight_layout()
    st.pyplot(plt)

# Insights from Visualizations
st.header("🔍 Insights from Visualizations")
st.markdown("""
    - **Delay During Different Parts of the Day:** Night-time buses tend to have the maximum average delays, while afternoon buses experience the least delays. An anomaly in December shows a sudden rise in afternoon delays.
    - **Incident Frequency and Median Delay:** Maximum delays occur between 3 pm and 8 pm, with longer delays observed during the night (1 am - 3 am).
    """)

# Section 2: Plotly Visualizations
st.header("📈 Plotly Visualizations")
col4, col5 = st.columns(2)

with col4:
    st.subheader("Top 10 Routes with Most Frequent Delays")
    delay_counts = df['Route Name'].value_counts().reset_index()
    delay_counts.columns = ['Route Name', 'Delay Count']
    top_10_routes = delay_counts.head(10)
    fig_top_10_routes = px.bar(top_10_routes, x='Route Name', y='Delay Count',
                               title='Top 10 Routes with Most Frequent Delays',
                               color='Delay Count', color_continuous_scale='Reds')
    fig_top_10_routes.update_layout(showlegend=False)
    st.plotly_chart(fig_top_10_routes)

with col5:
    st.subheader("Average Min Delay per Month - 2022 vs 2023")
    def plot_avg_min_delay(df_2022, df_2023):
        months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        avg_delay_2022 = df_2022.groupby('Month')['Min Delay'].mean().reindex(months_order).reset_index()
        avg_delay_2023 = df_2023.groupby('Month')['Min Delay'].mean().reindex(months_order).reset_index()
        fig = px.line(avg_delay_2022, x='Month', y='Min Delay', title='Average Min Delay per Month - Year 2022',
                      labels={'Min Delay': 'Average Min Delay', 'Month': 'Months Name'},
                      category_orders={'Month': months_order},
                      line_shape='linear', markers=True)
        fig.add_trace(go.Scatter(x=avg_delay_2023['Month'], y=avg_delay_2023['Min Delay'],
                                 mode='lines+markers', name='Year 2023', line=dict(color='orange')))
        fig.update_layout(title_text="Average Min Delay per Month - Year 2022 vs Year 2023", showlegend=False)
        return fig

    df_2022 = df[df['Year'] == 2022]
    df_2023 = df[df['Year'] == 2023]
    st.plotly_chart(plot_avg_min_delay(df_2022, df_2023))

# Insights from Plotly Visualizations
st.header("🔍 Insights from Plotly Visualizations")
st.markdown("""
    - **Routes with Highest Delay Frequency:** Routes 'Eglinton', 'Finch', and 'Lawrence' experience the highest delay frequencies due to higher traffic and longer route lengths.
    - **Comparison of Delay Frequency in 2022 and 2023:** Delay frequency in 2023 started higher but decreased from mid-September, indicating potential improvements in transit operations.
    """)

# Section 3: Tableau Reports
st.header("📊 Tableau Reports")
st.markdown("""
    Below are embedded Tableau dashboards providing insights into TTC delays and their geographic distribution.
    """)

def delay_dashboard_component():
    tableau_html_code = """
    <div class='tableauPlaceholder' id='viz1705632432757' style='position: relative'><noscript><a href='#'><img alt='TTC Bus Delay ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;TT&#47;TTCDelayDash&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='TTCDelayDash&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;TT&#47;TTCDelayDash&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1705632432757');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1650px';vizElement.style.height='887px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1650px';vizElement.style.height='887px';} else { vizElement.style.width='100%';vizElement.style.height='1727px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
    """
    return st.components.v1.html(tableau_html_code, height=960, width=1750, scrolling=True)

def map_component():
    tableau_map_html = """
    <div class='tableauPlaceholder' id='viz1743143274465' style='position: relative'><noscript><a href='#'><img alt='Map ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;CM&#47;CMCDZ55W8&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;CMCDZ55W8' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;CM&#47;CMCDZ55W8&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1743143274465');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
    """
    return st.components.v1.html(tableau_map_html, height=700, width=1750, scrolling=True)

delay_dashboard_component()
st.markdown("### Geographic Distribution of Subway Delays")
map_component()

# Insights from Tableau Reports
st.header("🔍 Insights from Tableau Reports")
st.markdown("""
    - **Heatmap Plot:** Higher delays during mid-night hours on weekdays and 6 am - 8 am on weekends.
    - **Top Delay Incidents:** 'Diversion', 'Mechanical', and 'Investigation' account for 50% of all delays.
    - **Delay Frequency Comparison (2022 vs. 2023):** Early 2023 saw higher delays, but improvements were noted from mid-September.
    - **Route Length and Delay Relationship:** Longer routes like 'Eglinton', 'Finch', and 'Lawrence' experience longer delays.
    - **Temporal Patterns of Delays:** Morning delays are influenced by increased passenger demand and potential carry-over delays from the night.
    """)

# Section 4: Rolling Average and ADX Analysis
st.header("📈 Rolling Average and ADX Analysis")

# Load and display the ADX image from the same folder
adx_image_path = os.path.join(os.path.dirname(__file__), "ADX.png")

st.image(adx_image_path, caption="Rolling Average and ADX of Delays", use_container_width=True)


st.title("Power BI Report with Images")



# Initialize session state for the current page
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0

# Display the Power BI report
st.markdown("[Click here to view the Power BI report](https://app.powerbi.com/reportEmbed?reportId=3aa46b8e-572e-44fa-80cb-ba8226145eed)")

# Display the note with the ArcGIS limitation
st.markdown(
    """
    <div style="background-color: #ffcc00; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center;">
        Note: Due to the limitations of the free version of ArcGIS, not all visuals may be visible to everyone. Thus, preview images are attached below.
    </div>
    """,
    unsafe_allow_html=True
)

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
image_files = ["img2.png", "img3.png", "img4.png", "img5.png"]
image_paths = [os.path.join(current_dir, img) for img in image_files]  # Full paths for images



# Check if the images exist
missing_images = [img for img in image_paths if not os.path.exists(img)]
if missing_images:
    st.error(f"Error: The following image files are missing: {', '.join(missing_images)}")
    st.stop()

try:
    total_images = len(image_paths)
    # Create two columns for displaying images side by side
    col1, col2 = st.columns(2)
    with col1:
        # Display the first image with reduced size (e.g., 800px width)
        st.image(image_paths[st.session_state.current_page], width=800)
    with col2:
        # Check if there is another image available for display
        if st.session_state.current_page + 1 < total_images:
            st.image(image_paths[st.session_state.current_page + 1], width=800)

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.button("◀ Previous", disabled=st.session_state.current_page == 0):
            st.session_state.current_page -= 1
            st.rerun()
    with col3:
        if st.button("Next ▶", disabled=st.session_state.current_page == total_images - 2):  # Adjust for two images
            st.session_state.current_page += 2  # Move by 2 for next page
            st.rerun()

    

except Exception as e:
    st.error(f"Error loading image: {str(e)}")


# Insights from Rolling Average and ADX
st.header("🔍 Insights from Rolling Average and ADX")
st.markdown("""
    - **30-Day Rolling Average:** The rolling average shows fluctuations in delays over time, indicating variability in delay patterns.
    - **ADX (Average Directional Index):** Despite the fluctuations in the rolling average, the ADX values are trending downward, suggesting a weakening trend in delay patterns.
    - **Trend Analysis:** The combination of the rolling average and ADX indicates that while delays may fluctuate, the overall trend strength is decreasing, which could signal improving conditions or reduced variability in delays.
    """)

# Section 4: Forecasting with Prophet
st.header("🔮 Delay Forecast with Prophet")
df = df[['Date', 'Min Delay']]
df.columns = ['ds', 'y']
m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True, weekly_seasonality=True)
m.add_seasonality(name='monthly', period=30.44, fourier_order=5)
m.fit(df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

col6, col7 = st.columns(2)
with col6:
    st.subheader("Delay Forecast")
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='blue')))
    fig_forecast.update_yaxes(range=[0, 40])
    fig_forecast.update_layout(legend=dict(x=0, y=1, traceorder="normal", orientation="h"))
    st.plotly_chart(fig_forecast)

with col7:
    st.subheader("Forecast Components")
    fig_components = m.plot_components(forecast, figsize=(8, 5))
    st.pyplot(fig_components)

# Insights from Forecasting Analysis
st.header("🔍 Insights from Forecasting Analysis")
st.markdown("""
    - **Forecasting for 2024:** The delay pattern in 2024 is expected to resemble previous years, aiding in strategic planning.
    - **Components of Prophet:** Maximum delays are anticipated on weekends and Wednesdays.
    - **Monthly Analysis:** Lower delays are forecasted for January, April, and December.
    - **Weekly Analysis:** Longer delays are expected mid-week, particularly on Wednesdays.
    """)

# Conclusion
st.header("🎯 Conclusion")
st.markdown("""
    The comprehensive analysis provides actionable insights for improving TTC operations. By understanding delay patterns, frequency, and forecasting future trends, transit management can implement targeted strategies to enhance efficiency and reduce delays.
    """)