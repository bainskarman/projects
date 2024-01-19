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
current_path = os.getcwd()
path = os.path.join(current_path, 'TTC_Delay_Analysis/data.csv')
df = pd.read_csv(path)
st.set_page_config(layout='wide')
st.title("TTC Delays Analysis Report")
st.subheader("Matplotlib and Seaborn Visualitzations")
col1, col2 = st.columns(2)

with col1:
    custom_palette = sns.color_palette("Set1", n_colors=len(df['day_part'].unique()))
    sns.set_style('darkgrid')
    sns.set_palette(custom_palette)

    g = sns.relplot(data=df, kind='line', y='Min Delay', x='Month', hue='day_part', style='day_part',
                    aspect=2, markers=True, ci=None, dashes=False)

    g.fig.suptitle("Delay During Different Parts of the Day in all Months", y=1.03)
    g.set(xlabel='Months Name', ylabel='Total Minimum Delay ')

    # Rotate the x-axis labels
    plt.xticks(rotation=75)

    # Display the plot in Streamlit
    st.pyplot(plt)
with col2:
    # Calculate the count of delays for each 'Hour'
    df_count = df.groupby('Hour')['Min Delay'].count().reset_index()

    # Calculate the median delay for each 'Hour'
    df_median = df.groupby('Hour')['Min Delay'].median().reset_index()

    # Set up Matplotlib subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # First subplot for incident frequency
    sns.lineplot(data=df_count, x='Hour', y='Min Delay', label='Freq. of incidents hours by hours', color='red', ax=axes[0])
    axes[0].set_xticks(np.arange(0, 24))
    axes[0].set_xlabel("Hour of the Day")
    axes[0].set_ylabel('Num of Incidents')
    axes[0].set_title("Incident Frequency in TTC")

    # Second subplot for median delay
    sns.lineplot(data=df_median, x='Hour', y='Min Delay', label='Median Delay', color='green', ax=axes[1])
    axes[1].set_xticks(np.arange(0, 24))
    axes[1].set_xlabel('Hour of the Day')
    axes[1].set_ylabel('Median Delay')
    axes[1].set_title('Median Delay of Incidents')

    # Ensures proper spacing between subplots
    plt.tight_layout()

    # Display the subplots in Streamlit
    st.pyplot(plt)
# Continue with the report
st.header("Insights from Visualizations")

st.write("The visualizations provide valuable insights into the delays in TTC throughout the year.")

st.subheader("1. Delay During Different Parts of the Day in All Months")

st.write("The first plot illustrates the variation in delay occurrences during different parts of the day across all months. "
         "It can be observed that throughout the year, night-time buses tend to have the maximum average delays, while afternoon buses experience the least delays. However, there is a notable anomaly in December, where there is a sudden rise in afternoon delays and a corresponding fall in night delays.")

st.subheader("2. Incident Frequency and Median Delay Analysis")

st.write("The second set of plots focuses on incident frequency and median delay analysis by hour of the day. Key observations include: "
         "- The maximum delays tend to occur between 3 pm and 8 pm, indicating a potential influence of office hours and increased traffic during these hours."
         "- Longer delay times are observed during the night, particularly from 1 am to 3 am. This could be attributed to reduced services during these hours, making replacements for buses or operators less readily available.")

st.subheader("Plotly Visualitzations")
col4,col5 = st.columns(2)

with col4:
    # Calculate the count of delays for each 'Route Name'
    delay_counts = df['Route Name'].value_counts().reset_index()
    delay_counts.columns = ['Route Name', 'Delay Count']

    # Select the top 10 routes
    top_10_routes = delay_counts.head(10)

    # Create a bar plot using Plotly Express
    fig_top_10_routes = px.bar(top_10_routes, x='Route Name', y='Delay Count',
                            title='Top 10 Routes with Most Frequent Delays',
                            color='Delay Count', color_continuous_scale='Reds')
    fig_top_10_routes.update_layout(showlegend=False)
    # Show the plot
    st.plotly_chart(fig_top_10_routes)
with col5:
    def plot_avg_min_delay(df_2022, df_2023):
        # Define the order of months
        months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

        # Calculate average 'Min Delay' for each month
        avg_delay_2022 = df_2022.groupby('Month')['Min Delay'].mean().reindex(months_order).reset_index()
        avg_delay_2023 = df_2023.groupby('Month')['Min Delay'].mean().reindex(months_order).reset_index()

        # Create a Plotly line graph
        fig = px.line(avg_delay_2022, x='Month', y='Min Delay', title='Average Min Delay per Month - Year 2022',
                    labels={'Min Delay': 'Average Min Delay', 'Month': 'Months Name'},
                    category_orders={'Month': months_order},
                    line_shape='linear', markers=True)

        # Add a line for Year 2023
        fig.add_trace(go.Scatter(x=avg_delay_2023['Month'], y=avg_delay_2023['Min Delay'],
                                mode='lines+markers', name='Year 2023', line=dict(color='orange')))

        # Update layout
        fig.update_layout(title_text="Average Min Delay per Month - Year 2022 vs Year 2023", showlegend=False)

        return fig

    df_2022 = df[df['Year'] == 2022]
    df_2023 = df[df['Year'] == 2023]
    df=df[['Date','Min Delay']]
    df.columns = ['ds','y']
    # Plot the average min delay graph
    st.plotly_chart(plot_avg_min_delay(df_2022, df_2023))
st.header("Insights from Plotly Visualizations")

st.write("The Plotly visualizations provide additional insights into specific aspects of TTC delays.")

st.subheader("1. Routes with Highest Delay Frequency")

st.write("Analyzing the bar plot of the top 10 routes with the most frequent delays, it is evident that routes 'Eglinton', 'Finch', and 'Lawrence' consistently experience the highest delay frequencies. This can be attributed to these routes being major transit arteries, leading to higher traffic and congestion. Additionally, these routes tend to be longer compared to others, contributing to increased potential for delays.")

st.subheader("2. Comparison of Delay Frequency in 2022 and 2023")

st.write("Examining the line graph comparing the average minimum delay per month for 2022 and 2023, a few notable trends emerge:")
st.write("- The delay frequency in 2023 started notably higher than in 2022, indicating potential challenges or external factors impacting the transit system.")
st.write("- However, from mid-September 2023 onward, the delay frequency consistently decreased, ultimately falling below the levels observed in the same period in 2022. This improvement could be attributed to proactive measures taken by the transit authority to address and mitigate delays.")
    # Filter data for Year 2022 and 2023
st.title("Tableau Reports")
def tableau_component():
    # Tableau HTML code
    tableau_html_code = """
    <div class='tableauPlaceholder' id='viz1705632432757' style='position: relative'><noscript><a href='#'><img alt='TTC Bus Delay ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;TT&#47;TTCDelayDash&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='TTCDelayDash&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;TT&#47;TTCDelayDash&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1705632432757');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1650px';vizElement.style.height='887px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1650px';vizElement.style.height='887px';} else { vizElement.style.width='100%';vizElement.style.height='1727px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
    """
    return html(tableau_html_code, height=960, width=1750, scrolling=True)
tableau_component()
st.header("Insights from Tableau Reports")
st.subheader("1. Heatmap Plot: Average Delay Time Analysis")

# Additional insights
st.write("- **Weekdays vs. Weekends:** The average delay is higher during the mid-night hours (2 am - 4 am) on weekdays, possibly indicating maintenance or operational challenges during the early morning hours. Interestingly, this pattern shifts to 6 am - 8 am on weekends, suggesting altered transit dynamics during different days of the week.")
st.write("- **Midday Resilience:** The middle parts of the day (e.g., 10 am - 2 pm) consistently exhibit lower average delays, indicating relative operational stability during these hours.")

# 2. Top Delay Incidents
st.subheader("2. Top Delay Incidents")
st.write(f"The top delay incidents are 'Diversion', 'Mechanical', and 'Investigation', accounting for approximately 50% of all delays. These incidents are often unpredictable and challenging to address, requiring proactive measures to mitigate their impact on transit operations.")
# 3. Delay Frequency Comparison (2022 vs. 2023)
# 3. Delay Frequency Comparison (2022 vs. 2023)
st.subheader("3. Delay Frequency Comparison (2022 vs. 2023)")


# Additional insights
st.write("- **Early 2023 Surge:** The delay frequency in the early months of 2023 was notably higher compared to 2022, suggesting potential challenges or external factors affecting transit operations.")
st.write("- **Mid-September Improvement:** Since mid-September 2023, there has been a consistent decrease in delay frequency, ultimately falling below the levels observed in the same period in 2022. This improvement may reflect proactive measures taken to address and mitigate delays.")

# 4. Route Length and Delay Relationship
st.subheader("4. Route Length and Delay Relationship")

# Additional insights
st.write("An analysis indicates a positive correlation between the length of transit routes and the duration of delays. Longer routes, such as 'Eglinton', 'Finch', and 'Lawrence', consistently experience longer delays. This insight emphasizes the need for targeted strategies to address delays on extended routes.")

# 5. Temporal Patterns of Delays
st.subheader("5. Temporal Patterns of Delays")

# Additional insights
st.write("Morning delays exhibit a unique pattern, approximating the mean of night and afternoon delays. This observation may be linked to a combination of factors such as increased passenger demand during morning rush hours and potential carry-over delays from the night.")

# Conclusion
st.subheader("Conclusion")

st.write("In conclusion, the interactive dashboard provides a comprehensive analysis of TTC delays, offering insights into temporal patterns, delay incidents, frequency comparisons, route length relationships, and more. These insights can inform strategic decisions to enhance operational efficiency, improve incident response, and optimize service planning.")
# Streamlit App
# Assuming df is your DataFrame with 'ds' and 'y' columns
m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True, weekly_seasonality=True)
m.add_seasonality(name='monthly', period=30.44, fourier_order=5)  # Adjust period based on your data
m.fit(df)
# Streamlit App
st.title("Delay Forecast with Prophet")

# Create a future DataFrame for forecasting
future = m.make_future_dataframe(periods=365)

# Make forecast
forecast = m.predict(future)

# Plotly figure for the forecast
fig_forecast = go.Figure()

# Plot only the forecast line
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='blue')))

# Set y-axis limits
fig_forecast.update_yaxes(range=[0, 40])

# Customize legend
fig_forecast.update_layout(legend=dict(x=0, y=1, traceorder="normal", orientation="h"))

# Set the title

col6,col7 = st.columns(2)
with col6:
    # Plot the forecast
    st.markdown('Delay Forecast')
    st.plotly_chart(fig_forecast)

with col7:
    # Plot the components
    st.markdown("Forecast Components")
    fig_components = m.plot_components(forecast, figsize=(8, 5))
    st.pyplot(fig_components)
import streamlit as st
st.header("Insights from Forecasting Analysis")
# 1. Forecasting for 2024
# 1. Forecasting for 2024
st.subheader("1. Forecasting for 2024")
st.write("The forecasting analysis leverages the Prophet model, capturing underlying patterns and trends in historical delay data. The model predicts that the delay pattern in 2024 will closely resemble that of the previous two years, providing a basis for strategic planning and operational decision-making.")

# 2. Components of Prophet
st.subheader("2. Components of Prophet")
st.write("The breakdown of Prophet's components sheds light on key drivers of delays. The model indicates that maximum delays are anticipated on weekends, potentially influenced by increased passenger demand or reduced operational capacity. Additionally, Wednesdays emerge as a day with heightened delays, suggesting unique operational challenges specific to mid-week.")

# 3. Monthly Analysis
st.subheader("3. Monthly Analysis")
st.write("The monthly analysis reveals notable variations in delay patterns. January, April, and December are forecasted to experience lower delays compared to other months. This could be attributed to factors such as reduced passenger load during holiday seasons, improved weather conditions, or altered operational schedules.")

# 4. Weekly Analysis
st.subheader("4. Weekly Analysis")
st.write("The weekly analysis highlights temporal patterns within individual days. Longer delays are anticipated in the middle of the week, particularly on Wednesdays. This observation may be linked to factors such as increased traffic, operational challenges, or specific events occurring during this timeframe. Weekends also exhibit higher delays, potentially influenced by leisure travel or altered service schedules.")

# Conclusion
st.subheader("Conclusion")
st.write("In conclusion, the comprehensive Prophet forecasting analysis provides actionable insights for transit management in 2024. Understanding the components, monthly trends, and weekly variations enables proactive planning and resource allocation. It is essential to use these insights as a foundation for continuous improvement, considering external factors and adapting strategies to ensure an efficient and resilient transit system.")

