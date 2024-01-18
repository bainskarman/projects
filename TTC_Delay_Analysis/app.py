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

# Streamlit App
st.title("TTC Delays Analysis Report")


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


# Calculate the count of delays for each 'Hour'
df_count = df.groupby('Hour')['Min Delay'].count().reset_index()

# Calculate the median delay for each 'Hour'
df_median = df.groupby('Hour')['Min Delay'].median().reset_index()

# Set up Matplotlib subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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

# Calculate the count of delays for each 'Route Name'
delay_counts = df['Route Name'].value_counts().reset_index()
delay_counts.columns = ['Route Name', 'Delay Count']

# Select the least 10 routes with the least frequent delays
least_10_routes = delay_counts.nsmallest(10, 'Delay Count')

# Create a bar plot using Plotly Express
fig_least_frequent_delays = px.bar(least_10_routes, x='Route Name', y='Delay Count',
                                   title='Top 10 Routes with Least Frequent Delays',
                                   color='Delay Count', color_continuous_scale='Reds')

# Show the plot
st.plotly_chart(fig_least_frequent_delays)

# Calculate the count of delays for each 'Route Name'
delay_counts = df['Route Name'].value_counts().reset_index()
delay_counts.columns = ['Route Name', 'Delay Count']

# Select the top 10 routes
top_10_routes = delay_counts.head(10)

# Create a bar plot using Plotly Express
fig_top_10_routes = px.bar(top_10_routes, x='Route Name', y='Delay Count',
                           title='Top 10 Routes with Most Frequent Delays',
                           color='Delay Count', color_continuous_scale='Reds')

# Show the plot
st.plotly_chart(fig_top_10_routes)

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
    fig.update_layout(title_text="Average Min Delay per Month - Year 2022 vs Year 2023")

    return fig

# Filter data for Year 2022 and 2023
df_2022 = df[df['Year'] == 2022]
df_2023 = df[df['Year'] == 2023]
df=df[['Date','Min Delay']]
df.columns = ['ds','y']
# Plot the average min delay graph
st.plotly_chart(plot_avg_min_delay(df_2022, df_2023))
# Assuming df is your DataFrame with 'ds' and 'y' columns
m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True, weekly_seasonality=True)
m.add_seasonality(name='monthly', period=30.44, fourier_order=5)  # Adjust period based on your data
m.fit(df)

st.subheader("Tableau Reports")
def tableau_component():
    # Tableau HTML code
    tableau_html_code = """
    <div class='tableauPlaceholder' id='viz1705566063091' style='position: relative'><noscript><a href='#'><img alt='Dashboard 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;TT&#47;TTCDelayDash&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='TTCDelayDash&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;TT&#47;TTCDelayDash&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>
    <script type='text/javascript'>
        var divElement = document.getElementById('viz1705566063091');
        var vizElement = divElement.getElementsByTagName('object')[0];
        if (divElement.offsetWidth > 800) {
            vizElement.style.minWidth='420px';
            vizElement.style.maxWidth='650px';
            vizElement.style.width='100%';
            vizElement.style.minHeight='587px';
            vizElement.style.maxHeight='887px';
            vizElement.style.height=(divElement.offsetWidth*0.75)+'px';
        } else if (divElement.offsetWidth > 500) {
            vizElement.style.minWidth='420px';
            vizElement.style.maxWidth='650px';
            vizElement.style.width='100%';
            vizElement.style.minHeight='587px';
            vizElement.style.maxHeight='887px';
            vizElement.style.height=(divElement.offsetWidth*0.75)+'px';
        } else {
            vizElement.style.width='100%';
            vizElement.style.height='877px';
        }
        var scriptElement = document.createElement('script');
        scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
        vizElement.parentNode.insertBefore(scriptElement, vizElement);
    </script>
    """
    return html(tableau_html_code, height=887, width=950, scrolling=True)
tableau_component()
def tableau_component2():
    # Tableau HTML code
    tableau_html_code = """
    <div class='tableauPlaceholder' id='viz1705573145388' style='position: relative'><noscript><a href='#'><img alt='Dashboard 2 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;TT&#47;TTCDelayDash2&#47;Dashboard2&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='TTCDelayDash2&#47;Dashboard2' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;TT&#47;TTCDelayDash2&#47;Dashboard2&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>
    <script type='text/javascript'>
        var divElement = document.getElementById('viz1705573145388');
        var vizElement = divElement.getElementsByTagName('object')[0];
        if (divElement.offsetWidth > 800) {
            vizElement.style.minWidth='520px';
            vizElement.style.maxWidth='920px';
            vizElement.style.width='100%';
            vizElement.style.minHeight='587px';
            vizElement.style.maxHeight='887px';
            vizElement.style.height=(divElement.offsetWidth*0.75)+'px';
        } else if (divElement.offsetWidth > 500) {
            vizElement.style.minWidth='520px';
            vizElement.style.maxWidth='920px';
            vizElement.style.width='100%';
            vizElement.style.minHeight='587px';
            vizElement.style.maxHeight='887px';
            vizElement.style.height=(divElement.offsetWidth*0.75)+'px';
        } else {
            vizElement.style.width='100%';
            vizElement.style.height='777px';
        }
        var scriptElement = document.createElement('script');
        scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
        vizElement.parentNode.insertBefore(scriptElement, vizElement);
    </script>
    """
    return html(tableau_html_code, height=887, width=950, scrolling=True)
tableau_component2()

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
fig_forecast.update_layout(title_text='Delay Forecast')

# Display the forecast plot in Streamlit
st.plotly_chart(fig_forecast)

# Plot components
st.subheader("Forecast Components")
fig_components = m.plot_components(forecast, figsize=(8, 8))
st.pyplot(fig_components)