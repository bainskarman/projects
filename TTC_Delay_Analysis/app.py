# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    layout="wide",
    page_title="TTC Delays Dashboard",
    page_icon="🚌",
    initial_sidebar_state="expanded",
)

# ==================== DARK MODE CSS ====================
st.markdown(
    """
<style>
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
    .js-plotly-plot, .plotly, .modebar {
        background: #2d2d2d !important;
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
    h2, h3, h4 { color: #ffffff !important; }
    .main-header h1 { font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; color: #ffffff !important; }
    .main-header p { font-size: 1.1rem; opacity: 0.9; color: rgba(255, 255, 255, 0.9) !important; }
    @media (max-width: 768px) {
        .main-header h1 { font-size: 1.8rem; }
        .metric-value { font-size: 1.5rem; }
        .section-header { font-size: 1.3rem; }
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==================== HELPERS ====================
MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

def get_day_part_from_hour(hour: int) -> str:
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 22:
        return "Evening"
    else:
        return "Night"

def _ensure_month_category(df: pd.DataFrame) -> pd.DataFrame:
    if "Month" in df.columns:
        df["Month"] = pd.Categorical(df["Month"], categories=MONTH_ORDER, ordered=True)
    return df

# ==================== DATA LOADING ====================
@st.cache_data(show_spinner=True)
def make_sample_data(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic sample dataset if data.csv is missing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=365, freq="D")
    route_names = [
        "Eglinton", "Finch", "Lawrence", "Queen", "King",
        "Bloor", "Danforth", "Sheppard", "Bathurst", "Dufferin",
        "Spadina", "Keele", "Steeles", "York Mills", "Bayview"
    ]
    # Randomly build rows
    ds = rng.choice(dates, size=n_rows, replace=True)
    hours = rng.integers(0, 24, size=n_rows)
    mins = np.clip(
        rng.normal(loc=8, scale=6, size=n_rows) + (hours >= 17) * 4 + (hours <= 5) * 3,
        a_min=0, a_max=None
    )
    routes = rng.choice(route_names, size=n_rows, replace=True, p=None)

    df = pd.DataFrame({
        "Date": pd.to_datetime(ds),
        "Year": pd.to_datetime(ds).year,
        "Month": pd.to_datetime(ds).month,
        "Hour": hours,
        "Min Delay": mins.round(1),
        "Route Name": routes,
        "Day": pd.to_datetime(ds).day_name(),
    })
    # Convert numeric month to name and categorical order
    df["Month"] = df["Date"].dt.month_name()
    df["day_part"] = df["Hour"].map(get_day_part_from_hour)
    df = _ensure_month_category(df)
    return df

@st.cache_data(show_spinner=True)
def load_data() -> tuple[pd.DataFrame, bool]:
    """Load TTC delay data; return (df, used_sample_flag)."""
    possible_paths = [
        "TTC_Delay_Analysis/data.csv",
        "data.csv",
        "./data.csv",
        os.path.join(os.getcwd(), "TTC_Delay_Analysis", "data.csv"),
    ]
    df = None
    for path in possible_paths:
        try:
            df_try = pd.read_csv(path)
            df = df_try
            break
        except Exception:
            continue

    if df is None:
        # Fallback to sample dataset for a runnable app
        return make_sample_data(), True

    # Preprocess
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # If Year/Month/Hour missing, derive where possible
    if "Year" not in df.columns and "Date" in df.columns:
        df["Year"] = df["Date"].dt.year

    if "Month" not in df.columns and "Date" in df.columns:
        df["Month"] = df["Date"].dt.month_name()

    if "Hour" not in df.columns and "Date" in df.columns:
        df["Hour"] = df["Date"].dt.hour

    if "day_part" not in df.columns and "Hour" in df.columns:
        df["day_part"] = df["Hour"].map(get_day_part_from_hour)

    df = _ensure_month_category(df)

    return df, False

df, used_sample = load_data()

# ==================== CACHED AGGREGATIONS ====================
@st.cache_data(show_spinner=False)
def time_of_day_monthly_avg(df_small: pd.DataFrame) -> pd.DataFrame:
    return (
        df_small.groupby(["Month", "day_part"], observed=True)["Min Delay"]
        .mean().reset_index()
    )

@st.cache_data(show_spinner=False)
def hourly_counts_and_median(df_small: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = df_small.groupby("Hour").size().reset_index(name="count")
    med = df_small.groupby("Hour")["Min Delay"].median().reset_index()
    return counts, med

@st.cache_data(show_spinner=False)
def monthly_trend(df_small: pd.DataFrame) -> pd.DataFrame:
    return (
        df_small.groupby(["Year", "Month"], observed=True)["Min Delay"]
        .mean().reset_index()
    )

@st.cache_data(show_spinner=False)
def top_routes_by_count(df_small: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    vc = df_small["Route Name"].value_counts().head(top_n).reset_index()
    vc.columns = ["Route Name", "Delay Count"]
    return vc

@st.cache_data(show_spinner=False)
def avg_delay_for_routes(df_small: pd.DataFrame, routes: list[str]) -> pd.DataFrame:
    return (
        df_small[df_small["Route Name"].isin(routes)]
        .groupby("Route Name")["Min Delay"].mean()
        .sort_values(ascending=False)
        .reset_index()
    )

@st.cache_data(show_spinner=False)
def monthly_route_avg(df_small: pd.DataFrame, routes: list[str]) -> pd.DataFrame:
    return (
        df_small[df_small["Route Name"].isin(routes)]
        .groupby(["Month", "Route Name"], observed=True)["Min Delay"]
        .mean().reset_index()
    )

@st.cache_data(show_spinner=False)
def top_routes_stats(df_small: pd.DataFrame, routes: list[str]) -> pd.DataFrame:
    g = (
        df_small[df_small["Route Name"].isin(routes)]
        .groupby("Route Name")["Min Delay"]
        .agg(["mean", "median", "std", "count"])
        .round(2)
        .rename(columns={"mean": "Avg Delay", "median": "Median Delay", "std": "Std Deviation", "count": "Incident Count"})
        .sort_values("Incident Count", ascending=False)
    )
    return g

@st.cache_data(show_spinner=False)
def peak_hours_data(df_small: pd.DataFrame, routes: list[str]) -> pd.DataFrame:
    rows = []
    for r in routes:
        d = df_small[df_small["Route Name"] == r]
        if len(d) == 0:
            continue
        peak_hour = d["Hour"].mode().iloc[0] if not d["Hour"].mode().empty else None
        rows.append({
            "Route": r,
            "Peak Hour": f"{int(peak_hour)}:00" if peak_hour is not None else "N/A",
            "Avg Delay": d["Min Delay"].mean(),
        })
    return pd.DataFrame(rows)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown(
        """
<div style="text-align: left; margin-bottom: 1rem;">
  <h2 style="color: #CC0000; font-size: 1.3rem;">🚌 TTC Dashboard</h2>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("### 📋 Navigation")
    sections = ["Overview", "Time Analysis", "Route Analysis", "Forecasting", "Tableau Dash", "Power BI Dash"]
    if "active_section" not in st.session_state:
        st.session_state.active_section = "Overview"

    st.session_state.active_section = st.radio(
        "Go to",
        sections,
        index=sections.index(st.session_state.active_section),
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 🔍 Filters")

    # Year filter
    selected_years = None
    if "Year" in df.columns:
        years = sorted([int(y) for y in df["Year"].dropna().unique()])
        selected_years = st.multiselect(
            "📅 Years",
            options=years,
            default=years[-1:] if len(years) >= 1 else years,
            key="year_filter",
        )

    # Month filter
    selected_months = None
    if "Month" in df.columns:
        months = MONTH_ORDER
        selected_months = st.multiselect(
            "📆 Months",
            options=months,
            default=months,
            key="month_filter",
        )

    # Time of day filter
    selected_day_parts = None
    if "day_part" in df.columns:
        day_parts = ["Morning", "Afternoon", "Evening", "Night"]
        selected_day_parts = st.multiselect(
            "⏰ Time of Day",
            options=day_parts,
            default=day_parts,
            key="daypart_filter",
        )

    # Route filter
    selected_routes = None
    if "Route Name" in df.columns:
        top_routes_sidebar = df["Route Name"].value_counts().head(20).index.tolist()
        selected_routes = st.multiselect(
            "🛣️ Routes (Top 20)",
            options=top_routes_sidebar,
            default=top_routes_sidebar[:5] if len(top_routes_sidebar) >= 5 else top_routes_sidebar,
            key="route_filter",
        )

    st.markdown("---")
    st.markdown("### ⚡ Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        if "Min Delay" in df.columns:
            st.metric("Avg Delay", f"{df['Min Delay'].mean():.1f}m")
    with col2:
        st.metric("Incidents", f"{len(df):,}")

# ==================== FILTERING ====================
df_filtered = df.copy()
if selected_years:
    df_filtered = df_filtered[df_filtered["Year"].isin(selected_years)]
if selected_months:
    df_filtered = df_filtered[df_filtered["Month"].isin(selected_months)]
if selected_day_parts:
    df_filtered = df_filtered[df_filtered["day_part"].isin(selected_day_parts)]
if selected_routes and st.session_state.active_section == "Route Analysis":
    df_filtered = df_filtered[df_filtered["Route Name"].isin(selected_routes)]

# ==================== HEADER ====================
st.markdown(
    """
<div class="main-header">
  <h1>🚌 TTC Transit Delays Analysis Dashboard</h1>
  <p>Comprehensive analysis of Toronto Transit Commission delays with interactive visualizations and forecasting</p>
</div>
""",
    unsafe_allow_html=True,
)

if used_sample:
    st.markdown(
        """
<div class="info-box">
  <b>Sample data mode:</b> <br/>
  No <code>data.csv</code> found. The charts use a generated sample dataset for demonstration. 
  Place your real <code>data.csv</code> next to <code>app.py</code> (or in <code>TTC_Delay_Analysis/data.csv</code>) and refresh.
</div>
""",
        unsafe_allow_html=True,
    )

# ==================== SECTIONS ====================
section = st.session_state.active_section

# -------- OVERVIEW --------
if section == "Overview":
    st.markdown('<h2 class="section-header">📊 Executive Dashboard</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_delay = df_filtered["Min Delay"].mean() if "Min Delay" in df_filtered.columns else 0
        st.markdown(f'<div class="metric-value">{avg_delay:.1f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Avg Delay (min)</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_incidents = len(df_filtered)
        st.markdown(f'<div class="metric-value">{total_incidents:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Incidents</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if "Min Delay" in df_filtered.columns and len(df_filtered) > 0:
            max_delay = df_filtered["Min Delay"].max()
            st.markdown(f'<div class="metric-value">{max_delay:.0f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Max Delay (min)</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if "Route Name" in df_filtered.columns:
            unique_routes = df_filtered["Route Name"].nunique()
            st.markdown(f'<div class="metric-value">{unique_routes}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Routes Affected</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if "Min Delay" in df_filtered.columns:
            median_delay = df_filtered["Min Delay"].median()
            st.markdown(f'<div class="metric-value">{median_delay:.1f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Median Delay</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if "Route Name" in df_filtered.columns and len(df_filtered) > 0:
            top_route = df_filtered["Route Name"].mode()[0]
            st.markdown(f'<div class="metric-value" style="font-size: 1.3rem;">{top_route}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Most Delayed Route</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if "Hour" in df_filtered.columns and len(df_filtered) > 0:
            peak_hour = df_filtered["Hour"].mode()[0]
            st.markdown(f'<div class="metric-value">{peak_hour}:00</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Peak Delay Hour</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if "Day" in df_filtered.columns and "Min Delay" in df_filtered.columns:
            weekday_delays = df_filtered[df_filtered["Day"].isin(
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            )]["Min Delay"].mean()
            st.markdown(f'<div class="metric-value">{weekday_delays:.1f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Avg Weekday Delay</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 📋 Data Overview")
        st.dataframe(df_filtered.head(10), height=300)
        st.caption(f"Showing 10 of {len(df_filtered):,} records")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 📊 Data Summary")
        if "Min Delay" in df_filtered.columns:
            delay_stats = df_filtered["Min Delay"].describe()
            stats_df = pd.DataFrame({
                "Statistic": ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"],
                "Value": [
                    f"{delay_stats['count']:.0f}",
                    f"{delay_stats['mean']:.1f}",
                    f"{delay_stats['std']:.1f}",
                    f"{delay_stats['min']:.1f}",
                    f"{delay_stats['25%']:.1f}",
                    f"{delay_stats['50%']:.1f}",
                    f"{delay_stats['75%']:.1f}",
                    f"{delay_stats['max']:.1f}",
                ],
            })
            st.dataframe(stats_df, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<h2 class="section-header">📋 Methodology & Tools</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### Data Sources")
        st.markdown(
            "- Toronto Open Data Portal\n- TTC Public Datasets\n- Historical Delay Records"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### Methodology")
        st.markdown(
            "- Data Cleaning & Merging\n- Time Series Analysis\n- Prophet Forecasting (optional)\n- Statistical Visualization"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### Tools Used")
        st.markdown(
            "- Streamlit for Dashboard\n- Pandas for Data Processing\n- Plotly for Visualization\n- Prophet for Forecasting"
        )
        st.markdown("</div>", unsafe_allow_html=True)

# -------- TIME ANALYSIS --------
elif section == "Time Analysis":
    st.markdown('<h2 class="section-header">⏰ Time-Based Analysis</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Delay Patterns by Time of Day")
        needed = {"day_part", "Month", "Min Delay"}
        if needed.issubset(df_filtered.columns):
            data = time_of_day_monthly_avg(df_filtered[list(needed)].copy())
            fig = px.line(
                data,
                x="Month",
                y="Min Delay",
                color="day_part",
                markers=True,
                line_shape="linear",
                color_discrete_sequence=["#CC0000", "#FF6B6B", "#4ECDC4", "#45B7D1"],
            )
            fig.update_layout(
                height=400,
                plot_bgcolor="#2d2d2d",
                paper_bgcolor="#2d2d2d",
                font_color="white",
                showlegend=True,
                legend=dict(title="Time of Day", yanchor="top", y=0.99, xanchor="left", x=1.02),
                margin=dict(r=100),
                xaxis_title="Month",
                yaxis_title="Average Delay (minutes)",
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<div class="info-box">Night-time delays often trend higher; afternoon typically lower.</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Hourly Incident Analysis")
        needed = {"Hour", "Min Delay"}
        if needed.issubset(df_filtered.columns):
            hour_counts, median_delays = hourly_counts_and_median(df_filtered[list(needed)].copy())
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("<b>Incident Frequency by Hour</b>", "<b>Median Delay by Hour</b>"),
                vertical_spacing=0.2,
                row_heights=[0.5, 0.5],
            )
            fig.add_trace(go.Bar(x=hour_counts["Hour"], y=hour_counts["count"], marker_color="#CC0000", opacity=0.7), row=1, col=1)
            fig.add_trace(go.Scatter(x=median_delays["Hour"], y=median_delays["Min Delay"], mode="lines+markers",
                                     line=dict(color="#2E86C1", width=3)), row=2, col=1)
            for r in (1, 2):
                fig.update_xaxes(title_text="Hour of Day", row=r, col=1, tickmode="linear", dtick=2, range=[-0.5, 23.5])
            fig.update_yaxes(title_text="Number of Incidents", row=1, col=1)
            fig.update_yaxes(title_text="Median Delay (minutes)", row=2, col=1)
            fig.update_layout(height=500, showlegend=False, plot_bgcolor="#2d2d2d", paper_bgcolor="#2d2d2d",
                              font_color="white", margin=dict(t=50, b=50))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<div class="info-box">Evening period (approx. 3–8 PM) tends to see more incidents; late-night shows longer delays.</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Monthly Delay Trends")
        needed = {"Year", "Month", "Min Delay"}
        if needed.issubset(df_filtered.columns):
            m_trend = monthly_trend(df_filtered[list(needed)].copy())
            fig = px.line(m_trend, x="Month", y="Min Delay", color="Year", markers=True, line_shape="linear")
            fig.update_layout(
                height=400,
                plot_bgcolor="#2d2d2d",
                paper_bgcolor="#2d2d2d",
                font_color="white",
                xaxis_title="Month",
                yaxis_title="Average Delay (minutes)",
                legend_title="Year",
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Peak Hours Analysis")
        if "Hour" in df_filtered.columns:
            hourly_counts = df_filtered["Hour"].value_counts().head(5).reset_index()
            hourly_counts.columns = ["Hour", "Count"]
            hourly_counts = hourly_counts.sort_values("Hour")
            fig = px.bar(hourly_counts, x="Hour", y="Count", color="Count", color_continuous_scale="Reds", text="Count")
            fig.update_layout(
                height=350,
                plot_bgcolor="#2d2d2d",
                paper_bgcolor="#2d2d2d",
                font_color="white",
                xaxis_title="Hour",
                yaxis_title="Incident Count",
                showlegend=False,
            )
            fig.update_traces(texttemplate="%{text}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
            peak_hours = ", ".join([f"{int(h)}:00" for h in hourly_counts["Hour"].tolist()])
            st.markdown(f"**Top Peak Hours:** {peak_hours}")
        st.markdown("</div>", unsafe_allow_html=True)

# -------- ROUTE ANALYSIS --------
elif section == "Route Analysis":
    st.markdown('<h2 class="section-header">🛣️ Route Performance Analysis</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Top 10 Routes by Delay Frequency")
        if "Route Name" in df_filtered.columns:
            route_counts = top_routes_by_count(df_filtered[["Route Name"]].copy(), top_n=10)
            fig = px.bar(
                route_counts, x="Delay Count", y="Route Name", orientation="h",
                color="Delay Count", color_continuous_scale="Reds", text="Delay Count"
            )
            fig.update_layout(
                height=500,
                plot_bgcolor="#2d2d2d",
                paper_bgcolor="#2d2d2d",
                font_color="white",
                xaxis_title="Number of Delays",
                yaxis_title="Route Name",
                yaxis={"categoryorder": "total ascending"},
                showlegend=False,
            )
            fig.update_traces(texttemplate="%{text}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<div class="info-box">High-frequency delays often occur on busy corridors and long routes.</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Top 10 Routes by Delay Severity")
        if {"Route Name", "Min Delay"}.issubset(df_filtered.columns):
            top10 = df_filtered["Route Name"].value_counts().head(10).index.tolist()
            avg_delay_per_route = avg_delay_for_routes(df_filtered[["Route Name", "Min Delay"]].copy(), top10)
            fig = px.bar(
                avg_delay_per_route.head(10), x="Route Name", y="Min Delay",
                color="Min Delay", color_continuous_scale="Viridis", text_auto=".1f"
            )
            fig.update_layout(
                height=500,
                plot_bgcolor="#2d2d2d",
                paper_bgcolor="#2d2d2d",
                font_color="white",
                xaxis_title="Route Name",
                yaxis_title="Average Delay (minutes)",
                xaxis_tickangle=45,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<div class="info-box">Average severity can correlate with route length, traffic, and transfer points.</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("#### Top 10 Routes Performance Over Time")
    if {"Route Name", "Month", "Min Delay"}.issubset(df_filtered.columns):
        top10 = df_filtered["Route Name"].value_counts().head(10).index.tolist()
        monthly_route = monthly_route_avg(df_filtered[["Route Name", "Month", "Min Delay"]].copy(), top10)
        fig = px.line(monthly_route, x="Month", y="Min Delay", color="Route Name", markers=True, line_shape="linear")
        fig.update_layout(
            height=450,
            plot_bgcolor="#2d2d2d",
            paper_bgcolor="#2d2d2d",
            font_color="white",
            xaxis_title="Month",
            yaxis_title="Average Delay (minutes)",
            xaxis_tickangle=45,
            legend_title="Route Name",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, bgcolor="rgba(45,45,45,0.8)"),
            margin=dict(r=150),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="info-box">Seasonal trends and operational changes can drive monthly variations.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Route Delay Statistics (Top 10)")
        if {"Route Name", "Min Delay"}.issubset(df_filtered.columns):
            top10 = df_filtered["Route Name"].value_counts().head(10).index.tolist()
            stats_df = top_routes_stats(df_filtered[["Route Name", "Min Delay"]].copy(), top10)
            st.dataframe(
                stats_df,
                column_config={
                    "Avg Delay": st.column_config.NumberColumn(format="%.1f min"),
                    "Median Delay": st.column_config.NumberColumn(format="%.1f min"),
                    "Std Deviation": st.column_config.NumberColumn(format="%.1f"),
                    "Incident Count": st.column_config.NumberColumn(format="%d"),
                },
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### Top 10 Routes Peak Hours")
        if {"Route Name", "Hour", "Min Delay"}.issubset(df_filtered.columns):
            top10 = df_filtered["Route Name"].value_counts().head(10).index.tolist()
            peak_df = peak_hours_data(df_filtered[["Route Name", "Hour", "Min Delay"]].copy(), top10)
            if not peak_df.empty:
                fig = px.bar(peak_df, x="Route", y="Avg Delay", color="Peak Hour", text="Peak Hour",
                             title="Peak Delay Hours by Route")
                fig.update_layout(
                    height=400,
                    plot_bgcolor="#2d2d2d",
                    paper_bgcolor="#2d2d2d",
                    font_color="white",
                    xaxis_title="Route",
                    yaxis_title="Average Delay (minutes)",
                    xaxis_tickangle=45,
                    showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -------- FORECASTING --------
elif section == "Forecasting":
    st.markdown('<h2 class="section-header">🔮 Delay Forecasting with Prophet</h2>', unsafe_allow_html=True)

    if {"Date", "Min Delay"}.issubset(df_filtered.columns):
        prophet_df = (
            df_filtered[["Date", "Min Delay"]].rename(columns={"Date": "ds", "Min Delay": "y"}).copy()
        )
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        prophet_df = prophet_df.groupby("ds", as_index=False)["y"].mean()

        col_config, col_plot = st.columns([1, 2])
        with col_config:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### Forecast Configuration")

            forecast_period = st.slider("Forecast Period (days)", 30, 365, 180, 30, key="forecast_days")
            confidence_level = st.slider("Confidence Interval", 0.80, 0.99, 0.95, 0.01, key="confidence")
            seasonality_mode = st.selectbox("Seasonality Mode", ["multiplicative", "additive"], index=0, key="seasonality")

            @st.cache_data(show_spinner=True)
            def _run_prophet_forecast(history_df: pd.DataFrame, period: int, conf: float, seas: str) -> pd.DataFrame:
                # Lazy import Prophet
                from prophet import Prophet
                m = Prophet(
                    seasonality_mode=seas,
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    interval_width=conf,
                )
                m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
                m.fit(history_df)
                future = m.make_future_dataframe(periods=period)
                forecast = m.predict(future)
                return forecast

            if st.button("Generate Forecast", type="primary", use_container_width=True):
                with st.spinner("Training Prophet model..."):
                    try:
                        forecast = _run_prophet_forecast(prophet_df, forecast_period, confidence_level, seasonality_mode)
                        st.session_state.forecast = forecast
                        st.success("Forecast generated successfully!")
                    except Exception as e:
                        st.error(
                            "Prophet not available or failed to run. "
                            "Install it via `pip install prophet` (or use precomputed forecasts)."
                        )

            if "forecast" in st.session_state:
                st.markdown("---")
                st.markdown("#### Forecast Summary")
                last = st.session_state.forecast.tail(forecast_period)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Average", f"{last['yhat'].mean():.1f} min")
                    st.metric("Minimum", f"{last['yhat'].min():.1f} min")
                with col_b:
                    st.metric("Maximum", f"{last['yhat'].max():.1f} min")
                    st.metric("Horizon", f"{forecast_period} days")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_plot:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### Delay Forecast Visualization")
            if "forecast" in st.session_state:
                f = st.session_state.forecast
                fig = go.Figure()
                # Historical
                fig.add_trace(go.Scatter(
                    x=prophet_df["ds"], y=prophet_df["y"], mode="markers", name="Historical",
                    marker=dict(color="#2E86C1", size=6, opacity=0.6)
                ))
                # Forecast
                fig.add_trace(go.Scatter(
                    x=f["ds"], y=f["yhat"], mode="lines", name="Forecast", line=dict(color="#CC0000", width=3)
                ))
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=f["ds"].tolist() + f["ds"].tolist()[::-1],
                    y=f["yhat_upper"].tolist() + f["yhat_lower"].tolist()[::-1],
                    fill="toself", fillcolor="rgba(204,0,0,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name=f"{int(confidence_level*100)}% Confidence Interval",
                    showlegend=True,
                ))
                fig.update_layout(
                    height=500,
                    plot_bgcolor="#2d2d2d",
                    paper_bgcolor="#2d2d2d",
                    font_color="white",
                    xaxis_title="Date",
                    yaxis_title="Delay (minutes)",
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(45,45,45,0.8)"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Configure your forecast settings and click 'Generate Forecast' to see predictions.")
                # Optional: recent history plot
                sample_hist = prophet_df.tail(120)
                fig = px.line(sample_hist, x="ds", y="y", title="Recent Historical Delay Data")
                fig.update_layout(height=450, plot_bgcolor="#2d2d2d", paper_bgcolor="#2d2d2d", font_color="white")
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Components (optional, requires Prophet model decomposition in forecast frame)
        if "forecast" in st.session_state and {"trend", "weekly", "yearly"}.issubset(st.session_state.forecast.columns):
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### Forecast Components Analysis")
            f = st.session_state.forecast
            figc = make_subplots(
                rows=3, cols=1,
                subplot_titles=("<b>Trend Component</b>", "<b>Weekly Seasonality</b>", "<b>Yearly Seasonality</b>"),
                vertical_spacing=0.17, row_heights=[0.4, 0.3, 0.3]
            )
            figc.add_trace(go.Scatter(x=f["ds"], y=f["trend"], mode="lines", line=dict(color="#CC0000", width=2)), row=1, col=1)
            if "weekly" in f:
                w = f[["ds", "weekly"]].tail(7 * 4)
                figc.add_trace(go.Scatter(x=w["ds"], y=w["weekly"], mode="lines", line=dict(color="#2E86C1", width=2)), row=2, col=1)
            if "yearly" in f:
                y = f[["ds", "yearly"]].tail(365)
                figc.add_trace(go.Scatter(x=y["ds"], y=y["yearly"], mode="lines", line=dict(color="#4CAF50", width=2)), row=3, col=1)
            figc.update_layout(height=600, plot_bgcolor="#2d2d2d", paper_bgcolor="#2d2d2d", font_color="white", showlegend=False)
            figc.update_xaxes(title_text="Date", row=1, col=1)
            figc.update_xaxes(title_text="Date", row=2, col=1)
            figc.update_xaxes(title_text="Date", row=3, col=1)
            figc.update_yaxes(title_text="Trend", row=1, col=1)
            figc.update_yaxes(title_text="Weekly Effect", row=2, col=1)
            figc.update_yaxes(title_text="Yearly Effect", row=3, col=1)
            st.plotly_chart(figc, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("`Date` or `Min Delay` columns not found in data. Forecasting requires temporal data.")

# -------- TABLEAU --------
elif section == "Tableau Dash":
    st.markdown('<h2 class="section-header">📈 Tableau Analytics Dashboard</h2>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Interactive Tableau dashboard showing comprehensive analysis of TTC delays.</div>',
        unsafe_allow_html=True,
    )

    from streamlit.components.v1 import html
    tableau_html_code = """
    <div class='tableauPlaceholder' id='viz1705632432757' style='position: relative; width: 100%;'>
      <noscript>
        <a href='#'>
          <img alt='TTC Bus Delay' src='https://public.tableau.com/static/images/TT/TTCDelayDash/Dashboard1/1.png' style='border: none; width: 100%;' />
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
      function setTableauDimensions() {
        var dashboardHeight = 900;
        vizElement.style.width = '100%';
        vizElement.style.height = dashboardHeight + 'px';
      }
      setTableauDimensions();
      window.addEventListener('resize', setTableauDimensions);
      var scriptElement = document.createElement('script');
      scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
      vizElement.parentNode.insertBefore(scriptElement, vizElement);
    </script>
    """
    html(tableau_html_code, height=950, scrolling=False)

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 📊 Key Insights from Tableau Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
**Temporal Patterns:**
- Higher delays during late night on weekdays
- Peak delays vary by weekend mornings
- Evening rush hour (5–7 PM) shows consistent patterns

**Incident Analysis:**
- Diversions often cause long delays
- Mechanical issues frequent
- Investigations contribute notably
"""
        )
    with c2:
        st.markdown(
            """
**Operational Insights:**
- Downtown core shows highest frequency
- Suburban terminals see fewer but longer delays
- Transfer stations are hotspots

**Comparative Analysis:**
- Year-over-year and seasonal differences are visible
- Improvements can appear after schedule changes
- Weekday vs weekend patterns differ materially
"""
        )
    st.markdown("</div>", unsafe_allow_html=True)

# -------- POWER BI --------
elif section == "Power BI Dash":
    st.markdown('<h2 class="section-header">📊 Power BI Analytics Dashboard</h2>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Power BI dashboard providing analytics and geographical insights into TTC delay patterns.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 🔗 Access Power BI Dashboard")
    st.markdown(
        """
<div style="background: linear-gradient(135deg, #333333 0%, #2a2a2a 100%);
            border-left: 4px solid #0078d4;
            padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0; text-align: center;">
  <h4 style="color: #0078d4; margin-bottom: 1rem;">Interactive Power BI Dashboard</h4>
  <p style="margin-bottom: 1.5rem; color: #cccccc;">For the complete interactive experience with all visualizations and filters, click the link below:</p>
  <a href="https://app.powerbi.com/reportEmbed?reportId=3aa46b8e-572e-44fa-80cb-ba8226145eed"
     target="_blank"
     style="background: linear-gradient(90deg, #0078d4 0%, #005a9e 100%);
            color: white; padding: 0.75rem 1.5rem; border-radius: 5px; text-decoration: none;
            font-weight: 600; display: inline-block; transition: all 0.3s ease;">
     📊 Open Power BI Dashboard
  </a>
  <p style="margin-top: 1rem; font-size: 0.9rem; color: #999;">
    <i>Note: You'll need appropriate permissions to access the Power BI report</i>
  </p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Optional previews (if you have local images)
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 📷 Power BI Report Previews")
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()
    image_files = ["img2.png", "img3.png", "img4.png", "img5.png"]
    image_paths = [os.path.join(current_dir, img) for img in image_files]
    existing_paths = [p for p in image_paths if os.path.exists(p)]

    if "powerbi_page" not in st.session_state:
        st.session_state.powerbi_page = 0

    if existing_paths:
        total_images = len(existing_paths)
        cols = st.columns(2)
        for i in range(2):
            idx = st.session_state.powerbi_page * 2 + i
            if idx < total_images:
                with cols[i]:
                    st.image(existing_paths[idx], caption=f"Power BI Report Preview {idx + 1}")
        if total_images > 2:
            c1, c2, c3 = st.columns([1, 2, 1])
            with c1:
                if st.button("◀ Previous", disabled=st.session_state.powerbi_page == 0):
                    st.session_state.powerbi_page -= 1
                    st.rerun()
            with c3:
                if st.button("Next ▶", disabled=(st.session_state.powerbi_page + 1) * 2 >= total_images):
                    st.session_state.powerbi_page += 1
                    st.rerun()
            with c2:
                current_page = st.session_state.powerbi_page + 1
                total_pages = (total_images + 1) // 2
                st.markdown(f"**Page {current_page} of {total_pages}**")
    else:
        st.info("Power BI preview images not found. Place img2.png, img3.png, img4.png, img5.png alongside app.py if you want to show previews.")
    st.markdown("</div>", unsafe_allow_html=True)