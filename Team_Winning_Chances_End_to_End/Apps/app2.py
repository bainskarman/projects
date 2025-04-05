import streamlit as st
import joblib
import pandas as pd
import os
teams = ['Rajasthan Royals',
 'Royal Challengers Bangalore',
 'Sunrisers Hyderabad',
 'Delhi Capitals',
 'Chennai Super Kings',
 'Gujarat Titans',
 'Lucknow Super Giants',
 'Kolkata Knight Riders',
 'Punjab Kings',
 'Mumbai Indians']

cities = ['Bangalore',
 'Chandigarh',
 'Delhi',
 'Mumbai',
 'Kolkata',
 'Jaipur',
 'Hyderabad',
 'Chennai',
 'Cape Town',
 'Port Elizabeth',
 'Durban',
 'Centurion',
 'East London',
 'Johannesburg',
 'Kimberley',
 'Bloemfontein',
 'Ahmedabad',
 'Cuttack',
 'Nagpur',
 'Dharamsala',
 'Visakhapatnam',
 'Pune',
 'Raipur',
 'Ranchi',
 'Abu Dhabi',
 'Sharjah',
 'Dubai',
 'Bengaluru',
 'Indore',
 'Navi Mumbai',
 'Lucknow',
 'Guwahati',
 'Mohali']

file_path = os.path.join(os.path.dirname(__file__), 'pipeline.pkl')
with open(file_path, 'rb') as file:
    pipe = joblib.load(file)

# Load data (replace with actual file paths or methods for retrieving stats)
data1_path = os.path.join(os.path.dirname(__file__), 'batter_stats.csv')
data2_path = os.path.join(os.path.dirname(__file__), 'bowler_stats.csv')
data3_path = os.path.join(os.path.dirname(__file__), 'venue_stats_overall.csv')

batter_stats = pd.read_csv(data1_path)
bowler_stats = pd.read_csv(data2_path)
venue_stats = pd.read_csv(data3_path)

st.title('Winner Predictor for IPL with Enhanced Model')

# Using st.columns
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target', min_value=0)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0)
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=9)

# Selecting batters and bowlers
col6, col7 = st.columns(2)
with col6:
    batters = st.multiselect('Select batters', sorted(batter_stats['batter'].unique()), max_selections=6)
with col7:
    bowlers = st.multiselect('Select bowlers', sorted(bowler_stats['bowler'].unique()), max_selections=6)

# Extracting venue stats
try:
    venue_stat = venue_stats[venue_stats['City'] == selected_city].iloc[0]
except IndexError:
    st.error(f"No venue stats found for {selected_city}")
    venue_stat = pd.Series(0, index=venue_stats.columns)

# Compute relevant features for batting and bowling stats
def calculate_batter_bowler_stats(batters, bowlers):
    # Default values if no batters/bowlers selected
    batter_defaults = {
        'total_matches': 0, 'total_balls_faced': 0, 'boundary_percentage': 0, 
        'dot_ball_percentage': 0, 'pp_runs': 0, 'pp_wickets': 0, 'pp_sr': 0,
        'death_runs': 0, 'death_wickets': 0, 'death_sr': 0
    }
    
    bowler_defaults = {
        'total_matches': 0, 'total_balls': 0, 'total_runs_conceded': 0, 
        'total_wickets': 0, 'bowling_average': 0, 'economy_rate': 0, 
        'strike_rate': 0, 'dot_percentage': 0, 'pp_wickets': 0, 
        'pp_economy': 0, 'middle_wickets': 0, 'middle_economy': 0,
        'death_wickets': 0, 'death_economy': 0
    }
    
    # Calculate batter stats
    if batters:
        top6_batters = batter_stats[batter_stats['batter'].isin(batters)]
        top6_batter_stats = top6_batters[list(batter_defaults.keys())].mean()
    else:
        top6_batter_stats = pd.Series(batter_defaults)
    
    # Calculate bowler stats
    if bowlers:
        top6_bowlers = bowler_stats[bowler_stats['bowler'].isin(bowlers)]
        top6_bowler_stats = top6_bowlers[list(bowler_defaults.keys())].mean()
    else:
        top6_bowler_stats = pd.Series(bowler_defaults)
    
    return top6_batter_stats, top6_bowler_stats

# Prepare input DataFrame for prediction
runs_left = max(target - score, 0)
balls_left = max(120 - (overs * 6), 0)
wickets_left = 10 - wickets
crr = score / overs if overs != 0 else 0
rrr = (runs_left * 6) / balls_left if balls_left != 0 else 0

# Get batter and bowler stats
top6_batter_stats, top6_bowler_stats = calculate_batter_bowler_stats(batters, bowlers)

# Prepare input features
input_df = pd.DataFrame({
    'BattingTeam': [batting_team], 
    'Team2': [bowling_team], 
    'City': [selected_city],
    'runs_left': [runs_left], 
    'balls_left': [balls_left], 
    'wickets_left': [wickets_left],
    'crr': [crr], 
    'rrr': [rrr],
    
    # Batter stats
    'top6_avg_matches_x': [top6_batter_stats['total_matches']],
    'top6_avg_balls_faced': [top6_batter_stats['total_balls_faced']],
    'top6_avg_boundary_pct': [top6_batter_stats['boundary_percentage']],
    'top6_avg_dot_pct': [top6_batter_stats['dot_ball_percentage']],
    'top6_avg_pp_runs': [top6_batter_stats['pp_runs']],
    'top6_avg_pp_wickets': [top6_batter_stats['pp_wickets']],
    'top6_avg_pp_sr': [top6_batter_stats['pp_sr']],
    'top6_avg_death_runs': [top6_batter_stats['death_runs']],
    'top6_avg_death_wickets': [top6_batter_stats['death_wickets']],
    'top6_avg_death_sr': [top6_batter_stats['death_sr']],
    
    # Bowler stats
    'top6_avg_balls_bowled': [top6_bowler_stats['total_balls']],
    'top6_avg_runs_conceded': [top6_bowler_stats['total_runs_conceded']],
    'top6_avg_wickets': [top6_bowler_stats['total_wickets']],
    'top6_avg_bowling_avg': [top6_bowler_stats['bowling_average']],
    'top6_avg_economy': [top6_bowler_stats['economy_rate']],
    'top6_avg_strike_rate': [top6_bowler_stats['strike_rate']],
    'top6_avg_dot_pct': [top6_bowler_stats['dot_percentage']],
    'top6_avg_pp_wkts': [top6_bowler_stats['pp_wickets']],
    'top6_avg_pp_eco': [top6_bowler_stats['pp_economy']],
    'top6_avg_middle_wkts': [top6_bowler_stats['middle_wickets']],
    'top6_avg_middle_eco': [top6_bowler_stats['middle_economy']],
    'top6_avg_death_wkts': [top6_bowler_stats['death_wickets']],
    'top6_avg_death_eco': [top6_bowler_stats['death_economy']],
    
    # Venue stats
    'avg_runs': [venue_stat.get('avg_runs', 0)],
    'avg_wickets': [venue_stat.get('avg_wickets', 0)],
    'avg_balls': [venue_stat.get('avg_balls', 0)],
    'avg_overs': [venue_stat.get('avg_overs', 0)],
    'avg_run_rate': [venue_stat.get('avg_run_rate', 0)],
    'avg_pp_runs': [venue_stat.get('avg_pp_runs', 0)],
    'avg_pp_wickets': [venue_stat.get('avg_pp_wickets', 0)],
    'avg_pp_run_rate': [venue_stat.get('avg_pp_run_rate', 0)],
    'innings_count': [venue_stat.get('innings_count', 0)],
})

# Make prediction using the model
if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win * 100)) + "%")
    st.header(bowling_team + "- " + str(round(loss * 100)) + "%")