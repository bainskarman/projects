import streamlit as st
import pickle
import pandas as pd

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

cities = ['Ahmedabad',
 'Kolkata',
 'Mumbai',
 'Navi Mumbai',
 'Pune',
 'Dubai',
 'Sharjah',
 'Abu Dhabi',
 'Delhi',
 'Chennai',
 'Hyderabad',
 'Visakhapatnam',
 'Chandigarh',
 'Bengaluru',
 'Jaipur',
 'Indore',
 'Bangalore',
 'Raipur',
 'Ranchi',
 'Cuttack',
 'Dharamsala',
 'Nagpur',
 'Johannesburg',
 'Centurion',
 'Durban',
 'Bloemfontein',
 'Port Elizabeth',
 'Kimberley',
 'East London',
 'Cape Town']

with open('pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)
st.title('IPL Win Predictor')

col1, col2 = st.beta_columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(teams))

selected_city = st.selectbox('Select host city',sorted(cities))

target = st.number_input('Target')

col3,col4,col5 = st.beta_columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'BattingTeam':[batting_team],'Team2':[bowling_team],'City':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets_left':[wickets],'total_runs':[target],'crr':[crr],'rrr':[rrr]})

    result = pipeline.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")