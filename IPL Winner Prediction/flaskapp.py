from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

teams = ['Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad', 'Delhi Capitals',
         'Chennai Super Kings', 'Gujarat Titans', 'Lucknow Super Giants', 'Kolkata Knight Riders',
         'Punjab Kings', 'Mumbai Indians']

cities = ['Ahmedabad', 'Kolkata', 'Mumbai', 'Navi Mumbai', 'Pune', 'Dubai', 'Sharjah', 'Abu Dhabi', 'Delhi',
          'Chennai', 'Hyderabad', 'Visakhapatnam', 'Chandigarh', 'Bengaluru', 'Jaipur', 'Indore', 'Bangalore',
          'Raipur', 'Ranchi', 'Cuttack', 'Dharamsala', 'Nagpur', 'Johannesburg', 'Centurion', 'Durban', 'Bloemfontein',
          'Port Elizabeth', 'Kimberley', 'East London', 'Cape Town']

file_path = 'my_pipeline.joblib'
pipe = joblib.load(file_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/method1', methods=['GET', 'POST'])
def method1():
    if request.method == 'POST':
        batting_team = request.form['batting_team']
        bowling_team = request.form['bowling_team']
        selected_city = request.form['selected_city']

        target = int(request.form['target'])
        score = int(request.form['score'])
        overs = int(request.form['overs'])
        wickets = int(request.form['wickets'])

        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets = 10 - wickets
        crr = score / overs
        rrr = (runs_left * 6) / balls_left

        input_df = pd.DataFrame({'BattingTeam': [batting_team], 'Team2': [bowling_team], 'City': [selected_city],
                                 'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets_left': [wickets],
                                 'team1_runs': [target], 'crr': [crr], 'rrr': [rrr]})

        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        return render_template('result.html', batting_team=batting_team, bowling_team=bowling_team,
                               win_percentage=round(win * 100), loss_percentage=round(loss * 100))

    return render_template('method1.html', teams=teams, cities=cities)

# Similar routes for other methods...

if __name__ == '__main__':
    app.run(debug=True)
