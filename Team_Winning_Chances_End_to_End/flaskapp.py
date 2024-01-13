from flask import Flask, render_template, request
import pickle
import pandas as pd
import base64

app = Flask(__name__)

teams = ['Rajasthan Royals', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad', 'Delhi Capitals', 'Chennai Super Kings', 'Gujarat Titans', 'Lucknow Super Giants', 'Kolkata Knight Riders', 'Punjab Kings', 'Mumbai Indians']

cities = ['Ahmedabad', 'Kolkata', 'Mumbai', 'Navi Mumbai', 'Pune', 'Dubai', 'Sharjah', 'Abu Dhabi', 'Delhi', 'Chennai', 'Hyderabad', 'Visakhapatnam', 'Chandigarh', 'Bengaluru', 'Jaipur', 'Indore', 'Bangalore', 'Raipur', 'Ranchi', 'Cuttack', 'Dharamsala', 'Nagpur', 'Johannesburg', 'Centurion', 'Durban', 'Bloemfontein', 'Port Elizabeth', 'Kimberley', 'East London', 'Cape Town']

file_path = 'C:/Users/bains/Downloads/GitHub24/projects/Team_Winning_Chances_End_to_End/pipeline.pkl'
with open(file_path, 'rb') as file:
    pipe = pickle.load(file)

# Encode the image to base64
with open('C:/Users/bains/Downloads/GitHub24/projects/Team_Winning_Chances_End_to_End/cricket-betting.jpg', 'rb') as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

def get_index_template():
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{
                background-image: url('data:image/jpeg;base64,{encoded_image}');
                background-size: cover;
                background-repeat: no-repeat;
                color: white;
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 0;
            }}
            .container {{
                max-width: 800px;
                margin: auto;
                margin-top: 50px;
                padding: 20px;
                background-color: rgba(0, 0, 0, 0.7);
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
            }}
            h1 {{
                text-align: center;
                margin-bottom: 20px;
            }}
            form {{
                display: flex;
                flex-direction: column;
            }}
            label, input, select, button {{
                margin: 10px 0;
                padding: 8px;
                border: none;
                border-radius: 5px;
            }}
            button {{
                background-color: #4CAF50;
                color: white;
                cursor: pointer;
            }}
        </style>
        <title>Winner Predictor for IPL</title>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to IPL Winner Predictor</h1>
            <form action="/predict" method="post">
                <label for="batting_team">Select the batting team:</label>
                <select name="batting_team">
                    {"".join([f"<option value='{team}'>{team}</option>" for team in teams])}
                </select><br>
                <label for="bowling_team">Select the bowling team:</label>
                <select name="bowling_team">
                    {"".join([f"<option value='{team}'>{team}</option>" for team in teams])}
                </select><br>
                <label for="selected_city">Select host city:</label>
                <select name="selected_city">
                    {"".join([f"<option value='{city}'>{city}</option>" for city in cities])}
                </select><br>
                <label for="target">Target:</label>
                <input type="number" name="target" class="form-control" required>
                <label for="score">Score:</label>
                <input type="number" name="score" class="form-control" required>
                <label for="overs">Overs completed:</label>
                <input type="number" name="overs" class="form-control" required>
                <label for="wickets">Wickets out:</label>
                <input type="number" name="wickets" class="form-control" required>
                <button type="submit" class="btn btn-success">Predict Probability</button>
            </form>
        </div>
    </body>
    </html>
    """

def get_result_template(batting_team, win, bowling_team, loss):
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{
                background-image: url('data:image/jpeg;base64,{encoded_image}');
                background-size: cover;
                background-repeat: no-repeat;
                color: white;
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 0;
            }}
            .container {{
                max-width: 800px;
                margin: auto;
                margin-top: 50px;
                padding: 20px;
                background-color: rgba(0, 0, 0, 0.7);
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
            }}
            h1, p {{
                text-align: center;
                margin-bottom: 20px;
            }}
        </style>
        <title>Result</title>
    </head>
    <body>
        <div class="container">
            <h1>Result</h1>
            <p>{batting_team} - {win}%</p>
            <p>{bowling_team} - {loss}%</p>
        </div>
    </body>
    </html>
    """

@app.route('/')
def home():
    return get_index_template()

@app.route('/predict', methods=['POST'])
def predict():
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

        return get_result_template(batting_team, round(win*100), bowling_team, round(loss*100))

if __name__ == '__main__':
    app.run(debug=True)
