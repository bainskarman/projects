# IPL Match Winner Predictor

## Overview

This project predicts the winning probabilities of cricket teams in IPL matches based on user-provided information. The application allows users to input details such as the batting team, bowling team, host city, target score, current score, overs completed, and wickets out. It utilizes a pre-trained machine learning model to predict the winning probabilities of both the batting and bowling teams.

## Project Structure

- **app.py**: Streamlit and Flask application script containing the main functionality.
- **pipeline.pkl**: Pre-trained machine learning model (serialized using joblib).
- **requirements.txt**: List of dependencies for the project.

## Installation and Usage

### Prerequisites

- Python 3.6 or later
- Streamlit
- Flask
- pandas
- numpy
- joblib
- scikit-learn

### Setup

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/yourusername/Team_Winning_Chances_End_to_End.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Team_Winning_Chances_End_to_End
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Run the Application

#### Streamlit App

1. Execute the following command in the terminal:

    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to the provided URL (typically http://localhost:8501).

3. Use the application to input various match details and predict the winning probabilities.

#### Flask App

1. Execute the following command in the terminal:

    ```bash
    python app.py
    ```

2. Open your web browser and go to http://localhost:5000.

3. Use the application to input various match details and predict the winning probabilities.

## App Structure

### 1. Streamlit App

- **Input Section**: Users can select the batting team, bowling team, host city, target score, current score, overs completed, and wickets out.

- **Match Winner Prediction**: After clicking the "Predict Probability" button, the application uses the pre-trained model to predict the winning probabilities of both the batting and bowling teams.

- **Results Display**: The application displays the predicted winning probabilities for the selected teams.

### 2. Flask App

- **Input Section**: Similar to the Streamlit app, users can input match details.

- **Match Winner Prediction**: After clicking the "Predict Probability" button, the Flask app also uses the pre-trained model to predict the winning probabilities.

- **Results Display**: The Flask app displays the predicted winning probabilities for the selected teams.

## Feedback and Issues

If you encounter any issues or have suggestions for improvement, please [report them on GitHub](https://github.com/yourusername/Team_Winning_Chances_End_to_End/issues).
