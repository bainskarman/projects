# Loan Default Prediction with Streamlit

## Overview

This Streamlit application predicts the probability of loan default based on user-provided information such as education level, loan purpose, employment type, loan amount, loan term, credit score, monthly income, interest rate, and months employed. It uses a pre-trained machine learning model.

## Project Structure

- **loan_default_app.py**: Streamlit application script containing the main functionality.
- **pipe.joblib**: Pre-trained machine learning model (serialized using joblib).

## Installation and Usage

### Prerequisites

- Python 3.6 or later
- Streamlit
- pandas
- numpy
- joblib
- scikit-learn

### Setup

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/bainskarman/projects.git
    ```

2. Navigate to the project directory:

    ```bash
    cd projects/Loan_Default_Probability
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Run the Application

1. Execute the following command in the terminal:

    ```bash
    streamlit run loan_default_app.py
    ```

2. Open your web browser and go to the provided URL (typically http://localhost:8501).

3. Use the application to input various parameters and predict the probability of loan default.

## Streamlit App Structure

The Streamlit application follows a structured flow:

- **Input Section**: Users can select their highest education level, loan purpose, employment type, and input various financial parameters.

- **Loan Default Prediction**: After clicking the "Predict Default Chances" button, the application uses the pre-trained model to predict the probability of loan default.

- **Results Display**: The application then displays the predicted default probability.

## Feedback and Issues

If you encounter any issues or have suggestions for improvement, please [report them on GitHub](https://github.com/bainskarman/projects/issues).

