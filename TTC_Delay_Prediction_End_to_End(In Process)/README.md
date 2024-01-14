# TTC Delay Predictor

## Overview

The TTC Delay Predictor is a Streamlit application that predicts the expected delay for TTC (Toronto Transit Commission) buses based on current date, time, location, and selected route. The application utilizes pre-trained machine learning models for delay prediction.

## Project Structure

- **app.py**: Streamlit application script containing the main functionality.
- **delay.pkl**: Pre-trained machine learning model for delay prediction.
- **incident.pkl**: Pre-trained machine learning model for incident prediction.
- **requirements.txt**: List of dependencies for the project.

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
    git clone https://github.com/yourusername/TTC_Delay_Prediction.git
    ```

2. Navigate to the project directory:

    ```bash
    cd TTC_Delay_Prediction
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1. Execute the following command in the terminal:

    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to the provided URL (typically http://localhost:8501).

3. Use the application to select the location and bus route.

4. Click the "Predict" button to get the expected delay for the selected route.

## App Structure

### 1. Input Section

Users can interact with the following input fields:

- **Location**: Select the location.
- **Bus Route**: Select the bus route.

### 2. Delay Prediction

- After clicking the "Predict" button, the application uses the pre-trained delay prediction model to estimate the expected delay for the selected route.

### 3. Results Display

- The application displays the predicted delay in minutes.

## Feedback and Issues

If you encounter any issues or have suggestions for improvement, please [report them on GitHub](https://github.com/yourusername/TTC_Delay_Prediction/issues).

---

*Note: Adjust the GitHub repository link and other details in the README based on your actual project details.*
