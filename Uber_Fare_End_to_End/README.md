# Uber Fare Estimation

## Overview

This Streamlit application predicts the fare amount for an Uber ride based on user-provided pickup and dropoff coordinates, passenger count, and the current date and time. The application uses a pre-trained machine learning model for fare estimation.

## Project Structure

- **app.py**: Streamlit application script containing the main functionality.
- **pipeline.joblib**: Pre-trained machine learning model (serialized using joblib).
- **requirements.txt**: List of dependencies for the project.

## Installation and Usage

### Prerequisites

- Python 3.6 or later
- Streamlit
- pandas
- joblib
- geopy

### Setup

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/yourusername/Uber_Fare_Estimation.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Uber_Fare_Estimation
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

3. Use the application to input pickup and dropoff coordinates, passenger count, and click the "Predict Fare" button to get the estimated fare amount.

## App Structure

### 1. Input Section

Users can interact with the following input fields:

- **Pickup Latitude**: Enter the latitude of the pickup location.
- **Pickup Longitude**: Enter the longitude of the pickup location.
- **Dropoff Latitude**: Enter the latitude of the dropoff location.
- **Dropoff Longitude**: Enter the longitude of the dropoff location.
- **Passenger Count**: Enter the number of passengers.

### 2. Fare Prediction

- After clicking the "Predict Fare" button, the application uses the pre-trained model to estimate the fare amount for the Uber ride.

### 3. Results Display

- The application displays the estimated fare amount.

## Feedback and Issues

If you encounter any issues or have suggestions for improvement, please [report them on GitHub](https://github.com/yourusername/Uber_Fare_Estimation/issues).

---

*Note: Adjust the GitHub repository link and other details in the README based on your actual project details.*
