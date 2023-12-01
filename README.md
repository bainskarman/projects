# IPL Winner Prediction App


## Overview

This machine learning-based app predicts the winner of an IPL (Indian Premier League) cricket match's second inning. It takes various factors into account, such as the batting team, bowling team, target, current score, overs, wickets, and the city where the match is being played.

The prediction model utilizes a machine learning algorithm trained on historical IPL match data. The video demonstration showcases how each factor influences the prediction outcome. For example, the winning percentage increases when the match is played at the high-scoring ground in 'Sharjah' but decreases if played at 'Ahmedabad'. Additionally, factors like wickets, overs, and current score contribute to the overall prediction accuracy.

## Features

- **Input Factors:**
  - Batting Team
  - Bowling Team
  - Target
  - Current Score
  - Overs
  - Wickets
  - City

- **City Influence:**
  -Example:
  - 'Sharjah' is considered a high-scoring ground, increasing the winning percentage.
  - 'Ahmedabad' may decrease the winning percentage.

- **Dynamic Prediction:**
  - The prediction model dynamically considers each input factor's impact on the outcome.

## Getting Started

### Prerequisites

- Python installed
- Dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/bainskarman/projects
    ```

2. Change into the project directory:

    ```bash
    cd IPL
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

2. Open your web browser and navigate to the provided URL (usually `http://localhost:8501`).

3. Input the required factors (batting team, bowling team, target, current score, overs, wickets, city) for the second inning.

4. Receive the dynamic prediction based on the machine learning model.

