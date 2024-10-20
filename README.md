# Data Science and Machine Learning Projects

Welcome to my portfolio of Data Science and Machine Learning projects! This repository contains various applications and models developed for different purposes. Below, you'll find a brief overview of each project along with instructions for installation and usage.

## Project List

### ATS Scanner

[Hugging Face Spaces - ATS Scanner](https://huggingface.co/spaces/bainskarman/ATSScanner)


Welcome to the **ATS Scanner**! This application evaluates your resume and provides an ATS (Applicant Tracking System) score, along with actionable feedback to improve your chances of getting noticed by recruiters.

Overview

The ATS Scanner utilizes advanced natural language processing models to analyze your resume content. It shortens your ATS score using the `paraphrase-MiniLM-L6-v2` model and generates a detailed review using the `nvidia/llama-3.1-nemotron-70b-instruct` model. The app highlights areas for improvement and what aspects of your resume are strong.

Features

- ATS Score Assessment: Receive a concise ATS score based on your resume content.
- Detailed Feedback: Get a comprehensive review of your resume, including:
- Strengths to emphasize
- Areas needing improvement
- Tips for optimization

How to Use

1. Input Your Resume: Upload your resume document or paste the text directly into the provided field.
2. Generate ATS Score: Click the button to analyze your resume.
3. Review Feedback: Examine the generated ATS score and feedback to understand how to enhance your resume.

Technologies Used

- paraphrase-MiniLM-L6-v2: For scoring the ATS effectiveness of your resume.
- nvidia/llama-3.1-nemotron-70b-instruct: For generating detailed, actionable feedback on your resume.

https://github.com/user-attachments/assets/c6ce0ad1-51ff-4cbd-bc7e-362b4cc5cf1f

### Wiqi 3 Level Classification

[Wiqi 3 Level Classification](https://huggingface.co/spaces/bainskarman/WiqiClassification)

Wiqi 3 Level Classification is a powerful tool designed to classify Wikipedia articles into up to three different categories. With an impressive accuracy rate of 88% and support for over 2000 unique categories, this project aims to enhance the organization and retrieval of information from Wikipedia.

Features
- Classifies Wikipedia articles into up to three categories
- Achieves 88% classification accuracy
- Supports over 2000 unique categories
- User-friendly interface hosted on Hugging Face
To get started with the Wiqi 3 Level Classification, you can either use the hosted app or run the code locally.


[Watch the Video](https://github.com/user-attachments/assets/c6ce0ad1-51ff-4cbd-bc7e-362b4cc5cf1f)



### Credit Score Analysis
[Credit Check App](https://creditchecker.streamlit.app/)
- **Description**: A Streamlit application for predicting credit scores based on user-provided information. Three profiles ("Poor," "Standard," "Good") are available for demonstration.
- **Files**:
  - `Credit_Classification_End_to_End/Apps/credit_score_app.py`: Streamlit application script
  - `Credit_Classification_End_to_End/Apps/model.pkl`: Pre-trained machine learning model
- **Usage**:
  - Clone the repository: `git clone https://github.com/bainskarman/projects.git`
  - Navigate to the project directory: `cd projects/Credit_Classification_End_to_End`
  - Install dependencies: `pip install -r requirements.txt`
  - Run the application: `streamlit run Apps/credit_score_app.py`

https://github.com/user-attachments/assets/384eaed3-f3d5-4ef0-b37e-16bed7e3de37

### Loan Default Prediction
[Loan Risk Analysis App](https://loanriskanalysis.streamlit.app/)
- **Description**: A Streamlit application for predicting the probability of loan default based on user-provided information such as education level, loan purpose, employment type, loan amount, loan term, credit score, monthly income, and interest rate.
- **Files**:
  - `Loan_Default_Probability_End_to_End/app.py`: Streamlit application script
  - `Loan_Default_Probability_End_to_End/pipe.joblib`: Pre-trained machine learning model
- **Usage**:
  - Clone the repository: `git clone https://github.com/bainskarman/projects.git`
  - Navigate to the project directory: `cd projects/Loan_Default_Probability_End_to_End`
  - Install dependencies: `pip install -r requirements.txt`
  - Run the application: `streamlit run app.py`
  - Access the application in your web browser at http://localhost:8501

https://github.com/bainskarman/projects/assets/122693789/6a52b5e0-67a0-4117-a8b0-a25e7ea0bf30

### Team Winning Chances

- **Description**: A Streamlit application for predicting cricket match outcomes based on user-selected teams, host city, target, score, overs completed, and wickets out.
- **Files**:
  - `Team_Winning_Chances_End_to_End/app.py`: Streamlit application script
  - `Team_Winning_Chances_End_to_End/pipeline.pkl`: Pre-trained machine learning model
- **Usage**:
  - Clone the repository: `git clone https://github.com/bainskarman/projects.git`
  - Navigate to the project directory: `cd projects/Team_Winning_Chances_End_to_End`
  - Install dependencies: `pip install -r requirements.txt`
  - Run the application: `streamlit run app.py`
  - Access the application in your web browser at http://localhost:8501

https://github.com/bainskarman/projects/assets/122693789/fa0793db-af29-4dc0-b4fd-c310449156ca

### TTC Delay Analysis

- **Description**: A Streamlit application for predicting delays in Toronto Transit Commission (TTC) buses based on date, time, location, and route information.
- **Files**:
  - `TTC_Delay_Prediction_End_to_End/app.py`: Streamlit application script
  - `TTC_Delay_Prediction_End_to_End/delay.pkl`: Pre-trained machine learning model for delay prediction
  - `TTC_Delay_Prediction_End_to_End/incident.pkl`: Pre-trained machine learning model for incident prediction
- **Usage**:
  - Clone the repository: `git clone https://github.com/bainskarman/projects.git`
  - Navigate to the project directory: `cd projects/TTC_Delay_Prediction_End_to_End`
  - Install dependencies: `pip install -r requirements.txt`
  - Run the application: `streamlit run app.py`

https://github.com/bainskarman/projects/assets/122693789/a38edcf1-22cd-4e0b-8886-4aebbd5f59f1

### Uber Fare Estimation
Vist Application: [WEB APP](https://ubercostestimation.streamlit.app/)
- **Description**: A Streamlit application for estimating Uber ride fares based on user-provided pickup and dropoff coordinates, passenger count, and the current date and time.
- **Files**:
  - `Uber_Fare_End_to_End/app.py`: Streamlit application script
  - `Uber_Fare_End_to_End/pipeline.joblib`: Pre-trained machine learning model
- **Usage**:
  - Clone the repository: `git clone https://github.com/bainskarman/projects.git`
  - Navigate to the project directory: `cd projects/Uber_Fare_End_to_End`
  - Install dependencies: `pip install -r requirements.txt`
  - Run the application: `streamlit run app.py`
  - Access the application in your web browser at http://localhost:8501

https://github.com/bainskarman/projects/assets/122693789/3b32f7fc-52f1-4a7c-9ff1-2e2f563f29e4
  
### Stocks Forcasting with LSTM and Prophet
Delving into Stock Forecasting, our project integrates LSTM, ANN, and Prophet models. By harnessing historical stock data, we aim to predict future trends, enabling insightful investment decisions through advanced time-series analysis.

[Prophet](https://github.com/bainskarman/projects/assets/122693789/4b77dbf8-a5fc-4cfd-867f-3832373c784f)

### Sales Analysis with SQL and Power BI.
This project focuses on data analysis using Microsoft SQL Server Management Studio (SSMS) and creating a comprehensive dashboard using Power BI. The analysis involves running SQL queries, saving query results, and leveraging the data to design and validate a Power BI dashboard.

[Dash](https://github.com/bainskarman/projects/assets/122693789/edcdf20f-40c9-495e-9ea4-762372562d92)

### Survey Data Dashboard with Power BI.
Surveys are a valuable tool for collecting information and feedback, but interpreting large datasets can be challenging. The Survey Data Dashboard addresses this challenge by utilizing Power BI, a robust business intelligence platform, to transform raw survey data into visually compelling and interactive dashboards.

[Dashboard](https://github.com/bainskarman/projects/assets/122693789/66d92832-228c-4fbe-86a3-1f14139c3e9f)
## Contact

If you have any questions or want to get in touch, you can reach me at [bsinghkarman@gmail.com].

Happy coding!
