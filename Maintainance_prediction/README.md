# Predictive Maintenance with Deep Learning

## Overview
This project focuses on predictive maintenance using deep learning techniques, specifically Long Short-Term Memory (LSTM) neural networks. The goal is to predict potential failures in machinery by analyzing telemetry data, errors, and maintenance records. The project involves comprehensive exploratory data analysis (EDA) and the implementation of LSTM models for time-series forecasting.

## Notebooks

### 1. Exploratory Data Analysis (EDA)
- **Notebook Name:** `EDA_Predictive_Maintenance.ipynb`
- **Description:**
  - The EDA notebook explores and analyzes the provided datasets, including telemetry data, failure records, errors, and maintenance logs.
  - Visualizations are used to understand the chronological relationship between errors and failures, as well as to identify patterns and anomalies in telemetry features.
  - The notebook provides insights into the data, helping in the selection of relevant features for predictive modeling.

### 2. LSTM Time-Series Forecasting
- **Notebook Name:** `LSTM_Predictive_Maintenance.ipynb`
- **Description:**
  - This notebook focuses on building LSTM models for time-series forecasting, particularly for predicting telemetry features such as volt, rotation, pressure, and vibration.
  - Data preprocessing steps, model architecture, and training procedures are detailed in the notebook.
  - The LSTM models are trained on a machine-specific basis to predict potential failures, contributing to a proactive maintenance strategy.
