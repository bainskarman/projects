import pandas as pd
import numpy as np
import joblib
import streamlit as st

Education= ["Bachelor's", "Master's" ,'High School', 'PhD']
LoanPurpose= ['Other', 'Auto', 'Business' ,'Home' ,'Education']
EmploymentType = ['Full-time' ,'Unemployed' ,'Self-employed' ,'Part-time']
 
file_path='/workspaces/projects/Loan/pipeline.joblib'
with open(file_path, 'rb') as file:
    pipe = joblib.load(file)

st.title('Loan Default Prediction')

col1, col2 = st.columns(2)

with col1:
    edu = st.selectbox('Select highest education level.', sorted(Education))
with col2:
    pur = st.selectbox('Purpose of Loan', sorted(LoanPurpose))

col3, col4 = st.columns(2)
with col3:
    emp = st.selectbox('Select Employment Type', sorted(EmploymentType))
with col4:
    loan = st.number_input('Select Loan Amount')

col5, col6 = st.columns(2)
with col5:
    term = st.number_input('Loan Term')
with col6:
    crs = st.number_input('Credit Score')

col8, col9, col10 = st.columns(3)
with col8:
    income = st.number_input('Monthly Income')
with col9:
    interest = st.number_input('Interest Rate')
with col10:
    employ = st.number_input('Employed Months till Now')

if st.button('Predict Default Chances'):
    input_df = pd.DataFrame({
        'Education': [edu],
        'LoanTerm': [term],
        'CreditScore': [crs],
        'Income': [income],
        'InterestRate': [interest],
        'LoanAmount': [loan],
        'MonthsEmployed': [employ],
        'LoanPurpose': [pur],
        'EmploymentType': [emp] 
    })

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header("Default Probability" + "- " + str(round(win * 100)) + "%")
 