import streamlit as st
import joblib
import pandas as pd

occupation= ['Scientist',
 'Teacher',
 'Engineer',
 'Entrepreneur',
 'Developer',
 'Lawyer',
 'Media_Manager',
 'Doctor',
 'Journalist',
 'Manager',
 'Accountant',
 'Musician',
 'Mechanic',
 'Writer',
 'Architect']

behaviour = ['High_spent_Small_value_payments',
 'Low_spent_Large_value_payments',
 'Low_spent_Medium_value_payments',
 'Low_spent_Small_value_payments',
 'High_spent_Medium_value_payments',
 'High_spent_Large_value_payments']

category = ['Good', 'Standard', 'Poor']

file_path='/workspaces/projects/Credit_Classification_End_to_End/Models/pipeline.joblib'
with open(file_path, 'rb') as file:
    pipe = joblib.load(file)

col1, col2,col21 = st.columns(3)

with col1:
    occupation = st.selectbox('Select the your Occupation', sorted(occupation))
with col2:
    salary = st.number_input('Monthly Inhand Salary')
with col21:
    cha= st.number_input('Credit History Age')

col5, col6= st.columns(2)

with col5:
    credit_cards = st.number_input('Number of Credit Cards')
with col6:
    loans = st.number_input('Number of Loans')

col7, col8= st.columns(2)  
with col7:
    delayed_pay = st.number_input('Number of Delayed Payments')
with col5:
    debt = st.number_input('Pending Debt Amount')


col12, col13 = st.columns(2)
with col12:
    cl = st.number_input('Total Credit Limit')
with col13:
    sp = st.number_input('Average Spendings')

if st.button('Predict CRS'):
    cur = (sp/cl)*100
    input_df = pd.DataFrame({
    'Occupation': [occupation],
    'Monthly_Inhand_Salary': [salary],
    'Num_Credit_Card': [credit_cards],
    'Num_of_Loan': [loans],
    'Num_of_Delayed_Payment': [delayed_pay],
    'Outstanding_Debt': [debt],
    'Credit_Utilization_Ratio': [cur],
    'Credit_History_Age':[cha]
})
    predicted_category = pipe.predict(input_df)[0]
    st.success(f'The predicted credit score category is: {predicted_category}')

