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

file_path='/workspaces/projects/Credit Score Classification/pipeline.joblib'
with open(file_path, 'rb') as file:
    pipe = joblib.load(file)

col1, col2 = st.columns(2)

with col1:
    occupation = st.selectbox('Select the your Occupation', sorted(occupation))
with col2:
    behaviour = st.selectbox('Select spending behavious similar to yours', sorted(behaviour))

col3, col4, col5, col6 = st.columns(4)

with col3:
    age = st.number_input('Age')
with col4:
    salary = st.number_input('Monthly Inhand Salary')
with col5:
    credit_cards = st.number_input('Number of Credit Cards')
with col6:
    interest_rate=st.number_input('Credit Card Interest Rate')

col7, col8, col9 = st.columns(3)
with col3:
    loans = st.number_input('Number of Loans')
with col4:
    delayed_pay = st.number_input('Number of Delayed Payments')
with col5:
    debt = st.number_input('Pending Debt Amount')

col10, col11 = st.columns(2)

with col10:
    emi = st.number_input('Total EMI Cost per Month')
with col11:
    invest = st.number_input('Total Investment per Month')

col12, col13 = st.columns(2)
with col12:
    cl = st.number_input('Total Credit Limit')
with col13:
    sp = st.number_input('Average Spendings')

if st.button('Predict CRS'):
    cur = (sp/cl)*100

    input_df = pd.DataFrame({
    'Age': [age],
    'Occupation': [occupation],
    'Monthly_Inhand_Salary': [salary],
    'Num_Credit_Card': [credit_cards],
    'Interest_Rate': [interest_rate],
    'Num_of_Loan': [loans],
    'Num_of_Delayed_Payment': [delayed_pay],
    'Outstanding_Debt': [debt],
    'Credit_Utilization_Ratio': [cur],
    'Total_EMI_per_month': [emi],
    'Amount_invested_monthly': [invest],
    'Payment_Behaviour': [behaviour],
    'Credit_Score': [credit_score]
})
    predicted_category = pipe.predict(input_df)[0]
    st.success(f'The predicted credit score category is: {predicted_category}')

