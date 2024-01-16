import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import pandas as pd
Education= ["Bachelor's", "Master's" ,'High School', 'PhD']
LoanPurpose= ['Other', 'Auto', 'Business' ,'Home' ,'Education']
EmploymentType = ['Full-time' ,'Unemployed' ,'Self-employed' ,'Part-time']
st.set_page_config(page_title='Credit Classification', layout='wide',initial_sidebar_state='auto', menu_items={
                        'Get Help': None,
                        'Report a bug': 'https://github.com/bainskarman/projects/issues'})
current_path = os.getcwd()
path = os.path.join(current_path, 'Loan_Default_Probability_End_to_End/pipe.joblib')
pipe =joblib.load(path)
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
importance = pipe.named_steps['model'].feature_importances_        
preprocessor = pipe.named_steps['preprocessor']
feature_names = preprocessor.get_feature_names_out() 
original_feature_names = [
    'num__LoanTerm', 'num__CreditScore', 'num__Income', 'num__InterestRate', 'num__LoanAmount', 'num__MonthsEmployed',
    "cat__Education_Bachelor's", 'cat__Education_High School', "cat__Education_Master's", 'cat__Education_PhD',
    'cat__LoanPurpose_Auto', 'cat__LoanPurpose_Business', 'cat__LoanPurpose_Education', 'cat__LoanPurpose_Home',
    'cat__LoanPurpose_Other', 'cat__EmploymentType_Full-time', 'cat__EmploymentType_Part-time',
    'cat__EmploymentType_Self-employed', 'cat__EmploymentType_Unemployed'
]

# Mapping original feature names to simpler names
name_mapping = {
    'num__LoanTerm': 'Loan Term',
    'num__CreditScore': 'Credit Score',
    'num__Income': 'Income',
    'num__InterestRate': 'Interest Rate',
    'num__LoanAmount': 'Loan Amount',
    'num__MonthsEmployed': 'Months Employed',
    "cat__Education_Bachelor's": 'Education: Bachelor\'s',
    'cat__Education_High School': 'Education: High School',
    "cat__Education_Master's": 'Education: Master\'s',
    'cat__Education_PhD': 'Education: PhD',
    'cat__LoanPurpose_Auto': 'Loan Purpose: Auto',
    'cat__LoanPurpose_Business': 'Loan Purpose: Business',
    'cat__LoanPurpose_Education': 'Loan Purpose: Education',
    'cat__LoanPurpose_Home': 'Loan Purpose: Home',
    'cat__LoanPurpose_Other': 'Loan Purpose: Other',
    'cat__EmploymentType_Full-time': 'Employment Type: Full-time',
    'cat__EmploymentType_Part-time': 'Employment Type: Part-time',
    'cat__EmploymentType_Self-employed': 'Employment Type: Self-employed',
    'cat__EmploymentType_Unemployed': 'Employment Type: Unemployed'
}

# Map the original feature names to simpler names
simplified_feature_names = [name_mapping.get(feature, feature) for feature in original_feature_names]
importance_df = pd.DataFrame({'Feature Importance': importance, 'Feature': simplified_feature_names})
importance_df = importance_df.sort_values(by='Feature Importance', ascending=True)
importance_df['Feature Importance'] = importance_df['Feature Importance'].round(2)
importance_df= importance_df[importance_df['Feature Importance'] > 0.01]
# Assuming 'importance_df' is your DataFrame
feature_names = importance_df['Feature']
importance_values = importance_df['Feature Importance']

fig = px.bar(
    x=importance_values,
    y=feature_names,
    text=importance_values,
    orientation='h',
    color=importance_values,
    color_continuous_scale='Viridis',
    labels={'x': 'Importance (%)', 'y': 'Feature'},
    title='Feature Importance'
)

# Customize layout for better visualization
fig.update_layout(yaxis=dict(autorange='reversed'))

# Customize hovertemplate to display percentage with 2 decimal places
fig.update_traces(texttemplate='%{text:.2%}', textposition='inside')

# Display the interactive plot using Streamlit
st.plotly_chart(fig)