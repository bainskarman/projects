import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
Education= ["Bachelor's", "Master's" ,'High School', 'PhD']
LoanPurpose= ['Other', 'Auto', 'Business' ,'Home' ,'Education']
EmploymentType = ['Full-time' ,'Unemployed' ,'Self-employed' ,'Part-time']
 
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

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader('Weightage of Each Feature')
    
    importance = pipe.named_steps['classifier'].feature_importances_            
    preprocessor = pipe.named_steps['preprocessor']
    
    # Assuming the preprocessor has a 'get_feature_names_out' method
    feature_names = preprocessor.get_feature_names_out()
    
    importance_df = pd.DataFrame({'Feature Importance': importance, 'Feature': feature_names})
    importance_df = importance_df.sort_values(by='Feature Importance', ascending=True)

    # Plotting the figure with vertical bar chart
    plt.figure(figsize=(8, 6))

    # Set a default color in case the number of features exceeds the number of custom colors
    default_color = '#FFD700'
    
    # Ensure the number of colors matches the number of features
    custom_colors = ['#FF6666', '#FF9933', '#99FF99', '#99CCFF', '#CC99FF', '#FFCC99', '#66FF99', '#FFD700', '#FF6347', '#8A2BE2'][:len(importance_df)]

    # Create bars with different colors and add legends individually
    for i, (feature, importance) in enumerate(zip(importance_df['Feature'], importance_df['Feature Importance'])):
        color = custom_colors[i] if i < len(custom_colors) else default_color
        bar = plt.bar(i, importance, color=color)
        plt.text(i, importance, f'{importance:.2f}%', ha='center', va='bottom', color='black')

    # Remove x labels
    plt.xticks([])

        # Create legends with different colors
        legend_labels = importance_df['Feature']
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in custom_colors]
        plt.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2)

        plt.grid(axis='y', linestyle='--', alpha=0.6)
        sns.despine(right=True, top=True)

        # Display the plot using Streamlit
        st.pyplot()

 