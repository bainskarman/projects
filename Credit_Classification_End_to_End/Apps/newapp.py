import streamlit as st
import pickle
import pandas as pd
from zipfile import ZipFile
import os
from src import transform_resp
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

st.set_page_config(page_title='Credit Classification', layout='wide',initial_sidebar_state='auto', menu_items={
                        'Get Help': None,
                        'Report a bug': 'https://github.com/bainskarman/projects/issues',
                        'About': '''Enter the following information to get your credit score for previous 12 months or select a profile from the given options. This is a mock-up intended for information only, if you wish to learn more about the model behind this please go to the GitHub [Credit Analysis](github.com/bainskarman/projects/Credit_Classification_End_to_End)''' })

current_path = os.getcwd()
path = os.path.join(current_path, 'Credit_Classification_End_to_End/Apps/model.pkl')
model =joblib.load(path)

st.title('Credit Score Analysis')

st.markdown('''Credit Industry uses credit scores to evaluate the potential risk posed by lending money to consumers and to mitigate losses due to bad debt. Credit scoring is not limited to banks. Other organizations, such as mobile phone companies, insurance companies, landlords, and government departments employ the same techniques. Credit scoring also has a lot of overlap with data mining, which uses many similar techniques. These techniques combine thousands of factors but are similar or identical. Hereby, I have used Machine Learning to predict the credit score of a person based on the information provided by them. ''')
age_default = None
annual_income_default = 0.00
accounts_default = 0
credit_cards_default = 0
delayed_payments_default = 0
credit_card_ratio_default = 0.00
emi_monthly_default = 0.00
credit_history_default = 0
loans_default = None
missed_payment_default = 0
minimum_payment_default = 0
profile = st.radio('Choose a profile:', options=['Poor', 'Standard', 'Good'], horizontal=True)
if profile == 'Poor':
    age_default = 19
    annual_income_default = 8000.00
    accounts_default = 10
    credit_cards_default = 10
    delayed_payments_default = 20
    credit_card_ratio_default = 28.00
    emi_monthly_default = 36.00
    credit_history_default = 87
    loans_default = ['Student Loan','Mortgage Loan','Debt Consolidation Loan','Payday Loan']
    missed_payment_default = 1
    minimum_payment_default = 1
elif profile == 'Standard':
    age_default = 30
    annual_income_default = 17000.00
    accounts_default = 5
    credit_cards_default = 5
    delayed_payments_default = 7
    credit_card_ratio_default = 30.00
    emi_monthly_default = 150.00
    credit_history_default = 204
    loans_default = ['Personal Loan', 'Auto Loan']
    missed_payment_default = 1
    minimum_payment_default = 1
elif profile == 'Good':
    age_default = 42
    annual_income_default = 90000.00
    accounts_default = 2
    credit_cards_default = 1
    delayed_payments_default = 0
    credit_card_ratio_default = 17.43
    emi_monthly_default = 500.00
    credit_history_default = 288
    loans_default = ['Auto Loan', 'Mortgage Loan']
    missed_payment_default = 1
    minimum_payment_default = 1

with st.sidebar:
    st.header('Credit Score Inputs')
    age = st.slider('Enter age', min_value=18, max_value=100, step=1, value=age_default)
    annual_income = st.number_input('Enter Annual Income', min_value=0.00, max_value=300000.00, value=annual_income_default)
    accounts = st.number_input('Number of Bank Accounts', min_value=0, max_value=20, step=1, value=accounts_default)
    credit_cards = st.number_input('Number of Credit Cards', min_value=0, max_value=12, step=1, value=credit_cards_default)
    delayed_payments = st.number_input('Number of Delayed Payments', min_value=0, max_value=20, step=1, value=delayed_payments_default)
    credit_card_ratio = st.slider('CUR Credit Utilization Ratio', min_value=0.00, max_value=100.00, value=credit_card_ratio_default)
    emi_monthly = st.number_input('Monthly EMI Payment', min_value=0.00, max_value=5000.00, value=emi_monthly_default)
    credit_history = st.number_input('Credit History in Number of Months', min_value=0, max_value=500, step=1, value=credit_history_default)
    loans = st.multiselect('Select the Loans', ['Auto Loan', 'Credit-Builder Loan', 'Personal Loan',
                                                'Home Equity Loan', 'Mortgage Loan', 'Student Loan',
                                                'Debt Consolidation Loan', 'Payday Loan'], default=loans_default)
    missed_payment = st.radio('Missed Payment Due Date', ['Yes', 'No'], index=missed_payment_default)
    minimum_payment = st.radio('Payed Minium Amount Only', ['Yes', 'No'], index=minimum_payment_default)
    run = st.button( 'Predict and Analyze')
placeholder = st.empty()
if run:
    st.header('Credit Score Results')
    resp = {
        'age': age,
        'annual_income': annual_income,
        'accounts': accounts,
        'credit_cards': credit_cards,
        'delayed_payments': delayed_payments,
        'credit_card_ratio': credit_card_ratio,
        'emi_monthly': emi_monthly,
        'credit_history': credit_history,
        'loans': loans,
        'missed_payment': missed_payment,
        'minimum_payment': minimum_payment
    }
    output = transform_resp(resp)
    output = pd.DataFrame(output, index=[0])
    output.loc[:,:] = output

    credit_score = model.predict(output)[0]
    if credit_score == 'Good':
        st.balloons()
        placeholder.markdown('Your credit score is **GOOD**! Congratulations!')
    elif credit_score == 'Standard':
        placeholder.markdown('Your credit score is **REGULAR**.')
        st.markdown('This credit score indicates that this person is likely to repay a loan, but can occasionally miss some payments. Meaning that the risk of giving them credit is medium.')
    elif credit_score == 'Poor':
        placeholder.markdown('Your credit score is **POOR**.')

    col1, col2 = st.columns(2)

    # Plot for Probability Plot
    with col1:
        st.subheader('Probability Plot for Each Category')
        predicted_probabilities = model.predict_proba(output)
        classes = model.classes_

        # Define colors for each category
        colors = {'Standard': 'yellow', 'Poor': 'red', 'Good': 'green'}

        # Create a pie chart
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.pie(predicted_probabilities[0], labels=classes, autopct='%.0f%%', startangle=90, wedgeprops=dict(width=0.4, edgecolor='w'), colors=[colors[c] for c in classes])
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Display the plot using Streamlit
        st.pyplot(fig)

    with col2:
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.subheader('Weightage of Each Feature')
        
        importance = model.named_steps['classifier'].feature_importances_            
        preprocessor = model.named_steps['preprocessor']
        
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


