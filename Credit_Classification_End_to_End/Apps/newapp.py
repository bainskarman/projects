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
import gzip
import plotly.express as px
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

st.set_page_config(page_title='Credit Classification', layout='wide',initial_sidebar_state='auto', menu_items={
                        'Get Help': None,
                        'Report a bug': 'https://github.com/bainskarman/projects/issues',
                        'About': '''Enter the following information to get your credit score for previous 12 months or select a profile from the given options. This is a mock-up intended for information only, if you wish to learn more about the model behind this please go to the GitHub [Credit Analysis](github.com/bainskarman/projects/Credit_Classification_End_to_End)''' })
current_path = os.getcwd()
path = os.path.join(current_path, 'Credit_Classification_End_to_End/Apps/final_pipeline.pkl.gz')
df_path = os.path.join(current_path, 'Credit_Classification_End_to_End/Apps/cleaned.csv')
data = pd.read_csv(df_path)
with gzip.open(path, 'rb') as file:
    model = pickle.load(file)

st.title('Credit Score Analysis')

st.markdown('''Credit Industry uses credit scores to evaluate the potential risk posed by lending money to consumers and to mitigate losses due to bad debt. Credit scoring is not limited to banks. Other organizations, such as mobile phone companies, insurance companies, landlords, and government departments employ the same techniques. Credit scoring also has a lot of overlap with data mining, which uses many similar techniques. These techniques combine thousands of factors but are similar or identical. Hereby, I have used Machine Learning to predict the credit score of a person based on the information provided by them with model accuracy of ~80 percent on more than 100k records.  ''')
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
        colors = {'Standard': 'yellow', 'Poor': 'red', 'Good': 'green'}

        # Create a DataFrame for Plotly Express
        plotly_df = pd.DataFrame({
            'Probability': predicted_probabilities[0],
            'Category': classes,
            'Color': [colors[c] for c in classes]
        })

        # Create an interactive pie chart
        fig = px.pie(plotly_df, values='Probability', names='Category',
                    title='Probability Plot for Each Category',
                    color='Category', color_discrete_map=colors,
                    hole=0.4,  # Adjust the hole size for a doughnut chart effect
                    labels={'Probability': 'Probability (%)'})

        # Adjust layout for better visualization
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=False, margin=dict(l=0, r=0, b=0, t=40))

        # Display the interactive plot using Streamlit
        st.plotly_chart(fig)

    # Custom feature names
    custom_feature_names = {
        'num__Age': 'Age',
        'num__Annual_Income': 'Annual Income',
        'num__Num_Bank_Accounts': 'Bank Accounts',
        'num__Num_Credit_Card': 'Number of Credit Cards',
        'num__Num_of_Delayed_Payment': 'Delayed Payments',
        'num__Credit_Utilization_Ratio': 'Utilization Ratio',
        'num__Total_EMI_per_month': 'Monthly EMI',
        'num__Credit_History_Age_Formated': 'Months of Credit History',
        'cat__Payment_of_Min_Amount_Yes_No': 'Min Payment if No',
        'cat__Payment_of_Min_Amount_Yes_Yes': 'Min Payment if Yes'
    }

    with col2:
        st.subheader('Weightage of Each Feature')
        
        importance = model.named_steps['classifier'].feature_importances_            
        original_feature_names =['num__Age', 'num__Annual_Income', 'num__Num_Bank_Accounts',
       'num__Num_Credit_Card', 'num__Num_of_Delayed_Payment',
       'num__Credit_Utilization_Ratio', 'num__Total_EMI_per_month',
       'num__Credit_History_Age_Formated',
       'cat__Payment_of_Min_Amount_Yes_No',
       'cat__Payment_of_Min_Amount_Yes_Yes']
        # Update feature names using the custom names
        feature_names = [custom_feature_names.get(feature, feature) for feature in original_feature_names]
        
        importance_df = pd.DataFrame({'Feature Importance': importance, 'Feature': feature_names})
        importance_df = importance_df.sort_values(by='Feature Importance', ascending=True)

        # Round the 'Feature Importance' column to 2 decimal places
        importance_df['Feature Importance'] = importance_df['Feature Importance'].round(2)

        fig = px.bar(importance_df, x='Feature Importance', y='Feature',
                    text='Feature Importance', orientation='h', color='Feature Importance',
                    color_continuous_scale='Viridis', labels={'Feature Importance': 'Importance (%)'},
                    title='Feature Importance')

        # Adjust layout for better visualization
        fig.update_layout(yaxis=dict(autorange='reversed'), margin=dict(l=0, r=0, b=0, t=40))

        # Customize hovertemplate to display percentage with 2 decimal places
        fig.update_traces(hovertemplate='%{x:.2%} Importance')

        # Display the interactive plot using Streamlit
        st.plotly_chart(fig)
columns_to_plot = [
    "Age",
    "Annual_Income",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Num_of_Delayed_Payment",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Credit_History_Age_Formated"
]

# Get unique credit score categories
if "Credit_Score" not in data.columns:
    st.error("The dataset doesn't contain a 'Credit_Score' column.")
    st.stop()

credit_score_categories = data["Credit_Score"].unique()

# Create tabs for each credit score category
tabs = st.tabs([f"Credit Score: {category}" for category in credit_score_categories])

for tab, category in zip(tabs, credit_score_categories):
    with tab:
        st.subheader(f"Distributions for Credit Score: {category}")
        
        # Filter data for current category
        category_data = data[data["Credit_Score"] == category]
        
        # Create a grid of plots
        cols_per_row = 2
        rows = (len(columns_to_plot) + cols_per_row - 1) // cols_per_row
        
        for i in range(rows):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i * cols_per_row + j
                if idx < len(columns_to_plot):
                    col = columns_to_plot[idx]
                    with cols[j]:
                        st.write(f"**{col}**")
                        
                        # Create figure
                        fig, ax = plt.subplots(figsize=(8, 4))
                        
                        # Plot distribution based on data type
                        if pd.api.types.is_numeric_dtype(category_data[col]):
                            sns.histplot(category_data[col], kde=True, ax=ax)
                            ax.set_title(f"Distribution of {col}")
                            ax.set_xlabel(col)
                            ax.set_ylabel("Frequency")
                        else:
                            # For categorical or string data
                            value_counts = category_data[col].value_counts().head(10)  # Limit to top 10
                            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                            ax.set_title(f"Top values for {col}")
                            ax.set_xlabel(col)
                            ax.set_ylabel("Count")
                            plt.xticks(rotation=45)
                        
                        st.pyplot(fig)
                        plt.close()

# Add some space at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)