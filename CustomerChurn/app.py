import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import os
import joblib
from plotly import graph_objects as go

# Set page config
st.set_page_config('Customer Churn Analysis', layout='wide')
st.title('Customer Churn Analysis')


# Load data
@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), 'data.csv')
    return pd.read_csv(data_path)

# Load pipeline
@st.cache_data
def load_pipeline():
    pipeline_path = os.path.join(os.path.dirname(__file__), 'churn_pipeline_v2.pkl')
    return joblib.load(pipeline_path)

# Clean data
def clean_data(data):
    # Convert empty strings to NaN and clean numerical columns
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        data[col] = pd.to_numeric(data[col].replace(' ', np.nan), errors='coerce')
    return data

data = load_data()
data = clean_data(data)
st.dataframe(data.head())

with st.expander("Data Description", expanded=False):
    st.write("""
             gender: Whether the customer is a male or a female\n
             SeniorCitizen: Whether the customer is a senior citizen or not (1, 0)\n
             Partner: Whether the customer has a partner or not (Yes, No)\n
             Dependents: Whether the customer has dependents or not (Yes, No)\n
             tenure: Number of months the customer has stayed with the company\n
             PhoneService: Whether the customer has a phone service or not (Yes, No)\n
             MultipleLines: Whether the customer has multiple lines or not (Yes, No, No phone service)\n
             InternetService: Customer's internet service provider (DSL, Fiber optic, No)\n
             OnlineSecurity: Whether the customer has online security or not (Yes, No, No internet service)\n
             OnlineBackup: Whether the customer has online backup or not (Yes, No, No internet service)\n
             DeviceProtection: Whether the customer has device protection or not (Yes, No, No internet service)\n
             TechSupport: Whether the customer has tech support or not (Yes, No, No internet service)\n
             StreamingTV: Whether the customer has streaming TV or not (Yes, No, No internet service)\n
             StreamingMovies: Whether the customer has streaming movies or not (Yes, No, No internet service)\n
             Contract: The contract term of the customer (Month-to-month, One year, Two year)\n
             PaperlessBilling: Whether the customer has paperless billing or not (Yes, No)\n
             PaymentMethod: The customer's payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card\n
             MonthlyCharges: The amount charged to the customer monthly\n
             TotalCharges: The total amount charged to the customer\n
             Churn: Whether the customer churned or not (Yes or No)\n
             """)

# Visualization 1: Payment Method vs Churn
col1, col2 = st.columns([2, 1], vertical_alignment='center')
with col1:
    contingency = pd.crosstab(data['PaymentMethod'], data['Churn'])
    tab = pd.DataFrame({
        "PaymentMethod": ["Bank transfer (automatic)", "Credit card (automatic)", 
                        "Electronic check", "Mailed check"] * 2,
        "Churn": ["No", "No", "No", "No", "Yes", "Yes", "Yes", "Yes"],
        "Count": contingency.values.T.reshape((-1))
    })
    payment_churn_contingency_tab = px.bar(tab, 
                 x="PaymentMethod", 
                 y="Count",  
                 color="Churn",
                 labels={"PaymentMethod": "Payment Method", "Churn": "Churn", "Count": "Number of Customers"})
    payment_churn_contingency_tab.update_layout(height=800)
    st.plotly_chart(payment_churn_contingency_tab, use_container_width=True)
    
with col2:
    st.write("This plot shows association between Churn Behaviour and Payment Method")
    st.write("""
            A lot of churned customer using Electronic Check as payment method compared
            to other method. There are some possible reasons\n
            1. Electronic Checks may offer customers more flexibility to cancel payments 
                compared to other methods like credit cards. This convenience 
                might encourage customers who are considering churn to proceed with 
                cancellation.\n
            2. Electronic Check payments could be prone to issues like failed transactions, 
                delays, or discrepancies, which could lead to dissatisfaction and, ultimately, 
                churn.
            """)

# Visualization 2: Monthly Charges vs Churn
col1, col2 = st.columns([2, 1], vertical_alignment='center')
with col1:
    churn_monthly_box = px.box(data_frame=data,
                               x='Churn',
                               y='MonthlyCharges',
                               color='Churn')
    churn_monthly_box.update_layout(xaxis_title="Churn",
                                    yaxis_title="MonthlyCharges",
                                    height=800)
    st.plotly_chart(churn_monthly_box, use_container_width=True)
    
with col2:
    st.write("""
            The median monthly charges for churned customers are visibly higher than for 
            loyal customers, suggesting that churned customers tend to pay more. 
            While there is some overlap in the ranges of monthly charges between the two groups, churned customers show a concentration at higher values.
            It is possible that expensive monthly charges is one of the reason why customer
            churn.
            """)

# Visualization 3: Contract vs Churn
col1, col2 = st.columns([2, 1], vertical_alignment='center')
with col1:
    tab = pd.crosstab(data['Contract'], data['Churn'])
    heatmap = px.imshow(tab, text_auto=True)
    heatmap.update_layout(height=800)
    st.plotly_chart(heatmap, use_container_width=True)
    
with col2:
    st.write("This plot show correlation between contract term and churn behaviour")
    st.write("""
            As seen from the plot, churned customer tend to have shorter contract term. 
            Possible explanations for
            this behaviour are:\n
             1. Trying out the service without a long-term intention to stay\n
             2. More likely to compare alternatives and switch providers when they find better 
             options.
            """)

st.markdown("""
<h3>Now lets see how each service affect customer behaviour</h3>
""", unsafe_allow_html=True)

service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies']
cols = service_cols + ['Churn']
services = data[cols].copy(True)

# Convert Churn to binary for correlation
services['Churn'] = services['Churn'].map({'Yes': 1, 'No': 0})

# Get feature importance from the pipeline
pipeline = load_pipeline()
rf_model = pipeline.named_steps['classifier']

# Get feature names after one-hot encoding
preprocessor = pipeline.named_steps['preprocessor']
# Define feature types (add this before the feature_names section)
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
binary_features = ['SeniorCitizen']

# Get feature names after one-hot encoding
preprocessor = pipeline.named_steps['preprocessor']
categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                      'MultipleLines', 'InternetService', 'OnlineSecurity', 
                      'OnlineBackup', 'DeviceProtection', 'TechSupport',
                      'StreamingTV', 'StreamingMovies', 'Contract', 
                      'PaperlessBilling', 'PaymentMethod']

# Get feature names
try:
    # For sklearn >= 1.0
    feature_names = (numerical_features + 
                    list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)) + 
                    binary_features)
except AttributeError:
    # For sklearn < 1.0
    feature_names = (numerical_features + 
                    list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names(categorical_features)) + 
                    binary_features)

# Display feature importance
st.markdown("""
<h3>Feature Importance from Random Forest Model</h3>
""", unsafe_allow_html=True)

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

fig = px.bar(importance_df.head(20), 
             x='Importance', 
             y='Feature', 
             orientation='h',
             title='Top 20 Important Features')
fig.update_layout(height=800)
st.plotly_chart(fig, use_container_width=True)

# Service vs Monthly Charges correlation
col1, col2 = st.columns([2, 1], vertical_alignment='center')
with col1:
    services = data[service_cols].copy()
    # Convert categorical services to numerical (simple approach)
    for col in service_cols:
        services[col] = services[col].map({'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0})
    
    services['MonthlyCharges'] = data['MonthlyCharges']
    service_monthly_corr = services.corr('kendall')['MonthlyCharges'].sort_values(ascending=False)[1:]
    service_monthly_barplot = px.bar(service_monthly_corr,
                                     orientation='h',
                                     color=service_monthly_corr.index)
    service_monthly_barplot.update_layout(
        xaxis_title="Kendall Tau Correlation Coefficient",
        yaxis_title="Columns",
        height=800,
        showlegend=False
    )
    st.plotly_chart(service_monthly_barplot, use_container_width=True)

with col2:
    st.write("""
    This plot shows correlation between MonthlyCharges and each service
    """)
    st.write("""
    From the plot we can see that Internet Service contribute a lot to
    monthly charge amount. This high contribution maybe one of the reason
    customers quit.
    """)

with st.container():
    st.markdown("""
    <h3>Conclusion</h3>
    """, unsafe_allow_html=True)
    st.write("""
    Based on the analysis of the data, several key factors were identified as having a significant impact on churn behavior. Specifically, the following features were found to be 
    highly correlated with customer churn:\n

    1. Contract Type: Customers with shorter-term contracts tend to have a higher likelihood of churning. This suggests that offering long-term contracts or incentives for contract 
    renewals could reduce churn rates.\n

    2. Internet Service: Customers with specific types of internet service (e.g., Fiber optic) are more likely to churn. Enhancing internet service offerings or lowering the monthly 
    charge for internet service could improve customer retention.\n

    3. Electronic Payment Method: Customers who use electronic payment methods (such as electronic checks) show a higher tendency to churn. 
    Encouraging customers to switch to more convenient payment methods or enhancing electronic payment method may help in reducing churn.\n
    """)

with st.container():
    st.markdown("""
    <h3>Suggestions</h3>
    """, unsafe_allow_html=True)
    st.write("""
            1. Explore how other features like tenure, Partner, Dependent and PaperBilling affecting churn behaviour.\n
            2. Use other machine learning model for analysis
            """)

# Add this section after your existing visualizations and before the conclusion

st.markdown("""
<h3>Customer Retention Prediction</h3>
""", unsafe_allow_html=True)

# Create a form for user input
with st.form("prediction_form"):
    st.write("Enter customer details to predict retention probability:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", 
                                    ["Electronic check", "Mailed check", 
                                     "Bank transfer (automatic)", "Credit card (automatic)"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        
    with col2:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        total_charges = st.number_input("Total Charges", min_value=0.0, value=tenure*monthly_charges)
        
    with col3:
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    
    submitted = st.form_submit_button("Predict Retention Probability")

if submitted:
    # Create a dataframe with the input data
    input_data = pd.DataFrame({
        'gender': ['Male'],  # Default value
        'SeniorCitizen': [senior_citizen],
        'Partner': ['No'],  # Default value
        'Dependents': ['No'],  # Default value
        'tenure': [tenure],
        'PhoneService': ['Yes'],  # Default value
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': ['No'],  # Default value
        'DeviceProtection': ['No'],  # Default value
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': ['No'],  # Default value
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })
    
    # Make prediction
    try:
        # Get the probability of retention (class 0)
        proba = pipeline.predict_proba(input_data)[0][0]
        retention_prob = round(proba * 100, 2)
        
        # Display results
        st.markdown(f"""
        <div style="background-color:#f0f2f6;padding:20px;border-radius:10px">
            <h3 style="color:#1f77b4">Prediction Results</h3>
            <p style="font-size:24px">
                Retention Probability: <strong>{retention_prob}%</strong>
            </p>
            <p style="font-size:18px;color:{"green" if retention_prob >= 50 else "red"}">
                This customer is <strong>{"likely" if retention_prob >= 50 else "unlikely"}</strong> to stay
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show feature importance for this prediction
        st.markdown("<h4>Key Factors Affecting This Prediction</h4>", unsafe_allow_html=True)
        
        # Get feature importance (fallback when SHAP is not available)
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': pipeline.named_steps['classifier'].feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        # Create plot with unique key
        fig = px.bar(importance_df, 
                     x='Importance', 
                     y='Feature', 
                     orientation='h',
                     title='Top 10 Important Features',
                     color='Importance')
        fig.update_layout(height=500)
        
        # Add unique key to prevent duplicate ID error
        st.plotly_chart(fig, use_container_width=True, key="prediction_feature_importance")
        
        st.info("Install the SHAP package (pip install shap) for more detailed explanations of individual predictions.")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")