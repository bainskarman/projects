import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

# Load models and data
@st.cache_resource
def load_models_and_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load models
    model = load(os.path.join(script_dir, "fertilizer_recommendation.joblib"))
    scaler = load(os.path.join(script_dir, "fertilizer_scaler.joblib"))
    fertilizer_encoder = load(os.path.join(script_dir, "fertilizer_encoder.joblib"))
    crop_encoder = load(os.path.join(script_dir, "crop_encoder.joblib"))
    feature_names = load(os.path.join(script_dir, "fertilizer_feature_names.joblib"))
    
    # Load dataset for visualization
    data_path = os.path.join(script_dir, "fertilizer_recommendation_dataset.csv")
    df = pd.read_csv(data_path)
    
    return model, scaler, fertilizer_encoder, crop_encoder, feature_names, df

model, scaler, fertilizer_encoder, crop_encoder, feature_names, df = load_models_and_data()

# Streamlit App
st.title("🌱 Fertilizer Recommendation System")

# Input Section
st.header("🔍 Input Parameters")
col1, col2 = st.columns(2)

with col1:
    temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
    moisture = st.slider("Moisture", 0.0, 1.0, 0.7)
    rainfall = st.slider("Rainfall (mm)", 0.0, 500.0, 200.0)
    ph = st.slider("Soil pH", 0.0, 14.0, 7.0)

with col2:
    nitrogen = st.slider("Nitrogen (kg/ha)", 0.0, 200.0, 70.0)
    phosphorous = st.slider("Phosphorous (kg/ha)", 0.0, 200.0, 80.0)
    potassium = st.slider("Potassium (kg/ha)", 0.0, 200.0, 100.0)
    carbon = st.slider("Carbon (%)", 0.0, 10.0, 1.0)

crop = st.selectbox("Crop", options=crop_encoder.classes_)
soil_type = st.selectbox("Soil Type", ['Acidic', 'Alkaline', 'Loamy', 'Neutral', 'Peaty'])

# Prediction Button
if st.button("🚀 Recommend Fertilizer"):
    # Prepare input
    soil_mapping = {
        'Acidic': [1, 0, 0, 0, 0],
        'Alkaline': [0, 1, 0, 0, 0],
        'Loamy': [0, 0, 1, 0, 0],
        'Neutral': [0, 0, 0, 1, 0],
        'Peaty': [0, 0, 0, 0, 1]
    }
    
    input_data = {
        'Temperature': [temperature],
        'Moisture': [moisture],
        'Rainfall': [rainfall],
        'PH': [ph],
        'Nitrogen': [nitrogen],
        'Phosphorous': [phosphorous],
        'Potassium': [potassium],
        'Carbon': [carbon],
        'Crops': [crop],
        'Acidic_Soil': [soil_mapping[soil_type][0]],
        'Alkaline_Soil': [soil_mapping[soil_type][1]],
        'Loamy_Soil': [soil_mapping[soil_type][2]],
        'Neutral_Soil': [soil_mapping[soil_type][3]],
        'Peaty_Soil': [soil_mapping[soil_type][4]]
    }
    
    input_df = pd.DataFrame(input_data)
    input_df['Crops'] = crop_encoder.transform(input_df['Crops'])
    input_df = input_df[feature_names]
    
    # Predict
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    recommended_fertilizer = fertilizer_encoder.inverse_transform(prediction)[0]
    
    # Display Results
    st.success(f"**Recommended Fertilizer:** `{recommended_fertilizer}` (Encoded: `{prediction[0]}`)")

    # Get remarks for the predicted fertilizer (using 'Fertilizer' column)
    remarks = df[df['Fertilizer'] == recommended_fertilizer]['Remark'].unique()
    
    if len(remarks) > 0:
        st.subheader("📝 Remarks for Recommended Fertilizer")
        for remark in remarks:
            st.write(f"- {remark}")
    else:
        st.warning("No remarks found for this fertilizer in the dataset.")

    # Visualization: Compare distributions
    st.header("📊 Predicted Fertilizer Deficit vs Other Deficieny")
    
    # Filter data
    predicted_data = df[df['Fertilizer'] == recommended_fertilizer]
    other_data = df[df['Fertilizer'] != recommended_fertilizer]
    
    # Select features to plot
    features_to_plot = ['Temperature', 'Moisture', 'Rainfall', 'PH', 
                       'Nitrogen', 'Phosphorous', 'Potassium', 'Carbon']
    
    # Plot distributions
    for feature in features_to_plot:
        plt.figure(figsize=(10, 4))
        sns.kdeplot(predicted_data[feature], label=f'Predicted ({recommended_fertilizer})', fill=True)
        sns.kdeplot(other_data[feature], label='Other Fertilizers', fill=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        st.pyplot(plt)
        plt.close()