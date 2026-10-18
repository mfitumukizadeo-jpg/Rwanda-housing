import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_artifacts():
    model = joblib.load('house_price_model.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    model_features = joblib.load('model_features.pkl')
    return model, scaler, model_features

model, scaler, model_features = load_artifacts()

numerical_cols = [
    'Area_m2', 'Bedrooms', 'Bathrooms', 
    'House_Age_Years', 'Distance_to_City_km', 'Parking_Spaces'
]

st.title("🏡 Rwanda House Price Predictor")
st.write("Enter the property details below to estimate its market price in Million RWF.")

area = st.number_input("Area (m²)", min_value=10.0, max_value=1000.0, value=120.0, step=5.0)
bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.slider("Number of Bathrooms", min_value=1, max_value=8, value=2)
age = st.number_input("House Age (Years)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
distance = st.number_input("Distance to City Center (km)", min_value=0.1, max_value=100.0, value=5.0, step=0.5)
parking = st.slider("Parking Spaces", min_value=0, max_value=5, value=1)

neighborhood_options = ['Kigali City', 'Nyarugenge', 'Gasabo', 'Kicukiro', 'Huye', 'Musanze']
selected_neighborhood = st.selectbox("Select Neighborhood", neighborhood_options)

if st.button("Predict House Price"):
    input_data = pd.DataFrame([{
        'Area_m2': area,
        'Bedrooms': float(bedrooms),
        'Bathrooms': float(bathrooms),
        'House_Age_Years': age,
        'Distance_to_City_km': distance,
        'Parking_Spaces': float(parking),
        'Neighborhood': selected_neighborhood
    }])
    
    input_encoded = pd.get_dummies(input_data, columns=['Neighborhood'], dtype=int)
    input_aligned = input_encoded.reindex(columns=model_features, fill_value=0)
    input_aligned[numerical_cols] = scaler.transform(input_aligned[numerical_cols])
    
    predicted_price = model.predict(input_aligned)[0]
    st.success(f"🎉 Estimated House Price: **{predicted_price:,.2f} Million RWF**")