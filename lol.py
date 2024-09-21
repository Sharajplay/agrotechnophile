import streamlit as st
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import base64
import gzip

hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Paths to the dataset files (replace with your actual paths)
price_data_path = 'price_dataset.csv'
yield_data_path = 'crop_yield_cleaned.csv'

# Load datasets
price_data = pd.read_csv(price_data_path)
yield_data = pd.read_csv(yield_data_path)

# Helper function to convert image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Paths to images (assuming they are in the main project folder)
logo_path = 'logo.jpg'
background_path = 'background.jpg'
home_image_path = 'home.png'

# Convert images to base64 strings
logo_base64 = get_base64_image(logo_path)
background_base64 = get_base64_image(background_path)

# Set background and logo globally
def set_background_and_logo():
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{background_base64}");
        background-size: cover;
        background-position: top left;
    }}

    .logo-container {{
        display: flex;
        justify-content: center;
        margin-top: 30px; /* Lower the logo */
        margin-bottom: 20px;
    }}

    .logo-container img {{
        border-radius: 50%;
        width: 120px;
        height: 120px;
    }}

    /* Lower the rest of the content */
    .content-container {{
        margin-top: 40px; /* Adjust the spacing to move the content down */
    }}

    footer {{
        text-align: center;
        font-size: 12px;
        font-weight: bold;
        padding: 10px;
        color: #777;
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(255, 255, 255, 0.8);
        z-index: 1000; /* Ensure the footer stays above other elements */
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Call function to set the background and logo
set_background_and_logo()

# Display the logo above the app selection box
st.markdown(
    f"""
    <div class="logo-container">
        <img src="data:image/jpg;base64,{logo_base64}" alt="logo">
    </div>
    """, unsafe_allow_html=True
)

# Wrap content in a container to adjust its positioning
st.markdown("<div class='content-container'>", unsafe_allow_html=True)

# Center the title using inline CSS
st.markdown("<h1 style='text-align: center;'>Agro Technophile</h1>", unsafe_allow_html=True)

# App selection (added Home as a default)
app_selection = st.selectbox("Choose an App To Run", ["Home", "Price Predictor - Main", "Yield Predictor - Feature"])

# Updated Price Predictor (Centering Crop Info)
if app_selection == "Price Predictor - Main":
    st.header("Crop Price Predictor")
    
    state = st.selectbox('Select State : ', price_data['state'].unique())
    crop = st.selectbox('Select Crop : ', price_data['crop'].unique())
    month = st.selectbox('Select Month You Want To Predict : ', ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])

    if st.button('Predict Price'):
        model_path = 'price_model.pkl'
        
        if not os.path.exists(model_path):
            # Train and save the price predictor model if it doesn't exist
            X = price_data[['state', 'crop']]  # Exclude 'month' from the dataset
            y = price_data['price']
            
            # Preprocessing: OneHotEncoding categorical columns + adding 'month' as categorical from user input
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(), ['state', 'crop'])
                ], remainder='passthrough')

            # Model pipeline
            model_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
            ])

            # Train the model
            model_pipeline.fit(X, y)
            
            # Save the model
            joblib.dump(model_pipeline, model_path)
        else:
            model_pipeline = joblib.load(model_path)
        
        # Predicting based on user input for 'state', 'crop', and 'month'
        input_data = pd.DataFrame([[state, crop]], columns=['state', 'crop'])  # Exclude 'month' in DataFrame
        prediction = model_pipeline.predict(input_data)[0]
        st.success(f"The predicted price is ₹{round(prediction, 2)}")
        
        # Additional information based on the selected crop
        crop_info = {
            'onion': "Onion - Best Soil: Well-draining sandy loam\n- Best Fertilizer: NPK (10:20:10) + organic compost\n- Best Month: September-October (cool, dry season)",
            'potato': "Potato - Best Soil: Cool, moist sandy loam\n- Best Fertilizer: NPK (15:15:15) + potassium nitrate\n- Best Month: January-February (cool season)",
            'tomato': "Tomato - Best Soil: Warm, well-draining loamy\n- Best Fertilizer: NPK (20:20:20) + calcium nitrate\n- Best Month: March-April (warm season)",
            'peas wet': "Peas Wet - Best Soil: Cool, moist clay loam\n- Best Fertilizer: NPK (10:20:10) + ammonium sulfate\n- Best Month: October-November (cool season)",
            'ginger': "Ginger - Best Soil: Warm, well-draining sandy loam\n- Best Fertilizer: Organic compost + NPK (10:10:10)\n- Best Month: April-May (warm season)",
            'apple': "Apple - Best Soil: Well-draining loamy\n- Best Fertilizer: NPK (20:20:20) + organic compost\n- Best Month: February-March (cool season)",
            'mango': "Mango - Best Soil: Warm, well-draining sandy loam\n- Best Fertilizer: Organic compost + NPK (15:15:15)\n- Best Month: May-June (warm season)"
        }

        # Center and print crop-specific information
        if crop in crop_info:
            st.markdown(f"<div style='text-align: center;'>{crop_info[crop]}</div>", unsafe_allow_html=True)

# Updated Yield Predictor (Centering Yield Info)
if app_selection == "Yield Predictor - Feature":
    st.header("Crop Yield Predictor")
    
    state = st.selectbox('Select State : ', yield_data['state'].unique())
    crop = st.selectbox('Select Crop :', yield_data['crop'].unique())
    season = st.selectbox('Select Season :', yield_data['season'].unique())
    rainfall = st.number_input('Enter Rainfall (mm) :', min_value=0.0, value=0.0)
    area = st.number_input('Enter Area (Hectares) :', min_value=0.0, value=0.0)
    fertilizer = st.number_input('Enter Fertilizer Used (kg) :', min_value=0.0, value=0.0)
    pesticide = st.number_input('Enter Pesticide Used (litres) :', min_value=0.0, value=0.0)
    
    if st.button('Predict Yield'):
        compressed_model_path = 'yield_model_compressed.pkl.gz'

        # Load the compressed model
        with gzip.open(compressed_model_path, 'rb') as f_in:
            model_pipeline = joblib.load(f_in)
        
        # Validate inputs before predicting
        if rainfall > 0 and area > 0 and fertilizer > 0 and pesticide > 0:
            input_data = pd.DataFrame([[state, crop, season, rainfall, area, fertilizer, pesticide]], 
                                      columns=['state', 'crop', 'season', 'rainfall', 'area', 'fertilizer', 'pesticide'])
            prediction = model_pipeline.predict(input_data)[0]
            st.success(f"The predicted yield is {round(prediction, 2)} tons per hectare")
            
            # Center and display yield information
            st.markdown(""" 
            <div style='text-align: center;'>
            Yield Information :<br>
            1. Hectare (ha): A unit of land area, equal to 10,000 square meters or 2.47 acres.<br>
            2. Annual Rainfall: Total rainfall in a year, usually measured in millimeters or inches.<br>
            3. Fertilizer: Substances added to soil to promote plant growth and fertility.<br>
            4. Pesticide: Chemicals used to control pests and diseases affecting crops.<br>
            5. Yield: Amount of crop produced on a given area of land, usually measured in tons per hectare.
            </div>
            """, unsafe_allow_html=True)

# Home app (default view)
if app_selection == "Home":
    # Display the welcome message and home image
    st.markdown("<h2 style='text-align: center;'>Welcome!  Thanks for your visit :)</h2>", unsafe_allow_html=True)
    st.image(home_image_path, use_column_width=True)
    
    # Display the About section
    st.markdown("""
    <h3 style='text-align: center;'>About :</h3>
    <p style='text-align: justify;'>
            Our predictors integrate historical data, machine learning models, and fine-tuning techniques to offer a reliable system for crop price prediction and crop selection tailored to farmer needs. 
    This system has the potential to address significant challenges in agriculture by optimizing decision-making and improving productivity.
    </p>
    <p style='text-align: justify;'>
    <strong>Our predictors benefit :</strong><br>
    Horticulture crop price forecasting helps farmers plan planting and harvesting, reducing market risk. 
    It aligns supply with demand, preventing wastage. Accurate forecasts improve food security, fair trading, and reduce intermediaries’ influence. 
    They encourage investment in agriculture by reducing uncertainty and promoting sustainability and rural development.
    </p>
    """, unsafe_allow_html=True)

# Close content container div
st.markdown("</div>", unsafe_allow_html=True)

# Add footer
st.markdown("""
<footer>
    @2024 - Agro Technophile - Data's are based on August 2024
</footer>
""", unsafe_allow_html=True)
