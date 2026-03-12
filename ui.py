import streamlit as st
from app.core.config import API_URL
import pandas as pd
import requests

st.set_page_config(page_title="Ames Housing Price Predictor",page_icon="🏠",layout="wide")
st.title("🏠 Ames Housing Price Predictor")
col1,col2 = st.columns([1,1.5])
# Real Ames Housing dataset neighborhood codes with Ames, Iowa coordinates
NEIGHBORHOOD_COORDS = {
    "NAmes":    (42.0347, -93.6203),   # North Ames
    "CollgCr":  (42.0213, -93.6855),   # College Creek
    "OldTown":  (42.0289, -93.6133),   # Old Town
    "Edwards":  (42.0275, -93.6611),   # Edwards
    "Somerst":  (42.0611, -93.6189),   # Somerset
    "NridgHt":  (42.0533, -93.6561),   # Northridge Heights  
    "Gilbert":  (42.1100, -93.6480),   # Gilbert
    "Sawyer":   (42.0347, -93.6525),   # Sawyer
    "NWAmes":   (42.0500, -93.6650),   # Northwest Ames
    "SawyerW":  (42.0347, -93.6700),   # Sawyer West
    "BrkSide":  (42.0219, -93.6078),   # Brookside
    "Crawfor":  (42.0244, -93.5986),   # Crawford
    "Mitchel":  (42.0100, -93.6100),   # Mitchell
    "NoRidge":  (42.0500, -93.6400),   # Northridge           
    "Timber":   (42.0019, -93.6572),   # Timberland
    "IDOTRR":   (42.0239, -93.6214),   # Iowa DOT & Rail Road
    "ClearCr":  (42.0083, -93.6761),   # Clear Creek
    "StoneBr":  (42.0608, -93.6461),   # Stone Brook           
    "SWISU":    (42.0161, -93.6597),   # SW of Iowa State
    "Blmngtn":  (42.0597, -93.6161),   # Bloomington Heights
    "MeadowV":  (41.9939, -93.6564),   # Meadow Village
    "BrDale":   (42.0572, -93.6022),   # Briardale
    "Veenker":  (42.0444, -93.6486),   # Veenker              
    "NPkVill":  (42.0572, -93.6089),   # Northpark Village
    "Blueste":  (42.0211, -93.6339),   # Bluestem
}
with col1:
    OverallQuall = st.slider("Overall Quality (1-10)",min_value=1,max_value=10,value=5)

    YearBuilt = st.number_input("Year Built",min_value=1872,max_value=2010,value=2000,step=1)

    YrSold = st.number_input("Year Sold",value=2026)

    YearRemodAdd = st.number_input("Year Since Remodel",min_value=1900)

    TotalBsmtSF = st.number_input("Total Basement SF",min_value=0.0,value=800.0)

    FirstFlrSF = st.number_input("First Floor Square Ft",min_value=0.0,value=400.0)

    SecondFlrSF = st.number_input("Second Floor Square Ft",min_value=0.0,value=400.0)

    BldgType = st.selectbox("Building Type",["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"])

    GarageCars = st.slider("Number Of Cars",min_value=0,max_value=5)

    GrLivArea = st.number_input("Above Ground Living Area (sqft)",min_value=0.0,value=1500.0)

    FullBath = st.number_input("Full Bathrooms",min_value=0,max_value=6,value=2,step=1)

    Fireplaces = st.number_input("Fireplaces",min_value=0,max_value=5,value=0,step=1)

    PoolQC = st.selectbox("Pool Quality",["missing","Ex","Gd","TA","Fa"])

    Neighborhood = st.selectbox("Neighborhood",list(NEIGHBORHOOD_COORDS.keys()))

with col2:

    if Neighborhood in NEIGHBORHOOD_COORDS:
        lat,lon = NEIGHBORHOOD_COORDS[Neighborhood]
        map_data = pd.DataFrame({"lat":[lat],"lon":[lon]})
        st.map(map_data,zoom=13)
    else:
        st.warning("Neighborhood not recognized - map unavailable")

    if st.button("🔍 Predict Price"):
        payload={
            "Neighborhood":Neighborhood,
            "OverallQual":OverallQuall,
            "YrSold":YrSold,
            "YearRemodAdd":YearRemodAdd,
            "YearBuilt":YearBuilt,
            "TotalBsmtSF":TotalBsmtSF,
            "FirstFlrSF":FirstFlrSF,
            "SecondFlrSF":SecondFlrSF,
            "GarageCars":GarageCars,
            "GrLivArea":GrLivArea,
            "FullBath":FullBath,
            "Fireplaces":Fireplaces,
            "PoolQC":PoolQC,
            "BldgType":BldgType,
        }
        with st.spinner("Predicting..."):
            response = requests.post(f"{API_URL}/predict",json=payload)

        if response.status_code == 200:
            price = response.json()["predicted_price"]
            st.success(f"💰 Estimated Price: **${price:,.0f} USD**")
        
        else:
            st.error(f"Error {response.status_code}")