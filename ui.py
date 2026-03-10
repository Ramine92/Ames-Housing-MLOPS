from matplotlib.pyplot import step
import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Ames Housing Price Predictor",page_icon="🏠",layout="wide")
st.title("🏠 Ames Housing Price Predictor")
col1,col2 = st.columns([1,1.5])
# Neighborhood coordinates (Ames, Iowa approximate centroids)
NEIGHBORHOOD_COORDS = {
    # Original neighborhoods from your request
    "NAmes": (42.034722, -93.620278),      # North Ames general area
    "CollgCr": (42.021328, -93.685522),    # College Creek
    "OldTown": (42.028890, -93.613330),    # Old Town Historic District
    "Edwards": (42.048550, -93.694489),    # Edwards neighborhood (50014 area)
    "Somerst": (42.033610, -93.588270),    # Somerset (50010 area)
    
    # Additional neighborhoods
    "ParkviewHeights": (42.038056, -93.619167),   # Parkview Heights area
    "Allenview": (42.036111, -93.625000),         # Allenview area
    "BloomingtonHts": (42.039722, -93.590278),    # Bloomington Heights
    "Campustown": (42.023333, -93.646389),        # South of ISU campus
    "CollegeCreek": (42.021111, -93.685556),      # Alternate/expanded College Creek
    "CollegeHeights": (42.025556, -93.609722),    # College Heights
    "CollegePark": (42.020000, -93.603889),       # College Park
    "ColonialVillage": (42.035000, -93.640556),   # Colonial Village
    "CountryGables": (42.031389, -93.665278),     # Country Gables
    "Dauntless": (42.031944, -93.602222),         # Dauntless area
    "DaytonPark": (42.033889, -93.593889),        # Dayton Park
    "EastHickoryPark": (42.043333, -93.588611),   # East Hickory Park
    "EstatesWest": (42.037222, -93.671944),       # Estates West
    "GatewayGreenHills": (42.018889, -93.651389), # Gateway Green Hills
    "GatewayHills": (42.019444, -93.649444),      # Gateway Hills
    "Hillside": (42.031111, -93.615278),          # Hillside area
    "ISU": (42.026667, -93.646944),               # Iowa State University
    "LittleHollywood": (42.029722, -93.628333),   # Little Hollywood
    "MainStreet": (42.027222, -93.615278),        # Main Street Cultural District
    "MelrosePark": (42.034722, -93.650000),       # Melrose Park
    "NorthridgeHeights": (42.047222, -93.619167), # Northridge Heights
    "NorthridgeParkway": (42.045556, -93.618333), # Northridge Parkway
    "Oakwood": (42.039167, -93.605556),           # Oakwood (50010 area)
    "OntarioHeights": (42.041389, -93.628889),    # Ontario Heights
    "Ridgewood": (42.036111, -93.632222),         # Ridgewood
    "RinggenbergPark": (42.044444, -93.623611),   # Ringgenberg Park
    "Roosevelt": (42.030583, -93.623639),         # Roosevelt neighborhood (near school)
    "SouthFork": (42.016944, -93.608889),         # South Fork
    "SouthGateway": (42.018056, -93.643333),      # South Gateway
    "SpringValley": (42.034167, -93.676111),      # Spring Valley
    "StoneBrooke": (42.036944, -93.581944),       # Stone Brooke (upscale area)
    "Suncrest": (42.032778, -93.623889),          # Suncrest
    "SunsetRidge": (42.043889, -93.610000),       # Sunset Ridge
    "TheReserve": (42.026111, -93.664444),        # The Reserve
    "WestAmes": (42.024000, -93.663000),          # West Ames (50014 centroid)
    
    # General area descriptors
    "Downtown": (42.027222, -93.615278),          # Downtown Ames
    "NorthAmes": (42.034722, -93.620278),         # North Ames general
    "SouthAmes": (42.018889, -93.635556),         # South Ames general
    "WestAmesGeneral": (42.024000, -93.663000),   # West Ames general
}
with col1:
    OverallQuall = st.slider("Overall Quality (1-10)",min_value=1,max_value=10,value=5)

    YearBuilt = st.number_input("Year Built",min_value=1872,max_value=2010,value=2000,step=1)

    TotalBsmtSF = st.number_input("Total Basement SF",min_value=0.0,value=800.0)

    FirstFlrSF = st.number_input("First Floor Square Ft",min_value=0.0,value=400.0)

    SecondFlrSF = st.number_input("Second Floor Square Ft",min_value=0.0,value=400.0)

    BldgType = st.selectbox("Building Type",["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"])

    GarageCars = st.slider("Number Of Cars",min_value=0,max_value=5)

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
            "YearBuilt":YearBuilt,
            "TotalBsmtSF":TotalBsmtSF,
            "FirstFlrSF":FirstFlrSF,
            "SecondFlrSF":SecondFlrSF,
            "GarageCars":GarageCars,
            "PoolQC":PoolQC,
            "BldgType":BldgType,
        }
        with st.spinner("Predicting..."):
            response = requests.post("http://localhost:8000/predict",json=payload)

        if response.status_code == 200:
            price = response.json()["predicted_price"]
            st.success(f"💰 Estimated Price: **${price:,.0f} USD**")
        
        else:
            st.error(f"Error {response.status_code}")