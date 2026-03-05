import numpy as np 
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

def _clean_and_enginner(X):
    df = X.copy()
    cols_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                 'BsmtFinType2', 'MasVnrType']
    
    cols_zero = ['GarageYrBlt', 'GarageArea', 'GarageCars', 
                 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 
                 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
    
    cols_mode = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 
                 'Exterior2nd', 'SaleType']

    for col in cols_none:
        if col in df.columns:
            df[col] = df[col].fillna("missing")

    for col in cols_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    for col in cols_mode:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

    # --- FEATURE ENGINEERING ---
    if all(col in df.columns for col in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]):
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    if all(col in df.columns for col in ["YrSold", "YearBuilt", "YearRemodAdd"]):
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
        df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]

    # Drop old redundant columns
    cols_to_drop = ["GarageArea", "1stFlrSF", "2ndFlrSF", "TotalBsmtSF", "YearBuilt", "YrSold", "YearRemodAdd"]
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True, errors='ignore')

    return df

housing_preprocessor = FunctionTransformer(func=_clean_and_enginner, validate=False)