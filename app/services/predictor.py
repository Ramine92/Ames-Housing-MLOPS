import joblib
from app.core.config import MODEL_PATH
import numpy as np
import pandas as pd
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully: {type(model)}")
except Exception as e:
    print("Model could not be loaded")


def make_prediction(customer_data: dict):
    expected_columns = model.feature_names_in_
    input_data = pd.DataFrame(columns=expected_columns)
    input_data.loc[0] = [np.nan] * len(expected_columns)

    if "FirstFlrSF" in customer_data:
        customer_data["1stFlrSF"] = customer_data.pop("FirstFlrSF")
    if "SecondFlrSF" in customer_data:
        customer_data["2ndFlrSF"] = customer_data.pop("SecondFlrSF")
    for key,val in customer_data.items():
        if key in input_data.columns:
            input_data[key] = val
    
    if hasattr(model, "best_estimator_"):
        pipeline = model.best_estimator_
    else:
        pipeline = model
        
    categorical_cols = pipeline.named_steps['preprocessor'].transformers_[0][2]
    
    for col in input_data.columns:
        if col in categorical_cols:
            input_data[col] = input_data[col].astype("object")
        else:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
    
    log_prediction = model.predict(input_data)[0]
    actual_price = np.expm1(log_prediction)
    return round(actual_price,2)

