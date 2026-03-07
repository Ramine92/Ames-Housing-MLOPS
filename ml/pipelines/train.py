from pandas.core.common import random_state
from sklearn.model_selection import GridSearchCV,train_test_split,cross_validate
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import r2_score,mean_squared_error
from app.core.config import MODEL_NAME,MODEL_VERSION
from ml.features.preprocessing import housing_preprocessor
import mlflow
from pathlib import Path
import numpy as np
import joblib
import pandas as pd 


BASE_DIR = Path(__file__).parent.parent.parent # root directory
DATA_PATH = BASE_DIR / "ml" / "data" / "raw" / "train.csv" 

def load_data(data_path=DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    
    if "Log_SalePrice" not in df.columns:
        X = df.drop(columns=["SalePrice"])
        y = np.log1p(df["SalePrice"])
    else:
        X = df.drop(columns=["Log_SalePrice","SalePrice"])
        y = df["Log_SalePrice"]
    return X,y

def get_pipeline(X,model_name):

    X_engineered = housing_preprocessor.fit_transform(X)
    categorical_columns = X_engineered.select_dtypes(include="object").columns
    numerical_columns = X_engineered.select_dtypes(exclude="object").columns

    categorical_transformer = Pipeline(steps=[
        ("cat_imputer",SimpleImputer(strategy="constant",fill_value="missing")),
        ("one_hot",OneHotEncoder(handle_unknown="ignore"))
        
    ])
    numerical_transformer = Pipeline(steps=[
        ("num_imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
        
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("cat_transformer",categorical_transformer,categorical_columns),
        ("num_transformer",numerical_transformer,numerical_columns)
    ])
    if model_name == "Ridge":
        model = Ridge(random_state=42)
    elif model_name == "LinearRegression":
        model = LinearRegression()
    
    pipeline = Pipeline(steps=[
        ("feature_engineering", housing_preprocessor),
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    return pipeline

def train_model(X,y,model_name):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    grid_params = {
        "model__alpha":[0.1,1.0,10.0,50.0,100.0,200.0]
    }
    pipeline = get_pipeline(X_train,model_name)
    if model_name == "Ridge":
        model = GridSearchCV(estimator=pipeline,param_grid=grid_params,cv=5,n_jobs=-1,scoring="neg_root_mean_squared_error")
        model.fit(X_train,y_train)
    else:
        model = pipeline
        model.fit(X_train,y_train)
    
    return model,X_test,y_test

def evaluate_model(model,X_test,y_test):
    y_preds = model.predict(X_test)
    r2_scoring = r2_score(y_test,y_preds)
    mse = mean_squared_error(y_test,y_preds)
    rmse = np.sqrt(mse)
    results = {"r2_score":r2_scoring,"mean_squared_error":mse,"root_mean_squared_error":rmse}
    return results

def save_model(model,model_name,version):
    artifacts_path = BASE_DIR / "ml" / "models" / "artifacts"
    artifacts_path.mkdir(parents=True,exist_ok=True)
    joblib.dump(model, artifacts_path / f"{model_name}_{version}.pkl")
    print(f"model saved succefully {model_name}")
if __name__ == "__main__":
    mlflow.set_tracking_uri("file://"+str(BASE_DIR/"mlruns"))
    mlflow.set_experiment("Ames_Housing_Regression")
    X,y = load_data(DATA_PATH)
    with mlflow.start_run(run_name=f"{MODEL_NAME}_{MODEL_VERSION}"):
        model,X_test,y_test = train_model(X,y,MODEL_NAME)
        rs = evaluate_model(model,X_test,y_test)
        print(rs)
        mlflow.log_metric("R2_score",rs["r2_score"])
        mlflow.log_metric("MSE",rs["mean_squared_error"])
        mlflow.log_metric("RMSE",rs["root_mean_squared_error"])
        if hasattr(model,"best_params_"):
            best_alpha = model.best_params_["model__alpha"]
            mlflow.log_param("best_alpha",best_alpha)
        mlflow.sklearn.log_model(model,f"{MODEL_NAME}")
        save_model(model,MODEL_NAME,MODEL_VERSION)




    




