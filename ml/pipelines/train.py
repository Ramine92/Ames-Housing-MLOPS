from sklearn.model_selection import GridSearchCV,train_test_split,cross_validate
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from pathlib import Path
import pandas as pd 


BASE_DIR = Path(__file__).parent.parent # root directory

def load_data():
    df = pd.read_csv("../data/raw/train.csv")
    X = df.drop(columns=["Log_SalePrice","SalePrice"])
    y = df["Log_SalePrice"]
    return X,y

def get_pipeline(X,model_name):
    categorical_columns = X.select_dtypes(include="object").columns
    numerical_columns = X.select_dtypes(exclude="object").columns

    categorical_transformer = Pipeline(steps=[
        ("one_hot",OneHotEncoder(handle_unknown="ignore")),
    ])
    numerical_transformer = Pipeline(steps=[
        ("scaler",StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("cat_transformer",categorical_transformer,categorical_columns),
        ("num_transformer",numerical_transformer,numerical_columns)
    ])
    if model_name == "Ridge":
        pipeline = Pipeline(steps=[
            ("preprocessor",preprocessor),
            ("model",Ridge())
        ])
    pipeline = Pipeline(steps=[
        ("preprocessor",preprocessor),
        ("model",Ridge())
    ])

    return pipeline

def train_model(X,y,model):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    grid_params = {
        "model__alpha":[0.1,1.0,10.0,50.0,100.0,200.0]
    }
    pipeline = get_pipeline(X_train,model)
    if model == "Ridge":
        gs_pipeline = GridSearchCV(estimator=pipeline,grid_params=grid_params,cv=5,n_jobs=-1,scoring="neg_root_mean_squared_error")
        gs_pipeline.fit(X_train,y_train)
    else:
        pipeline.fit(X_train,y_train)

    scoring = ["r2_score","mean_squared_error","root_mean_squared_error"]
    cv_results = cross_validate(gs_pipeline,X_train,y_train,cv=5,scoring=scoring,return_train_score=True)
    results = {"r2_score":cv_results["r2_score"].mean(),"mean_squared_error":cv_results["mean_squared_error"].mean(),"root_mean_squared_error":cv_results["root_mean_squared_error"].mean()}
    return gs_pipeline if model == "Ridge" else pipeline
    




