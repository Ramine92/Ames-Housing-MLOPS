from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_predict_endpoint_success():

    payload = {
    "Neighborhood": "CollgCr",
    "OverallQual": 7,
    "YearBuilt": 2003,
    "YrSold": 2008,
    "GrLivArea": 1710.0,
    "FullBath": 2,
    "Fireplaces": 1,
    "YearRemodAdd": 2003,
    "TotalBsmtSF": 856.0,
    "FirstFlrSF": 856.0,
    "SecondFlrSF": 854.0,
    "GarageCars": 2,
    "PoolQC": "missing",
    "BldgType": "1Fam"
    }


    response = client.post("/predict",json=payload)

    assert response.status_code == 200

    assert "predicted_price" in response.json()

    assert response.json()["currency"] == "USD"

def test_predict_returns_a_float():
    payload = {
    "Neighborhood": "CollgCr",
    "OverallQual": 7,
    "YearBuilt": 2003,
    "YrSold": 2008,
    "GrLivArea": 1710.0,
    "FullBath": 2,
    "Fireplaces": 1,
    "YearRemodAdd": 2003,
    "TotalBsmtSF": 856.0,
    "FirstFlrSF": 856.0,
    "SecondFlrSF": 854.0,
    "GarageCars": 2,
    "PoolQC": "missing",
    "BldgType": "1Fam"
    }

     
    response = client.post("/predict",json=payload)

    assert isinstance(response.json()["predicted_price"],float)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_missing_required_fields():
    payload= {}
    response = client.post("/predict",json=payload)

    assert response.status_code == 422


