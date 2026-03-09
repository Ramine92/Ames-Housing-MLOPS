from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
def test_make_prediction_returns_float():
    payload = {
        "Neighborhood":"CollgCr",
        "OverallQual":7,
        "YearBuilt":2003,
        "TotalBsmtSF": 856.0,
        "FirstFlrSF": 856.0,
        "SecondFlrSF": 854.0,
        "GarageCars": 2
    }
    response = client.post("/predict",json=payload)

    assert response.status_code == 200

    assert isinstance(response.json()["predicted_price"],float)

def test_make_prediction_positive_price():
    payload = {
        "Neighborhood":"CollgCr",
        "OverallQual":7,
        "YearBuilt":2003,
        "TotalBsmtSF": 856.0,
        "FirstFlrSF": 856.0,
        "SecondFlrSF": 854.0,
        "GarageCars": 2
    }
    response = client.post("/predict",json=payload)

    assert response.status_code == 200

    assert response.json()["predicted_price"] > 0

