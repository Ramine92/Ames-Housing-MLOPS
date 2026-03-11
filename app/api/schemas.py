from pydantic import BaseModel 
from typing import Optional

class PredictionRequest(BaseModel):
    Neighborhood: str
    OverallQual: int
    YearBuilt: int
    YrSold: int
    GrLivArea: float
    FullBath: int
    Fireplaces: int
    YearRemodAdd: int
    TotalBsmtSF: float
    FirstFlrSF: float
    SecondFlrSF: float
    GarageCars: int
    PoolQC: Optional[str] = "missing"
    BldgType: Optional[str] = "1Fam"

class PredictionResponse(BaseModel):
    predicted_price : float
    currency: str= "USD"


