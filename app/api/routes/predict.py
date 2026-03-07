from fastapi import APIRouter,HTTPException
from app.api.schemas import PredictionRequest,PredictionResponse
from app.services.predictor import make_prediction
router = APIRouter()

@router.post("/predict",response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        prediction = make_prediction(request.model_dump())
        return PredictionResponse(predicted_price=prediction)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")
        

