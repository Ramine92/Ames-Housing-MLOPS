from fastapi import APIRouter
from app.core.config import MODEL_NAME,MODEL_VERSION
router = APIRouter()

@router.get("/health")
def entry():
    return {"status":"ok","model_name":f"{MODEL_NAME}_{MODEL_VERSION}"}