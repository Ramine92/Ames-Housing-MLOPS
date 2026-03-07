from fastapi import FastAPI,APIRouter
from app.api.routes import health,predict


app = FastAPI(title="Ames-Housing Prediction")

@app.get("/")
def root():
    return {"message":"Welcome To Ml Ames-Housing Project"}

app.include_router(router=health.router)
app.include_router(router=predict.router)