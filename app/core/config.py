import os
from dotenv import load_dotenv
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent.parent
load_dotenv(BASE_DIR/ ".env")
MODEL_NAME = os.getenv("MODEL_NAME","LinearRegression")
MODEL_VERSION = "v1"
API_URL = "localhost:8000"
MODEL_PATH = BASE_DIR / "ml" / "models" / "artifacts" / f"{MODEL_NAME}_{MODEL_VERSION}.pkl"