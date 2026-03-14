# Ames Housing Price Predictor 🏠

An end-to-end Machine Learning project that predicts house prices in Ames, Iowa based on various property features. 

This project goes beyond a simple Jupyter Notebook and implements a full MLOps lifecycle, including experiment tracking, data versioning, API serving, an interactive UI, and continuous integration/deployment (CI/CD).

## 🛠️ Tech Stack
- **Algorithm**: `RandomForestRegressor` (scikit-learn)
- **Backend API**: `FastAPI`, `Uvicorn`, `Pydantic`
- **Frontend UI**: `Streamlit`
- **MLOps**: `MLflow` (Experiment Tracking), `DVC` (Data Versioning), `DagsHub` (Remote Storage), `GitHub Actions` (CI)
- **Deployment**: `Render.com`

---

## 🏃‍♂️ How to Run the Project Locally

### 1. Clone & Setup
Clone the repository and install the required dependencies inside a virtual environment:

```bash
git clone https://github.com/Ramine92/Ames-Housing-MLOPS.git
cd Ames-Housing-MLOPS/ml-project

# Create and activate virtual environment
python3 -m venv ml_project_env
source ml_project_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up Environment Variables
Create a `.env` file in the root of `ml-project/` to configure MLflow and DagsHub credentials (needed if you want to pull data or track experiments):

```env
# DagsHub/MLflow Credentials
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
```

### 3. Pull Data and Models
The actual dataset (`train.csv`) and the trained model (`.pkl`) are versioned using DVC and stored remotely on DagsHub. Pull them to your local machine:
```bash
dvc pull -r origin
```

### 4. Start the Application
You need to start both the backend API and the frontend UI in separate terminal windows (make sure your virtual environment is activated in both).

**Start the FastAPI Backend:**
```bash
uvicorn app.main:app --reload
```
*The API will be available at `http://localhost:8000`. You can view the interactive API docs at `http://localhost:8000/docs`.*

**Start the Streamlit UI:**
```bash
streamlit run ui.py
```
*The user interface will automatically open in your browser at `http://localhost:8501`.*

---

## 🔬 Training a New Model
If you want to tweak features or try a new algorithm, you can retrain the model.

1. Modify `ml/features/preprocessing.py` or edit the model algorithms in `ml/pipelines/train.py`.
2. Run the training script:
```bash
python -m ml.pipelines.train
```
3. The script will automatically log the experiment metrics (R², MSE, RMSE) and the model itself directly to **MLflow** hosted on DagsHub.
4. Push the new tracked model artifacts to DVC:
```bash
dvc add ml/models/artifacts/RandomForestRegressor_v1.pkl
dvc push
git add ml/models/artifacts/RandomForestRegressor_v1.pkl.dvc
git commit -m "Update model artifact"
git push
```

## 🚀 CI/CD Pipeline
- **Continuous Integration**: Every push to the `main` branch triggers a GitHub Action workflow (`.github/workflows/ci.yml`) that installs dependencies, pulls the latest model via DVC, and runs automated tests using `pytest`.
- **Continuous Deployment**: The FastAPI backend and Streamlit frontend are connected to Render.com and will automatically redeploy whenever the `main` branch is updated and passes CI.
