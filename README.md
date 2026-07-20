# 🍷 End-to-End MLOps Wine Quality Prediction Project

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-orange.svg)](https://mlflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ElasticNet-green.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-red.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/GitHub%20Actions-Continuous%20Deployment-black.svg)](https://github.com/features/actions)

## 📌 Executive Summary
Predicting wine quality based on physico-chemical characteristics is a classic tabular regression problem with high business impact in manufacturing quality control. Traditional batch model deployment often suffers from data drift, lack of reproducibility, and manual configuration overhead.

This project delivers an **industrial-grade, modular MLOps pipeline** that automates the entire machine learning lifecycle—from raw data ingestion and strict schema validation to hyperparameter tracking with **MLflow**, serving real-time predictions via a **Flask web dashboard**, containerizing with **Docker**, and continuous deployment via **GitHub Actions** to AWS.

---

## 🏗️ System Architecture & Workflow

```
[ Raw Dataset URL ] ──> 1. Data Ingestion ──> Extract Raw Data (.csv)
                                                     │
                                                     ▼
                                            2. Data Validation
                                         (Validates vs schema.yaml)
                                                     │
                                                     ▼
                                            3. Data Transformation
                                          (Train / Test 80-20 Split)
                                                     │
                                                     ▼
                                            4. Model Trainer
                                       (Trains ElasticNet Regressor)
                                                     │
                                                     ▼
                                            5. Model Evaluation
                                       (Logs Metrics & Models to MLflow)
                                                     │
                                                     ▼
                                       [ Flask Web Server & UI ]
```

---

## 🛠️ Tech Stack & Key Tools
- **Core Language**: Python 3.10+
- **Machine Learning**: Scikit-Learn (ElasticNet Regression)
- **MLOps & Tracking**: MLflow (Parameters, Metrics, Model Artifact Registry)
- **Data Engineering**: Pandas, NumPy, Python-Box, PyYAML
- **Web Application**: Flask, HTML5, Modern CSS (Glassmorphism Dashboard UI)
- **DevOps & Containerization**: Docker, Docker Hub, AWS ECR, AWS EC2
- **CI/CD Automation**: GitHub Actions (`.github/workflows/main.yaml`)

---

## 📂 Project Structure

```
├── .github/workflows/       # CI/CD deployment workflow file
│   └── main.yaml
├── config/
│   └── config.yaml          # Pipeline stage artifact directory paths
├── research/
│   └── trials.ipynb         # Experimental notebook
├── src/mlProject/
│   ├── components/          # Pipeline modular components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   ├── config/              # Configuration manager class
│   │   └── configuration.py
│   ├── constants/           # Path constants definitions
│   │   └── __init__.py
│   ├── entity/              # DataClass entity configuration schemas
│   │   └── config_entity.py
│   ├── pipeline/            # Stage runners & serving prediction pipeline
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_data_validation.py
│   │   ├── stage_03_data_transformation.py
│   │   ├── stage_04_model_trainer.py
│   │   ├── stage_05_model_evaluation.py
│   │   └── prediction.py
│   └── utils/               # Common file I/O and annotation helpers
│       └── common.py
├── templates/
│   └── index.html           # Dark glassmorphism web UI
├── app.py                   # Flask server application
├── main.py                  # End-to-end pipeline execution entrypoint
├── Dockerfile               # Container build instructions
├── params.yaml              # Hyperparameters (alpha, l1_ratio)
├── schema.yaml              # Dataset column specifications and data types
├── requirements.txt         # Project dependencies
├── setup.py                 # Setuptools package setup script
└── template.py              # Automated directory structure generator
```

---

## 📊 Pipeline Stages Overview

### Stage 01: Data Ingestion
- Automatically fetches zipped Red Wine Quality dataset from remote repository.
- Extracts `winequality-red.csv` into `artifacts/data_ingestion/`.

### Stage 02: Data Validation
- Reads column specifications from `schema.yaml`.
- Checks column presence and data type integrity.
- Outputs status report to `artifacts/data_validation/status.txt`.

### Stage 03: Data Transformation
- Verifies validation status (`status.txt == True`).
- Performs an 80/20 train/test split.
- Exports `train.csv` (1,279 rows) and `test.csv` (320 rows) into `artifacts/data_transformation/`.

### Stage 04: Model Trainer
- Reads hyperparameters (`alpha: 0.2`, `l1_ratio: 0.1`) from `params.yaml`.
- Fits an `ElasticNet` regression model on training data.
- Saves the trained model artifact to `artifacts/model_trainer/model.joblib`.

### Stage 05: Model Evaluation with MLflow
- Evaluates model performance on the test split.
- Computes **RMSE** (Root Mean Squared Error), **MAE** (Mean Absolute Error), and **R² Score**.
- Logs parameters, metrics, and registers the model in **MLflow** registry (`sqlite:///mlflow.db`).
- Exports evaluation summary to `artifacts/model_evaluation/metrics.json`.

---

## ⚡ Quickstart Guide

### 1️⃣ Clone & Setup Environment
```bash
git clone https://github.com/[YOUR_USERNAME]/End-to-End-MLOps-Wine-Quality-Project.git
cd End-to-End-MLOps-Wine-Quality-Project

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Execute Full MLOps Pipeline
```bash
python main.py
```

### 4️⃣ Launch MLflow Dashboard
To visually track experiments and compare model runs:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Navigate to `http://localhost:5000` in your web browser.

### 5️⃣ Launch Flask Web Application
```bash
python app.py
```
Open `http://localhost:8080` to interact with the wine quality predictor dashboard!

---

## 🐳 Docker Containerization & CI/CD Deployment

### Build & Run Docker Container Locally
```bash
docker build -t wine-quality-app .
docker run -p 8080:8080 wine-quality-app
```

### AWS EC2 & ECR GitHub Actions Pipeline
The repository includes `.github/workflows/main.yaml` for automated deployment to AWS EC2:
1. **Continuous Integration**: Code linting and automated test validation.
2. **Continuous Delivery**: Builds Docker image and pushes to Amazon ECR registry.
3. **Continuous Deployment**: Self-hosted runner pulls the latest Docker image and restarts the container on AWS EC2.

---

## 🎯 Benchmark Results

| Metric | Score |
|---|---|
| **RMSE (Root Mean Squared Error)** | `0.6543` |
| **MAE (Mean Absolute Error)** | `0.5186` |
| **R² Score** | `0.3379` |

---

## 🤝 Acknowledgments
- Inspired by Krish Naik's MLOps Guided Tutorial.
- Red Wine Quality dataset courtesy of UCI Machine Learning Repository.
