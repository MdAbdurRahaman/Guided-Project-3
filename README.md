# 🍷 End-to-End MLOps Wine Quality Prediction Project

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://guided-project-3.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-orange.svg)](https://mlflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ElasticNet-green.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-red.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/GitHub%20Actions-Continuous%20Deployment-black.svg)](https://github.com/features/actions)

> 🚀 **Live Interactive Web App**: [https://guided-project-3.streamlit.app](https://guided-project-3.streamlit.app)

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

### AWS EC2 & ECR GitHub Actions Deployment Setup

The repository includes a complete production CI/CD workflow in [.github/workflows/main.yaml](file:///.github/workflows/main.yaml). Follow these steps to connect your AWS Cloud environment:

#### 1. Create Amazon ECR Repository
- Open AWS Console ➔ Navigate to **Amazon ECR** ➔ Create repository (e.g., `wine-app`).
- Copy the ECR repository URI (e.g., `123456789012.dkr.ecr.us-east-1.amazonaws.com/wine-app`).

#### 2. Launch AWS EC2 Instance & Install Docker
- Launch an Ubuntu Server EC2 instance (`t2.medium` recommended).
- Configure **Security Group**: Inbound Rules ➔ Add Custom TCP rule for Port `8080` (Source: `0.0.0.0/0`).
- Connect via SSH and install Docker:
  ```bash
  sudo apt-get update -y
  sudo apt-get upgrade -y
  curl -fsSL https://get.docker.com -o get-docker.sh
  sudo sh get-docker.sh
  sudo usermod -aG docker ubuntu
  newgrp docker
  ```

#### 3. Setup GitHub Actions Self-Hosted Runner on EC2
- In your GitHub Repo ➔ **Settings** ➔ **Actions** ➔ **Runners** ➔ **New self-hosted runner**.
- Select **Linux** and run the provided command script inside your EC2 terminal.
- Start runner service: `./run.sh`.

#### 4. Configure GitHub Repository Secrets
- In your GitHub Repo ➔ **Settings** ➔ **Secrets and variables** ➔ **Actions**:
  - `AWS_ACCESS_KEY_ID`: Your AWS IAM User Access Key
  - `AWS_SECRET_ACCESS_KEY`: Your AWS IAM User Secret Key
  - `AWS_REGION`: AWS Region (e.g., `us-east-1`)
  - `AWS_ECR_LOGIN_URI`: Your ECR URI host (e.g., `123456789012.dkr.ecr.us-east-1.amazonaws.com`)
  - `ECR_REPOSITORY_NAME`: `wine-app`

#### 5. Trigger Automatic Deployment
- Any `git push origin main` triggers GitHub Actions to build the Docker image, push to AWS ECR, pull onto your EC2 runner, and launch the web server on `http://YOUR_EC2_PUBLIC_IP:8080`.

---

## 🎈 Free 1-Click Streamlit Community Cloud Deployment

If you do not have an active AWS account, you can deploy this application for **100% FREE** using **Streamlit Community Cloud**:

### 1️⃣ Run Streamlit Locally
```bash
streamlit run streamlit_app.py
```
Open `http://localhost:8501` to view your interactive dashboard locally.

### 2️⃣ Deploy to Streamlit Cloud (1-Click Free Hosting)
1. Push your repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **"New App"**.
4. Select your Repository: `[YOUR_USERNAME]/End-to-End-MLOps-Wine-Quality-Project`.
5. Set Main file path: `streamlit_app.py`.
6. Click **Deploy!** 🚀

Your live web app URL will be generated automatically (e.g. `https://wine-quality-mlops.streamlit.app`)!

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
