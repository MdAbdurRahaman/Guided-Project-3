# 🎥 Video Presentation Script (2–5 Minutes)

> 📌 **Instructions**: Use this transcript as a guideline when recording your 2 to 5 minute video presentation in English. Feel free to present naturally while covering all key points!

---

## 🎬 Video Overview

- **Target Duration**: ~3.5 minutes (210 seconds)
- **Language**: English
- **Tone**: Professional, confident, and engineering-focused
- **Visual Setup**: Split screen showing your presentation slide / web dashboard and code editor.

---

## 📜 Full Script & Timestamps

### ⏱️ 0:00 - 0:45 | 1️⃣ Problem Statement & Motivation
**[Speaker on Camera / Displaying Web UI Dashboard]**

"Hi everyone! In today's video, I'm excited to present my end-to-end MLOps Data Science Project: the **Wine Quality AI Engine with MLflow Experiment Tracking**.

In wine manufacturing and production, ensuring consistent chemical quality control is vital for brand reputation and price determination. Manual wine testing is subjective, expensive, and slow. 

However, building machine learning models in isolated Jupyter Notebooks creates a massive deployment gap. Data drift, unvalidated inputs, lack of reproducibility, and manual tracking often prevent models from surviving in real-world production environments.

To solve this, I designed a modular, production-ready MLOps pipeline that automates data ingestion, schema validation, ElasticNet regression modeling, experiment tracking with MLflow, and continuous deployment using Docker and GitHub Actions."

---

### ⏱️ 0:45 - 2:00 | 2️⃣ Key Architecture & Implementation Steps
**[Screen Share: Showing VS Code Project Structure & `main.py`]**

"Let's look at the core architecture and key implementation steps:

1. **Modular Setup & Configurations**: Rather than writing monolithic code, I structured the project cleanly under `src/mlProject`. All configuration paths are centralized in `config.yaml`, hyperparameters in `params.yaml`, and column schema rules in `schema.yaml`.
2. **5-Stage Automated Pipeline**:
   - **Stage 01 - Data Ingestion**: Downloads the Red Wine Quality dataset automatically and extracts it into artifact directories.
   - **Stage 02 - Data Validation**: Checks input data columns and data types against `schema.yaml`, logging a boolean status flag to prevent invalid data from corrupting downstream models.
   - **Stage 03 - Data Transformation**: Performs an 80/20 train/test split once validation passes.
   - **Stage 04 - Model Trainer**: Trains an `ElasticNet` regression model using hyperparameters parsed from configuration dataclasses.
   - **Stage 05 - Model Evaluation & MLflow Tracking**: Computes RMSE, MAE, and R² scores, logging hyperparameters, evaluation metrics, and registering the model binary directly into **MLflow**.
3. **Web Dashboard Serving**: I built a Flask application with a modern glassmorphism UI where users can input chemical parameters—like fixed acidity, pH, and alcohol content—and receive real-time quality predictions."

---

### ⏱️ 2:00 - 2:45 | 3️⃣ Technical Challenges & How They Were Overcome
**[Screen Share: Highlighting `components/model_evaluation.py` & GitHub Actions workflow]**

"During implementation, I faced two major engineering challenges:

1. **MLflow Tracking URI Handling in Windows Environments**:
   When configuring MLflow on Windows, path spaces and URI encoding scheme mismatches can crash experiment logging. I resolved this by enforcing explicit URI parsing logic and SQLite backend storage (`sqlite:///mlflow.db`) with fallback handling for local artifact logging.

2. **Ensuring Strict Data Validation in Automated Pipelines**:
   Machine learning pipelines often fail silently when raw schemas change. To solve this, I implemented an explicit schema guardrail stage that halts pipeline execution immediately if unexpected columns or mismatched data types are encountered."

---

### ⏱️ 2:45 - 3:30 | 4️⃣ Key Learnings & Portfolio Takeaways
**[Speaker on Camera / Showing Flask UI live prediction]**

"Through this guided project, I gained invaluable practical experience:
- **Modular Code Design (`src/`)**: Building pipeline components using custom loggers, utilities, and configuration managers makes code maintainable and enterprise-ready.
- **MLOps Best Practices**: Integrating MLflow ensures complete experiment reproducibility, versioning, and model registry governance.
- **Production Containerization**: Packaging the Flask app with Docker and configuring GitHub Actions CI/CD workflows demonstrates how data science models transition from local machines to cloud infrastructure on AWS.

Thank you so much for watching! The full GitHub repository link and project documentation are provided in the description. I would love to hear your feedback in the comments!"

---

## 🎯 Quick Recording Checklist Before You Click Record
- [ ] Good lighting and clear microphone audio
- [ ] Flask app running locally on `localhost:8080` to demo live prediction
- [ ] VS Code open with `main.py`, `src/mlProject/`, and `config.yaml` visible
- [ ] MLflow UI open on `localhost:5000` showing logged metrics and registered model
