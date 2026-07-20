import streamlit as st
import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
from mlProject.pipeline.prediction import PredictionPipeline

# Page configuration
st.set_page_config(
    page_title="Wine Quality AI Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #f472b6, #c084fc, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .prediction-box {
        background: linear-gradient(135deg, rgba(244, 114, 182, 0.15), rgba(192, 132, 252, 0.15));
        border: 1px solid rgba(192, 132, 252, 0.4);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Main Title Header
st.markdown('<div class="main-header">🍷 Wine Quality AI Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Industrial MLOps Engine with ElasticNet Regression & MLflow Tracking</div>', unsafe_allow_html=True)

# Sidebar Pipeline Controls & Metrics
with st.sidebar:
    st.header("⚙️ MLOps Dashboard")
    st.subheader("Model Status")
    st.success("✅ ElasticNet Model Loaded")
    st.info("📊 MLflow Tracking Active")
    
    st.markdown("---")
    st.subheader("Evaluation Metrics")
    
    metrics_path = Path("artifacts/model_evaluation/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        st.metric(label="RMSE", value=f"{metrics.get('rmse', 0):.4f}")
        st.metric(label="MAE", value=f"{metrics.get('mae', 0):.4f}")
        st.metric(label="R² Score", value=f"{metrics.get('r2', 0):.4f}")
    else:
        st.warning("Metrics file not generated yet. Run pipeline first.")

    st.markdown("---")
    if st.button("🚀 Trigger Full Training Pipeline"):
        with st.spinner("Executing main.py 5-stage MLOps pipeline..."):
            ret = os.system("python main.py")
            if ret == 0:
                st.success("Pipeline executed successfully!")
                st.rerun()
            else:
                st.error("Pipeline execution failed.")

# Two Column Layout for Form Inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧪 Chemical Composition (Part 1)")
    fixed_acidity = st.slider("Fixed Acidity (g/dm³)", 4.0, 16.0, 7.4, 0.1)
    volatile_acidity = st.slider("Volatile Acidity (g/dm³)", 0.1, 2.0, 0.70, 0.01)
    citric_acid = st.slider("Citric Acid (g/dm³)", 0.0, 1.0, 0.00, 0.01)
    residual_sugar = st.slider("Residual Sugar (g/dm³)", 0.5, 15.0, 1.9, 0.1)
    chlorides = st.slider("Chlorides (g/dm³)", 0.01, 0.6, 0.076, 0.001)
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide (mg/dm³)", 1.0, 72.0, 11.0, 1.0)

with col2:
    st.subheader("🧪 Chemical Composition (Part 2)")
    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide (mg/dm³)", 6.0, 289.0, 34.0, 1.0)
    density = st.slider("Density (g/cm³)", 0.9900, 1.0050, 0.9978, 0.0001)
    pH = st.slider("pH Level", 2.7, 4.0, 3.51, 0.01)
    sulphates = st.slider("Sulphates (g/dm³)", 0.3, 2.0, 0.56, 0.01)
    alcohol = st.slider("Alcohol (% vol)", 8.0, 15.0, 9.4, 0.1)

# Prediction Button
if st.button("🍷 Predict Wine Quality Score", use_container_width=True, type="primary"):
    try:
        input_data = np.array([
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
            free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol
        ]).reshape(1, 11)

        pipeline = PredictionPipeline()
        prediction = pipeline.predict(input_data)[0]
        score = round(float(prediction), 2)

        st.markdown(f"""
        <div class="prediction-box">
            <h2 style="margin: 0; color: #f472b6;">Predicted Quality Score</h2>
            <h1 style="font-size: 4rem; margin: 0.5rem 0; color: #fff;">{score} / 10</h1>
            <p style="color: #4ade80; font-size: 1.1rem; margin: 0;">✓ Inference generated via ElasticNet Pipeline</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")
