import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Fatigue Predictor", page_icon="🎮", layout="wide")

st.title("🎮 Gamification Fatigue Prediction System")
st.markdown("Real-time fatigue detection using Logistic Regression")

# Load the model
@st.cache_resource
def load_model():
    model_path = 'models/fatigue_model.joblib'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

if model is None:
    st.error("❌ Model not found! Please run: py -3.11 train_model_fixed.py")
    st.stop()

# Sidebar inputs
st.sidebar.header("📊 Student Parameters")

difficulty = st.sidebar.slider("Difficulty Level (0-100)", 0, 100, 30)
time_spent = st.sidebar.slider("Session Time (hours)", 0.0, 100.0, 20.0, 1.0)
streak = st.sidebar.slider("Streak (quizzes taken)", 0, 10, 5, 1)
threshold = st.sidebar.slider("Fatigue Threshold", 0.0, 1.0, 0.5, 0.05)

# Make prediction
features = np.array([[difficulty, time_spent, streak]])
probability = model.predict_proba(features)[0, 1]
prediction = "Fatigued" if probability >= threshold else "Not Fatigued"

# Display results
col1, col2, col3 = st.columns(3)
col1.metric("Fatigue Probability", f"{probability:.1%}")
col2.metric("Prediction", prediction)
col3.metric("Threshold", f"{threshold:.0%}")

# Show gauge
import plotly.graph_objects as go
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=probability * 100,
    title={'text': "Fatigue Risk"},
    gauge={
        'axis': {'range': [0, 100]},
        'steps': [
            {'range': [0, 30], 'color': "lightgreen"},
            {'range': [30, 70], 'color': "yellow"},
            {'range': [70, 100], 'color': "salmon"}
        ]
    }
))
fig.update_layout(height=300)
st.plotly_chart(fig, use_container_width=True)

# Recommendations
if probability >= 0.7:
    st.error("🚨 HIGH FATIGUE - Take a break!")
elif probability >= 0.5:
    st.warning("⚠️ MODERATE FATIGUE - Monitor closely")
elif probability >= 0.3:
    st.info("ℹ️ MILD FATIGUE - Continue monitoring")
else:
    st.success("✅ GOOD - Student engaged")

# Show model details
with st.expander("View Model Details"):
    st.write(f"**Intercept:** {model.intercept_[0]:.4f}")
    st.write(f"**Coefficients:**")
    st.write(f"- Difficulty: {model.coef_[0][0]:.4f}")
    st.write(f"- Session Time: {model.coef_[0][1]:.4f}")
    st.write(f"- Streak: {model.coef_[0][2]:.4f}")