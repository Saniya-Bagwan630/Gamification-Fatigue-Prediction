import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Fatigue Predictor", page_icon="🎯", layout="wide")

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Space+Grotesk:wght@400;500;700&display=swap');

        :root {
            --bg-start: #fff9ec;
            --bg-end: #d6f6f5;
            --panel: #ffffffcc;
            --ink: #16233b;
            --muted: #4a5a77;
            --accent: #f17105;
            --accent-soft: #ffd7a8;
            --safe: #2f9e44;
            --warn: #e67700;
            --risk: #c92a2a;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 14%, #ffe3bd 0 14%, transparent 15%),
                radial-gradient(circle at 88% 22%, #b9f4e7 0 18%, transparent 19%),
                linear-gradient(120deg, var(--bg-start), var(--bg-end));
            color: var(--ink);
            font-family: 'Space Grotesk', sans-serif;
        }

        .hero-card {
            background: linear-gradient(135deg, #fff8f0, #f4fffe);
            border: 1px solid #f5d3ae;
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 12px 24px #15243b1a;
            animation: riseIn 0.65s ease-out;
        }

        .hero-title {
            margin: 0;
            font-family: 'Fraunces', serif;
            color: #1f3359;
            font-size: 2.2rem;
            line-height: 1.2;
            letter-spacing: 0.3px;
        }

        .hero-sub {
            color: var(--muted);
            margin-top: 0.55rem;
            font-size: 1.02rem;
        }

        .small-pill {
            display: inline-block;
            margin-top: 0.7rem;
            background: var(--accent-soft);
            color: #8a3f00;
            padding: 0.2rem 0.7rem;
            border-radius: 999px;
            font-size: 0.86rem;
            font-weight: 600;
        }

        @keyframes riseIn {
            from {opacity: 0; transform: translateY(12px);}
            to {opacity: 1; transform: translateY(0);}
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    model_path = "models/fatigue_model.joblib"
    if not os.path.exists(model_path):
        return None

    loaded = joblib.load(model_path)
    if isinstance(loaded, dict) and "model" in loaded:
        return loaded["model"]
    return loaded


def infer_risk(probability):
    if probability >= 0.7:
        return "High", "Urgent intervention advised", "🚨", "error"
    if probability >= 0.5:
        return "Moderate", "Adjust pace and monitor closely", "⚠️", "warning"
    if probability >= 0.3:
        return "Mild", "Keep monitoring engagement", "ℹ️", "info"
    return "Low", "Student is currently stable", "✅", "success"


def get_defaults(preset_name):
    presets = {
        "Custom": (30, 20.0, 5),
        "High Risk Pattern": (82, 88.0, 1),
        "Moderate Risk Pattern": (58, 52.0, 4),
        "Engaged Pattern": (22, 18.0, 9),
    }
    return presets.get(preset_name, presets["Custom"])


model = load_model()
if model is None:
    st.error("Model not found. Run: py -3.11 train_model_fixed.py")
    st.stop()

feature_names = [
    "Difficulty",
    "TimeSpentOnCourse",
    "NumberOfQuizzesTaken",
]

st.markdown(
    """
    <div class="hero-card">
        <h1 class="hero-title">Gamification Fatigue Radar</h1>
        <p class="hero-sub">Interactive fatigue scoring from student behavior using Logistic Regression.</p>
        <span class="small-pill">Live Prediction Console</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Configuration")
scenario = st.sidebar.selectbox(
    "Quick Scenario",
    ["Custom", "High Risk Pattern", "Moderate Risk Pattern", "Engaged Pattern"],
)
default_difficulty, default_time, default_streak = get_defaults(scenario)

difficulty = st.sidebar.slider("Difficulty Level", 0, 100, default_difficulty, 1)
time_spent = st.sidebar.slider("Session Time (hours)", 0.0, 100.0, default_time, 1.0)
streak = st.sidebar.slider("Quiz Streak", 0, 10, default_streak, 1)
threshold = st.sidebar.slider("Alert Threshold", 0.1, 0.9, 0.5, 0.05)

if st.sidebar.button("Randomize Student"):
    difficulty = int(np.random.randint(0, 101))
    time_spent = float(np.random.randint(0, 101))
    streak = int(np.random.randint(0, 11))

features_df = pd.DataFrame(
    [[difficulty, time_spent, streak]],
    columns=feature_names,
)

probability = float(model.predict_proba(features_df)[0, 1])
prediction = "Fatigued" if probability >= threshold else "Not Fatigued"
risk_level, guidance, icon, level_type = infer_risk(probability)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Fatigue Probability", f"{probability:.1%}")
metric_col2.metric("Prediction", prediction)
metric_col3.metric("Risk Level", risk_level)
metric_col4.metric("Threshold", f"{threshold:.0%}")

tab1, tab2, tab3 = st.tabs(["Risk Dashboard", "What-If Analysis", "Model Details"])

with tab1:
    chart_col, info_col = st.columns([2, 1.1])

    with chart_col:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%", "font": {"size": 34, "color": "#1f3359"}},
                title={"text": "Fatigue Risk", "font": {"size": 24, "color": "#1f3359"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#1f3359"},
                    "bar": {"color": "#f17105"},
                    "steps": [
                        {"range": [0, 30], "color": "#d3f9d8"},
                        {"range": [30, 70], "color": "#ffe8cc"},
                        {"range": [70, 100], "color": "#ffc9c9"},
                    ],
                    "threshold": {
                        "line": {"color": "#1f3359", "width": 5},
                        "thickness": 0.8,
                        "value": threshold * 100,
                    },
                },
            )
        )
        gauge.update_layout(
            height=360,
            margin=dict(t=40, b=10, l=10, r=10),
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
        )
        st.plotly_chart(gauge, width="stretch")

    with info_col:
        st.subheader("Live Guidance")
        if level_type == "error":
            st.error(f"{icon} {guidance}")
        elif level_type == "warning":
            st.warning(f"{icon} {guidance}")
        elif level_type == "info":
            st.info(f"{icon} {guidance}")
        else:
            st.success(f"{icon} {guidance}")

        st.write("Risk intensity")
        st.progress(int(round(probability * 100)))

        st.caption("Current student profile")
        st.write(f"Difficulty: {difficulty}/100")
        st.write(f"Session Time: {time_spent:.1f} hours")
        st.write(f"Quiz Streak: {streak}")

with tab2:
    st.subheader("Scenario Comparison")
    adjustments = {
        "Current": (difficulty, time_spent, streak),
        "Reduce difficulty by 15": (max(0, difficulty - 15), time_spent, streak),
        "Shorten session by 20h": (difficulty, max(0.0, time_spent - 20.0), streak),
        "Increase streak by 3": (difficulty, time_spent, min(10, streak + 3)),
    }

    rows = []
    for name, values in adjustments.items():
        test_df = pd.DataFrame([values], columns=feature_names)
        test_prob = float(model.predict_proba(test_df)[0, 1])
        rows.append(
            {
                "Scenario": name,
                "Difficulty": values[0],
                "TimeSpent": values[1],
                "Streak": values[2],
                "FatigueProbability": round(test_prob, 4),
            }
        )

    compare_df = pd.DataFrame(rows)
    st.dataframe(compare_df, width="stretch")

    bar_fig = go.Figure(
        data=[
            go.Bar(
                x=compare_df["Scenario"],
                y=compare_df["FatigueProbability"] * 100,
                marker_color=["#f17105", "#2f9e44", "#1c7ed6", "#fab005"],
            )
        ]
    )
    bar_fig.update_layout(
        title="Projected Fatigue Probability by Scenario",
        yaxis_title="Probability (%)",
        height=360,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
    )
    st.plotly_chart(bar_fig, width="stretch")

with tab3:
    st.subheader("Logistic Regression Coefficients")
    coef = model.coef_[0]
    coef_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coef,
            "Effect": ["Increases risk" if c > 0 else "Decreases risk" for c in coef],
        }
    )
    st.dataframe(coef_df, width="stretch")
    st.write(f"Intercept: {model.intercept_[0]:.4f}")

    # Contribution view helps explain why current prediction moved up/down.
    contributions = coef * np.array([difficulty, time_spent, streak])
    contrib_fig = go.Figure(
        data=[
            go.Bar(
                x=feature_names,
                y=contributions,
                marker_color=["#c92a2a" if v > 0 else "#2f9e44" for v in contributions],
            )
        ]
    )
    contrib_fig.update_layout(
        title="Raw Contribution to Logit Score",
        yaxis_title="Coefficient × Value",
        height=320,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
    )
    st.plotly_chart(contrib_fig, width="stretch")