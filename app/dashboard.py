import os
import sqlite3

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Gym Willingness Prediction System",
    page_icon="\U0001F3CB",
    layout="wide",
    initial_sidebar_state="expanded",
)


BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "latest_model.pkl")
DB_PATH = os.path.join(BASE_DIR, "..", "database", "gym_data.db")
TABLE_NAME = "gym_willingness"
FEATURE_COLUMNS = [
    "STUDY_HOURS",
    "EXAM_SOON",
    "SLEEP_HOURS",
    "STRESS_LEVEL",
    "ENERGY_LEVEL",
    "PAST_WEEK_GYM_VISITS",
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(30, 136, 229, 0.14), transparent 28%),
                radial-gradient(circle at left top, rgba(0, 150, 136, 0.12), transparent 24%),
                linear-gradient(180deg, #f4f8fb 0%, #eef4f6 100%);
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        [data-testid="stSidebar"] * {
            color: #e2e8f0;
        }
        .hero-card, .section-card, .timeline-card, .prediction-card {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 22px;
            padding: 1.4rem 1.5rem;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
            backdrop-filter: blur(10px);
        }
        .hero-card {
            padding: 2rem 2.1rem;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 64, 175, 0.92));
            color: white;
        }
        .hero-kicker {
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-size: 0.8rem;
            opacity: 0.82;
            margin-bottom: 0.7rem;
        }
        .hero-title {
            font-size: 2.3rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }
        .hero-text {
            font-size: 1rem;
            line-height: 1.7;
            color: rgba(255, 255, 255, 0.88);
        }
        .mini-label {
            color: #475569;
            font-size: 0.9rem;
            margin-bottom: 0.2rem;
        }
        .result-box {
            border-radius: 20px;
            padding: 1.2rem 1.3rem;
            margin-top: 1rem;
            border: 1px solid transparent;
        }
        .result-positive {
            background: linear-gradient(135deg, rgba(22, 163, 74, 0.12), rgba(74, 222, 128, 0.18));
            border-color: rgba(22, 163, 74, 0.18);
        }
        .result-negative {
            background: linear-gradient(135deg, rgba(220, 38, 38, 0.1), rgba(248, 113, 113, 0.18));
            border-color: rgba(220, 38, 38, 0.18);
        }
        .result-title {
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .result-score {
            font-size: 2rem;
            font-weight: 800;
            margin: 0.35rem 0;
        }
        .timeline-item {
            position: relative;
            padding-left: 1.1rem;
            margin-bottom: 1rem;
            border-left: 3px solid #0ea5a4;
        }
        .timeline-date {
            font-size: 0.88rem;
            text-transform: uppercase;
            color: #0f766e;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }
        .timeline-title {
            font-size: 1.03rem;
            font-weight: 700;
            color: #0f172a;
        }
        .timeline-desc {
            color: #475569;
            margin-top: 0.1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Machine Learning Dashboard</div>
            <div class="hero-title">&#127947; Gym Willingness Prediction System</div>
            <div class="hero-text">
                A polished decision-support dashboard that estimates whether a student is likely
                to go to the gym based on study load, sleep, stress, energy, and recent workout behavior.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home(df, model):
    st.markdown("### Home")
    left, right = st.columns([1.8, 1.2], gap="large")

    with left:
        st.markdown(
            """
            <div class="section-card">
                <h4>Project Overview</h4>
                <p>
                    This system predicts <b>gym willingness</b> for a student using six lifestyle
                    and academic indicators. It is designed as a mini product demo for viva and
                    exhibition use, combining a trained model, SQLite-backed data, and interactive insights.
                </p>
                <p>
                    The key idea is simple: when workload, recovery, and energy shift, gym willingness
                    changes too. This dashboard helps present that relationship in a clear, modern format.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        positive_rate = (df["GYM_WILLING"].mean() * 100) if "GYM_WILLING" in df else 0
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.metric("Total Samples", f"{len(df):,}")
        col2.metric("Features", len(FEATURE_COLUMNS))
        col3, col4 = st.columns(2)
        col3.metric("Model", type(model).__name__)
        col4.metric("Positive Rate", f"{positive_rate:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    info1, info2, info3 = st.columns(3, gap="large")
    info1.markdown(
        """
        <div class="section-card">
            <h4>&#127919; Prediction Focus</h4>
            <p>Estimate whether a student is willing to go to the gym today using behavioral signals.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    info2.markdown(
        """
        <div class="section-card">
            <h4>&#128451; Data Source</h4>
            <p>Records are loaded from a SQLite database, enabling lightweight local storage and easy demos.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    info3.markdown(
        """
        <div class="section-card">
            <h4>&#128257; Model Lifecycle</h4>
            <p>The project includes retraining and model versioning so the deployed predictor stays current.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction(model):
    st.markdown("### Prediction")
    st.markdown(
        """
        <div class="section-card">
            <h4>Student Readiness Input</h4>
            <p>Adjust the sliders to simulate a student profile, then generate a prediction with confidence score.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        study_hours = st.slider("Study Hours", min_value=0, max_value=12, value=4)
        sleep_hours = st.slider("Sleep Hours", min_value=0, max_value=12, value=7)
        stress_level = st.slider("Stress Level", min_value=1, max_value=10, value=5)

    with col2:
        energy_level = st.slider("Energy Level", min_value=1, max_value=10, value=6)
        past_week_gym_visits = st.slider("Past Week Gym Visits", min_value=0, max_value=7, value=3)
        exam_soon_label = st.selectbox("Exam Soon", ["No", "Yes"])

    exam_soon = 1 if exam_soon_label == "Yes" else 0

    input_data = pd.DataFrame(
        {
            "STUDY_HOURS": [study_hours],
            "EXAM_SOON": [exam_soon],
            "SLEEP_HOURS": [sleep_hours],
            "STRESS_LEVEL": [stress_level],
            "ENERGY_LEVEL": [energy_level],
            "PAST_WEEK_GYM_VISITS": [past_week_gym_visits],
        }
    )

    if st.button("Predict Gym Willingness", type="primary", use_container_width=True):
        prediction = int(model.predict(input_data)[0])

        if hasattr(model, "predict_proba"):
            willingness_probability = float(model.predict_proba(input_data)[0][1])
        else:
            willingness_probability = 1.0 if prediction == 1 else 0.0

        is_willing = prediction == 1
        result_class = "result-positive" if is_willing else "result-negative"
        result_text = "Willing to go" if is_willing else "Not willing"
        interpretation = (
            "The profile suggests a good balance of recovery, energy, and recent workout momentum."
            if is_willing
            else "The profile indicates academic or physical fatigue may be reducing motivation for the gym."
        )

        st.markdown(
            f"""
            <div class="result-box {result_class}">
                <div class="result-title">{result_text}</div>
                <div class="result-score">{willingness_probability * 100:.1f}%</div>
                <div>Predicted probability of willingness</div>
                <div style="margin-top: 0.65rem; color: #334155;">{interpretation}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_insights(df):
    st.markdown("### Insights")
    st.markdown(
        """
        <div class="section-card">
            <h4>Interactive Model Insights</h4>
            <p>Explore how stress, energy, sleep, and workout history relate to gym willingness across the dataset.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_df = df.copy()
    chart_df["Willingness Label"] = chart_df["GYM_WILLING"].map({1: "Willing", 0: "Not Willing"})

    scatter_3d = px.scatter_3d(
        chart_df,
        x="STRESS_LEVEL",
        y="ENERGY_LEVEL",
        z="SLEEP_HOURS",
        color="Willingness Label",
        size="PAST_WEEK_GYM_VISITS",
        opacity=0.8,
        color_discrete_map={"Willing": "#16a34a", "Not Willing": "#dc2626"},
        title="3D View: Stress, Energy, Sleep, and Gym Willingness",
        hover_data=FEATURE_COLUMNS,
    )
    scatter_3d.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=520)

    visits_summary = (
        chart_df.groupby(["PAST_WEEK_GYM_VISITS", "Willingness Label"])
        .size()
        .reset_index(name="Count")
    )
    visits_bar = px.bar(
        visits_summary,
        x="PAST_WEEK_GYM_VISITS",
        y="Count",
        color="Willingness Label",
        barmode="group",
        color_discrete_map={"Willing": "#16a34a", "Not Willing": "#f97316"},
        title="Past Week Gym Visits vs Willingness",
    )
    visits_bar.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=380)

    stress_hist = px.histogram(
        chart_df,
        x="STRESS_LEVEL",
        color="Willingness Label",
        nbins=10,
        barmode="overlay",
        opacity=0.72,
        color_discrete_map={"Willing": "#0284c7", "Not Willing": "#ef4444"},
        title="Stress Level Distribution",
    )
    stress_hist.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=380)

    st.plotly_chart(scatter_3d, use_container_width=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.plotly_chart(visits_bar, use_container_width=True)
    with col2:
        st.plotly_chart(stress_hist, use_container_width=True)


def render_timeline():
    st.markdown("### Timeline")
    st.markdown(
        """
        <div class="timeline-card">
            <div class="timeline-item">
                <div class="timeline-date">January | 2nd Week</div>
                <div class="timeline-title">Planning</div>
                <div class="timeline-desc">Problem selection, feature definition, and dashboard concept planning.</div>
            </div>
            <div class="timeline-item">
                <div class="timeline-date">January to February</div>
                <div class="timeline-title">Data Generation + SQL Integration</div>
                <div class="timeline-desc">Dataset preparation, record design, and storage in SQLite.</div>
            </div>
            <div class="timeline-item">
                <div class="timeline-date">February 16</div>
                <div class="timeline-title">Mid-term + Baseline Model</div>
                <div class="timeline-desc">Initial model training and first evaluation milestone.</div>
            </div>
            <div class="timeline-item">
                <div class="timeline-date">February to March</div>
                <div class="timeline-title">Model Training + Dashboard</div>
                <div class="timeline-desc">Model comparison, retraining pipeline, and full Streamlit interface build.</div>
            </div>
            <div class="timeline-item">
                <div class="timeline-date">March 18</div>
                <div class="timeline-title">Exhibition</div>
                <div class="timeline-desc">Product-style demo presentation for judges and viva discussion.</div>
            </div>
            <div class="timeline-item" style="margin-bottom: 0;">
                <div class="timeline-date">March 23</div>
                <div class="timeline-title">Final Submission</div>
                <div class="timeline-desc">Project packaging, final validation, and submission handoff.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_about():
    st.markdown("### About")
    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown(
            """
            <div class="section-card">
                <h4>Models Used</h4>
                <p><b>Logistic Regression</b> provides a clean baseline for binary classification.</p>
                <p><b>Random Forest</b> captures non-linear behavior and feature interactions.</p>
                <p><b>SVM</b> supports robust classification when decision boundaries are less simple.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="section-card">
                <h4>System Design</h4>
                <p><b>SQL Database:</b> SQLite stores the project dataset in a portable local database.</p>
                <p><b>Model Versioning:</b> Best-performing models are archived and the latest version is deployed.</p>
                <p><b>Retraining Pipeline:</b> A retraining script reevaluates candidate models and refreshes <code>latest_model.pkl</code>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    inject_styles()
    model = load_model()
    df = load_data()

    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio(
            "Go to",
            ["Home", "Prediction", "Insights", "Timeline", "About"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("Gym Willingness Prediction System")
        st.caption("Model + SQL + Streamlit + Plotly")

    render_header()
    st.write("")

    if page == "Home":
        render_home(df, model)
    elif page == "Prediction":
        render_prediction(model)
    elif page == "Insights":
        render_insights(df)
    elif page == "Timeline":
        render_timeline()
    elif page == "About":
        render_about()


if __name__ == "__main__":
    main()
