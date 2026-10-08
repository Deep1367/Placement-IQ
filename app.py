import sqlite3
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

DATA_PATH = Path("data/placement_data.csv")
MODEL_PATH = Path("models/placement_model.pkl")
DB_PATH = Path("placement_iq.db")

st.set_page_config(page_title="Placement IQ", page_icon="🎓", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def create_tables():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS students (
        student_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        cgpa REAL NOT NULL,
        communication_score INTEGER NOT NULL,
        aptitude_score INTEGER NOT NULL,
        technical_score INTEGER NOT NULL,
        projects_count INTEGER NOT NULL,
        internship INTEGER NOT NULL,
        backlogs INTEGER NOT NULL,
        hackathons INTEGER NOT NULL,
        certifications INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        predicted_status TEXT NOT NULL,
        probability REAL NOT NULL,
        feedback TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(student_id) REFERENCES students(student_id)
    )
    """)
    conn.commit()
    conn.close()

def insert_student(data):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO students (
            name, cgpa, communication_score, aptitude_score, technical_score,
            projects_count, internship, backlogs, hackathons, certifications
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["name"], data["cgpa"], data["communication_score"], data["aptitude_score"],
        data["technical_score"], data["projects_count"], data["internship"],
        data["backlogs"], data["hackathons"], data["certifications"]
    ))
    student_id = cur.lastrowid
    conn.commit()
    conn.close()
    return student_id

def insert_prediction(student_id, predicted_status, probability, feedback_text):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions (student_id, predicted_status, probability, feedback)
        VALUES (?, ?, ?, ?)
    """, (student_id, predicted_status, probability, feedback_text))
    conn.commit()
    conn.close()

def fetch_history(limit=50):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT s.student_id, s.name, s.cgpa, p.predicted_status, p.probability, p.feedback, p.created_at
        FROM predictions p
        JOIN students s ON p.student_id = s.student_id
        ORDER BY p.prediction_id DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return rows

def generate_feedback(data):
    feedback = []
    if data["cgpa"] < 6.5:
        feedback.append("Improve CGPA through focused revision and consistent study.")
    if data["communication_score"] < 6:
        feedback.append("Practice speaking, presentations, and mock interviews.")
    if data["aptitude_score"] < 6:
        feedback.append("Solve aptitude and logical reasoning problems daily.")
    if data["technical_score"] < 6:
        feedback.append("Revise Python, DSA, and core computer science basics.")
    if data["projects_count"] < 2:
        feedback.append("Build at least 2 strong projects for your portfolio.")
    if data["internship"] == 0:
        feedback.append("Try for an internship or a virtual internship project.")
    if data["backlogs"] > 0:
        feedback.append("Clear backlogs as early as possible.")
    if data["hackathons"] == 0:
        feedback.append("Participate in hackathons to improve practical exposure.")
    if data["certifications"] == 0:
        feedback.append("Add relevant certifications to strengthen your resume.")
    if data["cgpa"] >= 8 and data["technical_score"] >= 8 and data["projects_count"] >= 3:
        feedback.append("Your profile is strong. Focus on mock interviews and company-specific prep.")
    if not feedback:
        feedback.append("Balanced profile. Keep practising interviews and project depth.")
    return feedback

def readiness_score(data):
    score = (
        data["cgpa"] * 10
        + data["communication_score"] * 6
        + data["aptitude_score"] * 6
        + data["technical_score"] * 7
        + data["projects_count"] * 4
        + data["internship"] * 5
        + data["hackathons"] * 2
        + data["certifications"] * 1.5
        - data["backlogs"] * 8
    )
    return round(max(0, min(100, score)), 1)

def readiness_label(probability):
    if probability >= 0.75:
        return "Very Strong"
    if probability >= 0.60:
        return "Strong"
    if probability >= 0.45:
        return "Moderate"
    return "Needs Improvement"

def metric_card(title, value, subtitle=""):
    st.markdown(f"""
    <div style="
        background-color:#f7f9fc;
        padding:20px;
        border-radius:12px;
        border:1px solid #e6eaf2;
        text-align:center;
        color:black;
    ">
        <h5 style="margin:0;color:#666;">{title}</h5>
        <h2 style="margin:10px 0;color:#111;">{value}</h2>
        <p style="margin:0;color:#777;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)
    
create_tables()
model = load_model()

st.title("Placement IQ")
st.caption("PLACEMENT PREDICTOR..."
"REVIEW ACADEMIC SCORE  - PROVIDES FEEDBACK AND IMPROVEMENT FIELDS")

tab1, tab2, tab3, tab4 = st.tabs(["PREDICT", "VISUALS", "HISTORY", "ABOUT"])

with tab1:
    st.subheader("Student Input")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Student Name", placeholder="Enter your name")
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, 0.1)
        communication_score = st.slider("Communication Score", 0, 10, 6)
        aptitude_score = st.slider("Aptitude Score", 0, 10, 6)
        technical_score = st.slider("Technical Score", 0, 10, 6)

    with col2:
        projects_count = st.number_input("Projects Count", 0, 10, 1)
        internship = st.selectbox("Internship Completed?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        backlogs = st.number_input("Backlogs", 0, 10, 0)
        hackathons = st.number_input("Hackathons Participated", 0, 10, 0)
        certifications = st.number_input("Certifications", 0, 10, 0)

    data = {
        "name": name.strip() if name.strip() else "Unnamed Student",
        "cgpa": float(cgpa),
        "communication_score": int(communication_score),
        "aptitude_score": int(aptitude_score),
        "technical_score": int(technical_score),
        "projects_count": int(projects_count),
        "internship": int(internship),
        "backlogs": int(backlogs),
        "hackathons": int(hackathons),
        "certifications": int(certifications),
    }

    if st.button("Predict Placement", type="primary"):
        features = pd.DataFrame([[
            data["cgpa"], data["communication_score"], data["aptitude_score"],
            data["technical_score"], data["projects_count"], data["internship"],
            data["backlogs"], data["hackathons"], data["certifications"]
        ]], columns=[
            "cgpa", "communication_score", "aptitude_score", "technical_score",
            "projects_count", "internship", "backlogs", "hackathons", "certifications"
        ])

        pred_label = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])
        feedback = generate_feedback(data)
        read_score = readiness_score(data)
        read_label = readiness_label(probability)

        student_id = insert_student(data)
        insert_prediction(student_id, "Placed" if pred_label == 1 else "Not Placed", probability, " | ".join(feedback))

        st.success(f"Prediction: {'Placed' if pred_label == 1 else 'Not Placed'}")
        st.info(f"Placement Probability: {probability:.2%}")
        st.warning(f"Readiness Level: {read_label}")

        a, b, c = st.columns(3)
        with a:
            metric_card("Readiness Score", read_score, "Out of 100")
        with b:
            metric_card("Placement Chance", f"{probability:.0%}", "Model estimate")
        with c:
            metric_card("Profile Status", read_label, "AI-style feedback")

        st.subheader("Feedback")
        for item in feedback:
            st.write(f"- {item}")

        st.subheader("Improvement Roadmap")
        roadmap = []
        if cgpa < 6.5:
            roadmap.append("Raise CGPA with focused revision.")
        if communication_score < 6:
            roadmap.append("Practice communication and interviews.")
        if technical_score < 6:
            roadmap.append("Revise DSA, Python, and core subjects.")
        if projects_count < 2:
            roadmap.append("Create more projects and upload them to GitHub.")
        if internship == 0:
            roadmap.append("Apply for internships.")
        if backlogs > 0:
            roadmap.append("Clear backlogs before placement season.")
        if not roadmap:
            roadmap.append("Keep practising mock interviews and coding problems.")
        for step in roadmap:
            st.write(f"- {step}")

with tab2:
    st.subheader("Data Analysis and Visuals")
    df = load_data()

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5,3))
        sns.countplot(data=df, x="placement_status", ax=ax)
        ax.set_title("Placement Distribution")
        ax.set_xlabel("Placement Status")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(5,3))
        sns.boxplot(data=df, x="placement_status", y="cgpa", ax=ax)
        ax.set_title("CGPA vs Placement")
        ax.set_xlabel("Placement Status")
        ax.set_ylabel("CGPA")
        st.pyplot(fig)

    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots(figsize=(5,3))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

    with c4:
        feature_cols = [col for col in df.columns if col != "placement_status"]
        importances = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.barplot(data=importances, x="Importance", y="Feature", ax=ax)
        ax.set_title("Feature Importance")
        st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(5,3))
    proj_summary = df.groupby("projects_count")["placement_status"].mean().reset_index()
    proj_summary["placement_status"] = (proj_summary["placement_status"] * 100).round(1)
    sns.lineplot(data=proj_summary, x="projects_count", y="placement_status", marker="o", ax=ax)
    ax.set_title("Placement Rate vs Projects")
    ax.set_xlabel("Projects Count")
    ax.set_ylabel("Placement Rate (%)")
    st.pyplot(fig)

with tab3:
    st.subheader("PREDICTION HISTORY")
    rows = fetch_history(limit=50)
    if rows:
        history = pd.DataFrame(rows, columns=["Student ID", "Name", "CGPA", "Prediction", "Probability", "Feedback", "Created At"])
        history["Probability"] = (history["Probability"] * 100).round(2).astype(str) + "%"
        st.dataframe(history, use_container_width=True)
    else:
        st.info("No prediction history yet.")

with tab4:
    st.subheader("ABOUT")
    st.write("Placement IQ is a learning project that combines Python, SQL, data analysis, ML, and simple AI-style feedback.")
    st.write("DESIGNED BY DIPANSHU  "
    "PURSUING BTECH  "
    "COMPLETED INTERNSHIP UNDER FALCON TECHNOLOGY.")
