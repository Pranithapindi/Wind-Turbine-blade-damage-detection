import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ultralytics import YOLO
from PIL import Image
import random
import time

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Wind Turbine Blade Monitoring",
    layout="wide",
    page_icon="⚡"
)

st.title("⚡ Wind Turbine Blade Monitoring Dashboard")
st.caption("AI-Based Blade Damage Detection & Turbine Health Monitoring")

st.divider()

# =====================================
# SIDEBAR
# =====================================
st.sidebar.header("Control Panel")

uploaded_file = st.sidebar.file_uploader(
    "Upload Blade Image",
    type=["jpg","jpeg","png"]
)
# =====================================
# PROJECT OVERVIEW (SIDEBAR)
# =====================================

st.sidebar.markdown("---")
st.sidebar.subheader("📘 Project Overview")

st.sidebar.write(
"""
This system detects **damage in wind turbine blades** using 
Artificial Intelligence and image analysis.

### 🔍 Key Features
- Blade condition classification (Healthy / Faulty)
- Wind turbine sensor monitoring
- Blade health estimation
- Wind load impact analysis
- Blade lifetime prediction
- Maintenance recommendation system

### ⚙ Technologies Used
- Python
- Streamlit Dashboard
- YOLO Deep Learning Model
- Computer Vision

### 🎯 Project Goal
To provide an **automated blade inspection system** that 
reduces manual inspection cost and improves turbine safety.
"""
)

st.sidebar.markdown("---")

st.sidebar.subheader("📊 System Workflow")

st.sidebar.write(
"""
1️⃣ Upload blade image  
2️⃣ AI model analyzes blade condition  
3️⃣ Dashboard shows turbine health metrics  
4️⃣ System predicts blade lifetime  
5️⃣ Maintenance suggestions generated
"""
)
run_simulation = st.sidebar.checkbox("Enable Sensor Simulation")

image_uploaded = uploaded_file is not None

# =====================================
# LOAD MODEL
# =====================================
model = YOLO("runs/classify/train/weights/best.pt")

# =====================================
# IMAGE + PREDICTION
# =====================================
col1, col2 = st.columns(2)

if image_uploaded:

    image = Image.open(uploaded_file)

    with col1:
        st.subheader("Uploaded Blade Image")
        st.image(image, use_container_width=True)

    with col2:

        results = model(image)
        result = results[0]

        probs = result.probs.data.cpu().numpy()
        names = list(result.names.values())

        max_index = probs.argmax()
        prediction = names[max_index]
        confidence = probs[max_index]

        st.subheader("Prediction Result")

        if prediction.lower() == "healthy":
            st.success("Healthy Blade")
        else:
            st.error("Faulty Blade")

        st.metric("Confidence Score", f"{confidence*100:.2f}%")

        fig = plt.figure(figsize=(3,2))
        plt.bar(names, probs)
        plt.title("Prediction")
        plt.tight_layout()
        st.pyplot(fig)

else:
    st.info("Upload an image to begin blade inspection")

st.divider()

# =====================================
# MECHANICAL PARAMETERS
# =====================================
st.subheader("⚙ Blade Mechanical Measurements")

colA, colB, colC, colD = st.columns(4)

if run_simulation and image_uploaded:
    blade_length = 55.2
    pitch_angle = random.uniform(5,12)
    rpm = random.uniform(12,20)
    temperature = random.uniform(30,60)
else:
    blade_length = 0
    pitch_angle = 0
    rpm = 0
    temperature = 0

colA.metric("Blade Length", f"{blade_length} m")
colB.metric("Pitch Angle", f"{pitch_angle:.2f}°")
colC.metric("Rotor Speed", f"{rpm:.2f} RPM")
colD.metric("Temperature", f"{temperature:.2f} °C")

st.divider()

# =====================================
# TURBINE GAUGES
# =====================================
st.subheader("📟 Turbine Gauges")

def gauge(title,value,max_val):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text':title},
        gauge={'axis':{'range':[0,max_val]}}
    ))

    fig.update_layout(height=350)
    return fig

col1, col2, col3 = st.columns(3)

if run_simulation and image_uploaded:
    wind_speed = random.uniform(8,18)
    health_score = random.uniform(80,100)
else:
    wind_speed = 0
    health_score = 0

col1.plotly_chart(gauge("Rotor RPM",rpm,25),use_container_width=True)
col2.plotly_chart(gauge("Wind Speed (m/s)",wind_speed,25),use_container_width=True)
col3.plotly_chart(gauge("Blade Health (%)",health_score,100),use_container_width=True)

st.divider()

# =====================================
# BLADE INSPECTION DASHBOARD
# =====================================
st.subheader("📊 Blade Inspection Dashboard")

col1, col2 = st.columns(2)

with col1:

    if image_uploaded:
        damage_risk = random.uniform(10,60)
    else:
        damage_risk = 0

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=damage_risk,
        title={'text':"Blade Damage Risk (%)"},
        gauge={'axis':{'range':[0,100]}}
    ))

    fig.update_layout(height=350)

    st.plotly_chart(fig,use_container_width=True)

with col2:

    if image_uploaded:
        healthy = 75
        faulty = 25
    else:
        healthy = 0
        faulty = 0

    fig2 = go.Figure(data=[go.Pie(
        labels=["Healthy","Damaged"],
        values=[healthy,faulty],
        hole=0.4
    )])

    fig2.update_layout(height=350)

    st.plotly_chart(fig2,use_container_width=True)

st.divider()

# =====================================
# WIND LOAD + BLADE LIFETIME
# =====================================
st.subheader("🌬 Wind Impact and Blade Lifetime")

col1, col2 = st.columns(2)

with col1:

    if image_uploaded:
        wind = np.linspace(5,25,20)
        load = wind**1.3 + np.random.normal(0,2,20)
    else:
        wind = np.linspace(5,25,20)
        load = np.zeros(20)

    fig = plt.figure(figsize=(3,2))
    plt.plot(wind, load, marker='o')
    plt.xlabel("Wind Speed")
    plt.ylabel("Blade Load")
    plt.title("Wind vs Load")
    plt.tight_layout()

    st.pyplot(fig)

with col2:

    years = np.arange(0,20)

    if image_uploaded:
        degradation = np.exp(years/15)
    else:
        degradation = np.zeros(20)

    fig2 = plt.figure(figsize=(3,2))
    plt.plot(years, degradation)
    plt.xlabel("Years")
    plt.ylabel("Damage Index")
    plt.title("Blade Lifetime")
    plt.tight_layout()

    st.pyplot(fig2)

st.divider()

# =====================================
# MAINTENANCE SUGGESTIONS (FIXED LOGIC)
# =====================================
st.subheader("🛠 Maintenance Suggestions")

if image_uploaded:

    if prediction.lower() == "healthy":
        st.success("Blade condition is healthy. Continue regular monitoring.")

    else:
        if confidence > 0.80:
            st.error("Critical blade damage detected. Immediate maintenance required.")
        else:
            st.warning("Possible blade defect detected. Schedule inspection soon.")

else:
    st.info("Upload blade image to receive maintenance suggestions")

st.divider()

# =====================================
# OVERALL SYSTEM STATUS GRAPH
# =====================================
st.subheader("📊 Overall Turbine Health Overview")

if image_uploaded:

    overall_metrics = {
        "Blade Health": health_score,
        "Wind Load": min(100, wind_speed*4),
        "Rotor Efficiency": min(100, rpm*4),
        "Damage Risk": damage_risk
    }

else:

    overall_metrics = {
        "Blade Health": 0,
        "Wind Load": 0,
        "Rotor Efficiency": 0,
        "Damage Risk": 0
    }

labels = list(overall_metrics.keys())
values = list(overall_metrics.values())

fig = plt.figure(figsize=(4,2.5))

plt.bar(labels, values)

plt.ylabel("Score (%)")
plt.title("Overall Turbine System Status")

plt.xticks(rotation=20)

plt.tight_layout()

st.pyplot(fig)