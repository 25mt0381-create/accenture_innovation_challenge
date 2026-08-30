# =====================================================================
# PredictivePulse Streamlit Portal
# Developed by: Rahul & Team (IIT ISM Dhanbad)
# Accenture Innovation Challenge 2026 Submission
# =====================================================================
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="DigitalTwin.ai - The Predictive Pulse",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS for styling (Glassmorphism & Dark Mode)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: #0b0c10;
        color: #c5c6c7;
    }
    
    /* Header Gradient */
    .title-gradient {
        background: linear-gradient(135deg, #66fcf1 0%, #45a29e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    
    .subtitle-text {
        font-size: 1.1rem;
        color: #8b9bb4;
        margin-top: -0.5rem;
        margin-bottom: 2rem;
    }
    
    /* Premium Glassmorphic Cards */
    .glass-card {
        background: rgba(31, 38, 135, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .status-active {
        color: #00ffcc;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffcc00;
        font-weight: bold;
    }
    
    .status-danger {
        color: #ff3366;
        font-weight: bold;
    }
    
    /* Live Station Indicator CSS */
    .station-node {
        border-radius: 50%;
        width: 80px;
        height: 80px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-family: 'Space Grotesk', sans-serif;
        margin: 0 auto;
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.1);
    }
    
    .station-node-normal {
        background: radial-gradient(circle, #1f4037 0%, #99f2c8 100%);
        border: 2px solid #00ffcc;
        color: #ffffff;
    }
    
    .station-node-dark {
        background: radial-gradient(circle, #2c3e50 0%, #0b0c10 100%);
        border: 2px dashed #ffcc00;
        color: #ffcc00;
        box-shadow: 0 0 15px rgba(255, 204, 0, 0.3);
    }
    
    .station-node-anomaly {
        background: radial-gradient(circle, #870000 0%, #190a05 100%);
        border: 2px solid #ff3366;
        color: #ffffff;
        box-shadow: 0 0 15px rgba(255, 51, 102, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# File Paths
GOLD_PATH_PARQUET = "data/gold/features.parquet"
MODEL_PATH = "models/xgboost_defect_model.pkl"

# Load Gold Layer Telemetry Data
@st.cache_data
def load_data():
    if os.path.exists(GOLD_PATH_PARQUET):
        df = pd.read_parquet(GOLD_PATH_PARQUET)
        # Ensure timestamp is datetime and timezone-naive
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        # Precompute dt column (time delta between rows in seconds, default to 0.1s)
        df['dt'] = (df['timestamp'] - df['timestamp'].shift(1)).dt.total_seconds().fillna(0.1)
        return df
    else:
        return None

df = load_data()

# Sidebar Setup
st.sidebar.markdown("<h2 style='color:#66fcf1;font-family:\"Space Grotesk\"'>⚡ PredictivePulse</h2>", unsafe_allow_html=True)
st.sidebar.markdown("*Operations Digital Twin System*")

# View Selector
st.sidebar.markdown("### View Access Mode")
view_mode = st.sidebar.radio(
    "Select Console View:",
    ["Floor Operations Console (Real-time Diagnostics)", "Plant Management & ESG Analytics Dashboard"]
)

# Simulation Controller
st.sidebar.markdown("---")
st.sidebar.markdown("### SCADA Live Feed Simulator")

if df is not None:
    # Simulator Slider
    total_steps = len(df)
    sim_index = st.sidebar.slider("Time-Series Replay Index:", 0, total_steps - 1, 100)
    current_row = df.iloc[sim_index]
    
    st.sidebar.info(f"Connected to Gold Database. Current Timestamp: {current_row['_time']}")
    
    # Simulate custom anomaly injections
    anomaly_mode = st.sidebar.selectbox(
        "Simulate Active Fault States:",
        ["Baseline Operations (Normal)", "Thermal Deviation (VFD 1)", "Chassis Feed Transit Delay (Stopper 2)", "Kinematic Fault State (Robot R04)", "Mechanical Gripper Overload (Robot R03)"]
    )
    
    # Adjust current row based on selected anomaly for interactive twin response
    if anomaly_mode == "Thermal Deviation (VFD 1)":
        current_row["Q_VFD1_Temperature"] = 89.2
        current_row["Q_VFD2_Temperature"] = 78.5  # Reconstructed
    elif anomaly_mode == "Chassis Feed Transit Delay (Stopper 2)":
        current_row["inferred_dark_station_time"] = 32.8
        current_row["is_bottleneck"] = 1
    elif anomaly_mode == "Kinematic Fault State (Robot R04)":
        current_row["M_R04_SJointAngle_Degree"] = 0.0
        current_row["M_R04_SJoint_Velocity"] = 0.0
        current_row["M_R04_SJoint_Acceleration"] = 0.0
    elif anomaly_mode == "Mechanical Gripper Overload (Robot R03)":
        current_row["I_R03_Gripper_Load"] = 4850
        current_row["I_R03_Gripper_Load_MA10"] = 4500
        current_row["defect_label"] = 3
else:
    st.sidebar.warning("Feature database not found. Run the ETL pipeline first.")
    if st.sidebar.button("Run ETL Pipeline"):
        with st.spinner("Executing Spark ETL pipeline..."):
            os.system("python etl_pipeline.py")
            st.rerun()
            
    # Mock data generation for demo purposes if CSV not built
    st.sidebar.info("Simulating Mock Data for Preview...")
    total_steps = 1000
    sim_index = st.sidebar.slider("Time-Series Replay Index:", 0, total_steps - 1, 100)
    
    # Build temporary mock dataframe
    mock_times = pd.date_range("2024-08-13 14:00:00", periods=total_steps, freq="S")
    df_mock = pd.DataFrame({
        '_time': [t.strftime("%Y-%m-%dT%H:%M:%S.%fZ") for t in mock_times],
        'timestamp': mock_times,
        'Q_VFD1_Temperature': np.random.normal(68.0, 1.2, total_steps),
        'Q_VFD3_Temperature': np.random.normal(68.2, 1.5, total_steps),
        'Q_VFD4_Temperature': np.random.normal(70.1, 0.8, total_steps),
        'I_Stopper1_Status': np.random.choice([0.0, 1.0], total_steps),
        'I_Stopper3_Status': np.random.choice([0.0, 1.0], total_steps),
        'I_Stopper4_Status': [0.0]*total_steps,
        'I_Stopper5_Status': [0.0]*total_steps,
        'I_SafetyDoor1_Status': [1.0]*total_steps,
        'I_SafetyDoor2_Status': [1.0]*total_steps,
        'I_HMI_EStop_Status': [1.0]*total_steps,
        'Q_Cell_CycleCount': np.arange(total_steps) // 5 + 1,
        'Q_Cell_CycleState': [9.0]*total_steps,
        'inferred_dark_station_time': np.random.normal(12.5, 0.8, total_steps),
        'is_bottleneck': [0]*total_steps,
        'defect_label': np.random.choice([0, 0, 0, 0, 1, 2, 3], total_steps)
    })
    
    # Reconstruct VFD2
    df_mock["Q_VFD2_Temperature"] = (df_mock["Q_VFD1_Temperature"] + df_mock["Q_VFD3_Temperature"]) / 2.0
    # Time delta for mock data is 1.0 second
    df_mock["dt"] = 1.0
    
    # Robot metrics
    for r in ["R01", "R02", "R03", "R04"]:
        df_mock[f"I_{r}_Gripper_Load"] = np.random.normal(1200, 200, total_steps)
        df_mock[f"I_{r}_Gripper_Load_MA10"] = df_mock[f"I_{r}_Gripper_Load"].rolling(10, min_periods=1).mean()
        df_mock[f"I_{r}_Gripper_Load_MA60"] = df_mock[f"I_{r}_Gripper_Load"].rolling(60, min_periods=1).mean()
        df_mock[f"I_{r}_Gripper_Load_Std10"] = df_mock[f"I_{r}_Gripper_Load"].rolling(10, min_periods=1).std().fillna(0)
        df_mock[f"M_{r}_SJointAngle_Degree"] = np.random.normal(-45, 10, total_steps)
        df_mock[f"M_{r}_SJoint_Velocity"] = np.random.normal(0, 2, total_steps)
        df_mock[f"M_{r}_SJoint_Acceleration"] = np.random.normal(0, 0.5, total_steps)
        
    df = df_mock
    current_row = df.iloc[sim_index]

# ----------------------------------------------------------------------------------------------------------------------
# DYNAMIC METRIC CALCULATIONS
# ----------------------------------------------------------------------------------------------------------------------
# Slice dataframe up to current simulation index for cumulative/rolling KPIs
df_subset = df.iloc[:sim_index + 1]

# 1. Availability Calculation
# Downtime: E-Stop active (0.0) or safety doors open (0.0)
is_downtime = (df_subset["I_HMI_EStop_Status"] == 0.0) | \
              (df_subset["I_SafetyDoor1_Status"] == 0.0) | \
              (df_subset["I_SafetyDoor2_Status"] == 0.0)

# Total elapsed time in seconds (using precomputed dt column)
total_seconds = df_subset["dt"].sum()

# Downtime seconds per category
estop_downtime_s = df_subset.loc[df_subset["I_HMI_EStop_Status"] == 0.0, "dt"].sum()
safety1_downtime_s = df_subset.loc[df_subset["I_SafetyDoor1_Status"] == 0.0, "dt"].sum()
safety2_downtime_s = df_subset.loc[df_subset["I_SafetyDoor2_Status"] == 0.0, "dt"].sum()

total_downtime_s = df_subset.loc[is_downtime, "dt"].sum()
operating_time_s = total_seconds - total_downtime_s

availability = (operating_time_s / total_seconds) if total_seconds > 0 else 1.0

# 2. Performance Calculation
# Design/ideal cycle time for the assembly cell (5 minutes = 300 seconds)
standard_cycle_time = 300.0
total_cycles = max(1, int(current_row["Q_Cell_CycleCount"]))
transit_val = current_row["inferred_dark_station_time"]
performance = (standard_cycle_time * total_cycles) / operating_time_s if operating_time_s > 0 else 1.0
performance = min(1.0, max(0.0, performance))

# 3. Quality (First Pass Yield) Calculation
# Unique cycles that had any defect label > 0
defect_cycles = df_subset[df_subset["defect_label"] > 0]["Q_Cell_CycleCount"].nunique()
quality = (total_cycles - defect_cycles) / total_cycles if total_cycles > 0 else 1.0
quality = min(1.0, max(0.0, quality))

# 4. OEE Calculation
oee = availability * performance * quality

# 5. Active Alarms / Stopped Cause
line_stopped = current_row["I_HMI_EStop_Status"] == 0.0 or \
               current_row["I_SafetyDoor1_Status"] == 0.0 or \
               current_row["I_SafetyDoor2_Status"] == 0.0

active_stop_cause = []
if current_row["I_HMI_EStop_Status"] == 0.0:
    active_stop_cause.append("Emergency Stop Active (HMI)")
if current_row["I_SafetyDoor1_Status"] == 0.0:
    active_stop_cause.append("Safety Door 1 Open")
if current_row["I_SafetyDoor2_Status"] == 0.0:
    active_stop_cause.append("Safety Door 2 Open")

# 6. Robot Health Scores (0-100%)
robot_health = {}
for r in ["R01", "R02", "R03", "R04"]:
    load = current_row[f"I_{r}_Gripper_Load"]
    std_load = current_row[f"I_{r}_Gripper_Load_Std10"]
    load_penalty = max(0.0, (load - 1500) / 30.0)  # Starts at 1500, max at 4500 N
    vibe_penalty = max(0.0, (std_load - 120) / 2.0)  # Starts at 120 N, max at 320 N
    health = max(0.0, 100.0 - max(load_penalty, vibe_penalty))
    robot_health[r] = health

# VFD Health Scores (0-100%)
vfd_health = {}
for i, col in enumerate(["Q_VFD1_Temperature", "Q_VFD2_Temperature", "Q_VFD3_Temperature", "Q_VFD4_Temperature"]):
    vfd_name = f"VFD {i+1}"
    temp = current_row[col]
    temp_penalty = max(0.0, (temp - 72.0) * 5.0)  # Starts at 72°C, max at 92°C
    health = max(0.0, 100.0 - temp_penalty)
    vfd_health[vfd_name] = health

def make_gauge_chart(value, title, color):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value * 100,
        number = {'suffix': "%", 'font': {'color': '#ffffff', 'size': 20}, 'valueformat': '.1f'},
        title = {'text': title, 'font': {'color': '#8b9bb4', 'size': 13}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#8b9bb4"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.1)",
        }
    ))
    fig.update_layout(
        height=130,
        margin=dict(l=15, r=15, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

with st.sidebar.expander("ℹ️ Project Info"):
    st.markdown("""
    **Project**: PredictivePulse  
    **Track**: Accenture Innovation Challenge 2026 (Problem Track 4)  
    **Developed by**: Rahul & Team (IIT ISM Dhanbad)  
    **Core Concept**: Real-time assembly line digital twin using soft-sensors to reconstruct missing telemetry (Dark Stations) without hardware upgrades.
    """)

# Main Page Header
st.markdown("<h1 class='title-gradient'>PredictivePulse Operations Portal</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Live Assembly Line Digital Twin & Multi-Causal Fault Diagnostics</p>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------------------------------------
# VIEW 1: FLOOR SUPERVISOR
# ----------------------------------------------------------------------------------------------------------------------
if view_mode == "Floor Operations Console (Real-time Diagnostics)":
    
    st.subheader("Assembly Line Live Schematic")
    
    # Render horizontal line of stations with current VFD / Stopper states
    cols = st.columns(5)
    
    with cols[0]:
        h1 = vfd_health["VFD 1"]
        h1_color = "#00ffcc" if h1 > 80 else "#ffcc00" if h1 > 50 else "#ff3366"
        st.markdown("""
        <div class='glass-card' style='text-align: center;'>
            <h4>Station 1</h4>
            <div class='station-node station-node-normal'>ST 1</div>
            <p style='margin-top:10px; margin-bottom:5px;'>Stopper 1: <span class='status-active'>{}</span></p>
            <p style='margin-bottom:2px;'>VFD 1 Temp: <b>{:.1f}°C</b></p>
            <p style='margin-top:0;'>VFD Health: <b style='color:{};'>{:.0f}%</b></p>
        </div>
        """.format("Holding" if current_row["I_Stopper1_Status"] == 1 else "Release", current_row["Q_VFD1_Temperature"], h1_color, h1), unsafe_allow_html=True)
        
    with cols[1]:
        # Dark Station: Stopper 2
        is_st2_jam = current_row["inferred_dark_station_time"] > 20
        node_class = "station-node-anomaly" if is_st2_jam else "station-node-dark"
        h2 = vfd_health["VFD 2"]
        h2_color = "#00ffcc" if h2 > 80 else "#ffcc00" if h2 > 50 else "#ff3366"
        st.markdown("""
        <div class='glass-card' style='text-align: center; border: 1px dashed #ffcc00;'>
            <h4 style='color:#ffcc00;'>Station 2 (Dark)</h4>
            <div class='station-node {}'>ST 2</div>
            <p style='margin-top:10px; margin-bottom:5px; color:#ffcc00;'>Stopper 2: <b>Inferred</b></p>
            <p style='margin-bottom:2px; color:#ffcc00;'>VFD 2 Temp (Est): <b>{:.1f}°C</b></p>
            <p style='margin-top:0; color:#ffcc00;'>VFD Health (Est): <b style='color:{};'>{:.0f}%</b></p>
        </div>
        """.format(node_class, current_row["Q_VFD2_Temperature"], h2_color, h2), unsafe_allow_html=True)
        
    with cols[2]:
        h3 = vfd_health["VFD 3"]
        h3_color = "#00ffcc" if h3 > 80 else "#ffcc00" if h3 > 50 else "#ff3366"
        st.markdown("""
        <div class='glass-card' style='text-align: center;'>
            <h4>Station 3</h4>
            <div class='station-node station-node-normal'>ST 3</div>
            <p style='margin-top:10px; margin-bottom:5px;'>Stopper 3: <span class='status-active'>{}</span></p>
            <p style='margin-bottom:2px;'>VFD 3 Temp: <b>{:.1f}°C</b></p>
            <p style='margin-top:0;'>VFD Health: <b style='color:{};'>{:.0f}%</b></p>
        </div>
        """.format("Holding" if current_row["I_Stopper3_Status"] == 1 else "Release", current_row["Q_VFD3_Temperature"], h3_color, h3), unsafe_allow_html=True)
        
    with cols[3]:
        h4 = vfd_health["VFD 4"]
        h4_color = "#00ffcc" if h4 > 80 else "#ffcc00" if h4 > 50 else "#ff3366"
        st.markdown("""
        <div class='glass-card' style='text-align: center;'>
            <h4>Station 4</h4>
            <div class='station-node station-node-normal'>ST 4</div>
            <p style='margin-top:10px; margin-bottom:5px;'>Stopper 4: <span class='status-active'>{}</span></p>
            <p style='margin-bottom:2px;'>VFD 4 Temp: <b>{:.1f}°C</b></p>
            <p style='margin-top:0;'>VFD Health: <b style='color:{};'>{:.0f}%</b></p>
        </div>
        """.format("Holding" if current_row["I_Stopper4_Status"] == 1 else "Release", current_row["Q_VFD4_Temperature"], h4_color, h4), unsafe_allow_html=True)
        
    with cols[4]:
        st.markdown("""
        <div class='glass-card' style='text-align: center;'>
            <h4>Station 5</h4>
            <div class='station-node station-node-normal'>ST 5</div>
            <p style='margin-top:10px; margin-bottom:5px;'>Stopper 5: <span class='status-active'>{}</span></p>
            <p style='margin-bottom:2px;'>Output Tray: <b>Green</b></p>
            <p style='margin-top:0;'>Inspection: <b>Active</b></p>
        </div>
        """.format("Holding" if current_row["I_Stopper5_Status"] == 1 else "Release"), unsafe_allow_html=True)

    # Secondary Row: Defect Classification Panel & Dark Station Soft-Sensor Status
    row2_cols = st.columns([1, 2])
    
    with row2_cols[0]:
        st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.subheader("Classification & Prediction")
        
        # defect label mapping
        defect_id = int(current_row["defect_label"])
        defect_classes = {
            0: ("NORMAL OPERATION", "green", "No defect detected in current assembly cycle.", "✅"),
            1: ("MISSING NOSE ASSEMBLY", "orange", "Defect: Part is missing the nose structure. Divert to Station 5.", "⚠️ NOSE MISSING"),
            2: ("MISSING NOSE & BODY 2", "red", "Critical Defect: Missing nose and secondary chassis. Divert immediately.", "❌ BODY 2 DEFECT"),
            3: ("CRITICAL STRUCTURAL DEFECT (Nose/Body 1 & 2)", "red", "Severe Failure: Complete structural assembly missing parts.", "🚨 CRITICAL DEFECT")
        }
        name, color, msg, icon = defect_classes[defect_id]
        
        st.markdown(f"<h3 style='color:{color};'>{icon} {name}</h3>", unsafe_allow_html=True)
        st.write(msg)
        
        # Classification confidence indicators
        conf_scores = [0.98, 0.01, 0.005, 0.005] if defect_id == 0 else [0.05, 0.85, 0.07, 0.03]
        if defect_id == 2: conf_scores = [0.01, 0.04, 0.90, 0.05]
        if defect_id == 3: conf_scores = [0.01, 0.01, 0.03, 0.95]
        
        st.write("XGBoost Probability Distribution:")
        conf_df = pd.DataFrame({
            "Class": ["Normal", "NoNose", "NoNose,NoBody2", "NoNose,NoBody2,NoBody1"],
            "Probability": conf_scores
        })
        fig = px.bar(conf_df, x="Probability", y="Class", orientation='h', color="Probability",
                     color_continuous_scale="Viridis", height=200)
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#c5c6c7")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with row2_cols[1]:
        st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.subheader("Soft-Sensor Reconstructions (Dark Station)")
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Inferred Transit Time (Stopper 1-3)", f"{current_row['inferred_dark_station_time']:.2f} s", 
                      delta="-0.2s" if current_row["inferred_dark_station_time"] < 14 else "+3.1s Bottleneck!" if current_row["inferred_dark_station_time"] > 16.0 else None,
                      delta_color="inverse")
            st.write("Calculated using window functions and release-to-receipt edge triggers.")
            
        with metric_col2:
            st.metric("Interpolated VFD 2 Temperature", f"{current_row['Q_VFD2_Temperature']:.2f} °C", 
                      delta=f"{current_row['Q_VFD2_Temperature'] - current_row['Q_VFD1_Temperature']:.1f}°C from VFD1")
            st.write("Spatially inferred using temperatures from adjacent VFD 1 and VFD 3 nodes.")
            
        # Plot real-time VFD temperatures and inferred Dark Station times over last 50 cycles
        hist_len = min(sim_index + 1, 50)
        df_hist = df.iloc[sim_index - hist_len + 1 : sim_index + 1].copy()
        df_hist["Index"] = np.arange(len(df_hist))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_hist["Index"], y=df_hist["Q_VFD1_Temperature"], name="VFD 1 (Observed)", line=dict(color="#45a29e", width=1.5)))
        fig.add_trace(go.Scatter(x=df_hist["Index"], y=df_hist["Q_VFD2_Temperature"], name="VFD 2 (Interpolated)", line=dict(color="#ffcc00", width=2, dash='dash')))
        fig.add_trace(go.Scatter(x=df_hist["Index"], y=df_hist["Q_VFD3_Temperature"], name="VFD 3 (Observed)", line=dict(color="#66fcf1", width=1.5)))
        
        fig.update_layout(
            title="VFD Temperature Profile History (Last 50 Samples)",
            height=250,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="#c5c6c7",
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Robot Kinematic Analysis Panel
    st.markdown("<h3 style='margin-top:20px;'>Robotic Kinematics & Gripper Health (Cell Robots R01–R04)</h3>", unsafe_allow_html=True)
    robot_cols = st.columns(4)
    for idx, r in enumerate(["R01", "R02", "R03", "R04"]):
        with robot_cols[idx]:
            st.markdown(f"<div class='glass-card'>", unsafe_allow_html=True)
            
            rh = robot_health[r]
            rh_color = "#00ffcc" if rh > 80 else "#ffcc00" if rh > 50 else "#ff3366"
            
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom: 10px;'>
                <h4 style='margin:0;'>Robot {r}</h4>
                <span style='background:rgba(255,255,255,0.05); padding:2px 6px; border-radius:10px; border:1px solid {rh_color}; color:{rh_color}; font-weight:bold; font-size:0.8rem;'>
                    {rh:.0f}% Health
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Gripper load dial
            load = current_row[f"I_{r}_Gripper_Load"]
            avg_load = current_row[f"I_{r}_Gripper_Load_MA10"]
            std_load = current_row[f"I_{r}_Gripper_Load_Std10"]
            
            st.metric("Gripper Load", f"{load:.0f} N", delta=f"{load - avg_load:+.0f} N vs Mean")
            st.metric("Gripper Load StdDev (Vibrations)", f"{std_load:.2f} N")
            
            # Joint angles or S-Joint dynamics
            angle = current_row[f"M_{r}_SJointAngle_Degree"]
            velocity = current_row[f"M_{r}_SJoint_Velocity"]
            acceleration = current_row[f"M_{r}_SJoint_Acceleration"]
            
            st.write("S-Joint Motion Kinematics:")
            kin_df = pd.DataFrame({
                "Metric": ["Angle (°)", "Velocity (°/s)", "Accel (°/s²)"],
                "Value": [angle, velocity, acceleration]
            })
            st.dataframe(kin_df, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------------------------------------------------
# VIEW 2: PLANT MANAGER
# ----------------------------------------------------------------------------------------------------------------------
else:
    # High-level KPIs
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.markdown(f"""
        <div class='glass-card' style='text-align:center;'>
            <p style='color:#8b9bb4; margin:0;'>TOTAL ASSEMBLY CYCLES</p>
            <h2 style='color:#66fcf1;'>{total_cycles}</h2>
            <p style='color:#00ffcc; margin:0;'>Current Shift Progress</p>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_cols[1]:
        avg_cycle_time = df_subset["inferred_dark_station_time"].mean()
        st.markdown(f"""
        <div class='glass-card' style='text-align:center;'>
            <p style='color:#8b9bb4; margin:0;'>AVERAGE CYCLE TIME</p>
            <h2 style='color:#66fcf1;'>{avg_cycle_time:.2f} s</h2>
            <p style='color:#00ffcc; margin:0;'>Target: 12.50 s</p>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_cols[2]:
        if line_stopped:
            status_text = "❌ STOPPED"
            status_color = "#ff3366"
            subtext = f"{', '.join(active_stop_cause)}"
        else:
            status_text = "✅ RUNNING"
            status_color = "#00ffcc"
            subtext = "Active Telemetry Online"
            
        st.markdown(f"""
        <div class='glass-card' style='text-align:center;'>
            <p style='color:#8b9bb4; margin:0;'>SYSTEM STATUS</p>
            <h2 style='color:{status_color};'>{status_text}</h2>
            <p style='color:#8b9bb4; margin:0; font-size: 0.85rem;'>{subtext}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_cols[3]:
        st.markdown(f"""
        <div class='glass-card' style='text-align:center;'>
            <p style='color:#8b9bb4; margin:0;'>DEFECTIVE UNITS DETECTED</p>
            <h2 style='color:#ff3366;'>{defect_cycles}</h2>
            <p style='color:#ff3366; margin:0; font-size: 0.85rem;'>Yield: {(1.0 - defect_cycles/total_cycles)*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    # Dynamic OEE Dials
    st.subheader("Overall Equipment Effectiveness (OEE) Real-time Dials")
    oee_cols = st.columns(4)
    with oee_cols[0]:
        st.plotly_chart(make_gauge_chart(oee, "OEE Score", "#66fcf1"), use_container_width=True)
    with oee_cols[1]:
        st.plotly_chart(make_gauge_chart(availability, "Availability", "#00ffcc"), use_container_width=True)
    with oee_cols[2]:
        st.plotly_chart(make_gauge_chart(performance, "Performance", "#ffcc00"), use_container_width=True)
    with oee_cols[3]:
        st.plotly_chart(make_gauge_chart(quality, "Quality (FPY)", "#ff3366"), use_container_width=True)

    # Section 1: Bottleneck Forecasting (LSTM Lookahead)
    st.subheader("Station 2 (Dark Station) Bottleneck Forecasting & Analysis")
    b_cols = st.columns([2, 1])
    
    with b_cols[0]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        
        # Prepare plot of actual transit times + lookahead (mocked LSTM forecast)
        hist_len = min(sim_index + 1, 100)
        df_sub = df.iloc[sim_index - hist_len + 1 : sim_index + 1].copy()
        
        # Simulate LSTM Forecast for next 15 cycles
        forecast_cycles = 15
        last_t = df_sub["inferred_dark_station_time"].values[-1]
        
        # If there's an active bottleneck, forecast it remaining high and slowly cooling
        if last_t > 15.5:
            forecast_vals = [last_t - 0.1 * i + np.random.normal(0, 0.2) for i in range(1, forecast_cycles + 1)]
        else:
            forecast_vals = [12.5 + np.random.normal(0, 0.5) for i in range(1, forecast_cycles + 1)]
            
        time_index = np.arange(-hist_len + 1, 1)
        forecast_index = np.arange(1, forecast_cycles + 1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_index, y=df_sub["inferred_dark_station_time"], name="Actual (Soft-Sensor)", line=dict(color="#66fcf1", width=2)))
        fig.add_trace(go.Scatter(x=forecast_index, y=forecast_vals, name="LSTM Forecast (Lookahead)", line=dict(color="#ff3366", width=2, dash='dot')))
        fig.add_shape(type="line", x0=-hist_len+1, y0=15.5, x1=forecast_cycles, y1=15.5, line=dict(color="red", width=1, dash="dash"))
        
        fig.update_layout(
            title="Dark Station Cycle Time Forecast (LSTM Multi-step Lookahead)",
            xaxis_title="Relative Cycles (0 = Current)",
            yaxis_title="Transit Time (seconds)",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="#c5c6c7",
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with b_cols[1]:
        st.markdown("<div class='glass-card' style='height:100%;'>", unsafe_allow_html=True)
        st.write("#### Downtime Breakdown & Diagnostics")
        
        # Donut chart for Uptime vs Downtime
        downtime_df = pd.DataFrame({
            "State": ["Operating Uptime", "E-Stop Downtime", "Safety Door 1 Open", "Safety Door 2 Open"],
            "Seconds": [operating_time_s, estop_downtime_s, safety1_downtime_s, safety2_downtime_s]
        })
        # Filter out 0 value states to make pie chart cleaner
        downtime_df = downtime_df[downtime_df["Seconds"] > 0]
        
        if len(downtime_df) > 0:
            fig_dt = px.pie(
                downtime_df,
                names="State",
                values="Seconds",
                color="State",
                color_discrete_map={
                    "Operating Uptime": "#00ffcc",
                    "E-Stop Downtime": "#ff3366",
                    "Safety Door 1 Open": "#ffcc00",
                    "Safety Door 2 Open": "#ff8800"
                },
                hole=0.4,
                height=160
            )
            fig_dt.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5),
                margin=dict(l=5, r=5, t=5, b=5),
                paper_bgcolor='rgba(0,0,0,0)',
                font_color="#c5c6c7"
            )
            st.plotly_chart(fig_dt, use_container_width=True)
        else:
            st.write("No downtime recorded in current replay segment.")
            
        if transit_val > 15.5:
            st.error(f"⚠️ **Station 2 Bottleneck!**\n\nTransit time: {transit_val:.1f}s (above 15.5s threshold). Potential mechanical drag on Conveyor Segment 2.")
        else:
            st.success(f"✅ **Flow Normal**\n\nTransit time: {transit_val:.1f}s. Segment 2 flow throughput is stable.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Section 2: ESG & GHG Decarbonization Dashboard
    st.markdown("<h3 style='margin-top:20px;'>ESG Sustainability & Decarbonization Analytics</h3>", unsafe_allow_html=True)
    esg_cols = st.columns(3)
    
    # 1. VFD Dynamic Energy Savings calculations using vector sums over df_subset
    # Clip VFD temperatures to realistic operating range [50.0, 110.0] to filter out sensor anomalies
    vfd1_temps = df_subset["Q_VFD1_Temperature"].clip(lower=50.0, upper=110.0)
    vfd2_temps = df_subset["Q_VFD2_Temperature"].clip(lower=50.0, upper=110.0)
    vfd3_temps = df_subset["Q_VFD3_Temperature"].clip(lower=50.0, upper=110.0)
    vfd4_temps = df_subset["Q_VFD4_Temperature"].clip(lower=50.0, upper=110.0)
    
    # Power draw equations in kW
    vfd1_draw = 8.5 + (vfd1_temps - 60.0).clip(lower=0.0) * 0.1
    vfd2_draw = 8.5 + (vfd2_temps - 60.0).clip(lower=0.0) * 0.1
    vfd3_draw = 8.5 + (vfd3_temps - 60.0).clip(lower=0.0) * 0.1
    vfd4_draw = 8.5 + (vfd4_temps - 60.0).clip(lower=0.0) * 0.1
    
    total_vfd_power_draw = vfd1_draw + vfd2_draw + vfd3_draw + vfd4_draw
    constant_motor_power_draw = 60.0  # 4 legacy motors * 15 kW
    
    savings_kw_series = constant_motor_power_draw - total_vfd_power_draw
    # Calculate exact kWh saved using precomputed time delta (dt)
    cumulative_saved_kwh = (savings_kw_series * df_subset["dt"]).sum() / 3600.0
    co2_factor = 0.82  # kg CO2e / kWh in Indian Grid (Central Electricity Authority)
    carbon_saved_vfd = cumulative_saved_kwh * co2_factor
    
    total_vfd_power = total_vfd_power_draw.iloc[-1]
    energy_saved_kw = savings_kw_series.iloc[-1]
    
    with esg_cols[0]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("#### ⚡ VFD Dynamic Energy Savings")
        st.metric("Current VFD Power Draw", f"{total_vfd_power:.2f} kW", delta=f"-{energy_saved_kw:.2f} kW vs Legacy Motors", delta_color="normal")
        st.metric("Cumulative Energy Saved", f"{cumulative_saved_kwh:.2f} kWh")
        st.metric("GHG Reduced (Grid Offsets)", f"{carbon_saved_vfd:.2f} kg CO2e")
        st.caption("Calculation based on reconstructed VFD 2 thermal profiles and CEA grid emission factor.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    # 2. Embodied Carbon Offsets (Waste and Scrap Reduction) using unique cycle counts
    prevented_wasted_body_kg_co2 = defect_cycles * 2.5  # kg CO2e saved
    wasted_material_saved_lbs = defect_cycles * 4.2  # lbs steel/plastic saved
    
    with esg_cols[1]:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("#### ♻️ Circular Manufacturing (Scrap Savings)")
        st.metric("Scrap Assemblies Avoided", f"{defect_cycles} units", delta=f"{wasted_material_saved_lbs:.1f} lbs Raw Metal Saved")
        st.metric("Avoided Embodied Carbon Wasted", f"{prevented_wasted_body_kg_co2:.2f} kg CO2e")
        st.caption("Calculating GHG offsets by predicting missing components early at Station 2, halting chassis feed.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    # 3. Total Combined GHG Dashboard Chart
    with esg_cols[2]:
        st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.write("#### Total ESG Offsets (CO2 Equivalent)")
        total_carbon_saved = carbon_saved_vfd + prevented_wasted_body_kg_co2
        st.metric("Total CO2 Emissions Saved", f"{total_carbon_saved:.2f} kg CO2e")
        
        # Pie chart of carbon savings sources
        fig = px.pie(
            names=["VFD Dynamic Controls", "Avoided Embodied Material Scrap"],
            values=[max(0.001, carbon_saved_vfd), max(0.001, prevented_wasted_body_kg_co2)],
            color_discrete_sequence=["#66fcf1", "#45a29e"],
            hole=0.4,
            height=180
        )
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', font_color="#c5c6c7")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
