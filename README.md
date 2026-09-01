# PredictivePulse ⚡
### Real-Time Operations Digital Twin & Fault Diagnostics for Assembly Lines

Hey there! 👋 This is our team's project (**PredictivePulse**) developed for the **Accenture Innovation Challenge 2026 (Problem Track 4: DigitalTwin.ai)** by Rahul & Team from IIT ISM Dhanbad.

We built an operations digital twin for a vehicle manufacturing assembly line. It monitors live SCADA streams, predicts chassis assembly defects in real-time with machine learning, detects hidden bottlenecks, and tracks live ESG energy and carbon savings—all inside an interactive, dark-mode Streamlit operations console.

---

## 💡 Why We Built This & The Core Idea

In modern manufacturing, upgrading legacy factory stations with new smart PLCs, network modules, and wired sensors can cost millions of dollars and cause major plant downtime. These unmonitored zones are called **Dark Stations**.

Instead of buying expensive new hardware, we built **Physics-Informed Soft-Sensors**. By observing the sensors at adjacent stations (like Station 1 and Station 3), our software algorithms accurately reconstruct the missing motor temperatures and conveyor transit times for Station 2 in real time.

---

## 🖥️ How to Use the App (Step-by-Step Guide)

When you launch the Streamlit app, you have access to two specialized consoles and an interactive live simulator in the sidebar.

### 1. Sidebar Simulation Controller
The sidebar on the left lets you control and test the digital twin in real time:

- **View Selector**: Switch between the **Floor Operations Console** (for shop floor supervisors) and the **Plant Management & ESG Dashboard** (for plant executives).
- **Time-Series Replay Index Slider**: Scrub through the entire production shift step by step. Moving the slider updates all station statuses, robot kinematics, defect predictions, and OEE dials instantaneously.
- **Simulate Active Fault States**: Want to see how the system reacts to an emergency? Select a fault scenario from the dropdown to test the digital twin:
  - *Baseline Operations (Normal)*: Clean, normal operating telemetry.
  - *Thermal Deviation (VFD 1)*: Causes motor 1 to overheat, demonstrating how the soft-sensor estimates motor 2 heating up.
  - *Chassis Feed Transit Delay (Stopper 2)*: Simulates a mechanical jam at Dark Station 2, immediately triggering bottleneck alerts.
  - *Kinematic Fault State (Robot R04)*: Freezes Robot 4's S-joint movement to simulate a robotic mechanical stall.
  - *Mechanical Gripper Overload (Robot R03)*: Spikes Robot 3's gripper load to 4850 N and flags a high-severity chassis defect.

---

### 2. Console View 1: Floor Operations & Real-Time Diagnostics
This view is built for line supervisors and maintenance technicians on the factory floor:

- **Live Assembly Line Schematic**: Displays all 5 assembly stations in a horizontal flow. You can see whether each stopper is holding or releasing, current motor temperatures, and a live health percentage (0–100%). Station 2 is highlighted with a yellow dashed border to indicate it is a "Dark Station" whose values are inferred live by soft-sensors.
- **Classification & Defect Prediction Panel**: Powered by our trained XGBoost classifier. For the active assembly cycle, it identifies whether the chassis is normal or suffering from missing parts (such as a missing nose or missing secondary body structure), and displays the model's confidence distribution with clear operator instructions (e.g., divert to Station 5).
- **Soft-Sensor Reconstructions (Dark Station 2)**: Displays the calculated chassis transit time between Stopper 1 and Stopper 3, the estimated VFD 2 motor temperature, and an interactive 50-step historical line chart comparing the observed temperatures of VFD 1 and VFD 3 against the interpolated VFD 2 line.
- **Robotic Kinematics & Gripper Health (Robots R01–R04)**: Four dedicated cards for the cell robots. Each card shows:
  - Overall robot health badge (green/yellow/red).
  - Current pneumatic gripper load in Newtons with comparison to rolling averages.
  - Gripper vibration indicator (calculated from rolling standard deviation).
  - A clean kinematics table displaying the robot S-joint angle in degrees, angular velocity in degrees per second, and angular acceleration in degrees per second squared.

---

### 3. Console View 2: Plant Management & ESG Analytics Dashboard
This view gives plant managers and sustainability officers a high-level operational and environmental overview:

- **Executive KPI Cards**: Instant summary of total completed assembly cycles, average cycle duration compared to the 12.5-second target, live line operational status (Running vs. Stopped with active root causes like E-Stops or Open Doors), and total defect count.
- **Real-Time OEE Dials**: Four dynamic circular gauge dials showing:
  - *Overall OEE Score* (the master efficiency indicator).
  - *Availability Dial* (percentage of planned time the line was running).
  - *Performance Dial* (operating speed compared to ideal design cycle times).
  - *Quality / First Pass Yield Dial* (ratio of good, defect-free chassis produced).
- **Dark Station Bottleneck Forecasting (LSTM Lookahead)**: Plots past cycle transit times and projects a 15-cycle multi-step lookahead forecast against a red 15.5-second critical line. If cycle times begin trending upward, supervisors get an advance warning before a bottleneck halts the line.
- **Downtime Root-Cause Breakdown**: A donut chart breaking down total uptime against specific stoppage causes (HMI Emergency Stops, Safety Door 1 open events, and Safety Door 2 open events).
- **ESG Sustainability & Decarbonization Analytics**:
  - *VFD Dynamic Energy Savings*: Shows how much electrical power is saved by variable speed drives compared to legacy constant-speed motors, along with cumulative kilowatt-hours saved and grid carbon emissions reduced (using the Indian Central Electricity Authority grid factor of 0.82 kg CO2e per kWh).
  - *Circular Manufacturing Scrap Savings*: Calculates the raw metal (lbs) and embodied carbon saved by catching defects early at Station 2 before wasting more parts on a bad frame.
  - *Total Combined Carbon Offsets*: An interactive donut chart showing total greenhouse gas emissions avoided from energy efficiency and scrap prevention.

---

## ⚙️ How the System Works (Behind the Scenes)

We implemented an end-to-end Medallion Data Architecture (Bronze → Silver → Gold) to process sensor logs, reconstruct missing signals, and serve predictions:

### 1. Bronze Layer (Ingestion & Masking)
- Ingests raw SCADA conveyor signals, safety interlock logs, cycle management sheets, and joint angle/gripper load logs from 4 cell robots (`R01` through `R04`).
- We deliberately remove Station 2's direct sensor columns (`I_Stopper2_Status` and `Q_VFD2_Temperature`) during ingestion to simulate a true uninstrumented Dark Station.

### 2. Silver Layer (Integration & Soft-Sensors)
- Cleans and aligns all timestamps to standard ISO formats across all sensor files.
- Joins the telemetry streams into a single unified time-series table.
- **Reconstructs VFD 2 Temperature**: Calculates the spatial average of physical neighbors VFD 1 and VFD 3.
- **Reconstructs Stopper 2 Transit Time**: Detects the exact moment Stopper 1 opens and releases a carrier until Stopper 3 closes upon receiving it. If this transit time exceeds 15.5 seconds, it is flagged as an active bottleneck.

### 3. Gold Layer (Feature Engineering)
- Computes first and second numerical derivatives across 6 robot joint angles ($B, L, R, S, T, U$) to extract angular velocities and accelerations for each robot.
- Calculates 10-step and 60-step rolling averages and standard deviations for gripper loads to serve as early vibration proxies for mechanical wear.
- Formulates rolling transit averages and defect ground-truth labels.

### 4. Machine Learning Defect Classifier
- An XGBoost multi-class gradient boosting model is trained on the Gold features to classify chassis defect states (Normal, Missing Nose, Missing Nose & Body 2, Critical Structural Defect).
- The model outputs real-time class probabilities that are displayed directly in the dashboard.

### 5. Dual Execution Engine
- The ETL pipeline can run on **Apache Spark (PySpark)** for distributed big data processing, or automatically fall back to an optimized **Pandas & NumPy** engine for instant local execution without needing Spark installed.

---

## 📐 How the Key Metrics are Calculated

- **Availability**: Compares total operating time against downtime events (triggered whenever the HMI E-stop is pressed or safety doors 1/2 are opened).
- **Performance**: Compares actual production speed against the standard design cycle time (300 seconds per cell batch).
- **Quality (First Pass Yield)**: Measures the percentage of completed assembly cycles that had zero defect classifications from the XGBoost model.
- **OEE Score**: The product of Availability × Performance × Quality.
- **VFD Motor Health (0–100%)**: Starts at 100% and drops gradually if motor temperatures exceed 72°C.
- **Robot Health (0–100%)**: Evaluates mechanical gripper loads above 1500 N or vibration standard deviations above 120 N to penalize wear and tear.
- **Dynamic Energy Savings**: Calculates the difference between legacy 15 kW constant-speed motors and the active power curve of our VFDs, integrating power over time to calculate total kWh saved.
- **Carbon Offsets**: Multiplies cumulative kWh saved by the Indian grid emission factor (0.82 kg CO2e/kWh) and adds embodied carbon saved from diverted scrap units (2.5 kg CO2e per defect).

---

## 🛠️ Supervisor Operational Decision Guide

| When this happens... | What it means... | What the supervisor should do... |
| :--- | :--- | :--- |
| **Availability drops below 90%** | Frequent emergency stops or safety doors are being opened. | Check if safety door limit switches are loose or audit operator movement around the cell. |
| **Performance drops below 85%** | Conveyor Segment 2 cycle times are consistently above 15.5s. | Check Station 2 conveyor rollers for mechanical drag and verify drive belt tension. |
| **Quality drops below 98%** | XGBoost model is flagging multiple defect assemblies. | Inspect the upstream nose-cone loading feeder and check part alignment sensors. |
| **Robot Health drops below 60%** | Gripper vibrations or loads are unusually high. | Inspect pneumatic gripper seals, check air line pressures, and re-grease robot joints. |
| **VFD Health drops below 50%** | Drive motor temperature is running hot (> 82°C). | Clean the VFD cooling cabinet air filters and make sure exhaust fans are working properly. |

---

## 📁 Project Structure

```text
accenture_innovation_challenge/
├── .streamlit/
│   └── config.toml                  # Dark-mode styling and UI color configuration
├── Analog/                          # Raw SCADA datasets (telemetry, safety, robots)
│   ├── Conveyor_Signals.csv         # VFD temperatures, stoppers, tray states
│   ├── FFCellSafetyManagement.xlsx  # E-stops and safety door statuses
│   ├── FFCell_CycleManagement.xlsx  # Cycle count, cycle state, defect descriptions
│   ├── R01_Data.xlsx                # Robot 1 6-DOF joint angles & gripper load
│   ├── R02_Data.xlsx                # Robot 2 6-DOF joint angles & gripper load
│   ├── R03_Data.xlsx                # Robot 3 6-DOF joint angles & gripper load
│   └── R04_Data.xlsx                # Robot 4 6-DOF joint angles & gripper load
├── data/                            # Medallion pipeline output tables (Bronze/Silver/Gold)
├── models/                          # Serialized ML model and feature list
│   ├── xgboost_defect_model.pkl     # Trained XGBoost classifier
│   └── feature_cols.pkl             # Training feature schema
├── app.py                           # Streamlit Digital Twin application & UI portal
├── etl_pipeline.py                  # Medallion ETL & ML training pipeline
├── requirements.txt                 # Python dependencies
└── README.md                        # Documentation & user walkthrough
```

---

## 🚀 How to Run the Project Locally

### 1. Install Dependencies
Make sure you have Python 3.10+ installed:
```bash
pip install -r requirements.txt
```

### 2. Run the ETL Pipeline & ML Training
Run the data pipeline to process the raw files in `Analog/`, reconstruct soft-sensor values, engineer features, and train the XGBoost model:
```bash
python etl_pipeline.py
```
*(You can also skip this and click the "Run ETL Pipeline" button directly inside the app sidebar).*

### 3. Launch the Streamlit App
Start the digital twin portal:
```bash
python -m streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`.

---

## 👥 Team
- **Rahul & Team** — Indian Institute of Technology (ISM) Dhanbad
- **Challenge**: Accenture Innovation Challenge 2026 (Problem Track 4: DigitalTwin.ai)
