# =====================================================================
# PredictivePulse ETL Pipeline
# Developed by: Rahul & Team (IIT ISM Dhanbad)
# Accenture Innovation Challenge 2026 Submission
# =====================================================================
import os
import sys
import numpy as np
import pandas as pd

# Ensure data output directories exist
os.makedirs("data/bronze", exist_ok=True)
os.makedirs("data/silver", exist_ok=True)
os.makedirs("data/gold", exist_ok=True)

# Datasets path
ANALOG_DIR = r"c:\Users\Rahul\OneDrive - Indian Institute of Technology Indian School of Mines Dhanbad\ISM\Hackathons\AIC\accenture_innovation_challenge\Analog"

print("Initializing Predictive Pulse ETL Pipeline...")

# Check for PySpark
use_pyspark = True
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import *
    from pyspark.sql.types import *
    from pyspark.sql.window import Window
    
    # Try starting Spark session
    spark = SparkSession.builder \
        .appName("PredictivePulse_ETL") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    print("PySpark successfully initialized.")
except Exception as e:
    print(f"Warning: PySpark initialization failed ({e}). Falling back to Pandas/NumPy emulation mode.")
    use_pyspark = False

def run_pyspark_pipeline():
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, lag, when, lit, last, coalesce, avg, stddev
    from pyspark.sql.window import Window
    
    # -------------------------------------------------------------
    # 1. BRONZE LAYER: Ingestion and Masking
    # -------------------------------------------------------------
    print("\n--- Processing Bronze Layer ---")
    
    # Helper to load data in PySpark (using Pandas fallback for local XLSX)
    def load_bronze_source(file_name):
        file_path = os.path.join(ANALOG_DIR, file_name)
        print(f"Loading {file_name} into Bronze...")
        
        # Databricks supports spark.read.format("excel") but locally we load via pandas & convert to spark df
        import pandas as pd
        if file_name.endswith('.csv'):
            try:
                # Conveyor_Signals.csv is actually an Excel zip workbook!
                pdf = pd.read_excel(file_path)
            except Exception:
                pdf = pd.read_csv(file_path)
        else:
            pdf = pd.read_excel(file_path)
            
        # Clean column names (strip spaces, resolve parentheses)
        pdf.columns = [c.strip().replace(" ", "_").replace("(", "").replace(")", "") for c in pdf.columns]
        
        # Ensure time is read as string for consistent parsing
        if '_time' in pdf.columns:
            pdf['_time'] = pdf['_time'].astype(str)
            
        # Coerce telemetry columns to float to avoid Spark/Arrow Double vs Long Type conflicts
        for col_name in pdf.columns:
            if col_name not in ["_time", "Description", "I_MHS_GreenRocketTray"]:
                pdf[col_name] = pd.to_numeric(pdf[col_name], errors='coerce').astype(float)
            
        return spark.createDataFrame(pdf)

    # Ingest telemetry sources
    df_conveyor_raw = load_bronze_source("Conveyor_Signals.csv")
    df_safety_raw = load_bronze_source("FFCellSafetyManagement.xlsx")
    df_cycle_raw = load_bronze_source("FFCell_CycleManagement.xlsx")
    df_r01_raw = load_bronze_source("R01_Data.xlsx")
    df_r02_raw = load_bronze_source("R02_Data.xlsx")
    df_r03_raw = load_bronze_source("R03_Data.xlsx")
    df_r04_raw = load_bronze_source("R04_Data.xlsx")

    # MASKING STRATEGY: Intentionally drop Stopper 2 Status and VFD 2 Temperature in Bronze
    print("Applying Masking Strategy on Conveyor Signals (Dark Station)...")
    df_conveyor_bronze = df_conveyor_raw.drop("I_Stopper2_Status", "Q_VFD2_Temperature")
    
    # Save to Bronze storage (as delta or parquet locally)
    df_conveyor_bronze.write.mode("overwrite").parquet("data/bronze/conveyor")
    df_safety_raw.write.mode("overwrite").parquet("data/bronze/safety")
    df_cycle_raw.write.mode("overwrite").parquet("data/bronze/cycle")
    df_r01_raw.write.mode("overwrite").parquet("data/bronze/r01")
    df_r02_raw.write.mode("overwrite").parquet("data/bronze/r02")
    df_r03_raw.write.mode("overwrite").parquet("data/bronze/r03")
    df_r04_raw.write.mode("overwrite").parquet("data/bronze/r04")
    print("Bronze tables saved successfully.")

    # -------------------------------------------------------------
    # 2. SILVER LAYER: Data Cleaning, Reconstruct Masked Fields, Soft Sensor
    # -------------------------------------------------------------
    print("\n--- Processing Silver Layer ---")
    
    # Reload from Bronze
    conveyor = spark.read.parquet("data/bronze/conveyor")
    safety = spark.read.parquet("data/bronze/safety")
    cycle = spark.read.parquet("data/bronze/cycle")
    r01 = spark.read.parquet("data/bronze/r01")
    r02 = spark.read.parquet("data/bronze/r02")
    r03 = spark.read.parquet("data/bronze/r03")
    r04 = spark.read.parquet("data/bronze/r04")

    # Clean & Cast columns to correct datatypes
    # Parse string times to Timestamp type
    def preprocess_silver(df, name):
        # Standardize timestamp parsing to handle formats with or without milliseconds
        df = df.withColumn(
            "timestamp", 
            coalesce(
                to_timestamp(col("_time"), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"),
                to_timestamp(col("_time"), "yyyy-MM-dd'T'HH:mm:ss'Z'")
            )
        )
        
        # Cast numeric telemetry columns to Double
        for col_name in df.columns:
            if col_name not in ["_time", "timestamp", "Description", "I_MHS_GreenRocketTray"]:
                df = df.withColumn(col_name, col(col_name).cast(DoubleType()))
        return df

    conveyor = preprocess_silver(conveyor, "conveyor")
    safety = preprocess_silver(safety, "safety")
    cycle = preprocess_silver(cycle, "cycle")
    r01 = preprocess_silver(r01, "r01")
    r02 = preprocess_silver(r02, "r02")
    r03 = preprocess_silver(r03, "r03")
    r04 = preprocess_silver(r04, "r04")

    # Join streams on timestamp
    # Outer join protects against slight timestamp drift, but since they are fully aligned, inner works too.
    # We will use inner join for unified multi-sensor telemetry rows.
    df_joined = conveyor.join(safety.drop("_time"), on="timestamp", how="inner") \
                        .join(cycle.drop("_time"), on="timestamp", how="inner") \
                        .join(r01.drop("_time"), on="timestamp", how="inner") \
                        .join(r02.drop("_time"), on="timestamp", how="inner") \
                        .join(r03.drop("_time"), on="timestamp", how="inner") \
                        .join(r04.drop("_time"), on="timestamp", how="inner")

    # RECONSTRUCT TEMPERATURE: Spatial interpolation of VFD2 using VFD1 and VFD3
    print("Reconstructing Q_VFD2_Temperature via spatial VFD interpolation...")
    df_joined = df_joined.withColumn("Q_VFD2_Temperature", (col("Q_VFD1_Temperature") + col("Q_VFD3_Temperature")) / 2.0)

    # SOFT SENSOR MATH: Infer Stopper 2 Transit Time
    # Order by timestamp for window functions
    window_spec = Window.orderBy("timestamp")
    
    # Detect transitions
    # Stopper 1 release: status goes from 1 (holding) to 0 (releasing)
    # Stopper 3 receipt: status goes from 0 (empty) to 1 (holding)
    df_joined = df_joined.withColumn("prev_stopper1", lag("I_Stopper1_Status", 1).over(window_spec)) \
                          .withColumn("prev_stopper3", lag("I_Stopper3_Status", 1).over(window_spec))
    
    df_joined = df_joined.withColumn(
        "is_release1", 
        (col("I_Stopper1_Status") == 0.0) & (col("prev_stopper1") == 1.0)
    )
    df_joined = df_joined.withColumn(
        "is_receipt3", 
        (col("I_Stopper3_Status") == 1.0) & (col("prev_stopper3") == 0.0)
    )

    # Extract timestamps
    df_joined = df_joined.withColumn(
        "t_release1", 
        when(col("is_release1"), col("timestamp").cast("double")).otherwise(lit(None))
    )
    df_joined = df_joined.withColumn(
        "t_receipt3", 
        when(col("is_receipt3"), col("timestamp").cast("double")).otherwise(lit(None))
    )

    # Forward fill Stopper 1 release timestamps
    ffill_window = Window.orderBy("timestamp").rowsBetween(Window.unboundedPreceding, 0)
    df_joined = df_joined.withColumn("last_release1_time", last("t_release1", ignorenulls=True).over(ffill_window))

    # Calculate delta on receipt at Stopper 3
    df_joined = df_joined.withColumn(
        "transit_time_seconds",
        when(col("is_receipt3"), col("timestamp").cast("double") - col("last_release1_time")).otherwise(lit(None))
    )

    # Forward fill the transit time to get a continuous feature
    df_joined = df_joined.withColumn("inferred_dark_station_time", last("transit_time_seconds", ignorenulls=True).over(ffill_window))
    
    # Fill leading null values with default (12.5 seconds, the median transit time)
    df_joined = df_joined.withColumn("inferred_dark_station_time", coalesce(col("inferred_dark_station_time"), lit(12.5)))

    # Clean intermediate columns
    df_joined = df_joined.drop("prev_stopper1", "prev_stopper3", "is_release1", "is_receipt3", "t_release1", "t_receipt3", "last_release1_time", "transit_time_seconds")

    # Save to Silver Storage
    df_joined.write.mode("overwrite").parquet("data/silver/integrated")
    print("Silver integrated telemetry table saved successfully.")

    # -------------------------------------------------------------
    # 3. GOLD LAYER: Feature Engineering
    # -------------------------------------------------------------
    print("\n--- Processing Gold Layer ---")
    
    silver_df = spark.read.parquet("data/silver/integrated")
    
    # Order for lag operations
    gold_window = Window.orderBy("timestamp")

    # Time delta between rows (in seconds)
    silver_df = silver_df.withColumn(
        "dt", 
        coalesce(col("timestamp").cast("double") - lag("timestamp", 1).over(gold_window).cast("double"), lit(1.0))
    )
    silver_df = silver_df.withColumn("dt", when(col("dt") == 0.0, 1.0).otherwise(col("dt")))

    # 3.1 Robot Joint Kinematics Features (Velocities & Accelerations)
    print("Engineering robot joint velocities and accelerations (R1 to R4)...")
    for r in ["R01", "R02", "R03", "R04"]:
        for joint in ["B", "L", "R", "S", "T", "U"]:
            col_name = f"M_{r}_{joint}JointAngle_Degree"
            vel_name = f"M_{r}_{joint}Joint_Velocity"
            acc_name = f"M_{r}_{joint}Joint_Acceleration"
            
            # Velocity: first derivative
            silver_df = silver_df.withColumn(
                vel_name, 
                (col(col_name) - lag(col_name, 1).over(gold_window)) / col("dt")
            )
            # Acceleration: second derivative
            silver_df = silver_df.withColumn(
                acc_name, 
                (col(vel_name) - lag(vel_name, 1).over(gold_window)) / col("dt")
            )
            
            # Fill nulls with 0.0
            silver_df = silver_df.withColumn(vel_name, coalesce(col(vel_name), lit(0.0)))
            silver_df = silver_df.withColumn(acc_name, coalesce(col(acc_name), lit(0.0)))

    # 3.2 Gripper Load Rolling Features
    print("Engineering rolling averages and standard deviations for robot grippers...")
    roll_window_10 = Window.orderBy("timestamp").rowsBetween(-10, 0)
    roll_window_60 = Window.orderBy("timestamp").rowsBetween(-60, 0)

    for r in ["R01", "R02", "R03", "R04"]:
        load_col = f"I_{r}_Gripper_Load"
        
        # Rolling averages
        silver_df = silver_df.withColumn(f"I_{r}_Gripper_Load_MA10", avg(load_col).over(roll_window_10))
        silver_df = silver_df.withColumn(f"I_{r}_Gripper_Load_MA60", avg(load_col).over(roll_window_60))
        
        # Rolling standard deviation (vibration proxy)
        silver_df = silver_df.withColumn(f"I_{r}_Gripper_Load_Std10", stddev(load_col).over(roll_window_10))
        
        # Fill nulls
        silver_df = silver_df.withColumn(f"I_{r}_Gripper_Load_Std10", coalesce(col(f"I_{r}_Gripper_Load_Std10"), lit(0.0)))

    # 3.3 Bottleneck Features
    # Rolling average of inferred Dark Station transit time (50 rows)
    silver_df = silver_df.withColumn(
        "Dark_Station_Time_MA50", 
        avg("inferred_dark_station_time").over(Window.orderBy("timestamp").rowsBetween(-50, 0))
    )
    # Binary bottleneck indicator flag (1 if transit time > 15.5 seconds, representing 20% slowdown)
    silver_df = silver_df.withColumn(
        "is_bottleneck",
        when(col("inferred_dark_station_time") > 15.5, 1).otherwise(0)
    )

    # 3.4 Target Variable Mapping
    # Convert Description string to integers:
    # 0: Good, 1: NoNose, 2: NoNose,NoBody2, 3: NoNose,NoBody2,NoBody1
    print("Mapping defect descriptions to multi-class classification labels...")
    silver_df = silver_df.withColumn(
        "defect_label",
        when(col("Description") == "NoNose", 1) \
        .when(col("Description") == "NoNose,NoBody2", 2) \
        .when(col("Description") == "NoNose,NoBody2,NoBody1", 3) \
        .otherwise(0)
    )

    # Drop intermediate columns
    silver_df = silver_df.drop("dt")

    # Save Gold Table to storage
    silver_df.write.mode("overwrite").parquet("data/gold/features")
    print("Gold features table saved as Parquet.")
    
    # Convert to pandas for XGBoost training
    pandas_gold = silver_df.toPandas()
    
    # Train the XGBoost model to classify defects
    train_xgboost(pandas_gold)

def train_xgboost(df):
    print("\n--- Training XGBoost Defect Classifier ---")
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    import pickle
    
    # Filter columns to use as features
    # Exclude identifier and string target columns
    exclude_cols = ["_time", "timestamp", "Description", "I_MHS_GreenRocketTray", "defect_label"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df["defect_label"].copy()
    
    # Convert any remaining objects or string values to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
        
    print(f"Training features count: {len(feature_cols)}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train classifier
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=4,
        random_state=42,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and features list
    os.makedirs("models", exist_ok=True)
    with open("models/xgboost_defect_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    print("XGBoost model saved to models/xgboost_defect_model.pkl")


def run_pandas_pipeline():
    # Standard library fallback logic for local runs with no PySpark configuration
    print("Executing fallback pipeline using Pandas and NumPy...")
    import pandas as pd
    
    def load_pandas(file_name):
        file_path = os.path.join(ANALOG_DIR, file_name)
        if file_name.endswith('.csv'):
            try:
                df = pd.read_excel(file_path)
            except Exception:
                df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        df.columns = [c.strip().replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]
        # Coerce numeric columns to float to prevent ArrowInvalid conversion failures in Parquet saving
        for col in df.columns:
            if col not in ["_time", "Description", "I_MHS_GreenRocketTray"]:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        return df

    # Ingest raw files (Bronze equivalent)
    print("Loading data files...")
    conveyor = load_pandas("Conveyor_Signals.csv")
    safety = load_pandas("FFCellSafetyManagement.xlsx")
    cycle = load_pandas("FFCell_CycleManagement.xlsx")
    r01 = load_pandas("R01_Data.xlsx")
    r02 = load_pandas("R02_Data.xlsx")
    r03 = load_pandas("R03_Data.xlsx")
    r04 = load_pandas("R04_Data.xlsx")

    # Bronze masking
    print("Masking Stopper 2 Status and VFD 2 Temperature...")
    conveyor_masked = conveyor.drop(columns=["I_Stopper2_Status", "Q_VFD2_Temperature"], errors='ignore')
    
    # Save bronze
    conveyor_masked.to_parquet("data/bronze/conveyor.parquet", index=False)
    safety.to_parquet("data/bronze/safety.parquet", index=False)
    cycle.to_parquet("data/bronze/cycle.parquet", index=False)
    r01.to_parquet("data/bronze/r01.parquet", index=False)
    r02.to_parquet("data/bronze/r02.parquet", index=False)
    r03.to_parquet("data/bronze/r03.parquet", index=False)
    r04.to_parquet("data/bronze/r04.parquet", index=False)

    # Silver cleaning & joining
    print("\n--- Silver Layer Fallback ---")
    for df in [conveyor_masked, safety, cycle, r01, r02, r03, r04]:
        df['_time'] = pd.to_datetime(df['_time'], format="ISO8601")
        for col in df.columns:
            if col not in ["_time", "Description", "I_MHS_GreenRocketTray"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # Join on time
    df_joined = conveyor_masked.merge(safety, on="_time", how="inner")
    df_joined = df_joined.merge(cycle, on="_time", how="inner")
    df_joined = df_joined.merge(r01, on="_time", how="inner")
    df_joined = df_joined.merge(r02, on="_time", how="inner")
    df_joined = df_joined.merge(r03, on="_time", how="inner")
    df_joined = df_joined.merge(r04, on="_time", how="inner")

    df_joined["timestamp"] = df_joined["_time"]

    # Spatial reconstruction of VFD 2
    df_joined["Q_VFD2_Temperature"] = (df_joined["Q_VFD1_Temperature"] + df_joined["Q_VFD3_Temperature"]) / 2.0

    # Soft Sensor Math
    print("Reconstructing Stopper 2 Transit Time (Soft Sensor)...")
    df_joined = df_joined.sort_values("_time").reset_index(drop=True)
    
    # Detect transitions
    df_joined["prev_stopper1"] = df_joined["I_Stopper1_Status"].shift(1)
    df_joined["prev_stopper3"] = df_joined["I_Stopper3_Status"].shift(1)
    
    df_joined["is_release1"] = (df_joined["I_Stopper1_Status"] == 0) & (df_joined["prev_stopper1"] == 1)
    df_joined["is_receipt3"] = (df_joined["I_Stopper3_Status"] == 1) & (df_joined["prev_stopper3"] == 0)

    # Mark timestamps
    df_joined["t_release1"] = np.where(df_joined["is_release1"], df_joined["_time"].astype(np.int64) // 10**9, np.nan)
    df_joined["t_receipt3"] = np.where(df_joined["is_receipt3"], df_joined["_time"].astype(np.int64) // 10**9, np.nan)

    # Forward fill Stopper 1 release timestamps
    df_joined["last_release1_time"] = df_joined["t_release1"].ffill()

    # Time delta
    df_joined["transit_time_seconds"] = np.where(df_joined["is_receipt3"], (df_joined["_time"].astype(np.int64) // 10**9) - df_joined["last_release1_time"], np.nan)
    df_joined["inferred_dark_station_time"] = df_joined["transit_time_seconds"].ffill().fillna(12.5)

    # Clean intermediate
    df_joined = df_joined.drop(columns=["prev_stopper1", "prev_stopper3", "is_release1", "is_receipt3", "t_release1", "t_receipt3", "last_release1_time", "transit_time_seconds"])
    df_joined.to_parquet("data/silver/integrated.parquet", index=False)

    # Gold features
    print("\n--- Gold Layer Fallback ---")
    df_gold = df_joined.copy()
    df_gold["dt"] = (df_gold["_time"] - df_gold["_time"].shift(1)).dt.total_seconds().fillna(1.0)
    df_gold.loc[df_gold["dt"] == 0, "dt"] = 1.0

    # Robot joint kinematics
    print("Calculating joint velocities and accelerations...")
    for r in ["R01", "R02", "R03", "R04"]:
        for joint in ["B", "L", "R", "S", "T", "U"]:
            col_name = f"M_{r}_{joint}JointAngle_Degree"
            vel_name = f"M_{r}_{joint}Joint_Velocity"
            acc_name = f"M_{r}_{joint}Joint_Acceleration"
            
            df_gold[vel_name] = (df_gold[col_name] - df_gold[col_name].shift(1)) / df_gold["dt"]
            df_gold[acc_name] = (df_gold[vel_name] - df_gold[vel_name].shift(1)) / df_gold["dt"]
            df_gold[vel_name] = df_gold[vel_name].fillna(0.0)
            df_gold[acc_name] = df_gold[acc_name].fillna(0.0)

    # Gripper loads
    print("Calculating gripper load rolling stats...")
    for r in ["R01", "R02", "R03", "R04"]:
        load_col = f"I_{r}_Gripper_Load"
        df_gold[f"I_{r}_Gripper_Load_MA10"] = df_gold[load_col].rolling(10, min_periods=1).mean()
        df_gold[f"I_{r}_Gripper_Load_MA60"] = df_gold[load_col].rolling(60, min_periods=1).mean()
        df_gold[f"I_{r}_Gripper_Load_Std10"] = df_gold[load_col].rolling(10, min_periods=1).std().fillna(0.0)

    # Bottleneck features
    df_gold["Dark_Station_Time_MA50"] = df_gold["inferred_dark_station_time"].rolling(50, min_periods=1).mean()
    df_gold["is_bottleneck"] = np.where(df_gold["inferred_dark_station_time"] > 15.5, 1, 0)

    # Target Mapping
    print("Mapping target descriptions...")
    defect_map = {"NoNose": 1, "NoNose,NoBody2": 2, "NoNose,NoBody2,NoBody1": 3}
    df_gold["defect_label"] = df_gold["Description"].map(defect_map).fillna(0).astype(int)

    df_gold = df_gold.drop(columns=["dt"])
    
    # Save gold
    df_gold.to_parquet("data/gold/features.parquet", index=False)
    
    # Train XGBoost
    train_xgboost(df_gold)

if __name__ == "__main__":
    if use_pyspark:
        try:
            run_pyspark_pipeline()
        except Exception as e:
            print(f"Error during PySpark pipeline execution: {e}")
            print("Swapping to Pandas fallback...")
            run_pandas_pipeline()
    else:
        run_pandas_pipeline()
    print("\nETL Pipeline completed successfully!")
