import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="AeroAI Workbench", page_icon="✈️", layout="wide")

# Personal Branding Header
st.title("✈️ AeroAI: Data-Driven Flight Workbench")
st.markdown("##### **Made by: Youssef Lafrem**")
st.markdown("---")

# Create Multi-Tab Layout based on Engineering Principles
tab1, tab2, tab3 = st.tabs([
    "🌍 1. Atmosphere & Physics", 
    "🤖 2. Airfoil AI Predictor", 
    "📊 3. Flight Telemetry Analysis"
])

# ==========================================
# TAB 1: ATMOSPHERE & FLIGHT MECHANICS
# ==========================================
with tab1:
    st.header("Standard Atmosphere & Basic Lift Calculator")
    st.write("Calculate fundamental aerodynamic properties using classic textbook formulas (Anderson, *Introduction to Flight*).")
    
    col1, col2 = st.columns(2)
    
    with col1:
        altitude_m = st.slider("Altitude (meters)", 0, 11000, 1000)
        velocity_ms = st.slider("Airspeed (m/s)", 10, 300, 100)
        
    with col2:
        # Simple troposphere approximation for density
        temp_0 = 288.15 # Kelvin at sea level
        rho_0 = 1.225   # kg/m^3 at sea level
        temp_k = temp_0 - (0.0065 * altitude_m)
        rho = rho_0 * ((temp_k / temp_0) ** 4.256) # Air density approximation
        
        dynamic_pressure = 0.5 * rho * (velocity_ms ** 2)
        
        st.metric(label="Estimated Air Density (rho)", value=f"{rho:.3f} kg/m³")
        st.metric(label="Dynamic Pressure (q)", value=f"{dynamic_pressure:.1f} Pa")

# ==========================================
# TAB 2: AIRFOIL AI PREDICTOR
# ==========================================
with tab2:
    st.header("Machine Learning Airfoil Performance Predictor")
    st.write("Predict Lift ($C_L$) and Drag ($C_D$) coefficients instantly without running heavy CFD simulations.")
    
    alpha = st.slider("Angle of Attack (Alpha - degrees)", -5.0, 20.0, 4.0)
    
    # Mock AI / Physics-informed calculation for demo purposes
    # In a later step, we will plug a trained scikit-learn model here!
    cl_pred = 0.09 * alpha + 0.2  # Simplified lift curve slope approximation
    cd_pred = 0.01 + 0.0005 * (alpha ** 2) # Induced drag approximation
    
    col_a, col_b = st.columns(2)
    col_a.metric("Predicted Lift Coefficient (Cl)", f"{cl_pred:.3f}")
    col_b.metric("Predicted Drag Coefficient (Cd)", f"{cd_pred:.3f}")
    
    # Generate a dynamic Lift Curve Slope plot
    st.subheading = "Lift Curve Preview"
    alphas_range = np.linspace(-5, 20, 50)
    cls_range = 0.09 * alphas_range + 0.2
    
    fig, ax = plt.subplots()
    ax.plot(alphas_range, cls_range, label="AI Model Prediction", color="blue", linewidth=2)
    ax.scatter([alpha], [cl_pred], color="red", zorder=5, label="Current Selection")
    ax.set_xlabel("Angle of Attack (deg)")
    ax.set_ylabel("Lift Coefficient (Cl)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    
    st.pyplot(fig)

# ==========================================
# TAB 3: FLIGHT TELEMETRY ANALYSIS
# ==========================================
with tab3:
    st.header("Flight Data & Telemetry Explorer")
    st.write("Upload or preview flight sensor logs to analyze performance trends (Data-Driven Engineering approach).")
    
    # Generate mock flight log data for demonstration
    time_sec = np.arange(0, 100, 1)
    altitude_log = 1000 + 50 * time_sec - 0.2 * (time_sec ** 2)
    
    df_telemetry = pd.DataFrame({
        "Time (s)": time_sec,
        "Altitude (m)": altitude_log,
        "Engine Temp (C)": 400 + np.random.normal(0, 5, len(time_sec))
    })
    
    st.dataframe(df_telemetry.head(10))
    
    # Plot telemetry
    st.line_chart(df_telemetry.set_index("Time (s)")["Altitude (m)"])
