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

# Create Multi-Tab Layout
tab1, tab2, tab3 = st.tabs([
    "🌍 1. Atmosphere & Lift Force", 
    "🤖 2. Airfoil AI Predictor", 
    "📊 3. Flight Telemetry Analysis"
])

# ==========================================
# TAB 1: ATMOSPHERE & LIFT FORCE CALCULATOR
# ==========================================
with tab1:
    st.header("Standard Atmosphere & Lift Force Calculator")
    st.write("Explore how altitude, airspeed, wing area, and angle of attack combine to generate total aerodynamic lift force.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        altitude_m = st.slider("Altitude (meters)", 0, 15000, 5217, key="t1_alt")
        velocity_ms = st.slider("Airspeed (m/s)", 10, 300, 234, key="t1_vel")
        wing_area = st.slider("Wing Surface Area (m²)", 5.0, 100.0, 25.0, key="t1_area")
        alpha_t1 = st.slider("Angle of Attack (degrees)", -5.0, 25.0, 5.0, key="t1_alpha")
        
    with col2:
        # --- Safety Guard: Restrict altitude to valid troposphere bounds (0 to 11,000 m) ---
        safe_altitude = max(0.0, min(11000.0, float(altitude_m)))
        
        # --- Calculations ---
        temp_0 = 288.15 # Sea-level standard temperature (K)
        temp_k = temp_0 - (0.0065 * safe_altitude) # Temperature at altitude
        
        rho_0 = 1.225   # Sea-level standard density (kg/m^3)
        rho = rho_0 * ((temp_k / temp_0) ** 4.256) # Air density
        
        dynamic_pressure = 0.5 * rho * (velocity_ms ** 2) # Dynamic pressure (Pa)
        
        # --- Stall Model Logic ---
        stall_angle = 14.0
        if alpha_t1 <= stall_angle:
            cl_calculated = 0.11 * alpha_t1 + 0.2
            stall_status = "Attached Flow (Normal)"
        else:
            max_cl = 0.11 * stall_angle + 0.2
            excess_alpha = alpha_t1 - stall_angle
            cl_calculated = max_cl - 0.015 * (excess_alpha ** 1.5)
            stall_status = "⚠️ AIRFOIL STALLED (Flow Separation)"
        
        # Lift Force Formula: L = q * S * CL (in Newtons)
        lift_force = dynamic_pressure * wing_area * cl_calculated
        
        # Display Results
        if altitude_m > 11000:
            st.warning("⚠️ Altitude exceeds 11,000m troposphere limit. Clamped to 11,000m for ISA calculations.")
            
        st.info(f"**Flow State:** {stall_status}")
        st.metric(label="Calculated Temperature", value=f"{temp_k:.2f} K")
        st.metric(label="Estimated Air Density (rho)", value=f"{rho:.3f} kg/m³")
        st.metric(label="Dynamic Pressure (q)", value=f"{dynamic_pressure:.1f} Pa")
        st.metric(label="Total Lift Force (L)", value=f"{lift_force:,.1f} N")

    # Live Step-by-Step Math Breakdown Section
    st.markdown("---")
    st.subheader("📝 Step-by-Step Calculation Breakdown")
    st.write("Here is the exact math evaluated live using your current inputs:")
    
    st.latex(rf"1. \text{{ Temperature: }} T = 288.15 - (0.0065 \times {safe_altitude:.0f}) = {temp_k:.2f} \text{{ K}}")
    st.latex(rf"2. \text{{ Air Density: }} \rho = 1.225 \times \left(\frac{{{temp_k:.2f}}}{{288.15}}\right)^{{4.256}} = {rho:.3f} \text{{ kg/m}}^3")
    st.latex(rf"3. \text{{ Dynamic Pressure: }} q = \frac{1}{2} \times ({rho:.3f}) \times ({velocity_ms})^2 = {dynamic_pressure:.1f} \text{{ Pa}}")
    if alpha_t1 <= stall_angle:
        st.latex(rf"4. \text{{ Lift Coefficient (Linear): }} C_L = 0.11 \times ({alpha_t1}) + 0.2 = {cl_calculated:.3f}")
    else:
        st.latex(rf"4. \text{{ Lift Coefficient (Post-Stall Decay): }} C_L = {cl_calculated:.3f}")
    st.latex(rf"5. \text{{ Lift Force: }} L = q \times S \times C_L = {dynamic_pressure:.1f} \times {wing_area} \times {cl_calculated:.3f} = {lift_force:,.1f} \text{{ N}}")

# ==========================================
# TAB 2: AIRFOIL AI PREDICTOR
# ==========================================
with tab2:
    st.header("Machine Learning Airfoil Performance Predictor")
    st.write("Predict Lift ($C_L$) and Drag ($C_D$) coefficients with realistic post-stall physics and drag spiking.")
    
    alpha = st.slider("Angle of Attack (Alpha - degrees)", -5.0, 25.0, 4.0, key="t2_alpha")
    
    # Advanced Stall & Drag Polar Logic
    stall_angle = 14.0
    if alpha <= stall_angle:
        cl_pred = 0.11 * alpha + 0.2  
        cd_pred = 0.008 + 0.045 * (cl_pred ** 2) # Standard parabolic drag polar
        flow_state_t2 = "Attached Flow (Linear Region)"
    else:
        max_cl = 0.11 * stall_angle + 0.2
        excess_alpha = alpha - stall_angle
        cl_pred = max_cl - 0.015 * (excess_alpha ** 1.5) # Smooth lift drop-off
        cd_base = 0.008 + 0.045 * (max_cl ** 2)
        cd_spike = 0.08 * (excess_alpha ** 1.8)         # Sharp drag spike post-stall
        cd_pred = cd_base + cd_spike
        flow_state_t2 = "⚠️ Stall Region (Separation Drag Spike)"
        
    st.info(f"**Airfoil Flow State:** {flow_state_t2}")
    
    col_a, col_b = st.columns(2)
    col_a.metric("Predicted Lift Coefficient (Cl)", f"{cl_pred:.3f}")
    col_b.metric("Predicted Drag Coefficient (Cd)", f"{cd_pred:.3f}")
    
    # Generate complete curve arrays
    alphas_range = np.linspace(-5, 25, 100)
    cls_range = np.where(
        alphas_range <= stall_angle, 
        0.11 * alphas_range + 0.2, 
        (0.11 * stall_angle + 0.2) - 0.015 * ((alphas_range - stall_angle) ** 1.5)
    )
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(alphas_range, cls_range, label="Advanced Airfoil Model ($C_L$)", color="blue", linewidth=2)
    ax.axvline(x=stall_angle, color="orange", linestyle=":", label=f"Critical Stall Angle (~{stall_angle}°)")
    ax.scatter([alpha], [cl_pred], color="red", zorder=5, label="Current Selection")
    ax.set_xlabel("Angle of Attack (deg)")
    ax.set_ylabel("Lift Coefficient ($C_L$)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    
    st.pyplot(fig)

    # --- Live Step-by-Step Math Breakdown for Tab 2 ---
    st.markdown("---")
    st.subheader("📝 Step-by-Step Airfoil Calculation Breakdown")
    st.write("Here is the exact mathematical model evaluated live using your current Angle of Attack:")
    
    if alpha <= stall_angle:
        st.latex(rf"1. \text{{ Lift Coefficient (Pre-Stall): }} C_L = 0.11 \times ({alpha}) + 0.2 = {cl_pred:.3f}")
        st.latex(rf"2. \text{{ Drag Coefficient (Parabolic Polar): }} C_D = 0.008 + 0.045 \times ({cl_pred:.3f})^2 = {cd_pred:.3f}")
    else:
        st.latex(rf"1. \text{{ Lift Coefficient (Post-Stall Decay): }} C_L = {max_cl:.3f} - 0.015 \times ({alpha} - 14)^{{1.5}} = {cl_pred:.3f}")
        st.latex(rf"2. \text{{ Drag Coefficient (Separation Spike): }} C_D = C_{{d0}} + k C_L^2 + \text{{Spike}} = {cd_pred:.3f}")

# ==========================================
# TAB 3: FLIGHT TELEMETRY ANALYSIS
# ==========================================
with tab3:
    st.header("Flight Data & Telemetry Explorer")
    st.write("Analyze performance trends using sample flight logs.")
    
    time_sec = np.arange(0, 100, 1)
    altitude_log = 1000 + 50 * time_sec - 0.2 * (time_sec ** 2)
    
    df_telemetry = pd.DataFrame({
        "Time (s)": time_sec,
        "Altitude (m)": altitude_log,
        "Engine Temp (C)": 400 + np.random.normal(0, 5, len(time_sec))
    })
    
    st.dataframe(df_telemetry.head(10))
    st.line_chart(df_telemetry.set_index("Time (s)")["Altitude (m)"])
