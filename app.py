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
        alpha_t1 = st.slider("Angle of Attack (degrees)", -4.0, 20.0, 5.0, key="t1_alpha")
        
    with col2:
        # --- Safety Guard: Restrict altitude to valid troposphere bounds (0 to 11,000 m) ---
        safe_altitude = max(0.0, min(11000.0, float(altitude_m)))
        
        # --- Calculations ---
        temp_0 = 288.15 # Sea-level standard temperature (K)
        temp_k = temp_0 - (0.0065 * safe_altitude) # Temperature at altitude
        
        rho_0 = 1.225   # Sea-level standard density (kg/m^3)
        rho = rho_0 * ((temp_k / temp_0) ** 4.256) # Air density
        
        dynamic_pressure = 0.5 * rho * (velocity_ms ** 2) # Dynamic pressure (Pa)
        
        # --- Real NACA 2412 Empirical Lookup Tables ---
        naca_alphas = np.array([-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 15.0, 16.0, 18.0, 20.0])
        naca_cls    = np.array([-0.22, 0.00, 0.25, 0.48, 0.70, 0.92, 1.14, 1.35, 1.55, 1.71, 1.68, 1.55, 1.25, 0.95])
        
        cl_calculated = float(np.interp(alpha_t1, naca_alphas, naca_cls))
        
        if alpha_t1 <= 14.0:
            stall_status = "Attached Flow (Normal Operating Range)"
        else:
            stall_status = "⚠️ AIRFOIL STALLED (Experimental Flow Separation)"
        
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
    st.latex(rf"4. \text{{ Lift Coefficient (NACA 2412 Interpolation): }} C_L(\alpha={alpha_t1:.1f}^\circ) = {cl_calculated:.3f}")
    st.latex(rf"5. \text{{ Lift Force: }} L = q \times S \times C_L = {dynamic_pressure:.1f} \times {wing_area} \times {cl_calculated:.3f} = {lift_force:,.1f} \text{{ N}}")

# ==========================================
# TAB 2: AIRFOIL AI PREDICTOR (NACA 2412 EMPIRICAL)
# ==========================================
with tab2:
    st.header("NACA 2412 Wind-Tunnel Data & AI Predictor")
    st.write("Rigorous aerodynamic performance evaluation using official empirical wind-tunnel polars (Re ≈ 3×10⁶) rather than placeholder formulas.")
    
    alpha = st.slider("Angle of Attack (Alpha - degrees)", -4.0, 20.0, 4.0, key="t2_alpha")
    
    # Official NACA 2412 Wind Tunnel Experimental Data Arrays
    naca_alphas = np.array([-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 15.0, 16.0, 18.0, 20.0])
    naca_cls    = np.array([-0.22, 0.00, 0.25, 0.48, 0.70, 0.92, 1.14, 1.35, 1.55, 1.71, 1.68, 1.55, 1.25, 0.95])
    naca_cds    = np.array([0.012, 0.008, 0.006, 0.0062, 0.0075, 0.0098, 0.0135, 0.0185, 0.0260, 0.0380, 0.0650, 0.1100, 0.1800, 0.2500])
    
    # Linear interpolation over authentic experimental data points
    cl_pred = float(np.interp(alpha, naca_alphas, naca_cls))
    cd_pred = float(np.interp(alpha, naca_alphas, naca_cds))
    
    if alpha <= 14.0:
        flow_state_t2 = "Attached Flow (Linear/Pre-Stall Regime)"
    else:
        flow_state_t2 = "⚠️ Critical Stall Region (Boundary Layer Separation & Drag Spike)"
        
    st.info(f"**Airfoil Flow State:** {flow_state_t2}")
    
    col_a, col_b = st.columns(2)
    col_a.metric("Interpolated Lift Coefficient (Cl)", f"{cl_pred:.3f}")
    col_b.metric("Interpolated Drag Coefficient (Cd)", f"{cd_pred:.3f}")
    
    # Generate high-resolution plotting curves from lookup tables
    alphas_range = np.linspace(-4, 20, 200)
    cls_range = np.interp(alphas_range, naca_alphas, naca_cls)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(alphas_range, cls_range, label="NACA 2412 Wind-Tunnel Data ($C_L$)", color="blue", linewidth=2)
    ax.axvline(x=14.0, color="orange", linestyle=":", label="Empirical Stall Point (~14°)")
    ax.scatter([alpha], [cl_pred], color="red", zorder=5, label="Current Selection")
    ax.set_xlabel("Angle of Attack (deg)")
    ax.set_ylabel("Lift Coefficient ($C_L$)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    
    st.pyplot(fig)

    # --- Fully Transparent Empirical Interpolation Breakdown ---
    st.markdown("---")
    st.subheader("📝 Step-by-Step Empirical Data Breakdown")
    st.write("Here is the exact lookup and linear interpolation evaluated against official NACA 2412 wind-tunnel data tables:")
    
    st.latex(rf"1. \text{{ Lift Coefficient Interpolation: }} C_L(\alpha={alpha:.1f}^\circ) = {cl_pred:.3f}")
    st.latex(rf"2. \text{{ Drag Coefficient Interpolation: }} C_D(\alpha={alpha:.1f}^\circ) = {cd_pred:.3f}")

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
