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

# Create Multi-Tab Layout FIRST
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
        alpha_t1 = st.slider("Angle of Attack (degrees)", -4.0, 18.0, 5.0, key="t1_alpha")
        
    with col2:
        # Safety Guard: Restrict altitude to valid troposphere bounds (0 to 11,000 m)
        safe_altitude = max(0.0, min(11000.0, float(altitude_m)))
        
        temp_0 = 288.15 
        temp_k = temp_0 - (0.0065 * safe_altitude)
        rho_0 = 1.225   
        rho = rho_0 * ((temp_k / temp_0) ** 4.256)
        dynamic_pressure = 0.5 * rho * (velocity_ms ** 2)
        
        # Verified Exact AirfoilTools NACA 2412 Re = 1,000,000 Datasets
        naca_alphas = np.array([-4.0, -2.0,  0.0,  2.0,  4.0,  6.0,  8.0, 10.0, 12.0, 14.0, 16.0, 18.0])
        naca_cls    = np.array([-0.1918, 0.0272, 0.2442, 0.4549, 0.7153, 0.9016, 1.0885, 1.2696, 1.4114, 1.5228, 1.5775, 1.5415])
        
        cl_calculated = float(np.interp(alpha_t1, naca_alphas, naca_cls))
        stall_status = "Attached Flow (Normal Operating Range)" if alpha_t1 <= 16.0 else "⚠️ AIRFOIL STALLED (XFOIL Separation Regime)"
        
        lift_force = dynamic_pressure * wing_area * cl_calculated
        
        if altitude_m > 11000:
            st.warning("⚠️ Altitude exceeds 11,000m troposphere limit. Clamped to 11,000m for ISA calculations.")
            
        st.info(f"**Flow State:** {stall_status}")
        st.metric(label="Calculated Temperature", value=f"{temp_k:.2f} K")
        st.metric(label="Estimated Air Density (rho)", value=f"{rho:.3f} kg/m³")
        st.metric(label="Dynamic Pressure (q)", value=f"{dynamic_pressure:.1f} Pa")
        st.metric(label="Total Lift Force (L)", value=f"{lift_force:,.1f} N")

    st.markdown("---")
    st.subheader("📝 Step-by-Step Calculation Breakdown")
    st.latex(rf"1. \text{{ Temperature: }} T = 288.15 - (0.0065 \times {safe_altitude:.0f}) = {temp_k:.2f} \text{{ K}}")
    st.latex(rf"2. \text{{ Air Density: }} \rho = 1.225 \times \left(\frac{{{temp_k:.2f}}}{{288.15}}\right)^{{4.256}} = {rho:.3f} \text{{ kg/m}}^3")
    st.latex(rf"3. \text{{ Dynamic Pressure: }} q = \frac{1}{2} \times ({rho:.3f}) \times ({velocity_ms})^2 = {dynamic_pressure:.1f} \text{{ Pa}}")
    st.latex(rf"4. \text{{ Lift Coefficient (Verified AirfoilTools Re=1M Interpolation): }} C_L(\alpha={alpha_t1:.1f}^\circ) = {cl_calculated:.3f}")
    st.latex(rf"5. \text{{ Lift Force: }} L = q \times S \times C_L = {dynamic_pressure:.1f} \times {wing_area} \times {cl_calculated:.3f} = {lift_force:,.1f} \text{{ N}}")

# ==========================================
# TAB 2: AIRFOIL AI PREDICTOR (AIRFOILTOOLS RE=1M)
# ==========================================
with tab2:
    st.header("NACA 2412 XFOIL Polar Predictor (Re = 1,000,000)")
    st.markdown("*Data Source Citing:* **AirfoilTools XFOIL Prediction Polar (`xf-naca2412-il-1000000`)**")
    st.write("This tab evaluates aerodynamic performance via piecewise linear interpolation over official public wind-tunnel/XFOIL simulation polars.")
    
    alpha = st.slider("Angle of Attack (Alpha - degrees)", -4.0, 18.0, 4.0, key="t2_alpha")
    
    # Verified Exact Citable AirfoilTools Re = 1,000,000 Arrays
    naca_alphas = np.array([-4.0, -2.0,  0.0,  2.0,  4.0,  6.0,  8.0, 10.0, 12.0, 14.0, 16.0, 18.0])
    naca_cls    = np.array([-0.1918, 0.0272, 0.2442, 0.4549, 0.7153, 0.9016, 1.0885, 1.2696, 1.4114, 1.5228, 1.5775, 1.5415])
    naca_cds    = np.array([0.0077, 0.0065, 0.0057, 0.0058, 0.0071, 0.0095, 0.0127, 0.0159, 0.0202, 0.0262, 0.0394, 0.0680])
    
    # Perform direct array interpolation
    cl_pred = float(np.interp(alpha, naca_alphas, naca_cls))
    cd_pred = float(np.interp(alpha, naca_alphas, naca_cds))
    
    flow_state_t2 = "Attached Flow (Linear/Pre-Stall Regime)" if alpha <= 16.0 else "⚠️ Critical Stall Region (Boundary Layer Separation & Drag Rise)"
    st.info(f"**Airfoil Flow State:** {flow_state_t2}")
    
    col_a, col_b = st.columns(2)
    col_a.metric("Interpolated Lift Coefficient (Cl)", f"{cl_pred:.3f}")
    col_b.metric("Interpolated Drag Coefficient (Cd)", f"{cd_pred:.3f}")
    
    # Plotting Lookup Curves
    alphas_range = np.linspace(-4, 18, 150)
    cls_range = np.interp(alphas_range, naca_alphas, naca_cls)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(alphas_range, cls_range, label="XFOIL Re=1M Interpolation ($C_L$)", color="blue", linewidth=2)
    ax.axvline(x=16.0, color="orange", linestyle=":", label="Empirical Stall Point (~16°)")
    ax.scatter([alpha], [cl_pred], color="red", zorder=5, label="Current Selection")
    ax.set_xlabel("Angle of Attack (deg)")
    ax.set_ylabel("Lift Coefficient ($C_L$)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    
    st.pyplot(fig)

    # --- Academic Transparency Section ---
    st.markdown("---")
    st.subheader("🔍 Academic Transparency & Citable Data Source")
    st.write("Inspecting the underlying citable dataset confirms public reproducibility against AirfoilTools:")
    
    with st.expander("📂 Click to view verified AirfoilTools Re=1M coordinate arrays used by `np.interp()`"):
        df_lookup = pd.DataFrame({
            "Angle of Attack (deg)": naca_alphas,
            "Lift Coeff (Cl)": naca_cls,
            "Drag Coeff (Cd)": naca_cds
        })
        st.dataframe(df_lookup, use_container_width=True)

    st.markdown("**Active Lookup Evaluation:**")
    st.latex(rf"C_L = \text{{np.interp}}(\alpha = {alpha:.1f}^\circ) \rightarrow \mathbf{{{cl_pred:.3f}}}")
    st.latex(rf"C_D = \text{{np.interp}}(\alpha = {alpha:.1f}^\circ) \rightarrow \mathbf{{{cd_pred:.3f}}}")

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
