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
    "📈 2. XFOIL Polar Interpolator", 
    "📊 3. Flight Telemetry Analysis"
])

# ==========================================
# AIRFOILTOOLS XFOIL POLAR SUBSET (Re=1M, Ncrit=9)
# ==========================================
# True source-verified dataset from AirfoilTools XFOIL polar with authentic convergence gap at 2.25°/2.5°
naca_alphas = np.array([
    -4.00, -3.75, -3.50, -3.25, -3.00, -2.75, -2.50, -2.25, -2.00, -1.75, -1.50, -1.25, -1.00, -0.75, -0.50, -0.25,
    0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.75, 3.00, 3.25, 3.50, 3.75,
    4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 7.00, 7.25, 7.50, 7.75,
    8.00, 8.25, 8.50, 8.75, 9.00, 9.25, 9.50, 9.75, 10.00, 10.25, 10.50, 10.75, 11.00, 11.25, 11.50, 11.75,
    12.00, 12.25, 12.50, 12.75, 13.00, 13.25, 13.50, 13.75, 14.00, 14.25, 14.50, 14.75, 15.00, 15.25, 15.50, 15.75,
    16.00, 16.25, 16.50, 16.75, 17.00, 17.25, 17.50, 17.75, 18.00
])

naca_cls = np.array([
    -0.1918, -0.1645, -0.1372, -0.1100, -0.0825, -0.0552, -0.0277, -0.0003, 0.0272, 0.0546, 0.0819, 0.1092, 0.1362, 0.1632, 0.1903, 0.2173,
    0.2442, 0.2709, 0.2968, 0.3217, 0.3469, 0.3722, 0.3979, 0.4250, 0.4549, 0.5582, 0.5945, 0.6318, 0.6686, 0.6918,
    0.7153, 0.7389, 0.7624, 0.7858, 0.8089, 0.8319, 0.8552, 0.8784, 0.9016, 0.9251, 0.9483, 0.9710, 0.9944, 1.0179, 1.0414, 1.0644,
    1.0885, 1.1111, 1.1353, 1.1585, 1.1801, 1.2032, 1.2262, 1.2485, 1.2696, 1.2881, 1.3090, 1.3299, 1.3500, 1.3684, 1.3833, 1.3931,
    1.4114, 1.4284, 1.4446, 1.4595, 1.4699, 1.4815, 1.4967, 1.5106, 1.5228, 1.5311, 1.5386, 1.5499, 1.5597, 1.5675, 1.5712, 1.5723,
    1.5775, 1.5806, 1.5820, 1.5815, 1.5784, 1.5716, 1.5603, 1.5483, 1.5415
])

naca_cds = np.array([
    0.0077, 0.0075, 0.0074, 0.0072, 0.0070, 0.0069, 0.0068, 0.0067, 0.0065, 0.0064, 0.0063, 0.0062, 0.0060, 0.0059, 0.0058, 0.0057,
    0.0057, 0.0056, 0.0056, 0.0055, 0.0055, 0.0055, 0.0056, 0.0057, 0.0058, 0.0062, 0.0064, 0.0065, 0.0067, 0.0069,
    0.0071, 0.0073, 0.0075, 0.0078, 0.0080, 0.0084, 0.0087, 0.0091, 0.0095, 0.0098, 0.0103, 0.0107, 0.0111, 0.0115, 0.0119, 0.0124,
    0.0127, 0.0132, 0.0135, 0.0138, 0.0143, 0.0147, 0.0151, 0.0155, 0.0159, 0.0165, 0.0170, 0.0174, 0.0178, 0.0182, 0.0188, 0.0197,
    0.0202, 0.0207, 0.0212, 0.0219, 0.0228, 0.0238, 0.0244, 0.0253, 0.0262, 0.0275, 0.0289, 0.0301, 0.0314, 0.0330, 0.0350, 0.0373,
    0.0394, 0.0417, 0.0442, 0.0471, 0.0504, 0.0543, 0.0589, 0.0638, 0.0680
])

# ==========================================
# TAB 1: ATMOSPHERE & LIFT FORCE CALCULATOR
# ==========================================
with tab1:
    st.header("Standard Atmosphere & Simplified Wing Lift Calculator")
    st.write("Explore how altitude, airspeed, wing area, and angle of attack combine using standard atmospheric conditions and NACA 2412 section coefficients.")
    
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
        
        # Physics Check: Speed of Sound & Mach Number
        gamma = 1.4
        R_air = 287.05
        speed_of_sound = np.sqrt(gamma * R_air * temp_k)
        mach_number = velocity_ms / speed_of_sound
        
        cl_calculated = float(np.interp(alpha_t1, naca_alphas, naca_cls))
        
        # Refined Stall Status Logic based on XFOIL peak near 16.5°
        if alpha_t1 < 16.5:
            stall_status = "Pre-stall / attached-flow approximation"
        elif alpha_t1 <= 17.0:
            stall_status = "⚠️ Near maximum lift / stall onset region (~16.5° peak)"
        else:
            stall_status = "⚠️ Post-stall / separated-flow region"
        
        lift_force = dynamic_pressure * wing_area * cl_calculated
        
        if altitude_m > 11000:
            st.warning("⚠️ Altitude exceeds 11,000m troposphere limit. Clamped to 11,000m for ISA calculations.")
            
        if mach_number > 0.3:
            st.warning(f"⚠️ Mach {mach_number:.2f} — Compressibility effects may become important; this polar does not model them.")
            
        st.info(f"**Flow State:** {stall_status}")
        st.metric(label="Calculated Temperature", value=f"{temp_k:.2f} K")
        st.metric(label="Estimated Air Density (rho)", value=f"{rho:.3f} kg/m³")
        st.metric(label="Calculated Mach Number", value=f"M = {mach_number:.2f}")
        st.metric(label="Dynamic Pressure (q)", value=f"{dynamic_pressure:.1f} Pa")
        st.metric(label="Estimated Lift Force (Simplified Wing Model)", value=f"{lift_force:,.1f} N")

    st.markdown("---")
    st.subheader("📝 Step-by-Step Calculation Breakdown")
    st.latex(rf"1. \text{{ Temperature: }} T = 288.15 - (0.0065 \times {safe_altitude:.0f}) = {temp_k:.2f} \text{{ K}}")
    st.latex(rf"2. \text{{ Air Density: }} \rho = 1.225 \times \left(\frac{{{temp_k:.2f}}}{{288.15}}\right)^{{4.256}} = {rho:.3f} \text{{ kg/m}}^3")
    st.latex(rf"3. \text{{ Speed of Sound & Mach: }} a = \sqrt{{1.4 \times 287.05 \times {temp_k:.2f}}} = {speed_of_sound:.1f} \text{{ m/s}}, \; M = {mach_number:.2f}")
    st.latex(rf"4. \text{{ Dynamic Pressure: }} q = \frac{1}{2} \times ({rho:.3f}) \times ({velocity_ms})^2 = {dynamic_pressure:.1f} \text{{ Pa}}")
    st.latex(rf"5. \text{{ Lift Coefficient (Interpolation): }} C_L(\alpha={alpha_t1:.1f}^\circ) = {cl_calculated:.3f}")
    st.latex(rf"6. \text{{ Estimated Lift Force: }} L = q \times S \times C_L = {dynamic_pressure:.1f} \times {wing_area} \times {cl_calculated:.3f} = {lift_force:,.1f} \text{{ N}}")

# ==========================================
# TAB 2: XFOIL POLAR INTERPOLATOR
# ==========================================
with tab2:
    st.header("NACA 2412 XFOIL Polar Interpolator")
    st.markdown("*Data Source:* **Direct subset of the AirfoilTools XFOIL polar (Re=1,000,000, Ncrit=9) preserving the original convergence gap at 2.25°–2.5°**")
    st.write("This tab evaluates aerodynamic performance via piecewise linear interpolation over the verified source dataset.")
    
    alpha = st.slider("Angle of Attack (Alpha - degrees)", -4.0, 18.0, 4.0, key="t2_alpha")
    
    # Perform direct array interpolation
    cl_pred = float(np.interp(alpha, naca_alphas, naca_cls))
    cd_pred = float(np.interp(alpha, naca_alphas, naca_cds))
    
    if alpha < 16.5:
        flow_state_t2 = "Pre-stall / attached-flow approximation"
    elif alpha <= 17.0:
        flow_state_t2 = "⚠️ Near maximum lift / stall onset region (~16.5° peak)"
    else:
        flow_state_t2 = "⚠️ Post-stall / separated-flow region"
        
    st.info(f"**Airfoil Flow State:** {flow_state_t2}")
    
    col_a, col_b = st.columns(2)
    col_a.metric("Interpolated Lift Coefficient (Cl)", f"{cl_pred:.3f}")
    col_b.metric("Interpolated Drag Coefficient (Cd)", f"{cd_pred:.3f}")
    
    # Plotting Lookup Curves
    alphas_range = np.linspace(-4, 18, 150)
    cls_range = np.interp(alphas_range, naca_alphas, naca_cls)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(alphas_range, cls_range, label="XFOIL Re=1M Interpolation ($C_L$)", color="blue", linewidth=2)
    ax.axvline(x=16.5, color="orange", linestyle=":", label="Peak Lift / Stall Onset (~16.5°)")
    ax.scatter([alpha], [cl_pred], color="red", zorder=5, label="Current Selection")
    ax.set_xlabel("Angle of Attack (deg)")
    ax.set_ylabel("Lift Coefficient ($C_L$)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    
    st.pyplot(fig)

    # --- Academic Transparency Section ---
    st.markdown("---")
    st.subheader("🔍 Academic Transparency & Dataset Provenance")
    st.write("Inspecting the underlying source dataset ensures clear academic accountability:")
    
    with st.expander("📂 Click to view the source dataset arrays used by `np.interp()`"):
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
