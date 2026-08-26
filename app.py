import streamlit as st

st.title("✈️ AeroAI: Flight & Airfoil Predictor")
st.write("Welcome to your first interactive aerospace engineering workbench!")

# Create a sidebar for user inputs (like a real engineering tool)
st.sidebar.header("Flight Parameters")
angle_of_attack = st.sidebar.slider("Angle of Attack (deg)", -5.0, 25.0, 5.0)
mach_number = st.sidebar.slider("Mach Number", 0.1, 2.0, 0.5)

# Main panel display
st.subheader("Current Configuration")
st.write(f"You have selected an Angle of Attack of **{angle_of_attack}°** at **Mach {mach_number}**.")

# A placeholder button to trigger your future AI model
if st.button("Run AI Prediction"):
    # Later, your machine learning model will calculate this!
    mock_lift_coefficient = 0.09 * angle_of_attack 
    st.success(f"Predicted Lift Coefficient (Cl): {mock_lift_coefficient:.2f}")
