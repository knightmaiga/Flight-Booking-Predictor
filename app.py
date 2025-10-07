import streamlit as st
import pandas as pd
import pickle
import json

# Load model and artifacts
with open('booking_predictor.pkl', 'rb') as f:
    model = pickle.load(f)

with open('deployment_artifacts.json', 'r') as f:
    artifacts = json.load(f)

st.set_page_config(page_title="Flight Booking Predictor", page_icon="✈️")

st.title("✈️ Flight Booking Completion Predictor")
st.markdown("Predict whether a customer will complete their flight booking")

# Sidebar with model info
st.sidebar.header("Model Information")
st.sidebar.metric("Accuracy", f"{artifacts['model_metrics']['accuracy']:.1%}")
st.sidebar.metric("ROC AUC", f"{artifacts['model_metrics']['roc_auc']:.3f}")
st.sidebar.metric("F1 Score", f"{artifacts['model_metrics']['f1_score']:.3f}")

# Main input form
st.header("Booking Details")

col1, col2 = st.columns(2)

with col1:
    num_passengers = st.slider("Number of Passengers", 1, 10, 1)
    sales_channel = st.selectbox("Sales Channel", ["Internet", "Mobile"])
    trip_type = st.selectbox("Trip Type", ["RoundTrip", "OneWay", "CircleTrip"])
    purchase_lead = st.number_input("Purchase Lead (days)", min_value=0, max_value=365, value=30)
    length_of_stay = st.number_input("Length of Stay (days)", min_value=1, max_value=365, value=7)

with col2:
    flight_hour = st.slider("Flight Hour", 0, 23, 12)
    flight_day = st.selectbox("Flight Day", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    booking_origin = st.selectbox("Booking Origin", ["Australia", "Malaysia", "China", "Japan", "UK", "US", "Germany"])
    flight_duration = st.number_input("Flight Duration (hours)", min_value=1.0, max_value=24.0, value=5.0)

# Additional preferences
st.subheader("Customer Preferences")
col3, col4, col5 = st.columns(3)
with col3:
    wants_extra_baggage = st.checkbox("Extra Baggage")
with col4:
    wants_preferred_seat = st.checkbox("Preferred Seat")
with col5:
    wants_in_flight_meals = st.checkbox("In-Flight Meals")

# Prepare input data
flight_day_mapping = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}

input_data = pd.DataFrame({
    'num_passengers': [num_passengers],
    'sales_channel': [sales_channel],
    'trip_type': [trip_type],
    'purchase_lead': [purchase_lead],
    'length_of_stay': [length_of_stay],
    'flight_hour': [flight_hour],
    'flight_day': [flight_day_mapping[flight_day]],
    'booking_origin': [booking_origin],
    'wants_extra_baggage': [1 if wants_extra_baggage else 0],
    'wants_preferred_seat': [1 if wants_preferred_seat else 0],
    'wants_in_flight_meals': [1 if wants_in_flight_meals else 0],
    'flight_duration': [flight_duration]
})

# Prediction
if st.button("Predict Booking Completion"):
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        st.header("Prediction Result")
        
        if prediction == 1:
            st.success(f"✅ High likelihood of booking completion ({probability:.1%} probability)")
        else:
            st.error(f"❌ Low likelihood of booking completion ({probability:.1%} probability)")
        
        # Show probability gauge
        st.subheader("Confidence Level")
        st.progress(float(probability))
        st.write(f"Confidence: {probability:.1%}")
        
        # Business recommendations
        st.subheader("Recommendations")
        if prediction == 0 and probability < 0.3:
            st.info("💡 Consider offering promotional discounts or flexible booking options")
        elif prediction == 0 and probability < 0.6:
            st.info("💡 Suggest adding travel insurance or highlighting popular routes")
        else:
            st.info("💡 Customer shows strong intent - focus on smooth checkout process")
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Built with XGBoost • Model updated with latest booking data*")
