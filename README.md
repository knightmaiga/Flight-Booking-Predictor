# Flight Booking Prediction System

## 📊 Project Overview
Machine learning system to predict customer flight booking completion using XGBoost and Random Forest models.

## 🚀 Quick Deployment

### Option 1: Local Deployment
```bash
# Install requirements
pip install -r requirements.txt

# Run the web app
streamlit run app.py
```

### Option 2: Cloud Deployment (Heroku)
```bash
# Create Heroku app
heroku create your-booking-predictor

# Deploy
git push heroku main
```

### Option 3: Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 📈 Model Performance
- **Accuracy**: 85.2%
- **ROC AUC**: 0.891
- **F1 Score**: 0.634

## 🔧 API Endpoint (FastAPI Example)
```python
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()
model = pickle.load(open('booking_predictor.pkl', 'rb'))

class BookingRequest(BaseModel):
    num_passengers: int
    sales_channel: str
    trip_type: str
    purchase_lead: int
    length_of_stay: int
    flight_hour: int
    flight_day: str
    booking_origin: str
    flight_duration: float
    wants_extra_baggage: bool
    wants_preferred_seat: bool
    wants_in_flight_meals: bool

@app.post("/predict")
async def predict_booking(request: BookingRequest):
    input_data = pd.DataFrame([request.dict()])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "completed_booking": bool(prediction)
    }
```

## 📋 Requirements
See `requirements.txt` for full package list
