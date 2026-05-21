import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import joblib
import openrouteservice
import folium
import os
from streamlit_folium import st_folium
from dotenv import load_dotenv


load_dotenv()


BASE_DIR  = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_metrics():
    metrics = {}
    try:
        with open(os.path.join(MODEL_DIR, "metrics.txt"), "r") as f:
            for line in f:
                k, v = line.strip().split(":")
                metrics[k] = v
    except Exception:
        pass
    return metrics


@st.cache_resource
def load_all():
    model = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "traffic_congestion_model.keras")
    )
    scaler          = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoders  = joblib.load(os.path.join(MODEL_DIR, "label_encoders_all.pkl"))
    seq_feature_cols = joblib.load(os.path.join(MODEL_DIR, "seq_feature_cols.pkl"))

    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        le_target = pickle.load(f)

    return model, scaler, label_encoders, seq_feature_cols, le_target

model, scaler, label_encoders, seq_feature_cols, le_target = load_all()


client  = openrouteservice.Client(key=API_KEY)


st.title("🚦 Traffic Congestion Prediction And Alternative Route Finder")
st.write("Click on the map to select start and destination locations.")


for key, default in [("start", None), ("destination", None),
                     ("routes", None), ("congestion_level", None),
                     ("alternative", 1)]:
    if key not in st.session_state:
        st.session_state[key] = default

travel_mode = st.selectbox("Select Travel Mode", ["Car 🚗", "Bike 🏍️"])
profile     = "driving-car" if "Car" in travel_mode else "cycling-regular"


# map
m = folium.Map(location=[17.38, 78.47], zoom_start=12)
if st.session_state.start:
    folium.Marker(st.session_state.start, popup="Start",
                  icon=folium.Icon(color="blue")).add_to(m)
if st.session_state.destination:
    folium.Marker(st.session_state.destination, popup="Destination",
                  icon=folium.Icon(color="red")).add_to(m)

clicked = st_folium(m, width=700, height=500)
if clicked and clicked.get("last_clicked"):
    lat = clicked["last_clicked"]["lat"]
    lon = clicked["last_clicked"]["lng"]
    if not st.session_state.start:
        st.session_state.start = (lat, lon)
        st.success("Start location selected")
    elif not st.session_state.destination:
        st.session_state.destination = (lat, lon)
        st.success("Destination selected")

if st.button("Reset Locations"):
    st.session_state.update(
        start=None, destination=None, routes=None,
        congestion_level=None, alternative=1
    )
    st.rerun()


def le_classes(col):
    return list(label_encoders[col].classes_) if col in label_encoders else []

city       = st.selectbox("City",      le_classes("City")      or ["Hyderabad"])
road_name  = st.selectbox("Road Name", le_classes("Road_Name") or ["NH65"])
road_type  = st.selectbox("Road Type", le_classes("Road_Type") or ["Highway", "Main Road", "Street"])
weather    = st.selectbox("Weather Condition", le_classes("Weather_Condition") or ["Cloudy","Foggy","Rainy","Sunny"])
pub_trans  = st.selectbox("Public Transport Level", le_classes("Public_Transport") or ["Low","Medium","High"])
day_of_week= st.selectbox("Day of Week", le_classes("Day_of_Week") or
                           ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

traffic_density = st.slider("Traffic Density",    0, 100, 50)
avg_speed       = st.slider("Average Speed",      0, 120, 30)
temperature     = st.slider("Temperature",       -10,  50, 25)
accidents       = st.slider("Accidents Reported", 0,  20,  2)

road_closure = st.selectbox("Road Closure", le_classes("Road_Closure") or ["Yes","No"])
hour         = st.slider("Hour", 0, 23, 14)
urban_rural  = st.selectbox("Urban/Rural", le_classes("urban_rural") or ["Urban","Rural"])
high_risk    = 1 if st.selectbox("High Risk Zone", ["Yes","No"]) == "Yes" else 0

is_weekend = 1 if day_of_week in ["Saturday","Sunday"] else 0
rush_hour  = 1 if (7 <= hour <= 10 or 17 <= hour <= 20) else 0


def safe_le_encode(col_name, value):
    """Encode a value using its saved LabelEncoder; fallback to 0 if unseen."""
    if col_name not in label_encoders:
        return 0
    le = label_encoders[col_name]
    if value in le.classes_:
        return int(le.transform([value])[0])
    st.warning(f"⚠️ '{value}' not seen during training for '{col_name}'. Defaulting to 0.")
    return 0

def preprocess(user_input: dict) -> np.ndarray:
    """
    Mirrors train_model.py step by step:
      1. Build raw DataFrame row
      2. Feature engineering
      3. Drop Timestamp / Area_Name / Latitude / Longitude
      4. LabelEncode every object column using SAVED encoders
      5. Convert to numeric, fillna
      6. Scale with SAVED scaler (same column order as training)
      7. Drop Road_Name (done inside group loop during training)
      8. Reorder to seq_feature_cols
      9. Reshape to (1, 1, n_features) for LSTM
    """
    df = pd.DataFrame([user_input])

    
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["prev_congestion"]   = 0   
    df["rolling_congestion"] = 0
    df["traffic_speed_ratio"] = df["Traffic_Density"] / (df["Avg_Speed"] + 1)
    df["speed_efficiency"]    = df["Avg_Speed"] / (df["Traffic_Density"] + 1)
    df["density_x_rush"]      = df["Traffic_Density"] * df["Rush_Hour"]
    df["peak_hour_flag"]      = df["hour"].apply(lambda x: 1 if (7 <= x <= 10 or 17 <= x <= 20) else 0)

    
    df = df.drop(columns=["Timestamp","Area_Name","Latitude","Longitude"], errors="ignore")

    
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(lambda v: safe_le_encode(col, v))

    
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    scaler_cols = list(scaler.feature_names_in_)  

    for col in scaler_cols:
        if col not in df.columns:
            df[col] = 0 

    df = df[scaler_cols]  

    
    df_scaled = pd.DataFrame(scaler.transform(df), columns=scaler_cols)

    
    df_lstm = df_scaled[seq_feature_cols]

    return df_lstm.values.reshape(1, 1, df_lstm.shape[1])


def predict(arr: np.ndarray) -> str:
    pred = model.predict(arr)
    idx  = np.argmax(pred, axis=1)[0]
    return le_target.inverse_transform([idx])[0]


if st.button("Predict Congestion"):
    user_input = {
        "City":             city,
        "Road_Name":        road_name,
        "Road_Type":        road_type,
        "Weather_Condition":weather,
        "Public_Transport": pub_trans,
        "Day_of_Week":      day_of_week,
        "Traffic_Density":  traffic_density,
        "Avg_Speed":        avg_speed,
        "Temperature":      temperature,
        "Accidents_Reported": accidents,
        "Road_Closure":     road_closure,
        "hour":             hour,
        "is_weekend":       is_weekend,
        "Rush_Hour":        rush_hour,
        "urban_rural":      urban_rural,
        "high_risk_zone":   high_risk,
        
        "Timestamp": "", "Area_Name": "", "Latitude": 0.0, "Longitude": 0.0,
    }
    try:
        arr   = preprocess(user_input)
        level = predict(arr)
        st.session_state.congestion_level = level
        st.success(f"Predicted Congestion: **{level}**")
        st.session_state.alternative = 3 if level in ["High","Moderate"] else 1
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.exception(e)


if st.session_state.congestion_level and st.session_state.start and st.session_state.destination:
    try:
        st.session_state.routes = client.directions(
            coordinates=[
                st.session_state.start[::-1],
                st.session_state.destination[::-1],
            ],
            profile=profile,
            alternative_routes={"target_count": st.session_state.alternative},
            format="geojson",
        )
    except Exception as e:
        st.error(f"Route error: {e}")


if st.session_state.routes:
    color  = "red" if st.session_state.congestion_level == "High" else "green"
    routes = st.session_state.routes["features"]
    for i, r in enumerate(routes):
        dist = r["properties"]["segments"][0]["distance"] / 1000
        dur  = r["properties"]["segments"][0]["duration"] / 60
        st.write(f"**Route {i+1}:** {dist:.2f} km — {dur:.1f} min")

        center = [
            (st.session_state.start[0] + st.session_state.destination[0]) / 2,
            (st.session_state.start[1] + st.session_state.destination[1]) / 2,
        ]
        rm = folium.Map(location=center, zoom_start=13)
        coords = [(lat, lon) for lon, lat in r["geometry"]["coordinates"]]
        folium.PolyLine(coords, color=color, weight=5).add_to(rm)
        folium.Marker(st.session_state.start, popup="Start",
                      icon=folium.Icon(color="blue")).add_to(rm)
        folium.Marker(st.session_state.destination, popup="Destination",
                      icon=folium.Icon(color="red")).add_to(rm)
        st_folium(rm, width=700, height=500)

metrics = load_metrics()
if metrics:
    st.divider()
    st.subheader("% Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    def fmt(key):
        try:
            return f"{float(metrics[key]):.2f}"
        except Exception:
            return "—"
    col1.metric("Accuracy",  fmt("Accuracy"))
    col2.metric("Precision", fmt("Precision"))
    col3.metric("Recall",    fmt("Recall"))
    col4.metric("F1 Score",  fmt("F1"))