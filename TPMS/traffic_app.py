import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import openrouteservice
import folium
import os
from streamlit_folium import st_folium

# ==============================
# API CONFIG
# ==============================

API_KEY = "5b3ce3597851110001cf6248eb117ba2f9774812b0e9a0f752bfa7f1"
client = openrouteservice.Client(key=API_KEY)

st.title("🚦 Traffic Congestion Prediction And Alternative Route Finder")
st.write("Click on the map to select start and destination locations.")

# ==============================
# SESSION STATE INITIALIZATION
# ==============================

if "start" not in st.session_state:
    st.session_state.start = None

if "destination" not in st.session_state:
    st.session_state.destination = None

if "routes" not in st.session_state:
    st.session_state.routes = None

if "congestion_level" not in st.session_state:
    st.session_state.congestion_level = None

if "alternative" not in st.session_state:
    st.session_state.alternative = 1

# ==============================
# TRAVEL MODE
# ==============================

travel_mode = st.selectbox("Select Travel Mode", ["Car 🚗", "Bike 🏍️"])

profile = "driving-car" if travel_mode == "Car 🚗" else "cycling-regular"

# ==============================
# MAP
# ==============================

m = folium.Map(location=[17.38, 78.47], zoom_start=12)

if st.session_state.start:
    folium.Marker(
        st.session_state.start,
        popup="Start",
        icon=folium.Icon(color="blue")
    ).add_to(m)

if st.session_state.destination:
    folium.Marker(
        st.session_state.destination,
        popup="Destination",
        icon=folium.Icon(color="red")
    ).add_to(m)

clicked_location = st_folium(m, width=700, height=500)

if clicked_location and clicked_location["last_clicked"]:
    lat = clicked_location["last_clicked"]["lat"]
    lon = clicked_location["last_clicked"]["lng"]

    if not st.session_state.start:
        st.session_state.start = (lat, lon)
        st.success("Start location selected")

    elif not st.session_state.destination:
        st.session_state.destination = (lat, lon)
        st.success("Destination selected")

# ==============================
# RESET BUTTON
# ==============================

if st.button("Reset Locations"):
    st.session_state.start = None
    st.session_state.destination = None
    st.session_state.routes = None
    st.session_state.congestion_level = None
    st.session_state.alternative = 1
    st.rerun()

# ==============================
# MODEL + PICKLE LOADING
# ==============================

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_model():
    path = os.path.join(MODEL_DIR, "traffic_congestion_model.h5")
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_pickle(name):
    with open(os.path.join(MODEL_DIR, name), "rb") as f:
        return pickle.load(f)

model = load_model()
scaler = load_pickle("scaler.pkl")
le_urban_rural = load_pickle("le_urban_rural.pkl")
le_Road_Closure = load_pickle("le_Road_Closure.pkl")
le_congestion = load_pickle("le_congestion_level.pkl")
ohe = load_pickle("ohe.pkl")

# ==============================
# INPUT UI
# ==============================

road_type = st.selectbox("Road Type", ["Highway", "Main Road", "Street"])
weather = st.selectbox("Weather", ["Cloudy", "Foggy", "Rainy", "Sunny"])
public_transport = st.selectbox("Public Transport Level", ["Low", "Medium", "High"])

day_of_week = st.selectbox(
    "Day of Week",
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
)

traffic_density = st.slider("Traffic Density", 0, 100, 50)
avg_speed = st.slider("Average Speed", 0, 120, 30)
temperature = st.slider("Temperature", -10, 50, 25)
accidents = st.slider("Accidents Reported", 0, 20, 2)

road_closure = st.selectbox("Road Closure", ["Yes","No"])

hour = st.slider("Hour",0,23,14)

urban_rural = st.selectbox("Urban/Rural",["Urban","Rural"])

high_risk = st.selectbox("High Risk Zone",["Yes","No"])
high_risk = 1 if high_risk=="Yes" else 0

is_weekend = 1 if day_of_week in ["Saturday","Sunday"] else 0
rush_hour = 1 if hour in [8,9,17,18] else 0

# ==============================
# DATAFRAME
# ==============================

input_df = pd.DataFrame({
"Road_Type":[road_type],
"Weather_Condition":[weather],
"Public_Transport":[public_transport],
"Day_of_Week":[day_of_week],
"Traffic_Density":[traffic_density],
"Avg_Speed":[avg_speed],
"Temperature":[temperature],
"Accidents_Reported":[accidents],
"Road_Closure":[road_closure],
"hour":[hour],
"is_weekend":[is_weekend],
"rush_hour":[rush_hour],
"urban_rural":[urban_rural],
"high_risk_zone":[high_risk]
})

# ==============================
# PREPROCESSING
# ==============================

def preprocess(df):

    df = df.copy()

    df["urban_rural"] = le_urban_rural.transform(df["urban_rural"])
    df["Road_Closure"] = le_Road_Closure.transform(df["Road_Closure"])

    num = ["Temperature","Avg_Speed","Accidents_Reported","Traffic_Density"]

    df[num] = scaler.transform(df[num])

    cat = ["Road_Type","Weather_Condition","Public_Transport","Day_of_Week"]

    enc = ohe.transform(df[cat])

    enc_df = pd.DataFrame(enc,columns=ohe.get_feature_names_out(cat))

    final = pd.concat([enc_df,df.drop(columns=cat)],axis=1)

    return final

# ==============================
# PREDICTION
# ==============================

def predict(df):

    arr = df.values.reshape(1,df.shape[0],df.shape[1])

    pred = model.predict(arr)

    label = np.argmax(pred,axis=1)[0]

    return le_congestion.inverse_transform([label])[0]

# ==============================
# PREDICT BUTTON
# ==============================

if st.button("Predict Congestion"):

    processed = preprocess(input_df)

    level = predict(processed)

    st.session_state.congestion_level = level

    st.success(f"Predicted Congestion: {level}")

    if level in ["High","Moderate"]:
        st.session_state.alternative = 3
    else:
        st.session_state.alternative = 1

# ==============================
# ROUTE GENERATION
# ==============================

if st.session_state.congestion_level and st.session_state.start and st.session_state.destination:

    try:

        st.session_state.routes = client.directions(
            coordinates=[
                st.session_state.start[::-1],
                st.session_state.destination[::-1]
            ],
            profile=profile,
            alternative_routes={"target_count":st.session_state.alternative},
            format="geojson"
        )

    except Exception as e:
        st.error(e)

# ==============================
# SHOW ROUTES
# ==============================

if st.session_state.routes:

    routes = st.session_state.routes["features"]

    color = "red" if st.session_state.congestion_level=="High" else "green"

    for i,r in enumerate(routes):

        distance = r["properties"]["segments"][0]["distance"]/1000
        duration = r["properties"]["segments"][0]["duration"]/60

        st.write(f"Route {i+1}: {distance:.2f} km , {duration:.1f} minutes")

        m = folium.Map(
            location=[
                (st.session_state.start[0]+st.session_state.destination[0])/2,
                (st.session_state.start[1]+st.session_state.destination[1])/2
            ],
            zoom_start=13
        )

        coords = [(lat,lon) for lon,lat in r["geometry"]["coordinates"]]

        folium.PolyLine(coords,color=color,weight=5).add_to(m)

        st_folium(m,width=700,height=500)