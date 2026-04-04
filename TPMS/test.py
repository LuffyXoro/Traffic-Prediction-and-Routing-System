## test.py

import streamlit as st
import openrouteservice
import folium
from streamlit_folium import st_folium

API_KEY = "5b3ce3597851110001cf6248eb117ba2f9774812b0e9a0f752bfa7f1"

client = openrouteservice.Client(key=API_KEY)
st.title("🚗🏍️ Alternative Route Finder with Dynamic Location Selection")
st.write("Click on the map to select start and destination locations.")

if "start" not in st.session_state:
    st.session_state.start = None
if "destination" not in st.session_state:
    st.session_state.destination = None
if "routes" not in st.session_state:
    st.session_state.routes = None

travel_mode = st.selectbox("Select Travel Mode", ["Car 🚗", "Bike 🏍️"])
profile = "driving-car" if travel_mode == "Car 🚗" else "cycling-regular"


m = folium.Map(location=[17.38, 78.47], zoom_start=12)

if st.session_state.start:
    folium.Marker(st.session_state.start, popup="Start", icon=folium.Icon(color="blue")).add_to(m)
if st.session_state.destination:
    folium.Marker(st.session_state.destination, popup="Destination", icon=folium.Icon(color="red")).add_to(m)

clicked_location = st_folium(m, width=700, height=500)

if clicked_location and clicked_location["last_clicked"]:
    lat, lon = clicked_location["last_clicked"]["lat"], clicked_location["last_clicked"]["lng"]

    if not st.session_state.start:
        st.session_state.start = (lat, lon)
        st.write("✅ Start location selected!")
    elif not st.session_state.destination:
        st.session_state.destination = (lat, lon)
        st.write("✅ Destination location selected!")

if st.button("Reset Locations"):
    st.session_state.start = None
    st.session_state.destination = None
    st.session_state.routes = None
    st.experimental_rerun()

if st.session_state.start and st.session_state.destination :
    try:
        st.session_state.routes = client.directions(
            coordinates=[st.session_state.start[::-1], st.session_state.destination[::-1]],
            profile=profile,
            
            format="geojson",
        )
    except Exception as e:
        st.error(f"Error fetching routes: {e}")

if st.session_state.routes:
    routes = st.session_state.routes
    num_routes = len(routes["features"])
    st.write(f"### 🚦 Found {num_routes} alternative routes for {travel_mode}")

    colors = ["red", "green", "blue"]

    
    for i, feature in enumerate(routes["features"]):
        
        distance_meters = feature["properties"]["segments"][0]["distance"]
        duration_seconds = feature["properties"]["segments"][0]["duration"]

        
        distance_km = round(distance_meters / 1000, 2)
        duration_minutes = round(duration_seconds / 60, 2)

        st.write(f"### 🗺️ Route {i+1}")
        st.write(f"🛣️ **Distance:** {distance_km} km")
        st.write(f"⏱️ **Estimated Time:** {duration_minutes} minutes")

        
        center_lat = (st.session_state.start[0] + st.session_state.destination[0]) / 2
        center_lon = (st.session_state.start[1] + st.session_state.destination[1]) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

        
        folium.Marker(st.session_state.start, popup="Start", icon=folium.Icon(color="blue")).add_to(m)
        folium.Marker(st.session_state.destination, popup="Destination", icon=folium.Icon(color="red")).add_to(m)

        
        coords = [(lat, lon) for lon, lat in feature["geometry"]["coordinates"]]
        folium.PolyLine(coords, color=colors[i % len(colors)], weight=5, opacity=0.7).add_to(m)

        
        st_folium(m, width=700, height=500)
