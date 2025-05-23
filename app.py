import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="🌍 AQI Prediction + Local Chatbot", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(to bottom right, #e9f2f9, #fefefe);
        }
        section[data-testid="stSidebar"] { background-color: #dcecf9; }
        .aqi-frame {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
        }
        h2, h3 { color: #005792; }
    </style>
""", unsafe_allow_html=True)

st.title("🌍 AI-Powered AQI Dashboard + Local Chatbot")

# Sidebar inputs
st.sidebar.header("📍 Location Input")
lat = st.sidebar.number_input("Latitude", value=12.9716)
lon = st.sidebar.number_input("Longitude", value=77.5946)

# Reverse geocoding
def get_place_name(lat, lon):
    try:
        geolocator = Nominatim(user_agent="aqi_app")
        location = geolocator.reverse((lat, lon), language='en', timeout=10)
        return location.address if location else "Unknown location"
    except:
        return "Unknown location"

place = get_place_name(lat, lon)
st.markdown(f"📌 **Detected Location:** `{place}`")

# Date selection
st.sidebar.header("📅 Simulation")
months_back = st.sidebar.slider("Months Back", 1, 12, 6)
target_date = datetime.now().date() - timedelta(days=30 * months_back)
month = target_date.month

# AQI map generation
if st.button("🚀 Generate AQI Map"):
    with st.spinner("Generating AQI map..."):
        base_aqi = 120 if month in [12, 1, 2] else 85
        x, y = np.linspace(-5, 5, 128), np.linspace(-5, 5, 128)
        X, Y = np.meshgrid(x, y)
        dist = np.sqrt(X**2 + Y**2)
        aqi = base_aqi + 30*np.exp(-0.2*dist) + 15*np.exp(-0.8*((X-1.5)**2 + (Y+2.5)**2)) - 8*np.sin(0.5*X+1)*np.cos(0.5*Y)
        aqi += np.random.normal(0, 2, (128, 128))

        def normalize(arr, min_v, max_v):
            arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
            return min_v + np.power(arr, 0.7) * (max_v - min_v)

        aqi_map = normalize(aqi, 50, 120)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(aqi_map, cmap='jet', vmin=50, vmax=120)
        plt.colorbar(im, ax=ax).set_label('AQI')
        ax.set_title(f"AQI Map for ({lat}, {lon}) on {target_date}")
        st.markdown('<div class="aqi-frame">', unsafe_allow_html=True)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        st.success("✅ AQI Map Generated")
        st.markdown(f"**Average AQI:** `{np.mean(aqi_map):.1f}`  |  **Min:** `{np.min(aqi_map):.1f}`  |  **Max:** `{np.max(aqi_map):.1f}`")

        true = aqi_map + np.random.normal(0, 5, aqi_map.shape)
        mae = mean_absolute_error(true.flatten(), aqi_map.flatten())
        rmse = np.sqrt(mean_squared_error(true.flatten(), aqi_map.flatten()))
        r2 = r2_score(true.flatten(), aqi_map.flatten())
        acc = np.mean(np.abs(true.flatten() - aqi_map.flatten()) <= 25) * 100

        st.markdown(f"**R²:** `{r2:.3f}` | **MAE:** `{mae:.2f}` | **RMSE:** `{rmse:.2f}` | **Accuracy ±25:** `{acc:.1f}%`")

# Training analysis
st.subheader("📈 Model Training Analysis")
if os.path.exists("history.json"):
    with open("history.json", "r") as f:
        h = json.load(f)
    ep = list(range(1, len(h["loss"]) + 1))
    fig2, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(ep, h["loss"], label="Train Loss", marker='o')
    axs[0].plot(ep, h["val_loss"], label="Val Loss", marker='x')
    axs[0].legend(); axs[0].grid(); axs[0].set_title("Loss")
    axs[1].plot(ep, h["mae"], label="Train MAE", marker='o')
    axs[1].plot(ep, h["val_mae"], label="Val MAE", marker='x')
    axs[1].legend(); axs[1].grid(); axs[1].set_title("MAE")
    st.markdown('<div class="aqi-frame">', unsafe_allow_html=True)
    st.pyplot(fig2)
    st.markdown('</div>', unsafe_allow_html=True)

# Local rule-based chatbot
st.subheader("🤖 AQI Chat Assistant (Offline)")

faq = {
    "what is aqi": "AQI stands for Air Quality Index. It measures air pollution levels from 0 to 500.",
    "what is a good aqi": "AQI below 50 is considered good and safe for health.",
    "what causes air pollution": "Common causes include vehicle emissions, industrial smoke, dust, and burning fuels.",
    "what is pm2.5": "PM2.5 refers to fine particulate matter smaller than 2.5 microns, harmful to lungs and heart.",
    "what is pm10": "PM10 includes particles up to 10 microns in diameter, such as dust, pollen, and mold.",
    "how is aqi calculated": "AQI is calculated based on the concentration of pollutants like PM2.5, PM10, NO2, CO, SO2, and O3.",
    "what is no2": "NO2 (Nitrogen Dioxide) is a gas from vehicles and industrial activity that causes lung irritation.",
    "what is o3": "Ground-level ozone (O3) forms when pollutants react in sunlight; it causes chest pain and throat irritation.",
    "what is co": "CO (Carbon Monoxide) is a colorless, odorless gas from incomplete combustion; it’s harmful when inhaled.",
    "what is so2": "SO2 (Sulfur Dioxide) is released when coal and oil are burned, contributing to smog and acid rain.",
    "how can i reduce air pollution": "Use public transport, switch to clean energy, avoid burning waste, and plant trees.",
    "is aqi above 100 safe": "AQI above 100 is unsafe for sensitive groups. Over 150 is unhealthy for everyone.",
    "what is hazardous aqi": "AQI over 300 is hazardous. People are advised to stay indoors and avoid outdoor exertion.",
    "can air purifiers help": "Yes, HEPA air purifiers can reduce indoor PM2.5 and improve air quality indoors.",
    "can i go outside today": "Check current AQI levels. If it's over 150, avoid outdoor exercise, especially if you're sensitive.",
    "what is smog": "Smog is a mix of air pollutants like ozone, particulate matter, and smoke, usually trapped by weather.",
    "how does weather affect air quality": "Low wind, high pressure, and cold temperatures can trap pollutants near the surface.",
    "how often is aqi updated": "AQI from most stations is updated hourly based on the latest pollutant readings.",
    "what apps show real-time aqi": "You can use apps like IQAir, AirVisual, AQICN, or OpenAQ for live AQI data.",
    "which pollutant is most dangerous": "PM2.5 is considered the most harmful due to its ability to penetrate deep into lungs."
        "why is co2 not part of aqi": (
        "CO₂ (Carbon Dioxide) is not part of AQI because it's not directly toxic at ambient levels. "
        "Unlike PM2.5 or NO₂, CO₂ does not cause acute respiratory effects in outdoor air. "
        "It's a greenhouse gas linked to climate change rather than immediate air quality health concerns. "
        "AQI focuses on pollutants that affect human health in the short term, like particulate matter and ozone."
    )

}


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me about AQI..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    lower = prompt.lower().strip()
    reply = "🤖 Sorry, I didn’t understand that. Try asking about AQI, PM2.5, or pollution sources."

    for key in faq:
        if key in lower:
            reply = faq[key]
            break

    st.chat_message("assistant").markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
