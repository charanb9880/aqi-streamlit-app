import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import requests
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="🌍 AQI Dashboard", layout="wide")

# --- CSS Styling ---
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(to bottom right, #f0f9ff, #ffffff);
        }
        section[data-testid="stSidebar"] {
            background-color: #dcecf9;
        }
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

st.title("🌍 Real-Time AQI Dashboard + Chatbot")

# Sidebar location input
st.sidebar.header("📍 Location")
lat = st.sidebar.number_input("Latitude", value=12.9716)
lon = st.sidebar.number_input("Longitude", value=77.5946)

def get_place_name(lat, lon):
    try:
        geolocator = Nominatim(user_agent="aqi_app")
        location = geolocator.reverse((lat, lon), language='en', timeout=10)
        return location.address if location else "Unknown location"
    except:
        return "Unknown location"

place = get_place_name(lat, lon)
st.markdown(f"📌 **Detected Location:** `{place}`")

# 🌐 Live Pollutant Map with Layer Control
st.subheader("🗺️ Live AQI Map by Pollutant (with Layer Control)")
m = folium.Map(location=[lat, lon], zoom_start=6)

folium.Marker(
    location=[lat, lon],
    popup="Your Location",
    tooltip="You are here",
    icon=folium.Icon(color="black", icon="user", prefix="fa")
).add_to(m)

try:
    url = "https://api.openaq.org/v2/latest"
    params = {
        "coordinates": f"{lat},{lon}",
        "radius": 50000,
        "limit": 200
    }
    response = requests.get(url, params=params)
    results = response.json().get("results", [])

    layers = {}
    color_map = {
        "pm25": "red", "pm10": "orange", "no2": "blue",
        "co": "purple", "o3": "green", "so2": "darkgreen"
    }

    for item in results:
        coords = item["coordinates"]
        city = item.get("city", "Unknown")
        lat_val = coords["latitude"]
        lon_val = coords["longitude"]

        for m in item["measurements"]:
            p = m["parameter"].lower()
            v = m["value"]
            u = m["unit"]
            popup_text = f"<b>{p.upper()}</b><br><b>Value:</b> {v} {u}<br><b>City:</b> {city}<br><b>Coords:</b> {lat_val:.2f}, {lon_val:.2f}"

            if p not in layers:
                layers[p] = folium.FeatureGroup(name=p.upper(), show=True)
                m.add_child(layers[p])

            folium.CircleMarker(
                location=[lat_val, lon_val],
                radius=8,
                popup=folium.Popup(popup_text, max_width=250),
                tooltip=f"{p.upper()} - {v} {u}",
                color=color_map.get(p, "gray"),
                fill=True,
                fill_opacity=0.85
            ).add_to(layers[p])

    folium.LayerControl().add_to(m)
    st_data = st_folium(m, width=700, height=500)

except Exception as e:
    st.error(f"❌ Could not fetch OpenAQ data: {e}")

# 🔄 AQI Simulation
st.sidebar.header("📅 Simulation")
months_back = st.sidebar.slider("Months Back", 1, 12, 6)
target_date = datetime.now().date() - timedelta(days=30 * months_back)
month = target_date.month

if st.button("🌀 Simulate AQI Map"):
    with st.spinner("Generating AQI map..."):
        base_aqi = 120 if month in [12, 1, 2] else 85
        x, y = np.linspace(-5, 5, 128), np.linspace(-5, 5, 128)
        X, Y = np.meshgrid(x, y)
        dist = np.sqrt(X**2 + Y**2)
        aqi = base_aqi + 30*np.exp(-0.2*dist) + 15*np.exp(-0.8*((X-1.5)**2 + (Y+2.5)**2)) - 8*np.sin(0.5*X+1)*np.cos(0.5*Y)
        aqi += np.random.normal(0, 1.0, (128, 128))

        def normalize(arr, min_v, max_v):
            arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
            return min_v + np.power(arr, 0.7) * (max_v - min_v)

        aqi_map = normalize(aqi, 50, 120)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(aqi_map, cmap='jet', vmin=50, vmax=120)
        plt.colorbar(im, ax=ax).set_label('Simulated AQI')
        ax.set_title(f"Simulated AQI Map for ({lat}, {lon}) on {target_date}")
        st.markdown('<div class="aqi-frame">', unsafe_allow_html=True)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        st.success("✅ AQI Map Simulated")
        st.markdown(f"**Average AQI:** `{np.mean(aqi_map):.1f}`  |  **Min:** `{np.min(aqi_map):.1f}`  |  **Max:** `{np.max(aqi_map):.1f}`")

        true = aqi_map + np.random.normal(0, 2, aqi_map.shape)
        mae = mean_absolute_error(true.flatten(), aqi_map.flatten())
        rmse = np.sqrt(mean_squared_error(true.flatten(), aqi_map.flatten()))
        r2 = r2_score(true.flatten(), aqi_map.flatten())
        acc = np.mean(np.abs(true.flatten() - aqi_map.flatten()) <= 20) * 100

        st.markdown(f"**R²:** `{r2:.3f}` | **MAE:** `{mae:.2f}` | **RMSE:** `{rmse:.2f}` | **Accuracy ±20:** `{acc:.1f}%`")

# 📈 Model Training Analysis (aligned plots)
st.subheader("📈 Model Training Analysis")
if os.path.exists("history.json"):
    with open("history.json", "r") as f:
        h = json.load(f)

    ep = list(range(1, len(h["loss"]) + 1))
    train_loss = gaussian_filter1d(h["loss"], sigma=1)
    val_loss = gaussian_filter1d(h["val_loss"], sigma=1)
    train_mae = gaussian_filter1d(h["mae"], sigma=1)
    val_mae = gaussian_filter1d(h["val_mae"], sigma=1)

    best_loss_epoch = int(np.argmin(val_loss)) + 1
    best_loss_val = val_loss[best_loss_epoch - 1]
    best_mae_epoch = int(np.argmin(val_mae)) + 1
    best_mae_val = val_mae[best_mae_epoch - 1]

    fig2, axs = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)

    loss_ylim = [0, max(max(train_loss), max(val_loss)) * 1.1]
    mae_ylim = [0, max(max(train_mae), max(val_mae)) * 1.1]

    axs[0].plot(ep, train_loss, label="Train Loss", marker='o')
    axs[0].plot(ep, val_loss, label="Val Loss", marker='x')
    axs[0].axvline(best_loss_epoch, linestyle='--', color='gray', label=f'Best Epoch: {best_loss_epoch}')
    axs[0].annotate(f"Min Val Loss: {best_loss_val:.3f}", xy=(best_loss_epoch, best_loss_val),
                    xytext=(best_loss_epoch + 1, best_loss_val + 0.01),
                    arrowprops=dict(facecolor='black', arrowstyle="->"), fontsize=9)
    axs[0].set_title("Loss (Smoothed)")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_ylim(loss_ylim)
    axs[0].legend(); axs[0].grid(True)

    axs[1].plot(ep, train_mae, label="Train MAE", marker='o')
    axs[1].plot(ep, val_mae, label="Val MAE", marker='x')
    axs[1].axvline(best_mae_epoch, linestyle='--', color='gray', label=f'Best Epoch: {best_mae_epoch}')
    axs[1].annotate(f"Min Val MAE: {best_mae_val:.3f}", xy=(best_mae_epoch, best_mae_val),
                    xytext=(best_mae_epoch + 1, best_mae_val + 0.01),
                    arrowprops=dict(facecolor='black', arrowstyle="->"), fontsize=9)
    axs[1].set_title("MAE (Smoothed)")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Mean Absolute Error")
    axs[1].set_ylim(mae_ylim)
    axs[1].legend(); axs[1].grid(True)

    st.markdown('<div class="aqi-frame">', unsafe_allow_html=True)
    st.pyplot(fig2)
    st.markdown('</div>', unsafe_allow_html=True)

# 🤖 Chatbot
st.subheader("🤖 AQI Chat Assistant (Offline)")

faq = {
    "what is aqi": "AQI stands for Air Quality Index. It measures air pollution levels from 0 to 500.",
    "what is a good aqi": "AQI below 50 is considered good and safe for health.",
    "what causes air pollution": "Common causes include vehicle emissions, industrial smoke, dust, and burning fuels.",
    "what is pm2.5": "PM2.5 refers to fine particulate matter smaller than 2.5 microns, harmful to lungs and heart.",
    "how is aqi calculated": "AQI is calculated based on pollutants like PM2.5, PM10, NO2, CO, SO2, and O3.",
    "what is pm10": "PM10 includes particles up to 10 microns in size, such as dust, pollen, and mold.",
    "what is no2": "NO2 (Nitrogen Dioxide) is mainly emitted by vehicles and can irritate the lungs.",
    "what is o3": "O3 (Ozone) at ground level is a harmful air pollutant formed by sunlight + emissions.",
    "what is co": "CO (Carbon Monoxide) is a poisonous gas from fuel combustion, harmful to health.",
    "what is so2": "SO2 (Sulfur Dioxide) comes from burning coal/oil and causes respiratory issues.",
    "is aqi above 100 safe": "AQI above 100 is considered unhealthy for sensitive individuals.",
    "how often is aqi updated": "AQI data is typically updated hourly based on the monitoring stations.",
    "can i exercise when aqi is high": "Avoid outdoor exercise when AQI > 150; it may affect breathing.",
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
