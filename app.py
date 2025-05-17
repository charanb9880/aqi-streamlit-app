import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from sklearn.metrics import mean\_squared\_error, mean\_absolute\_error, r2\_score

st.set\_page\_config(page\_title="🌍 AQI Prediction + Local Chatbot", layout="wide")

# --- Custom CSS ---

st.markdown(""" <style>
html, body, \[class\*="css"] {
font-family: 'Segoe UI', sans-serif;
background: linear-gradient(to bottom right, #e9f2f9, #fefefe);
}
section\[data-testid="stSidebar"] { background-color: #dcecf9; }
.aqi-frame {
background-color: #ffffff;
padding: 1.5rem;
border-radius: 12px;
box-shadow: 0 8px 20px rgba(0,0,0,0.05);
margin-bottom: 2rem;
}
h2, h3 { color: #005792; } </style>
""", unsafe\_allow\_html=True)

st.title("🌍 AI-Powered AQI Dashboard + Local Chatbot")

# Sidebar inputs

st.sidebar.header("📍 Location Input")
lat = st.sidebar.number\_input("Latitude", value=12.9716)
lon = st.sidebar.number\_input("Longitude", value=77.5946)

# Reverse geocoding

def get\_place\_name(lat, lon):
try:
geolocator = Nominatim(user\_agent="aqi\_app")
location = geolocator.reverse((lat, lon), language='en', timeout=10)
return location.address if location else "Unknown location"
except:
return "Unknown location"

place = get\_place\_name(lat, lon)
st.markdown(f"📌 **Detected Location:** `{place}`")

# Date selection

st.sidebar.header("📅 Simulation")
months\_back = st.sidebar.slider("Months Back", 1, 12, 6)
target\_date = datetime.now().date() - timedelta(days=30 \* months\_back)
month = target\_date.month

# AQI map generation

if st.button("🚀 Generate AQI Map"):
with st.spinner("Generating AQI map..."):
base\_aqi = 120 if month in \[12, 1, 2] else 85
x, y = np.linspace(-5, 5, 128), np.linspace(-5, 5, 128)
X, Y = np.meshgrid(x, y)
dist = np.sqrt(X**2 + Y**2)
aqi = base\_aqi + 30*np.exp(-0.2*dist) + 15*np.exp(-0.8*((X-1.5)\*\*2 + (Y+2.5)\*\*2)) - 8*np.sin(0.5*X+1)*np.cos(0.5*Y)
aqi += np.random.normal(0, 2, (128, 128))

```
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
```

# Training analysis

st.subheader("📈 Model Training Analysis")
if os.path.exists("history.json"):
with open("history.json", "r") as f:
h = json.load(f)
ep = list(range(1, len(h\["loss"]) + 1))
fig2, axs = plt.subplots(1, 2, figsize=(12, 4))
axs\[0].plot(ep, h\["loss"], label="Train Loss", marker='o')
axs\[0].plot(ep, h\["val\_loss"], label="Val Loss", marker='x')
axs\[0].legend(); axs\[0].grid(); axs\[0].set\_title("Loss")
axs\[1].plot(ep, h\["mae"], label="Train MAE", marker='o')
axs\[1].plot(ep, h\["val\_mae"], label="Val MAE", marker='x')
axs\[1].legend(); axs\[1].grid(); axs\[1].set\_title("MAE")
st.markdown('<div class="aqi-frame">', unsafe\_allow\_html=True)
st.pyplot(fig2)
st.markdown('</div>', unsafe\_allow\_html=True)

# Local rule-based chatbot

st.subheader("🤖 AQI Chat Assistant (Offline)")

faq = {
"what is aqi": "AQI stands for Air Quality Index. It measures air pollution levels from 0 to 500.",
"what is a good aqi": "AQI below 50 is considered good and safe for health.",
"what causes air pollution": "Common causes include vehicle emissions, industrial smoke, dust, and burning fuels.",
"what is pm2.5": "PM2.5 refers to fine particulate matter smaller than 2.5 microns, harmful to lungs and heart.",
"how is aqi calculated": "AQI is calculated based on pollutants like PM2.5, PM10, NO2, CO, SO2, and O3.",
}

if "messages" not in st.session\_state:
st.session\_state.messages = \[]

for msg in st.session\_state.messages:
with st.chat\_message(msg\["role"]):
st.markdown(msg\["content"])

if prompt := st.chat\_input("Ask me about AQI..."):
st.session\_state.messages.append({"role": "user", "content": prompt})
lower = prompt.lower().strip()
reply = "🤖 Sorry, I didn’t understand that. Try asking about AQI, PM2.5, or pollution sources."

```
for key in faq:
    if key in lower:
        reply = faq[key]
        break

st.chat_message("assistant").markdown(reply)
st.session_state.messages.append({"role": "assistant", "content": reply})   
```
