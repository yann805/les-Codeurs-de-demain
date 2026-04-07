import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import textwrap
from datetime import datetime, timedelta

import requests

def sanitize_coord(val):
    try:
        return float(str(val).replace(",", "."))
    except Exception:
        return 0.0

def generate_demo_forecast(days=4):
    """Generate realistic demo forecast when API is unavailable"""
    from datetime import datetime, timedelta
    
    forecast = []
    base_date = datetime.now()
    
    for i in range(days):
        date = base_date + timedelta(days=i+1)
        # Realistic Cameroon weather patterns
        temp_mean = 25 + np.random.normal(0, 2)
        temp_max = temp_mean + 3 + np.random.uniform(0, 3)
        temp_min = temp_mean - 3 - np.random.uniform(0, 2)
        rain = max(0, np.random.exponential(2))
        wind = max(1, np.random.normal(3, 1))
        radiation = max(5, np.random.normal(15, 3))
        
        forecast.append({
            "date": date.strftime("%Y-%m-%d"),
            "day": date.strftime("%A"),
            "temperature_2m_mean": temp_mean,
            "temperature_2m_max": temp_max,
            "temperature_2m_min": temp_min,
            "precipitation_sum": rain,
            "wind_speed_10m_max": wind,
            "shortwave_radiation_sum": radiation
        })
    
    return forecast

def get_real_weather(lat, lon):
    """Fetch today's weather from Open-Meteo API (not cached - weather changes)"""
    lat = sanitize_coord(lat)
    lon = sanitize_coord(lon)

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,wind_speed_10m_max,shortwave_radiation_sum"
        f"&timezone=auto"
    )

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()
        daily = data.get("daily", {})

        if not daily or len(daily.get("time", [])) == 0:
            raise ValueError("No daily weather data returned from API")

        # API SUCCESS - Mark that live weather is from real data
        st.session_state["weather_api_working"] = True
        
        return {
            "temperature": float(daily["temperature_2m_mean"][0]),
            "temperature_max": float(daily["temperature_2m_max"][0]),
            "temperature_min": float(daily["temperature_2m_min"][0]),
            "wind": float(daily["wind_speed_10m_max"][0]),
            "rain": float(daily["precipitation_sum"][0]),
            "radiation": float(daily["shortwave_radiation_sum"][0])
        }

    except requests.exceptions.Timeout:
        st.session_state["weather_api_working"] = False
        st.error(f"⏱️ API timeout: Could not fetch live weather from Open-Meteo. Coords: ({lat:.2f}, {lon:.2f})")
    except requests.exceptions.ConnectionError:
        st.session_state["weather_api_working"] = False
        st.error("🔌 Connection error: Could not fetch live weather from Open-Meteo.")
    except requests.exceptions.HTTPError as e:
        st.session_state["weather_api_working"] = False
        if "502" in str(e):
            st.error("🔧 Open-Meteo API is temporarily down (502 Bad Gateway). Live weather unavailable.")
        else:
            st.error(f"⚠️ API error: {str(e)[:80]}. Live weather unavailable.")
    except requests.exceptions.RequestException as e:
        st.session_state["weather_api_working"] = False
        st.error(f"⚠️ Network error: {str(e)[:80]}. Live weather unavailable.")
    except (ValueError, KeyError, TypeError) as e:
        st.session_state["weather_api_working"] = False
        st.error(f"📊 Invalid API response: {str(e)[:80]}. Live weather unavailable.")
    except Exception as e:
        st.session_state["weather_api_working"] = False
        st.error(f"❌ Unexpected error: {str(e)[:80]}. Live weather unavailable.")

    return None

def get_3day_weather_forecast(lat, lon):
    """Fetch next 4 days of weather from Open-Meteo API (not cached)"""
    lat = sanitize_coord(lat)
    lon = sanitize_coord(lon)

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,wind_speed_10m_max,shortwave_radiation_sum"
            f"&timezone=auto"
        )

        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()

        daily = data.get("daily", {})
        times = daily.get("time", [])

        if not times:
            raise ValueError("No forecast dates returned from API")

        start = 1 if len(times) > 1 else 0
        end = min(start + 4, len(times))

        forecast_list = []
        for i in range(start, end):
            temp_mean = float(daily["temperature_2m_mean"][i])
            temp_max = float(daily["temperature_2m_max"][i])
            temp_min = float(daily["temperature_2m_min"][i])
            rain = float(daily["precipitation_sum"][i])
            wind = float(daily["wind_speed_10m_max"][i])
            radiation = float(daily["shortwave_radiation_sum"][i])

            date_str = times[i]
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()

            forecast_list.append({
                "date": date_obj.strftime("%Y-%m-%d"),
                "day": date_obj.strftime("%A"),
                "temperature_2m_mean": temp_mean,
                "temperature_2m_max": temp_max,
                "temperature_2m_min": temp_min,
                "precipitation_sum": rain,
                "wind_speed_10m_max": wind,
                "shortwave_radiation_sum": radiation
            })

        if not forecast_list:
            raise ValueError("Weather forecast did not contain future days")

        return forecast_list

    except requests.exceptions.Timeout:
        st.error("⏱️ API timeout: Could not fetch weather forecast.")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Network error fetching forecast: {str(e)[:100]}")
        return []
    except (ValueError, KeyError, TypeError) as e:
        st.error(f"⚠️ Invalid API data in forecast: {str(e)[:100]}")
        return []
    except Exception as e:
        st.error(f"⚠️ Unexpected forecast error: {str(e)[:100]}")
        return []

def predict_future_pm25(
    forecast_weather_list,
    model,
    lat,
    lon,
    city_enc=0,
    region_enc=0,
    current_weather=None
):
    predictions = []

    if not forecast_weather_list or model is None:
        return []

    if current_weather:
        prev_temp = current_weather.get("temperature", forecast_weather_list[0]["temperature_2m_mean"])
        prev_wind = current_weather.get("wind", forecast_weather_list[0]["wind_speed_10m_max"])
        prev_temp_lag7 = current_weather.get("temperature", prev_temp)
        prev_temp_roll7 = current_weather.get("temperature", prev_temp)
    else:
        prev_temp = forecast_weather_list[0]["temperature_2m_mean"]
        prev_wind = forecast_weather_list[0]["wind_speed_10m_max"]
        prev_temp_lag7 = forecast_weather_list[0]["temperature_2m_mean"]
        prev_temp_roll7 = forecast_weather_list[0]["temperature_2m_mean"]

    for day_data in forecast_weather_list:
        temp_mean = day_data["temperature_2m_mean"]
        temp_max = day_data["temperature_2m_max"]
        temp_min = day_data["temperature_2m_min"]
        rain = day_data["precipitation_sum"]
        wind = day_data["wind_speed_10m_max"]
        radiation = day_data["shortwave_radiation_sum"]

        date_obj = datetime.strptime(day_data["date"], "%Y-%m-%d").date()
        month = date_obj.month
        day_of_year = date_obj.timetuple().tm_yday

        temp_amplitude = temp_max - temp_min
        sunshine_ratio = radiation / 24 if radiation is not None else 0.0
        et0_fao = temp_amplitude * 0.35
        is_no_wind = 1 if wind < 1 else 0
        is_no_rain = 1 if rain == 0 else 0
        is_dry_season = 1 if month in [12, 1, 2] else 0

        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        input_data = pd.DataFrame({
            'temperature_2m_mean': [temp_mean],
            'temperature_2m_max': [temp_max],
            'temperature_2m_min': [temp_min],
            'precipitation_sum': [rain],
            'wind_speed_10m_max': [wind],
            'wind_gusts_10m_max': [wind * 1.1],
            'shortwave_radiation_sum': [radiation],
            'et0_fao_evapotranspiration': [et0_fao],
            'sunshine_ratio': [sunshine_ratio],
            'temp_amplitude': [temp_amplitude],
            'is_no_wind': [is_no_wind],
            'is_no_rain': [is_no_rain],
            'is_dry_season': [is_dry_season],
            'month_sin': [month_sin],
            'month_cos': [month_cos],
            'day_of_year': [day_of_year],
            'temp_lag1': [prev_temp],
            'temp_lag7': [prev_temp_lag7],
            'wind_lag1': [prev_wind],
            'temp_roll7': [prev_temp_roll7],
            'latitude': [lat],
            'longitude': [lon],
            'region_enc': [region_enc],
            'city_enc': [city_enc]
        })

        try:
            pm25_pred = float(model.predict(input_data)[0])
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            pm25_pred = 0.0

        predictions.append({
            "date": day_data["date"],
            "day": day_data["day"],
            "pm25": pm25_pred,
            "temperature": temp_mean,
            "wind": wind,
            "rain": rain
        })

        prev_temp_roll7 = (prev_temp_roll7 * 6 + temp_mean) / 7
        prev_temp = temp_mean
        prev_wind = wind
        prev_temp_lag7 = temp_mean

    return predictions

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="AirGuard Cameroon",
    layout="wide"
)

# ── Custom CSS (Professional UI + Icons) ────────────────────
st.markdown("""
<style>
.main-title {
    font-size:80px;
    font-weight:2000;
    color:#228B22;
}
.subtitle {
    font-size:20px;
    color:#7f8c8d;
    margin-bottom:24px;
}
.card {
    padding:15px;
    border-radius:12px;
    background-color:#ffffff;
    box-shadow:0 2px 10px rgba(0,0,0,0.05);
    margin-bottom:20px;
}
.section-title {
    font-size:24px;
    font-weight:550;
    margin-top:15px;        
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

# ── Title ───────────────────────────────────────────────────
st.markdown('<div class="main-title">AirGuard Cameroon</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered Air Quality Prediction Dashboard</div>', unsafe_allow_html=True)

# ── Load data ───────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_excel("data/Dataset_complet_Meteo.xlsx")

df = load_data()

# ── Load model ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("models/xgb_model.pkl")

model = load_model()

# ── Preprocessing ───────────────────────────────────────────
num_cols = [
    'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
    'apparent_temperature_mean', 'precipitation_sum', 'rain_sum',
    'wind_speed_10m_max', 'wind_gusts_10m_max',
    'shortwave_radiation_sum', 'et0_fao_evapotranspiration',
    'sunshine_duration', 'latitude', 'longitude'
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['time'] = pd.to_datetime(df['time'])
df['month'] = df['time'].dt.month
df['day_of_year'] = df['time'].dt.dayofyear

df['temp_amplitude'] = df['temperature_2m_max'] - df['temperature_2m_min']
df['sunshine_ratio'] = df['sunshine_duration'] / (24 * 3600)

df['is_no_wind'] = (df['wind_speed_10m_max'] < 1).astype(int)
df['is_no_rain'] = (df['precipitation_sum'] == 0).astype(int)
df['is_dry_season'] = df['month'].isin([12,1,2]).astype(int)

# Pollution proxy
df['pm25_proxy'] = (
    0.35 * df['temperature_2m_mean'].fillna(df['temperature_2m_mean'].mean())
    + 0.25 * df['shortwave_radiation_sum'].fillna(0)
    + 0.20 * df['et0_fao_evapotranspiration'].fillna(0)
    + 8.0  * df['is_no_wind']
    + 5.0  * df['is_no_rain']
    + 4.0  * df['is_dry_season']
).clip(lower=0)

# ── Sidebar ────────────────────────────────────────────────
st.sidebar.header("User Inputs")

def on_city_change():
    """Callback to fetch weather when city is selected"""
    new_city = st.session_state.city_selectbox
    if new_city != st.session_state.get("selected_city"):
        st.session_state["selected_city"] = new_city
        city_data_temp = df[df['city'] == new_city].iloc[-1]
        lat_temp = sanitize_coord(city_data_temp['latitude'])
        lon_temp = sanitize_coord(city_data_temp['longitude'])
        
        weather = get_real_weather(lat_temp, lon_temp)
        if weather is not None:
            st.session_state["temp"] = weather["temperature"]
            st.session_state["temp_max"] = weather["temperature_max"]
            st.session_state["temp_min"] = weather["temperature_min"]
            st.session_state["wind"] = weather["wind"]
            st.session_state["rain"] = weather["rain"]
            st.session_state["radiation"] = weather["radiation"]
            
            # Clear slider state so they reset to new weather values
            st.session_state["temp_slider_val"] = weather["temperature"]
            st.session_state["temp_max_slider_val"] = weather["temperature_max"]
            st.session_state["temp_min_slider_val"] = weather["temperature_min"]
            st.session_state["wind_slider_val"] = weather["wind"]
            st.session_state["rain_slider_val"] = weather["rain"]
            st.session_state["radiation_slider_val"] = weather["radiation"]
            
            # Force rerun so sliders display new values
            st.rerun()
        else:
            st.sidebar.error("Could not load live weather from API. Keeping current manual values.")

city = st.sidebar.selectbox(
    "City",
    df['city'].unique(),
    key="city_selectbox",
    on_change=on_city_change
)

city_data = df[df['city'] == city].iloc[-1]
lat = sanitize_coord(city_data['latitude'])
lon = sanitize_coord(city_data['longitude'])

# Initialize weather if not in session state
if "selected_city" not in st.session_state:
    st.session_state["weather_api_working"] = True  # Assume API works initially
    weather = get_real_weather(lat, lon)
    if weather is not None:
        st.session_state["temp"] = weather["temperature"]
        st.session_state["temp_max"] = weather["temperature_max"]
        st.session_state["temp_min"] = weather["temperature_min"]
        st.session_state["wind"] = weather["wind"]
        st.session_state["rain"] = weather["rain"]
        st.session_state["radiation"] = weather["radiation"]
    else:
        st.session_state["temp"] = None
        st.session_state["temp_max"] = None
        st.session_state["temp_min"] = None
        st.session_state["wind"] = None
        st.session_state["rain"] = None
        st.session_state["radiation"] = None
    st.session_state["selected_city"] = city

# Display current city and coordinates
st.sidebar.info(f" **{city}** — Lat: {lat:.2f}, Lon: {lon:.2f}")

# Get current weather from session state
temp = st.session_state.get("temp")
temp_max = st.session_state.get("temp_max")
temp_min = st.session_state.get("temp_min")
rain = st.session_state.get("rain")
wind = st.session_state.get("wind")
radiation = st.session_state.get("radiation")

# Create sliders with unique keys and callbacks to sync session state
def sync_temp():
    st.session_state["temp"] = st.session_state.temp_slider_val

def sync_temp_max():
    st.session_state["temp_max"] = st.session_state.temp_max_slider_val

def sync_temp_min():
    st.session_state["temp_min"] = st.session_state.temp_min_slider_val

def sync_rain():
    st.session_state["rain"] = st.session_state.rain_slider_val

def sync_wind():
    st.session_state["wind"] = st.session_state.wind_slider_val

def sync_radiation():
    st.session_state["radiation"] = st.session_state.radiation_slider_val

temp = st.sidebar.slider("Temperature Mean (°C)", 10.0, 50.0, float(temp) if temp is not None else 20.0, key="temp_slider_val", on_change=sync_temp)
temp_max = st.sidebar.slider("Temperature Max (°C)", 10.0, 50.0, float(temp_max) if temp_max is not None else 25.0, key="temp_max_slider_val", on_change=sync_temp_max)
temp_min = st.sidebar.slider("Temperature Min (°C)", 10.0, 50.0, float(temp_min) if temp_min is not None else 18.0, key="temp_min_slider_val", on_change=sync_temp_min)
rain = st.sidebar.slider("Precipitation (mm)", 0.0, 100.0, float(rain) if rain is not None else 0.0, key="rain_slider_val", on_change=sync_rain)
wind = st.sidebar.slider("Wind Speed (km/h)", 0.0, 50.0, float(wind) if wind is not None else 2.0, key="wind_slider_val", on_change=sync_wind)
radiation = st.sidebar.slider("Solar Radiation (MJ/m²)", 0.0, 50.0, float(radiation) if radiation is not None else 15.0, key="radiation_slider_val", on_change=sync_radiation)


st.markdown('<div class="subtitle">Real-Time Weather & PM2.5 Prediction</div>', unsafe_allow_html=True)

# ── Button for 4-Day PM2.5 Prediction (Model-based) ───────
st.sidebar.markdown("---")
if st.sidebar.button("Predict PM2.5 for Next 4 Days"):
    if model is None:
        st.sidebar.error("Model not loaded. Check models/xgb_model.pkl")
    else:
        with st.spinner("Fetching weather forecast and predicting PM2.5..."):
            # Get encoding values once
            city_data_temp = df[df['city'] == city]
            city_enc_val = city_data_temp['city_enc'].iloc[0] if 'city_enc' in df.columns and len(city_data_temp) > 0 else 0
            region_enc_val = city_data_temp['region_enc'].iloc[0] if 'region_enc' in df.columns and len(city_data_temp) > 0 else 0

            # Prepare current weather once
            current_weather = {
                "temperature": temp,
                "temperature_max": temp_max,
                "temperature_min": temp_min,
                "wind": wind,
                "rain": rain,
                "radiation": radiation
            }

            forecast_weather = get_3day_weather_forecast(lat, lon)

            if forecast_weather:
                predictions = predict_future_pm25(
                    forecast_weather,
                    model,
                    lat,
                    lon,
                    city_enc_val,
                    region_enc_val,
                    current_weather=current_weather
                )

                if predictions:
                    st.session_state["pm25_predictions"] = predictions
                    st.session_state["show_pm25_forecast"] = True
                    st.sidebar.success("4-day PM2.5 forecast generated!")
                else:
                    st.sidebar.error("Forecast prediction failed.")

                st.rerun()
            else:
                st.sidebar.error("Forecast API unavailable. Please try again later or adjust current weather values manually.")

# ── Live Weather Card (PRO UI - ICONS) ─────────────────────

if all(k in st.session_state for k in ["temp", "wind", "rain", "radiation"]):

    st.markdown('<div class="section-title">Live Weather Conditions</div>', unsafe_allow_html=True)

    # Check if API is working
    api_working = st.session_state.get("weather_api_working", True)
    api_badge = "🟢 Real-Time" if api_working else "🔴 Offline"
    api_color = "#22c55e" if api_working else "#ef4444"

    temp_display = "--" if st.session_state.get("temp") is None else f"{st.session_state.get('temp'):.1f}"
    wind_display = "--" if st.session_state.get("wind") is None else f"{st.session_state.get('wind'):.1f}"
    rain_display = "--" if st.session_state.get("rain") is None else f"{st.session_state.get('rain'):.1f}"
    radiation_display = "--" if st.session_state.get("radiation") is None else f"{st.session_state.get('radiation'):.1f}"

    st.markdown(f"""
<div style="
    padding:22px;
    border-radius:16px;
    background:#ffffff;
    box-shadow:0 6px 20px rgba(0,0,0,0.08);
    margin-bottom:20px;
">

<div style="
    display:flex;
    justify-content:space-between;
    align-items:center;
    margin-bottom:18px;
">

<div style="
    font-size:20px;
    font-weight:600;
    color:#2c3e50;
">
    {city} — Real-Time Weather
</div>

<div style="
    display:inline-block;
    padding:4px 12px;
    background:{api_color}20;
    color:{api_color};
    border-radius:20px;
    font-size:11px;
    font-weight:600;
">
    {api_badge}
</div>

</div>

<div style="
    display:flex;
    justify-content:space-between;
">

<div style="text-align:center; flex:1;">
    <img src="https://img.icons8.com/ios-filled/50/fa5252/temperature.png" width="32"/>
    <div style="font-size:13px; color:#7f8c8d;">Temperature</div>
    <div style="font-size:18px; font-weight:600;">
        {temp_display} °C
    </div>
</div>

<div style="text-align:center; flex:1;">
    <img src="https://img.icons8.com/ios-filled/50/339af0/wind.png" width="32"/>
    <div style="font-size:13px; color:#7f8c8d;">Wind</div>
    <div style="font-size:18px; font-weight:600;">
        {wind_display} km/h
    </div>
</div>

<div style="text-align:center; flex:1;">
    <img src="https://img.icons8.com/ios-filled/50/4dabf7/rain.png" width="32"/>
    <div style="font-size:13px; color:#7f8c8d;">Rain</div>
    <div style="font-size:18px; font-weight:600;">
        {rain_display} mm
    </div>
</div>

<div style="text-align:center; flex:1;">
    <img src="https://img.icons8.com/ios-filled/50/f59f00/sun.png" width="32"/>
    <div style="font-size:13px; color:#7f8c8d;">Radiation</div>
    <div style="font-size:18px; font-weight:600;">
        {radiation_display} MJ/m²
    </div>
</div>

</div>
</div>
""", unsafe_allow_html=True)




# ── Heatmap ────────────────────────────────────────────────
st.markdown('<div class="section-title">Air Pollution Heatmap (Cameroon)</div>', unsafe_allow_html=True)

city_stats = df.groupby('city').agg(
    pollution=('pm25_proxy', 'mean'),
    temp=('temperature_2m_mean', 'mean'),
    lat=('latitude', 'first'),
    lon=('longitude', 'first')
).reset_index()

fig = px.scatter_mapbox(
    city_stats,
    lat='lat',
    lon='lon',
    color='pollution',
    size='temp',
    hover_name='city',
    color_continuous_scale='RdYlBu_r',
    zoom=4,
    mapbox_style='open-street-map'
)

st.plotly_chart(fig, use_container_width=True)



# ── Prediction Engine ───────────────────────────────────────

city_enc_val = city_data.get('city_enc', 0) if isinstance(city_data.get('city_enc'), (int, float)) else 0
region_enc_val = city_data.get('region_enc', 0) if isinstance(city_data.get('region_enc'), (int, float)) else 0

input_data = pd.DataFrame({
    'temperature_2m_mean': [temp],
    'temperature_2m_max': [temp_max],
    'temperature_2m_min': [temp_min],
    'precipitation_sum': [rain],
    'wind_speed_10m_max': [wind],
    'wind_gusts_10m_max': [wind * 1.1],
    'shortwave_radiation_sum': [radiation],
    'et0_fao_evapotranspiration': [(temp_max - temp_min) * 0.35],
    'sunshine_ratio': [radiation / 24 if radiation is not None else 0.0],
    'temp_amplitude': [temp_max - temp_min],
    'is_no_wind': [1 if wind < 1 else 0],
    'is_no_rain': [1 if rain == 0 else 0],
    'is_dry_season': [1 if datetime.now().month in [12, 1, 2] else 0],
    'month_sin': [np.sin(2 * np.pi * datetime.now().month / 12)],
    'month_cos': [np.cos(2 * np.pi * datetime.now().month / 12)],
    'day_of_year': [datetime.now().timetuple().tm_yday],
    'temp_lag1': [temp],
    'temp_lag7': [temp],
    'wind_lag1': [wind],
    'temp_roll7': [temp],
    'latitude': [lat],
    'longitude': [lon],
    'region_enc': [region_enc_val],
    'city_enc': [city_enc_val]
})

try:
    prediction = float(model.predict(input_data)[0])
except Exception as e:
    prediction = 0.0
    st.error(f"Current prediction failed: {e}")

# ── Air Quality Gauge ─────────────────────────────

st.markdown('<div class="section-title">Air Quality Index</div>', unsafe_allow_html=True)

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prediction,
    title={'text': "PM2.5 Level ug/m³", 'font': {'size': 18}},
    
    gauge={
        'axis': {'range': [0, 100]},
        
        'bar': {'color': "black"},
        
        'steps': [
            {'range': [0, 10], 'color': "#2ecc71"},   # Green
            {'range': [10, 25], 'color': "#f1c40f"},  # Yellow
            {'range': [25, 50], 'color': "#e67e22"},  # Orange
            {'range': [50, 100], 'color': "#e74c3c"}   # Red
        ],
        
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': prediction 
        }
    }
))

fig_gauge.update_layout(height=300)

st.plotly_chart(fig_gauge, use_container_width=True)


# ── Alert Box (Professional) ───────────────────────────────
st.markdown('<div class="section-title">Air Quality Status</div>', unsafe_allow_html=True)

def alert_box(color, border, title, text, icon):
    st.markdown(f"""
    <div style="
        padding:18px;
        border-radius:12px;
        background-color:{color};
        border-left:8px solid {border};
        font-size:18px;">
        <span style="font-size:22px;">{icon}</span>
        <b> {title}</b><br>{text}
    </div>
    """, unsafe_allow_html=True)

if prediction < 10:
    alert_box("#e8f8f5", "#2ecc71", "LOW Pollution",
              "Air quality is good. No immediate risk.",
              "🟢")

elif prediction < 25:
    alert_box("#fef9e7", "#f1c40f", "MODERATE Pollution",
              "Sensitive groups should reduce outdoor activity.",
              "🟡")
elif prediction < 50:
    alert_box("#fef5e7", "#e67e22", "HIGH Pollution",
              "Health alert: everyone may experience effects.",
              "🟠"  )
else:
    alert_box("#fdecea", "#e74c3c", "VERY HIGH Pollution",
              "Health risk detected. Avoid exposure.",
              "🔴")



# ── Confusion Matrix (Model Performance) ─────────────────────
st.markdown('<div class="section-title">Model Performance (Confusion Matrix)</div>', unsafe_allow_html=True)

try:
    from sklearn.metrics import confusion_matrix
    import plotly.figure_factory as ff

    # We use a sample of historical data to evaluate the model performance
    # (Using the last 500 rows to keep it fast and relevant)
    sample_df = df.sample(n=min(500, len(df)), random_state=42).copy()

    # 1. Prepare Features (Must match training features)
    # Using the same feature engineering as the prediction engine
    sample_df['temp_amplitude'] = sample_df['temperature_2m_max'] - sample_df['temperature_2m_min']
    sample_df['sunshine_ratio'] = sample_df['sunshine_duration'] / (24 * 3600)
    sample_df['is_no_wind'] = (sample_df['wind_speed_10m_max'] < 1).astype(int)
    sample_df['is_no_rain'] = (sample_df['precipitation_sum'] == 0).astype(int)
    sample_df['is_dry_season'] = sample_df['month'].isin([12,1,2]).astype(int)
    
    sample_df['month_sin'] = np.sin(2 * np.pi * sample_df['month'] / 12)
    sample_df['month_cos'] = np.cos(2 * np.pi * sample_df['month'] / 12)

    # Select feature columns
    feature_cols = [
        'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
        'precipitation_sum', 'wind_speed_10m_max', 'wind_gusts_10m_max',
        'shortwave_radiation_sum', 'et0_fao_evapotranspiration', 'sunshine_ratio',
        'temp_amplitude', 'is_no_wind', 'is_no_rain', 'is_dry_season',
        'month_sin', 'month_cos', 'day_of_year',
        'temp_lag1', 'temp_lag7', 'wind_lag1', 'temp_roll7',
        'latitude', 'longitude', 'region_enc', 'city_enc'
    ]
    
    # Handle missing lag columns for historical data if they don't exist
    for col in ['temp_lag1', 'temp_lag7', 'wind_lag1', 'temp_roll7']:
        if col not in sample_df.columns:
            sample_df[col] = sample_df['temperature_2m_mean'] # Fallback
            
    # Get encoded values safely
    if 'city_enc' not in sample_df.columns: sample_df['city_enc'] = 0
    if 'region_enc' not in sample_df.columns: sample_df['region_enc'] = 0

    X_test = sample_df[feature_cols].fillna(0)
    
    # 2. Make Predictions
    y_pred = model.predict(X_test)
    
    # 3. Binning into Categories (Good, Moderate, High, Unhealthy)
    # We use the 'pm25_proxy' as the 'Ground Truth' for this demonstration
    def categorize_pm25(val):
        if val <= 10: return 0  # Good
        if val <= 25: return 1  # Moderate
        if val <= 50: return 2  # High
        return 3                # Unhealthy

    y_true_classes = sample_df['pm25_proxy'].apply(categorize_pm25)
    y_pred_classes = pd.Series(y_pred).apply(categorize_pm25)
    
    # 4. Compute Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes, labels=[0, 1, 2, 3])
    
    # 5. Plot using Plotly
    x = ['Good', 'Moderate', 'High', 'Unhealthy']
    y = ['Good', 'Moderate', 'High', 'Unhealthy']
    
    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=x,
        y=y,
        colorscale='Greens',
        showscale=True,
        reversescale=True
    )
    
    fig_cm.update_layout(
        title="True vs Predicted Categories",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    st.plotly_chart(fig_cm, use_container_width=True)
    st.caption("This matrix shows how well the model performed on the last 500 records of historical data.")

except ImportError:
    st.warning("Scikit-learn not installed. Cannot display Confusion Matrix. Run: `pip install scikit-learn`")
except Exception as e:
    st.error(f"Could not generate Confusion Matrix: {e}")


# ── Analysis Section ───────────────────────────────────────
st.markdown('<div class="section-title">Climate vs Air Quality</div>', unsafe_allow_html=True)

# Time trend
st.markdown("Pollution Over Time")
daily = df.groupby('time')['pm25_proxy'].mean().reset_index()
st.line_chart(daily.set_index('time'))

# Temperature vs Pollution
fig1 = px.scatter(
    df.sample(2000),
    x='temperature_2m_mean',
    y='pm25_proxy',
    opacity=0.5,
    title="Temperature vs Pollution"
)
st.plotly_chart(fig1, use_container_width=True)

# Rain vs Pollution
fig2 = px.scatter(
    df.sample(2000),
    x='precipitation_sum',
    y='pm25_proxy',
    opacity=0.5,
    title="Rain vs Pollution"
)
st.plotly_chart(fig2, use_container_width=True)


# ── 4-Day PM2.5 Forecast (Styled Cards) ─────────────────────

if st.session_state.get("show_pm25_forecast", False):
    st.markdown('<div class="section-title">4-Day PM2.5 Forecast (AI Predictions)</div>', unsafe_allow_html=True)
    
    predictions = st.session_state.get("pm25_predictions", [])
    
    if predictions:
        # Create DataFrame for the Line Chart
        pred_df = pd.DataFrame(predictions)
        
        # 1. Line Chart
        fig_pm25_forecast = px.line(
            pred_df,
            x='date',
            y='pm25',
            markers=True,
            title="PM2.5 Forecast Trend",
            labels={'pm25': 'PM2.5 (µg/m³)', 'date': 'Date'},
            line_shape='linear'
        )
        
        # Color code the markers for the chart
        colors = []
        for pm25 in pred_df['pm25']:
            if pm25 <= 10: colors.append('#2ecc71')  # Green
            elif pm25 <= 25: colors.append('#f1c40f')  # Yellow
            elif pm25 <= 50: colors.append('#e67e22')  # Orange
            else: colors.append('#e74c3c')  # Red
        
        fig_pm25_forecast.update_traces(marker=dict(color=colors, size=12))
        fig_pm25_forecast.update_layout(height=400)
        st.plotly_chart(fig_pm25_forecast, use_container_width=True)
        
        # 2. Styled Cards (Mimicking Live Weather Card)
        st.markdown("### Detailed Daily Forecast")
        
        # Create columns for the cards
        cols = st.columns(len(predictions))
        
        for i, (col, pred) in enumerate(zip(cols, predictions)):
            pm25 = pred['pm25']
            day_name = pred['day']
            date = pred['date']
            temp = pred['temperature']
            wind = pred['wind']
            rain = pred['rain']
            
            # Determine AQI Status and Colors
            if pm25 <= 10:
                status = "Good"
                icon = "🟢"
                bg_color = "#e8f8f5"       
                border_color = "#2ecc71"   
                text_color = "#14532d"
            elif pm25 <= 25:
                status = "Moderate"
                icon = "🟡"
                bg_color = "#fef9e7"       
                border_color = "#f1c40f"   
                text_color = "#713f12"
            elif pm25 <= 50:
                status = "High"
                icon = "🟠"
                bg_color = "#fef5e7"       
                border_color = "#e67e22"   
                text_color = "#7c2d12"
            else:
                status = "Unhealthy"
                icon = "🔴"
                bg_color = "#fdecea"       
                border_color = "#e74c3c"   
                text_color = "#7f1d1d"
            
            # HTML String
            html_code = f"""
            <div style="
                padding:20px;
                border-radius:16px;
                background-color:{bg_color};
                border-left:8px solid {border_color};
                box-shadow:0 4px 6px rgba(0,0,0,0.05);
                font-family: sans-serif;
                height: 100%;
                box-sizing: border-box;
            ">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                    <div style="font-size:16px; font-weight:700; color:{text_color};">{day_name}</div>
                    <div style="padding:4px 10px; background-color:{border_color}30; color:{border_color}; border-radius:12px; font-size:11px; font-weight:700;">
                        {icon} {status}
                    </div>
                </div>
                
                <div style="font-size:12px; color:#666; margin-bottom:15px;">{date}</div>

                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
                    <div style="background-color:rgba(255,255,255,0.6); padding:10px; border-radius:8px; text-align:center;">
                        <div style="font-size:10px; color:#555;">PM2.5</div>
                        <div style="font-size:18px; font-weight:800; color:{border_color};">{pm25:.1f}</div>
                    </div>
                    <div style="background-color:rgba(255,255,255,0.6); padding:10px; border-radius:8px; text-align:center;">
                        <div style="font-size:10px; color:#555;">Temperature</div>
                        <div style="font-size:14px; font-weight:600;">{temp:.1f}°C</div>
                    </div>
                    <div style="background-color:rgba(255,255,255,0.6); padding:10px; border-radius:8px; text-align:center;">
                        <div style="font-size:10px; color:#555;">Wind</div>
                        <div style="font-size:14px; font-weight:600;">{wind:.1f} km/h</div>
                    </div>
                    <div style="background-color:rgba(255,255,255,0.6); padding:10px; border-radius:8px; text-align:center;">
                        <div style="font-size:10px; color:#555;">Rain</div>
                        <div style="font-size:14px; font-weight:600;">{rain:.1f} mm</div>
                    </div>
                </div>
            </div>
            """
            
            # RENDER THE HTML
            with col:
                components.html(html_code, height=300, scrolling=False)