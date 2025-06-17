import streamlit as st
import json
import urllib.parse
import requests
import folium
from streamlit_folium import folium_static
from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from firecrawl import FirecrawlApp
import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# MODELS AND DATA CLASSES (Section 1 & 2 - Unchanged)
# -------------------------------------------------------------------
class AQIResponse(BaseModel):
    success: bool; data: Dict[str, float]; status: str; expiresAt: Union[str, datetime]

class ExtractSchema(BaseModel):
    aqi: float = Field(description="Air Quality Index"); temperature: float = Field(description="Temperature in degrees Celsius"); humidity: float = Field(description="Humidity percentage"); wind_speed: float = Field(description="Wind speed in kilometers per hour"); pm25: float = Field(description="Particulate Matter 2.5 micrometers"); pm10: float = Field(description="Particulate Matter 10 micrometers"); co: float = Field(description="Carbon Monoxide level")

@dataclass
class UserInput:
    city: str; state: str; country: str; medical_conditions: Optional[str]; planned_activity: str

# -------------------------------------------------------------------
# AQI MAP GENERATOR (Unchanged)
# -------------------------------------------------------------------
class AQIMapGenerator:
    """Fetches AQI data for a region and generates an interactive map."""
    def __init__(self, waqi_token: str):
        if not waqi_token or waqi_token == "your_waqi_token_here":
            if waqi_token != "dummy": # Allow dummy token for internal use
                raise ValueError("WAQI API Token is missing or invalid.")
        self.api_token = waqi_token
        # Bounding boxes for different countries [lat_min, lon_min, lat_max, lon_max]
        self.country_bounds = {
            "India": "6.55,68.11,35.67,97.39",
            "USA": "24.39,-125.0,49.38,-66.94",
            "China": "18.10,73.5,53.56,134.77",
            "Europe": "35.0,-10.0,71.0,40.0"
        }

    def get_color(self, aqi: int) -> str:
        """Returns a color based on the AQI value."""
        if aqi <= 50: return 'green'      # Good
        if aqi <= 100: return 'orange'     # Moderate
        if aqi <= 150: return '#ff8c00' # darkorange
        if aqi <= 200: return 'red'        # Unhealthy
        if aqi <= 300: return '#8b0000'    # darkred
        return 'purple'                   # Hazardous

    def create_map(self, country: str = "India"):
        """Fetches data and creates a Folium map."""
        bounds = self.country_bounds.get(country)
        if not bounds:
            raise ValueError(f"Country '{country}' not supported. Choose from {list(self.country_bounds.keys())}.")
        
        url = f"https://api.waqi.info/map/bounds/?latlng={bounds}&token={self.api_token}"
        try:
            response = requests.get(url)
            response.raise_for_status() # Raise an exception for bad status codes
            api_data = response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch map data from WAQI API: {e}")
            return None
        
        if api_data.get("status") != "ok":
            st.error(f"WAQI API Error: {api_data.get('data', 'Unknown error')}")
            return None

        # Calculate map center
        lat_min, lon_min, lat_max, lon_max = map(float, bounds.split(','))
        map_center = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]

        # Create Folium map
        m = folium.Map(location=map_center, zoom_start=5, tiles="CartoDB positron")

        for station in api_data["data"]:
            try:
                aqi = int(station["aqi"])
                lat, lon = station["lat"], station["lon"]
                station_name = station["station"]["name"]
                
                popup_html = f"<b>{station_name}</b><br>AQI: {aqi}"
                iframe = folium.IFrame(popup_html, width=200, height=80)
                popup = folium.Popup(iframe, max_width=200)

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=8,
                    popup=popup,
                    color=self.get_color(aqi),
                    fill=True,
                    fill_color=self.get_color(aqi),
                    fill_opacity=0.7
                ).add_to(m)
            except (ValueError, KeyError):
                continue
        return m

# -------------------------------------------------------------------
# SINGLE-CITY ANALYSIS CLASSES (Unchanged)
# -------------------------------------------------------------------
class AQIAnalyzer:
    def __init__(self, firecrawl_key: str): self.firecrawl = FirecrawlApp(api_key=firecrawl_key)
    def _format_url(self, country: str, state: str, city: str) -> str:
        country_clean, city_clean = country.strip().lower().replace(" ", "-"), city.strip().lower().replace(" ", "-")
        if not state or state.strip().lower() in {"", "none"}: return f"https://www.aqi.in/dashboard/{country_clean}/{city_clean}"
        state_clean = state.strip().lower().replace(" ", "-")
        return f"https://www.aqi.in/dashboard/{country_clean}/{state_clean}/{city_clean}"
    def fetch_aqi_data(self, city: str, state: str, country: str) -> Tuple[Dict[str, float], str]:
        try:
            url = self._format_url(country, state, city)
            info_msg, extract_res = f"Accessing URL: {url}", self.firecrawl.extract([f"{url}/*"], prompt="Extract the current real-time AQI, temperature, humidity, wind speed, PM2.5, PM10, and CO levels from the page.", schema=ExtractSchema.model_json_schema())
            resp_dict = extract_res.model_dump() if hasattr(extract_res, "model_dump") else vars(extract_res)
            aqi_response = AQIResponse(**resp_dict)
            if not aqi_response.success: raise ValueError(f"Failed to fetch AQI data: {aqi_response.status}")
            return aqi_response.data, info_msg
        except Exception as e:
            error_msg, default_data = f"Error fetching AQI data: {str(e)}", {"aqi": 0.0, "temperature": 0.0, "humidity": 0.0, "wind_speed": 0.0, "pm25": 0.0, "pm10": 0.0, "co": 0.0}
            return default_data, error_msg

class HealthRecommendationAgent:
    def __init__(self, openai_key: str): self.llm = OpenAIChat(api_key=openai_key)
    def _create_shopping_link(self, search_term: str) -> str:
        return f"https://www.amazon.in/s?k={urllib.parse.quote_plus(search_term)}"
    def get_recommendations(self, aqi_data: Dict[str, float], user_input: UserInput) -> str:
        aqi_value, advice, product_suggestions = aqi_data.get("aqi", 0.0), [], []
        if aqi_value > 150:
            advice.append("**🚨 HIGH AQI ALERT: WEAR A MASK!** It is strongly recommended to wear a high-quality mask if you must go outdoors.")
            advice.append("Air quality is poor. Avoid all strenuous outdoor activity. Keep windows closed.")
            product_suggestions.extend([('Buy High-filtration N95 or FFP2 masks', 'N95 FFP2 mask'), ('Buy HEPA Air Purifiers for your home', 'HEPA air purifier for home'), ('Buy Indoor air-purifying plants', 'air purifying indoor plants'), ('Buy Car cabin air filters', 'car cabin air filter')])
        elif aqi_value > 100:
            advice.append("Air quality is unhealthy for sensitive groups. Limit outdoor exertion."); advice.append("Consider wearing a mask as a precaution.")
            product_suggestions.extend([('Buy Surgical or KN95 face masks', 'KN95 surgical mask'), ('Buy replacement filters for your air purifier', 'HEPA filter replacement'), ('Buy Saline nasal sprays', 'saline nasal spray')])
        else:
            advice.append("Air quality is good. It's a great time for your planned outdoor activities!")
            product_suggestions.extend([('Buy Broad-spectrum sunscreen', 'broad spectrum SPF 50 sunscreen'), ('Buy Hydration packs or water bottles', 'hydration pack for running'), ('Buy Sunglasses and a wide-brimmed hat', 'UV protection sunglasses')])
        if user_input.medical_conditions: advice.append(f"With your mentioned condition of **{user_input.medical_conditions}**, please be extra cautious.")
        advice.append(f"Regarding your planned activity ('{user_input.planned_activity}'), please follow the health advice above.")
        recommendation_str = "\n\n".join(advice)
        if product_suggestions:
            recommendation_str += "\n\n---\n\n### 🛒 Product Recommendations (with links)\n"
            for name, search_term in product_suggestions: recommendation_str += f"- [{name}]({self._create_shopping_link(search_term)})\n"
        return recommendation_str

# -------------------------------------------------------------------
# MAIN ANALYSIS & HELPER FUNCTIONS
# -------------------------------------------------------------------
def analyze_single_city(city, state, country, medical_conditions, planned_activity, firecrawl_key, openai_key):
    try:
        aqi_analyzer, health_agent = AQIAnalyzer(firecrawl_key=firecrawl_key), HealthRecommendationAgent(openai_key=openai_key)
        user_input = UserInput(city.strip(), state.strip(), country.strip(), medical_conditions.strip() if medical_conditions else None, planned_activity.strip())
        aqi_data, info_msg = aqi_analyzer.fetch_aqi_data(user_input.city, user_input.state, user_input.country)
        aqi_json = json.dumps({"Air Quality Index (AQI)": aqi_data.get("aqi", 0.0), "PM2.5": f"{aqi_data.get('pm25', 0.0)} µg/m³", "PM10": f"{aqi_data.get('pm10', 0.0)} µg/m³", "Carbon Monoxide (CO)": f"{aqi_data.get('co', 0.0)} ppb", "Temperature": f"{aqi_data.get('temperature', 0.0)}°C", "Humidity": f"{aqi_data.get('humidity', 0.0)}%", "Wind Speed": f"{aqi_data.get('wind_speed', 0.0)} km/h"}, indent=2)
        recommendations = health_agent.get_recommendations(aqi_data, user_input)
        warning_msg = """⚠️ **Note:** Single-city data is scraped and may not match real-time values. Shopping links are auto-generated. Always cross-reference for critical decisions."""
        return aqi_json, recommendations, info_msg, warning_msg
    except Exception as e:
        return "", "Analysis failed.", f"An error occurred: {e}", ""

# --- NEW: Functions for Weekly Prediction ---
def get_aqi_category(aqi: float) -> Tuple[str, str]:
    """Returns a category label and emoji based on the AQI value."""
    if aqi <= 50: return "Good", "🟢"
    if aqi <= 100: return "Moderate", "🟡"
    if aqi <= 150: return "Unhealthy (S)", "🟠"
    if aqi <= 200: return "Unhealthy", "🔴"
    if aqi <= 300: return "V. Unhealthy", "🟣"
    return "Hazardous", "💀"

def simulate_weekly_aqi(base_aqi: float) -> list[float]:
    """Simulates a 7-day AQI forecast by adding random fluctuations."""
    if base_aqi <= 0: return [0.0] * 7
    fluctuation = base_aqi * 0.15
    weekly_forecast = []
    current_val = base_aqi
    for _ in range(7):
        change = np.random.uniform(-fluctuation, fluctuation)
        current_val = max(1, current_val + change)
        weekly_forecast.append(round(current_val, 1))
    return weekly_forecast

@st.cache_data(ttl=3600) # Cache for 1 hour
def get_prediction_data(city: str, state: str, country: str, firecrawl_key: str):
    """Fetches current AQI and generates a simulated 7-day forecast."""
    if not firecrawl_key:
        return None, "Please enter your Firecrawl API Key in the sidebar.", None
    try:
        aqi_analyzer = AQIAnalyzer(firecrawl_key=firecrawl_key)
        aqi_data, info_msg = aqi_analyzer.fetch_aqi_data(city, state, country)
        if "error" in info_msg.lower() or aqi_data.get('aqi') == 0.0:
            return None, f"Could not fetch base AQI data for {city}. Error: {info_msg}", None
        base_aqi = aqi_data.get("aqi", 0.0)
        forecast_values = simulate_weekly_aqi(base_aqi)
        today = date.today()
        dates = [(today + timedelta(days=i)).strftime("%A, %b %d") for i in range(7)]
        df = pd.DataFrame({'Date': dates, 'Predicted AQI': forecast_values}).set_index('Date')
        return df, f"Successfully fetched current AQI ({base_aqi}) for {city} and generated a forecast.", base_aqi
    except Exception as e:
        return None, f"An unexpected error occurred: {str(e)}", None

# -------------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------------
st.set_page_config(page_title="Global AQI Analysis Agent", layout="wide")
st.title("🌍 Global AQI Analysis & Recommendation Agent")

# --- SIDEBAR: API KEY CONFIGURATION & NAVIGATION ---
st.sidebar.header("🔑 API Configuration")
waqi_token = st.sidebar.text_input("WAQI API Token (for Map)", type="password", help="Get a free token from https://aqicn.org/data-platform/token/")
firecrawl_key = st.sidebar.text_input("Firecrawl API Key (for City Analysis & Prediction)", type="password")
openai_key = st.sidebar.text_input("OpenAI API Key (for City Analysis)", type="password")

st.sidebar.markdown("---")
page = st.sidebar.radio("Select a Feature", ["🏠 Home", "📅 Weekly AQI Prediction"])

# --- PAGE 1: HOME (MAP & SINGLE CITY ANALYSIS) ---
if page == "🏠 Home":
    st.header("🗺️ Live AQI Map")
    st.markdown("Visually identify air quality hotspots and safe zones. Select a region and click 'Load Map'.")
    selected_country = st.selectbox("Select a Region to Map", ["India", "USA", "China", "Europe"])
    if st.button("Load Live AQI Map"):
        if not waqi_token:
            st.error("Please enter your WAQI API Token in the sidebar to load the map.")
        else:
            with st.spinner(f"Generating live AQI map for {selected_country}..."):
                try:
                    map_generator = AQIMapGenerator(waqi_token)
                    aqi_map = map_generator.create_map(selected_country)
                    if aqi_map:
                        folium_static(aqi_map, width=1100, height=500)
                except ValueError as e:
                    st.error(str(e))
    st.markdown("---")
    st.header("🏙️ Detailed City Analysis & Recommendations")
    st.markdown("Get personalized health advice and product links for a specific city.")
    with st.form(key="aqi_form"):
        col1, col2, col3 = st.columns(3)
        with col1: city = st.text_input("City", placeholder="e.g., Delhi")
        with col2: state = st.text_input("State", placeholder="e.g., Delhi")
        with col3: country_input = st.text_input("Country", value="India")
        medical_conditions = st.text_area("Medical Conditions (optional)", placeholder="e.g., asthma, allergies")
        planned_activity = st.text_area("Planned Activity", placeholder="e.g., a 5km run in the park")
        analyze_button = st.form_submit_button(label="🔍 Analyze City & Get Recommendations")
    if analyze_button:
        if not all([city, country_input, planned_activity, firecrawl_key, openai_key]):
            st.error("For city analysis, please fill in: City, Country, Planned Activity, and both Firecrawl & OpenAI API keys in the sidebar.")
        else:
            with st.spinner(f"Analyzing {city}..."):
                aqi_json, recommendations, info_msg, warning_msg = analyze_single_city(
                    city, state, country_input, medical_conditions, planned_activity, firecrawl_key, openai_key
                )
            if "error" in info_msg.lower() or "failed" in recommendations.lower():
                st.error(f"Analysis failed. Status: {info_msg}")
            else:
                st.info(info_msg)
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📊 Current Air Quality Data")
                    st.json(json.loads(aqi_json))
                with col2:
                    st.subheader("🏥 Health Advice & Product Links")
                    st.markdown(recommendations, unsafe_allow_html=True)
                st.warning(warning_msg)

# --- PAGE 2: WEEKLY AQI PREDICTION (MODIFIED) ---
elif page == "📅 Weekly AQI Prediction":
    st.header("📅 Weekly AQI Prediction (Simulation)")
    st.markdown("Get a simulated 7-day AQI forecast for a specific city. This is based on the most recent available data and is for illustrative purposes only.")
    st.warning("**Disclaimer:** This is NOT a real weather forecast. It's a simulation that adds random variation to the latest fetched AQI value.")

    with st.form(key="predict_form"):
        st.subheader("Enter Location for Prediction")
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1: city_predict = st.text_input("City", placeholder="e.g., Mumbai")
        with p_col2: state_predict = st.text_input("State", placeholder="e.g., Maharashtra")
        with p_col3: country_predict = st.text_input("Country", value="India")
        predict_button = st.form_submit_button("🔮 Predict 7-Day AQI Trend")

    if predict_button:
        if not all([city_predict, country_predict, firecrawl_key]):
            st.error("Please fill in City, Country, and your Firecrawl API key in the sidebar.")
        else:
            with st.spinner(f"Fetching current data and simulating forecast for {city_predict}..."):
                prediction_df, message, base_aqi = get_prediction_data(
                    city_predict.strip(), state_predict.strip(), country_predict.strip(), firecrawl_key
                )
            st.info(message)
            if prediction_df is not None and base_aqi is not None:
                st.subheader(f"Simulated 7-Day AQI Score Card for {city_predict.title()}")
                
                # Display base AQI for context
                base_category, base_emoji = get_aqi_category(base_aqi)
                st.info(f"Prediction based on most recent fetched AQI: **{int(base_aqi)}** ({base_emoji} {base_category})")
                st.markdown("---")

                # Create 7 columns for the weekly score card
                cols = st.columns(len(prediction_df))
                map_gen = AQIMapGenerator(waqi_token="dummy") # For color utility

                for i, (date, row) in enumerate(prediction_df.iterrows()):
                    with cols[i]:
                        aqi_val = row['Predicted AQI']
                        category, emoji = get_aqi_category(aqi_val)
                        day_name = date.split(',')[0]
                        
                        # Use a container with border to create a "card"
                        with st.container(border=True):
                            st.markdown(f"<p style='text-align: center; font-weight: bold;'>{day_name}</p>", unsafe_allow_html=True)
                            
                            # Get color and display the AQI value prominently
                            color = map_gen.get_color(int(aqi_val))
                            st.markdown(f"<h3 style='text-align: center; color: {color};'>{int(aqi_val)}</h3>", unsafe_allow_html=True)
                            
                            # Display the category label with an emoji
                            st.markdown(f"<p style='text-align: center;'>{emoji} {category}</p>", unsafe_allow_html=True)