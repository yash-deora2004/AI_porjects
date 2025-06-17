import streamlit as st
import json
import urllib.parse  # <-- Added for URL encoding
from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass
from datetime import datetime
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from firecrawl import FirecrawlApp

# -------------------------------------------------------------------
# 1. Define the Pydantic models for schema validation
# -------------------------------------------------------------------

class AQIResponse(BaseModel):
    success: bool
    data: Dict[str, float]
    status: str
    expiresAt: Union[str, datetime]

class ExtractSchema(BaseModel):
    aqi: float = Field(description="Air Quality Index")
    temperature: float = Field(description="Temperature in degrees Celsius")
    humidity: float = Field(description="Humidity percentage")
    wind_speed: float = Field(description="Wind speed in kilometers per hour")
    pm25: float = Field(description="Particulate Matter 2.5 micrometers")
    pm10: float = Field(description="Particulate Matter 10 micrometers")
    co: float = Field(description="Carbon Monoxide level")

# -------------------------------------------------------------------
# 2. Define a simple dataclass to hold user inputs
# -------------------------------------------------------------------

@dataclass
class UserInput:
    city: str
    state: str
    country: str
    medical_conditions: Optional[str]
    planned_activity: str

# -------------------------------------------------------------------
# 3. AQIAnalyzer: wraps Firecrawl to fetch AQI + related metrics
# -------------------------------------------------------------------

class AQIAnalyzer:
    def __init__(self, firecrawl_key: str) -> None:
        self.firecrawl = FirecrawlApp(api_key=firecrawl_key)

    def _format_url(self, country: str, state: str, city: str) -> str:
        country_clean = country.strip().lower().replace(" ", "-")
        city_clean = city.strip().lower().replace(" ", "-")
        if not state or state.strip().lower() in {"", "none"}:
            return f"https://www.aqi.in/dashboard/{country_clean}/{city_clean}"
        state_clean = state.strip().lower().replace(" ", "-")
        return f"https://www.aqi.in/dashboard/{country_clean}/{state_clean}/{city_clean}"

    def fetch_aqi_data(self, city: str, state: str, country: str) -> Tuple[Dict[str, float], str]:
        try:
            url = self._format_url(country, state, city)
            info_msg = f"Accessing URL: {url}"
            extract_res = self.firecrawl.extract(
                [f"{url}/*"],
                prompt=(
                    "Extract the current real-time AQI, temperature, humidity, wind speed, "
                    "PM2.5, PM10, and CO levels from the page. Also extract the timestamp of the data."
                ),
                schema=ExtractSchema.model_json_schema()
            )
            resp_dict = extract_res.model_dump() if hasattr(extract_res, "model_dump") else vars(extract_res)
            aqi_response = AQIResponse(**resp_dict)
            if not aqi_response.success:
                raise ValueError(f"Failed to fetch AQI data: {aqi_response.status}")
            return aqi_response.data, info_msg
        except Exception as e:
            error_msg = f"Error fetching AQI data: {str(e)}"
            default_data = {
                "aqi": 0.0, "temperature": 0.0, "humidity": 0.0, "wind_speed": 0.0,
                "pm25": 0.0, "pm10": 0.0, "co": 0.0,
            }
            return default_data, error_msg

# -------------------------------------------------------------------
# 4. HealthRecommendationAgent with Clickable Shopping Links
# -------------------------------------------------------------------

class HealthRecommendationAgent:
    """
    Generates health advice and a recommendation engine with clickable
    product links for cross-selling.
    - Feature 1: Issues a "wear mask" warning if AQI is bad.
    - Feature 2a: Provides clickable shopping links for relevant products.
    """

    def __init__(self, openai_key: str) -> None:
        self.llm = OpenAIChat(api_key=openai_key)

    def _create_shopping_link(self, search_term: str) -> str:
        """Creates a URL-encoded Amazon search link."""
        encoded_term = urllib.parse.quote_plus(search_term)
        # Using amazon.in as a default, can be changed to amazon.com etc.
        return f"https://www.amazon.in/s?k={encoded_term}"

    def get_recommendations(
        self, aqi_data: Dict[str, float], user_input: UserInput
    ) -> str:
        aqi_value = aqi_data.get("aqi", 0.0)
        advice: List[str] = []
        # Suggestions are now (display_name, search_term) tuples
        product_suggestions: List[Tuple[str, str]] = []

        # --- FEATURE IMPLEMENTATION: MASK WARNING & CROSS-SELLING LINKS ---
        if aqi_value > 150: # Bad AQI
            advice.append("**🚨 HIGH AQI ALERT: WEAR A MASK!** It is strongly recommended to wear a high-quality mask if you must go outdoors.")
            advice.append("Air quality is poor. Avoid all strenuous outdoor activity. Keep windows closed.")
            product_suggestions.extend([
                # Special emphasis on masks when AQI is bad
                ('Buy High-filtration N95 or FFP2 masks', 'N95 FFP2 mask'),
                ('Buy HEPA Air Purifiers for your home', 'HEPA air purifier for home'),
                ('Buy Indoor air-purifying plants', 'air purifying indoor plants'),
                ('Buy Car cabin air filters', 'car cabin air filter')
            ])
        elif aqi_value > 100: # Moderate AQI
            advice.append("Air quality is unhealthy for sensitive groups. Limit outdoor exertion.")
            advice.append("Consider wearing a mask as a precaution.")
            product_suggestions.extend([
                ('Buy Surgical or KN95 face masks', 'KN95 surgical mask'),
                ('Buy replacement filters for your air purifier', 'HEPA filter replacement'),
                ('Buy Saline nasal sprays', 'saline nasal spray')
            ])
        else: # Good AQI
            advice.append("Air quality is good. It's a great time for your planned outdoor activities!")
            product_suggestions.extend([
                ('Buy Broad-spectrum sunscreen', 'broad spectrum SPF 50 sunscreen'),
                ('Buy Hydration packs or water bottles', 'hydration pack for running'),
                ('Buy Sunglasses and a wide-brimmed hat', 'UV protection sunglasses'),
                ('Buy Comfortable athletic wear', 'moisture wicking athletic shirts')
            ])

        # --- Personalize advice ---
        if user_input.medical_conditions:
            advice.append(
                f"With your mentioned condition of **{user_input.medical_conditions}**, please be extra cautious."
            )
        advice.append(f"Regarding your planned activity ('{user_input.planned_activity}'), please follow the health advice above.")

        # --- Construct the final markdown output ---
        recommendation_str = "\n\n".join(advice)
        if product_suggestions:
            recommendation_str += "\n\n---\n\n"
            recommendation_str += "### 🛒 Product Recommendations (with links)\n"
            recommendation_str += "Based on the current conditions, consider these products:\n"
            for name, search_term in product_suggestions:
                link = self._create_shopping_link(search_term)
                # Format as a clickable Markdown link
                recommendation_str += f"- [{name}]({link})\n"

        return recommendation_str

# -------------------------------------------------------------------
# 5. Main analysis function
# -------------------------------------------------------------------

def analyze_conditions(
    city: str, state: str, country: str, medical_conditions: Optional[str],
    planned_activity: str, firecrawl_key: str, openai_key: str,
) -> Tuple[str, str, str, str]:
    try:
        aqi_analyzer = AQIAnalyzer(firecrawl_key=firecrawl_key)
        health_agent = HealthRecommendationAgent(openai_key=openai_key)
        user_input = UserInput(
            city=city.strip(), state=state.strip(), country=country.strip(),
            medical_conditions=medical_conditions.strip() if medical_conditions else None,
            planned_activity=planned_activity.strip(),
        )
        aqi_data, info_msg = aqi_analyzer.fetch_aqi_data(
            user_input.city, user_input.state, user_input.country
        )
        aqi_json = json.dumps({
            "Air Quality Index (AQI)": aqi_data.get("aqi", 0.0),
            "PM2.5": f"{aqi_data.get('pm25', 0.0)} µg/m³",
            "PM10": f"{aqi_data.get('pm10', 0.0)} µg/m³",
            "Carbon Monoxide (CO)": f"{aqi_data.get('co', 0.0)} ppb",
            "Temperature": f"{aqi_data.get('temperature', 0.0)}°C",
            "Humidity": f"{aqi_data.get('humidity', 0.0)}%",
            "Wind Speed": f"{aqi_data.get('wind_speed', 0.0)} km/h",
        }, indent=2)
        recommendations = health_agent.get_recommendations(aqi_data, user_input)
        warning_msg = """⚠️ **Note:** Data is scraped and may not match real-time values. Shopping links are auto-generated for Amazon and are for suggestion purposes. Always cross-reference for critical decisions."""
        return aqi_json, recommendations, info_msg, warning_msg
    except Exception as e:
        error_msg = f"An error occurred during analysis: {str(e)}"
        return "", "Analysis failed.", error_msg, ""

# -------------------------------------------------------------------
# 6. Streamlit interface
# -------------------------------------------------------------------

st.set_page_config(page_title="AQI Analysis & Recommendation Agent", layout="centered")
st.title("🌍 AQI Analysis & Recommendation Agent")

st.markdown(
    """
    Get personalized health advice and **clickable product links** based on local air quality.
    This tool provides critical warnings and acts as a dynamic shopping assistant.
    """
)

st.sidebar.header("API Configuration")
firecrawl_key = st.sidebar.text_input("Firecrawl API Key", type="password")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")

with st.form(key="aqi_form"):
    st.subheader("📍 Location Details")
    city = st.text_input("City", placeholder="e.g., Delhi")
    state = st.text_input("State", placeholder="e.g., Delhi (or leave blank)")
    country = st.text_input("Country", value="India")

    st.subheader("Personal Details")
    medical_conditions = st.text_area("Medical Conditions (optional)", placeholder="e.g., asthma, allergies")
    planned_activity = st.text_area("Planned Activity", placeholder="e.g., a 5km run in the park")

    analyze_button = st.form_submit_button(label="🔍 Analyze & Get Recommendations")

if analyze_button:
    if not all([city, country, planned_activity, firecrawl_key, openai_key]):
        st.error("Please fill in all fields: City, Country, Planned Activity, and API keys in the sidebar.")
    else:
        with st.spinner("Analyzing air quality and generating recommendations..."):
            aqi_json, recommendations, info_msg, warning_msg = analyze_conditions(
                city=city, state=state, country=country,
                medical_conditions=medical_conditions,
                planned_activity=planned_activity,
                firecrawl_key=firecrawl_key,
                openai_key=openai_key,
            )

        if info_msg:
            st.info(info_msg)

        if "failed" in recommendations.lower() or "error" in info_msg.lower():
             st.error(info_msg)
        else:
            st.subheader("📊 Current Air Quality Data")
            st.json(json.loads(aqi_json))

            st.subheader("🏥 Health Advice & Product Recommendations")
            st.markdown(recommendations, unsafe_allow_html=True) # Allow HTML for links

            st.warning(warning_msg)