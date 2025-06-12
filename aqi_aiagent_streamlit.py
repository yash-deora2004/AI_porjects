import streamlit as st
import json
from typing import Dict, Optional, Tuple, Union
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
    # Accept either a string or a datetime for expiresAt, since Firecrawl may return a datetime
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
        """
        Format the target URL based on location, handling cases with and without state.
        """
        country_clean = country.strip().lower().replace(" ", "-")
        city_clean = city.strip().lower().replace(" ", "-")

        if not state or state.strip().lower() in {"", "none"}:
            return f"https://www.aqi.in/dashboard/{country_clean}/{city_clean}"

        state_clean = state.strip().lower().replace(" ", "-")
        return f"https://www.aqi.in/dashboard/{country_clean}/{state_clean}/{city_clean}"

    def fetch_aqi_data(self, city: str, state: str, country: str) -> Tuple[Dict[str, float], str]:
        """
        Fetch AQI data from the website using Firecrawl. Returns a tuple:
          (data_dict, info_message)
        On failure, returns default zeroed metrics + an error message.
        """
        try:
            url = self._format_url(country, state, city)
            info_msg = f"Accessing URL: {url}"

            # Firecrawl: extract real-time AQI metrics according to our schema.
            extract_res = self.firecrawl.extract(
                [f"{url}/*"],
                prompt=(
                    "Extract the current real-time AQI, temperature, humidity, wind speed, "
                    "PM2.5, PM10, and CO levels from the page. Also extract the timestamp of the data."
                ),
                schema=ExtractSchema.model_json_schema()
            )

            # Convert ExtractResponse to a dict before validation
            if hasattr(extract_res, "dict"):
                resp_dict = extract_res.dict()
            elif hasattr(extract_res, "model_dump"):
                resp_dict = extract_res.model_dump()
            else:
                resp_dict = vars(extract_res)

            # Validate with AQIResponse (now accepts datetime for expiresAt)
            aqi_response = AQIResponse(**resp_dict)
            if not aqi_response.success:
                raise ValueError(f"Failed to fetch AQI data: {aqi_response.status}")

            return aqi_response.data, info_msg

        except Exception as e:
            # On error, return zeros and log the exception
            error_msg = f"Error fetching AQI data: {str(e)}"
            default_data = {
                "aqi": 0.0,
                "temperature": 0.0,
                "humidity": 0.0,
                "wind_speed": 0.0,
                "pm25": 0.0,
                "pm10": 0.0,
                "co": 0.0,
            }
            return default_data, error_msg

# -------------------------------------------------------------------
# 4. HealthRecommendationAgent placeholder
#    (Replace or implement this class according to your actual logic)
# -------------------------------------------------------------------

class HealthRecommendationAgent:
    """
    A placeholder agent that, in reality, should use OpenAIChat (or similar)
    to generate personalized health recommendations based on AQI + user profile.

    In production, replace the get_recommendations method with your own logic,
    using OpenAIChat or Agno's agent system.
    """

    def __init__(self, openai_key: str) -> None:
        # Example: initialize an OpenAIChat client inside Agno's Agent framework
        self.llm = OpenAIChat(api_key=openai_key)
        # If you're using Agno's Agent wrapper:
        # self.agent = Agent(model=self.llm)

    def get_recommendations(
        self, aqi_data: Dict[str, float], user_input: UserInput
    ) -> str:
        """
        Build a prompt from aqi_data + user_input, send to LLM, and return the answer.
        For now, we'll return a very basic, hardcoded recommendation string.
        """
        aqi_value = aqi_data.get("aqi", 0.0)
        pm25 = aqi_data.get("pm25", 0.0)

        # Example logic:
        advice = []
        if aqi_value > 150:
            advice.append("Air quality is poor. It's best to stay indoors and avoid strenuous outdoor activity.")
        elif aqi_value > 100:
            advice.append("Air quality is moderate. Consider reducing prolonged or heavy exertion outdoors.")
        else:
            advice.append("Air quality is good. You can proceed with your planned activity, but stay hydrated.")

        if user_input.medical_conditions:
            advice.append(
                f"Since you mentioned having {user_input.medical_conditions}, "
                "monitor your symptoms closely and have your medication handy."
            )

        advice.append(f"Your planned activity: '{user_input.planned_activity}'.")

        # In a real implementation, you'd send something like:
        # prompt = (
        #     f"User is in {user_input.city}, {user_input.state}, {user_input.country}. "
        #     f"AQI data: {json.dumps(aqi_data)}. "
        #     f"Medical conditions: {user_input.medical_conditions}. "
        #     f"Planned activity: {user_input.planned_activity}. "
        #     "Give personalized health recommendations."
        # )
        # response = self.agent.run(prompt)
        # return response

        return "\n\n".join(advice)

# -------------------------------------------------------------------
# 5. Main analysis function
# -------------------------------------------------------------------

def analyze_conditions(
    city: str,
    state: str,
    country: str,
    medical_conditions: Optional[str],
    planned_activity: str,
    firecrawl_key: str,
    openai_key: str,
) -> Tuple[str, str, str, str]:
    """
    Given user inputs and API keys, fetch AQI data, generate recommendations,
    and return four strings:
      1) JSON-formatted AQI data for display
      2) Markdown recommendations
      3) Info/status message
      4) Warning message (if needed)
    """
    try:
        # Initialize the AQI analyzer and health recommendation agent
        aqi_analyzer = AQIAnalyzer(firecrawl_key=firecrawl_key)
        health_agent = HealthRecommendationAgent(openai_key=openai_key)

        # Bundle user input into the dataclass
        user_input = UserInput(
            city=city.strip(),
            state=state.strip(),
            country=country.strip(),
            medical_conditions=medical_conditions.strip() if medical_conditions else None,
            planned_activity=planned_activity.strip(),
        )

        # 1. Fetch AQI + related data
        aqi_data, info_msg = aqi_analyzer.fetch_aqi_data(
            city=user_input.city,
            state=user_input.state,
            country=user_input.country,
        )

        # 2. Format the raw numbers into a JSON string for display
        aqi_json = json.dumps(
            {
                "Air Quality Index (AQI)": aqi_data.get("aqi", 0.0),
                "PM2.5": f"{aqi_data.get('pm25', 0.0)} µg/m³",
                "PM10": f"{aqi_data.get('pm10', 0.0)} µg/m³",
                "Carbon Monoxide (CO)": f"{aqi_data.get('co', 0.0)} ppb",
                "Temperature": f"{aqi_data.get('temperature', 0.0)}°C",
                "Humidity": f"{aqi_data.get('humidity', 0.0)}%",
                "Wind Speed": f"{aqi_data.get('wind_speed', 0.0)} km/h",
            },
            indent=2,
        )

        # 3. Get health recommendations from the agent
        recommendations = health_agent.get_recommendations(aqi_data, user_input)

        # 4. A static warning in case Firecrawl data isn't perfectly real-time
        warning_msg = """⚠️ **Note:** The data shown may not match real-time values on the website.
This could be due to:
- Cached data in Firecrawl
- Rate limiting on the source site
- Website updates not being captured immediately

Consider refreshing or checking the original site for the latest values."""

        return aqi_json, recommendations, info_msg, warning_msg

    except Exception as e:
        error_msg = f"Error occurred during analysis: {str(e)}"
        # If anything goes wrong, return placeholders
        return "", "Analysis failed", error_msg, ""

# -------------------------------------------------------------------
# 6. Streamlit interface
# -------------------------------------------------------------------

st.set_page_config(page_title="AQI Analysis Agent", layout="centered")
st.title("🌍 AQI Analysis Agent")

st.markdown(
    """
    Get personalized health recommendations based on air quality conditions.
    Enter your location details, any medical conditions, and your planned activity,
    then click **Analyze**.  
    """
)

# ---- Sidebar: API Keys ----

st.sidebar.header("API Configuration")
firecrawl_key = st.sidebar.text_input(
    "Firecrawl API Key", type="password", placeholder="Enter your Firecrawl API key"
)
openai_key = st.sidebar.text_input(
    "OpenAI API Key", type="password", placeholder="Enter your OpenAI API key"
)

# ---- Main form: Location + Personal Details ----

with st.form(key="aqi_form"):
    st.subheader("Location Details")
    city = st.text_input("City", placeholder="e.g., Mumbai")
    state = st.text_input(
        "State", placeholder="Leave blank for Union Territories or US cities"
    )
    country = st.text_input("Country", value="India")

    st.subheader("Personal Details")
    medical_conditions = st.text_area(
        "Medical Conditions (optional)", placeholder="e.g., asthma, allergies"
    )
    planned_activity = st.text_area(
        "Planned Activity", placeholder="e.g., morning jog for 2 hours"
    )

    # Submit button
    analyze_button = st.form_submit_button(label="🔍 Analyze AQI & Health Recommendations")

# ---- When the button is pressed ----

if analyze_button:
    # Validate mandatory fields
    if not city.strip() or not country.strip() or not planned_activity.strip():
        st.error("Please provide at least City, Country, and Planned Activity.")
    elif not firecrawl_key.strip() or not openai_key.strip():
        st.error("Please provide both your Firecrawl API Key and OpenAI API Key in the sidebar.")
    else:
        # Call our analysis function
        aqi_json, recommendations, info_msg, warning_msg = analyze_conditions(
            city=city,
            state=state,
            country=country,
            medical_conditions=medical_conditions,
            planned_activity=planned_activity,
            firecrawl_key=firecrawl_key,
            openai_key=openai_key,
        )

        # Display status/info
        if info_msg:
            st.info(info_msg)

        # Display the JSON data
        if aqi_json:
            st.subheader("📊 Current Air Quality Data (JSON)")
            st.json(json.loads(aqi_json))

        # Display the recommendations
        if recommendations:
            st.subheader("🏥 Health Recommendations")
            st.markdown(recommendations)

        # Display the warning (always shown)
        st.warning(warning_msg)