from smolagents import Tool, DuckDuckGoSearchTool
from huggingface_hub import list_models

import random, requests
import os

OPENWEATHER_API = "REMOVED"
# OPENWEATHER_API = os.getenv('OPENWEATHER_API')


class WeatherInfoTool(Tool):
    name = "weather_info"
    description = "Fetches weather information for a given location."
    inputs = {
        "location": {
            "type": "string",
            "description": "The location to get weather information for.",
        }
    }
    output_type = "string"

    def forward(self, location: str):
        # Dummy weather data
        # weather_conditions = [
        #     {"condition": "Rainy", "temp_c": 15},
        #     {"condition": "Clear", "temp_c": 25},
        #     {"condition": "Windy", "temp_c": 20},
        # ]
        # # Randomly select a weather condition
        # data = random.choice(weather_conditions)
        # return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

        geo_response = requests.get(f"http://api.openweathermap.org/geo/1.0/direct?q={location}&appid={OPENWEATHER_API}")
        lat = geo_response.json()[0]["lat"]
        lon = geo_response.json()[0]["lon"]

        weather_response = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API}")
        data = weather_response.json()

        weather = data['weather'][0]
        main = data['main']
        wind = data['wind']

        # Convert Kelvin to Celsius
        temp_celsius = main['temp'] - 273.15

        weather_info = (
            f"📍 Weather Report for {location}:\n"
            f"🌤 Condition   : {weather['main']} - {weather['description'].capitalize()}\n"
            f"🌡 Temperature : {temp_celsius:.1f}°C\n"
            f"💧 Humidity    : {main['humidity']}%\n"
            f"🌬 Wind Speed  : {wind['speed']} m/s"
        )

        return weather_info

class HubStatsTool(Tool):
    name = "hub_stats"
    description = "Fetches the most downloaded model from a specific author on the Hugging Face Hub."
    inputs = {
        "author": {
            "type": "string",
            "description": "The username of the model author/organization to find models from.",
        }
    }
    output_type = "string"

    def forward(self, author: str):
        try:
            # List models from the specified author, sorted by downloads
            models = list(
                list_models(author=author, sort="downloads", direction=-1, limit=1)
            )

            if models:
                model = models[0]
                return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
            else:
                return f"No models found for author {author}."
        except Exception as e:
            return f"Error fetching models for {author}: {str(e)}"
