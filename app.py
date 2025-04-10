import os
from huggingface_hub import InferenceClient
from smolagents import CodeAgent, HfApiModel, GradioUI
from tools import HubStatsTool, WeatherInfoTool, DuckDuckGoSearchTool
from retriever import load_guest_dataset

hf_api = "REMOVED"

client = InferenceClient(
	provider="fireworks-ai",
    model='Qwen/Qwen2.5-Coder-32B-Instruct',
	api_key=hf_api
)

model = HfApiModel(
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    provider='fireworks-ai',
    token=hf_api,
)

search_tool = DuckDuckGoSearchTool()

weather_info_tool = WeatherInfoTool()

hub_stats_tool = HubStatsTool()

guest_info_tool = load_guest_dataset()

alfred = CodeAgent(
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool],
    model=model,
    add_base_tools=False,  # Add any additional base tools
    planning_interval=3,   # Enable planning every 3 steps
)

if __name__ == "__main__":
    GradioUI(alfred).launch()