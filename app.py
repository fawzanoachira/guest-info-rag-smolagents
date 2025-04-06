from smolagents import CodeAgent, HfApiModel, GradioUI
from tools import HubStatsTool, WeatherInfoTool, DuckDuckGoSearchTool
from retriever import GuestInfoRetrieverTool

model = HfApiModel()

search_tool = DuckDuckGoSearchTool()

weather_info_tool = WeatherInfoTool()

hub_stats_tool = HubStatsTool()

guest_info_tool = GuestInfoRetrieverTool()

alfred = CodeAgent(
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool], 
    model=model,
    add_base_tools=True,  # Add any additional base tools
    planning_interval=3   # Enable planning every 3 steps
)

if __name__ == "__main__":
    GradioUI(alfred).launch()