import requests

API_key = 'API_KEY'

lat_response = requests.get(
    f'http://api.openweathermap.org/geo/1.0/direct?q=alappuzha&appid={API_key}'
)

lat = lat_response.json()[0]['lat']
lon = lat_response.json()[0]['lon']

response = requests.get(
    f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_key}'
)

data = response.json()

print(f"Weather in Alappuzha: {data['weather'][0]['main']}, {data['weather'][0]['description']}. Temperature: {data['main']['temp']}. Humidity: {data['main']['humidity']}. Wind Speed: {data['wind']['speed']}")
