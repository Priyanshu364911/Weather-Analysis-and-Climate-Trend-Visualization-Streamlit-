import requests, pandas as pd

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 23.02, "longitude": 72.57,
    "start_date": "2015-01-01", "end_date": "2025-12-31",
    "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,"
             "precipitation_sum,wind_speed_10m_max,relative_humidity_2m_mean",
    "timezone": "Asia/Kolkata",
}
data = requests.get(url, params=params).json()
df = pd.DataFrame(data["daily"])
df.to_csv("weather_ahmedabad.csv", index=False)