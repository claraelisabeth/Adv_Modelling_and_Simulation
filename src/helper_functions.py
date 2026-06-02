import numpy as np
import pandas as pd
import requests

from datetime import datetime, timezone


def calculate_rect_dimensions(lat_range: tuple[float, float], lon_range: tuple[float, float]) -> tuple[float, float]:
    """ Calculates the width and height (in km) of a geographic bounding box using the Flat Earth approximation. """
    km_per_degree = 111.32

    # Unpack the coordinate tuples
    lat1, lat2 = lat_range
    lon1, lon2 = lon_range

    # Calculate absolute differences
    delta_lat = np.abs(lat2 - lat1)
    delta_lon = np.abs(lon2 - lon1)

    # Calculate average latitude in radians
    avg_lat = (lat1 + lat2) / 2
    avg_lat_rad = np.radians(avg_lat)

    # Calculate physical dimensions
    height_km = delta_lat * km_per_degree
    width_km = delta_lon * km_per_degree * np.cos(avg_lat_rad)

    return width_km, height_km


def collect_weather_data(lat: float, lon: float, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Collects wind data from the https://open-meteo.com API for the given latitude, longitude and time range.

    Parameters
    ----------
    lat: float
        Latitude of the point of interest.
    lon: float
        Longitude of the point of interest.
    start_date: datetime
        Start date and time for the weather data collection. This needs to be a timezone-aware datetime object.
    end_date: datetime
        End date and time for the weather data collection. This needs to be a timezone-aware datetime.

    Returns
    -------
    pd.DataFrame
        Returns a pandas dataframe with hourly wind data for the given time range.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"

    assert isinstance(start_date, datetime) and isinstance(end_date, datetime), \
        f"Start and/or end date not a datetime object"
    assert start_date.tzinfo == timezone.utc, f"Start date must be UTC timezone."
    assert end_date.tzinfo == timezone.utc, f"End date must be UTC timezone."

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ["wind_speed_10m", "wind_direction_10m"],
        "wind_speed_unit": "kmh",
        "timezone": "UTC",
    }

    response = requests.get(url, params=params).json()
    response = response["hourly"]

    times = response["time"]
    wind_speed = response["wind_speed_10m"]
    wind_direction = response["wind_direction_10m"]

    results = pd.DataFrame({"Time": times, "Wind Speed (kph)": wind_speed, "Wind Direction (°)": wind_direction})
    results["Time"] = pd.to_datetime(results["Time"], utc=True)
    results = results[(results["Time"] >= start_date) & (results["Time"] < end_date)]

    return results
