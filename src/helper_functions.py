import numpy as np


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

