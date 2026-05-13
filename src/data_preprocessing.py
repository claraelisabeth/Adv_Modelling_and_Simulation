import requests
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from datetime import datetime
import io
import matplotlib.pyplot as plt
import numpy as np
import tifffile

class sentinel_client:
    """
    A client to interact with the Copernicus Sentinel Hub API to fetch 
    satellite imagery and calculate environmental indices.
    """
    def __init__(self, client_id, client_secret):
        # API Endpoints for processing data and fetching OAuth2 tokens
        self.url = "https://sh.dataspace.copernicus.eu/api/v1/process"
        self.token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        
        # Initialize OAuth2 session using Client Credentials flow
        client = BackendApplicationClient(client_id=client_id)
        self.oauth = OAuth2Session(client=client)
        # Fetch and store the access token
        self.oauth.fetch_token(token_url=self.token_url, client_secret=client_secret, include_client_id=True)

    def get_data(self, lon_min, lat_min, lon_max, lat_max, date_start, date_end, width, height):
        """
        Fetches raw satellite bands and calculates indices (NDVI, NDWI, NDMI, NBR).
        Returns the data as 32-bit float arrays.
        """
        data = {
            "input": {
                "bounds": {
                    "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
                    "bbox": [lon_min, lat_min, lon_max, lat_max]
                },
                "data": [{
                    "dataFilter": {"timeRange": {"from": f"{date_start}T00:00:00Z", "to": f"{date_end}T23:59:59Z"}},
                    "type": "sentinel-2-l2a"
                }]
            },
            "output": {
                "width": width, "height": height,
                "responses": [{
                    "identifier": "default",
                    "format": {"type": "image/tiff"}
                }]
            },
            "evalscript": """
                //VERSION=3
                function setup() {
                  return {
                    input: ["B03", "B04", "B08", "B11", "B12"],
                    output: { bands: 4, sampleType: "FLOAT32" }
                  };
                }
                function evaluatePixel(sample) {
                  // Indices with protection against Division by Zero
                  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 0.0001);
                  let ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08 + 0.0001);
                  let ndmi = (sample.B08 - sample.B11) / (sample.B08 + sample.B11 + 0.0001);
                  let nbr  = (sample.B08 - sample.B12) / (sample.B08 + sample.B12 + 0.0001);

                  return [ndvi, ndwi, ndmi, nbr];
                }
            """
        }

        # Sending the request to the Sentinel Hub API
        response = self.oauth.post(self.url, json=data)

        if response.status_code == 200:
            # Read TIFF content (preserves 32-bit floats)
            image = tifffile.imread(io.BytesIO(response.content))
            return image[:,:,0], image[:,:,1], image[:,:,2], image[:,:,3]
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None, None, None, None
        
    def get_photo(self, lon_min, lat_min, lon_max, lat_max, date_start, date_end, width, height, mode="true_colour"):
        """
        Fetches visual imagery (True Colour or False Colour) for display.
        Returns a standard RGB image (PNG).
        """
        if mode == "true_colour":
            bands = '["B02", "B03", "B04"]'
            mapping = "[2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02]"
        elif mode == "false_colour":
            bands = '["B12", "B08", "B04"]'
            mapping = "[2.5 * sample.B12, 1.5 * sample.B08, 2.5 * sample.B04]"
        else:
            raise ValueError("Mode must be 'true_colour' or 'false_colour'")

        data = {
            "input": {
                "bounds": {
                    "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
                    "bbox": [lon_min, lat_min, lon_max, lat_max]
                },
                "data": [{"dataFilter": {"timeRange": {"from": f"{date_start}T00:00:00Z","to": f"{date_end}T23:59:59Z"}},
                          "type": "sentinel-2-l2a"}]
            },
            "output": {
                "width": width, 
                "height": height, 
                "responses": [{"identifier": "default", "format": {"type": "image/png"}}]
            },
            "evalscript": f"""
                //VERSION=3
                function setup() {{
                  return {{
                    input: {bands},
                    output: {{ bands: 3 }}
                  }};
                }}
                function evaluatePixel(sample) {{
                  return {mapping};
                }}
            """
        }

        response = self.oauth.post(self.url, json=data)

        if response.status_code == 200:
            return plt.imread(io.BytesIO(response.content))
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
        
    def get_topo(self, lon_min, lat_min, lon_max, lat_max, width, height):
        """
        Fetches Digital Elevation Model (DEM) data from Copernicus.
        Returns a 2D numpy array with elevation in meters.
        """
        data = {
            "input": {
                "bounds": {
                    "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
                    "bbox": [lon_min, lat_min, lon_max, lat_max]
                },
                "data": [{
                    "type": "dem",
                    "dataFilter": {
                        "demInstance": "COPERNICUS_30" # 30m Auflösung ist ideal für Waldbrand
                    }
                }]
            },
            "output": {
                "width": width, "height": height,
                "responses": [{
                    "identifier": "default",
                    "format": {"type": "image/tiff"}
                }]
            },
            "evalscript": """
                //VERSION=3
                function setup() {
                  return {
                    input: ["DEM"],
                    output: { bands: 1, sampleType: "FLOAT32" }
                  };
                }
                function evaluatePixel(sample) {
                  return [sample.DEM];
                }
            """
        }

        response = self.oauth.post(self.url, json=data)

        if response.status_code == 200:
            # Das TIFF enthält die Höhenmeter direkt als Float32
            image = tifffile.imread(io.BytesIO(response.content))
            return image
        else:
            print(f"Error fetching DEM: {response.status_code}, {response.text}")
            return None
        
    def get_simulation_time(self, fire_start_date, observation_end_date):
        """
        Calculates the number of days between two date strings.
        """
        fmt = "%Y-%m-%d"
        d1 = datetime.strptime(fire_start_date, fmt)
        d2 = datetime.strptime(observation_end_date, fmt)
        delta = d2 - d1

        return abs(delta.days)
    
    def get_severity_mask(self, dnbr_array):
        h, w = dnbr_array.shape
        color_mask = np.zeros((h, w, 3))
        colors = {
            "enhanced_high": [0.2, 0.4, 0.1],   
            "enhanced_low":  [0.7, 0.8, 0.5],   
            "unburned":      [0.6, 1.0, 0.4],  
            "low":           [1.0, 1.0, 0.2],   
            "mod_low":       [1.0, 0.6, 0.2],    
            "mod_high":      [0.9, 0.2, 0.2],   
            "high":          [0.7, 0.0, 0.7]     
        }

        # thresholds based on standard dNBR classification
        color_mask[dnbr_array <= -0.251] = colors["enhanced_high"]
        color_mask[(dnbr_array > -0.251) & (dnbr_array <= -0.101)] = colors["enhanced_low"]
        color_mask[(dnbr_array > -0.100) & (dnbr_array <= 0.099)] = colors["unburned"]
        color_mask[(dnbr_array >= 0.100) & (dnbr_array <= 0.269)] = colors["low"]
        color_mask[(dnbr_array >= 0.270) & (dnbr_array <= 0.439)] = colors["mod_low"]
        color_mask[(dnbr_array >= 0.440) & (dnbr_array <= 0.659)] = colors["mod_high"]
        color_mask[dnbr_array >= 0.660] = colors["high"]

        # create binary mask for burned vs unburned (threshold at 0.270)
        binary_burned_mask = np.where(dnbr_array >= 0.270, 1, 0)

        return color_mask, binary_burned_mask
    


