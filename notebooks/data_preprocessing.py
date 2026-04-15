import requests
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from datetime import datetime
import io
import matplotlib.pyplot as plt
import numpy as np

class sentinel_client:
    def __init__(self, client_id, client_secret):
        self.url = "https://sh.dataspace.copernicus.eu/api/v1/process"
        self.token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        
        client = BackendApplicationClient(client_id=client_id)
        self.oauth = OAuth2Session(client=client)
        self.oauth.fetch_token(token_url=self.token_url, client_secret=client_secret, include_client_id=True)


    def get_data(self, lon_min, lat_min, lon_max, lat_max, date_start, date_end, width, height):
        if lat_min > lat_max or lon_min > lon_max:
            raise ValueError("Minimum latitude/longitude must be less than maximum latitude/longitude.")
        if date_start > date_end:
            raise ValueError("Start date must be before end date.")
        
        data = {
          "input": {
            "bounds": {
              "properties": {
                "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
              },
              "bbox": [lon_min, lat_min, lon_max, lat_max]
            },
            "data": [
              {
                "dataFilter": {
                  "timeRange": {
                    "from": f"{date_start}T00:00:00Z",
                    "to": f"{date_end}T23:59:59Z"
                  }
                },
                "type": "sentinel-2-l2a"
              }
            ]
          },
          "output": {
            "width": width,
            "height": height,
            "responses": [
              {
                "identifier": "default",
                "format": {"type": "image/png"}
              }
            ]
          },
          "evalscript": """
            //VERSION=3
            function setup() {
              return {
                input: ["B08", "B04", "B03"],
                output: { bands: 3 }
              };
            }
            function evaluatePixel(sample) {
              let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
              let ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08);
              return [ndvi, 0, ndwi];
            }
          """
        }

        response = self.oauth.post(self.url, json=data)

        if response.status_code == 200:
            image = plt.imread(io.BytesIO(response.content))
            fuel_matrix = image[:,:,0]
            water_matrix = image[:,:,2]
            return fuel_matrix, water_matrix
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None, None
