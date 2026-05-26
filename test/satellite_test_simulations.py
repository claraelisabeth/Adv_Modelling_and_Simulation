"""
Combines satellite live-data extraction via Sentinel Hub with your custom
Cellular Automata class to simulate the historical Dixie Fire footprint.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Connect local directory mapping to pick up modules from the src folder
project_root = os.path.abspath(os.path.join(os.getcwd(), '.'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.data_preprocessing import sentinel_client
    from src.fire_spreading_model import FireSpreadingAdvanced, Parameters
except ImportError:
    print("Error: Could not import custom classes. Ensure you are running this")
    print("script from the parent directory containing the 'src/' folder.")
    sys.exit(1)


def main():

    # setup authentication for Sentinel Hub API 
    print("Connecting to Sentinel Hub API...")
    client_id = 'sh-83b3baad-681c-49ff-967c-a1fe3eed19a4'
    client_secret = 'L9cE8ASdvwyY64jWo8nUJQRIZ37A0XpS'
    sc = sentinel_client(client_id=client_id, client_secret=client_secret)

    # Historical Dixie Fire ignition spatial coordinates area
    lon_min, lat_min, lon_max, lat_max = -121.50, 39.80, -121.10, 40.15
    fire_start_date_1, fire_start_date_2 = "2021-07-13", "2021-07-14"
    observation_end_date = "2021-08-15"
    
    # Grid Dimensions for the simulation (500x500 pixels to cover the area with reasonable resolution)
    pixel_x, pixel_y = 500, 500 

    print("Downloading satellite environmental layers...")
    fuel_before, water_before, moisture_before, _ = sc.get_data(
        lon_min, lat_min, lon_max, lat_max,
        fire_start_date_1, fire_start_date_2,
        pixel_x, pixel_y
    )

    print("Downloading DEM topography...")
    topo_mask = sc.get_topo(
        lon_min, lat_min, lon_max, lat_max,
        pixel_x, pixel_y
    )

    # Historical Ignition Center point located inside the canyon grid space
    historical_ignition = (393, 151) 
    
    # Calculate simulation runtime step framework (Hours * steps)
    # time_steps = int(sc.get_simulation_time(fire_start_date_1, observation_end_date) * 24)
    time_steps = 700  # for testing
    print(f"Calculated Simulation Duration: {time_steps} operational steps.")

    params = Parameters(
        n=pixel_x,
        m=pixel_y,
        mu_H=[0.35, 0.1625, 0.1625, 0.1625, 0.1625],  # Diffusion core profile
        mu_O=[0.45, 0.1375, 0.1375, 0.1375, 0.1375],  # Oxygen weight distribution
        dF=0.015,                                     # Fuel burn consumption step
        dO=0.025,                                     # Oxygen draw rate
        dW=0.04,                                      # Evaporation rate
        ignition_temp=0.38,                           # Ignition temperature threshold
        ignition_oxy=0.72,                            # Minimum oxygen level
        ignition_fuel=0.25,                           # Minimum fuel asset limit
        extinction_fuel_ratio=0.10,                   # Bare-ground threshold
        extinction_oxy=0.06,                          # Air starvation limit
        
        # Wind blowing towards the Northeast (Southwest wind)
        wind=(0.4, -0.2),                             
        wind_strength_factor=0.5,
        k_slope=0.005,                                # Kept low to handle real meter scales safely
        
        start_cells=[historical_ignition],
        random_F=False,                               # Deactivate procedural noise generator
        fuel_mask=fuel_before,
        water_mask=water_before,
        moisture_mask=moisture_before,
        topo_mask=topo_mask
    )

    
    print("Initializing FireSpreadingAdvanced Simulation Model Engine...")
    sim = FireSpreadingAdvanced(params)

    output_dir = "results"
    output_path = os.path.join(output_dir, "dixie_fire_historical")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running simulation loop. Writing frames to {output_path}.gif...")
    sim.run_simulation(T=time_steps, gif_name=output_path)
    print("Simulation finished processing successfully!")


if __name__ == "__main__":
    main()