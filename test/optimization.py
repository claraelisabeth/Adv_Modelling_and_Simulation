'''
This module contains the optimization routine for tuning the parameters of the fire spreading model.

We considered three loss functions:
1. General overlap loss function:       Jaccard’s similarity coefficient
2. Partial overlap loss functions:      Precision and recall
3. Distance loss functions:             Internal baddeley??? 

For more details on the loss functions used and other possibilities, see the paper:
"Loss functions for spatial wildfire applications" https://www.sciencedirect.com/science/article/pii/S1364815224000057

Geostack https://gitlab.com/geostack/library/-/blob/master/README.md?ref_type=heads
Geospatial toolkit
'''
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.ndimage import distance_transform_edt
from src.fire_spreading_model import FireSpreadingAdvanced, Parameters
from src.data_preprocessing import SentinelClient as sentinel_client




def objective_function(params_2optimize, target_mask, params_static, T, state_tracker, param_names):
    """
    Optimizes wildfire shapes by combining General Overlap and Spatial Distance
    """
    # distance_to_target[x, y] = how many pixels away this cell is from the REAL fire
    distance_to_target = distance_transform_edt(~target_mask)
    
    # Run Simulation
    # print("Running simulation")

    optimized_params = {name: float(val) for name, val in zip(param_names, params_2optimize)}

    params = Parameters(
        **optimized_params,
        **params_static
    )
    
    sim = FireSpreadingAdvanced(params)
    try:
        sim.run_simulation(T, gif_name=None)
    except:
        sim.run_simulation(params_static["delta_T"])

    pred_mask = sim.calculate_simulation_burned_mask()
    
    # General Overlap (IoU)
    intersection = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or(pred_mask, target_mask).sum()
    
    if union == 0:
        print("Warning: target_mask and pred_mask are both empty. Returning loss of 1.0")
        return 1.0  

    # keeping it a min problem
    iou_loss = 1.0 - (intersection / union)
    

    # Combined Loss ? if we choose to add another loss function
    total_loss = iou_loss 

    if total_loss < state_tracker["best_loss"]:
        state_tracker["best_loss"] = total_loss
        state_tracker["best_mask"] = pred_mask.copy()

    return total_loss



def main():
    # setup authentication for Sentinel Hub API 
    print("Connecting to Sentinel Hub API...")
    client_id = 'sh-83b3baad-681c-49ff-967c-a1fe3eed19a4'
    client_secret = 'L9cE8ASdvwyY64jWo8nUJQRIZ37A0XpS'
    sc = sentinel_client(client_id=client_id, client_secret=client_secret)

    # Historical Dixie Fire ignition spatial coordinates area
    lon_min, lat_min, lon_max, lat_max = -121.50, 39.80, -121.10, 40.15
    fire_start_date_1, fire_start_date_2 = "2021-07-13", "2021-07-14"
    observation_end_date_1, observation_end_date_2 = "2021-08-15", "2021-08-30"

    # Grid Dimensions for the simulation (500x500 pixels to cover the area with reasonable resolution)
    pixel_x, pixel_y = 500, 500 

    print("Downloading satellite environmental layers before the fire...")
    fuel_before, water_before, moisture_before, burnt_before = sc.get_data(
        lon_min, lat_min, lon_max, lat_max,
        fire_start_date_1, fire_start_date_2,
        pixel_x, pixel_y
    )

    print("Downloading DEM topography...")
    topo_mask = sc.get_topo(
        lon_min, lat_min, lon_max, lat_max,
        pixel_x, pixel_y
    )

    print("Downloading satellite environmental layers after the fire...")
    _, _, _, burnt_after = sc.get_data(
        lon_min, lat_min, lon_max, lat_max, 
        observation_end_date_1, observation_end_date_2, 
        pixel_x, pixel_y
    )

    #dNBR = (NBR_before - NBR_after)
    # Create the Target Burn Mask based on NBR and water levels
    dnbr = (burnt_before - burnt_after)
    target_mask = (dnbr > 0.1)

    # Historical Ignition Center point located inside the canyon grid space
    historical_ignition = (393, 151) 
    
    # Calculate simulation runtime step framework (Hours * steps)
    time_steps = int(sc.get_simulation_time(fire_start_date_1, observation_end_date_1) * 24)
    print(f"Calculated Simulation Duration: {time_steps} operational steps.")

    
    params_static = {
        #"mu_H": 0.2,  
        "mu_O": 0.4,                                      
        #"dF": 0.15,                                     
        "dO": 0.025,                                    
        "dW": 0.1,                           
        "ignition_temp": 0.27,              
        "ignition_oxy": 0.72,                  
        "ignition_fuel": 0.3,                  
        "extinction_fuel_ratio": 0.10,            
        "extinction_oxy": 0.06,               
        
        "wind": (0.4, -0.2),                             
        #"wind_strength_factor": 0.5,
        "k_slope": 0.005,                                
        
        "start_cells": [historical_ignition],
        "random_F": False,                               # Deactivate procedural noise generator
        "fuel_mask": fuel_before,
        "water_mask": water_before,
        "moisture_mask": moisture_before,
        "topo_mask": topo_mask                            
    }

    # define bounds for parameters [min, max]
    # these matrics are getting optimized
    bounds = [
        (0.05, 0.5),    # mu_H
        (0.01, 0.2),    # dF 
        (0.1, 1.5)      # wind_strength_factor 
    ]

    tracker = {"best_loss": float("inf"), "best_mask": None}

    # Run the calibration optimizer
    print(f"Running optimization...")
    result = differential_evolution(
        objective_function,
        bounds=bounds,
        args=(target_mask, params_static, time_steps, tracker), # passes these to objective function
        strategy='best1bin',
        maxiter=20, # Start small to test pipeline execution speed
        popsize=5,  # Start small to test pipeline execution speed, later maybe 15
        disp=True,
        polish=False
    )

    print("Optimized Parameters Found:", result.x)
    print("Best Loss Achieved (1 - IoU):", result.fun)

    #final_burnt_mask = tracker["best_mask"]
    #print()
    




if __name__ == "__main__":
    main()