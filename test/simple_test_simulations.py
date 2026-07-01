'''
This code is for testing the creation of simulations with the FireSpreadingAdvanced class. 
It sets up a simple simulation with default parameters and runs it for a specified number of time steps, 
saving the results as a GIF. The parameters can be adjusted to create different types of simulations, such as varying
the fuel distribution, wind conditions, or ignition points.
'''
from src.fire_spreading_model import FireSpreadingAdvanced, Parameters
import numpy as np
n,m = 100,100
params = Parameters(
    n=n,
    m=m,
    mu_H=[0.52, 0.12, 0.12, 0.12, 0.12],
    dF=0.02,
    dW=0.1,
    ignition_temp=0.3,
    ignition_fuel=0.3,
    extinction_fuel=0.1,
    wind_velocity=0,
    wind_direction=0,
    start_cells=[(50, 50)],
    random_F=True,
    fuel_mask=None,
    water_mask=None,
    moisture_mask=None,
    topo_mask=None,
    k_slope=0.1,
    wind_strength_factor=0
)

#### First Test: Simulation with wind
# wind blowing to the right and slightly downward (approx 110 degrees East-Southeast)
params.start_cells = [(50, 50), (20, 80)]
params.wind_velocity = 50
params.wind_direction = 110
params.mu_H = 0.4
params.wind_strength_factor = 0.6

params.__post_init__()

T=50
wind_fire_spreading = FireSpreadingAdvanced(params)
wind_fire_spreading.run_simulation(T, gif_name="results/test1_wind", visualization=True)



#### Second Test: Simulation with slope effect
# fire should spread faster uphill 
# single ignition point in the center, and a slope
params.start_cells = [(50, 50)]
params.k_slope = 1.6
params.wind_velocity = 20
params.wind_direction = 30
params.wind_strength_factor = 0.1

topo = np.zeros((n, m))
for col in range(m):
    topo[:, col] = col * 20

params.topo_mask = topo
params.moisture_mask = None
params.water_mask = None

params.__post_init__()

T = 70
topo_fire = FireSpreadingAdvanced(params)
topo_fire.run_simulation(T, gif_name="results/test2_slope", visualization=True)


#### Third Test: Airplane Drop
# Airplane dynamically drops a wall of water on the advancing edge of the fire at specific timesteps
params.topo_mask = None
params.water_mask = None
params.moisture_mask = None

params.start_cells = [(50, 20)]
params.wind_velocity = 50
params.wind_direction = 90
params.wind_strength_factor = 0.5

# 3. Schedule the dynamic airplane drops
drops = {
    20: [
        {'auto_target': True, 'target_edge': True, 'height': 5, 'width': 3, 'water_intensity': 0.8, 'cooling_effect': 0.8}
    ],
    30: [
        {'auto_target': True, 'target_edge': True, 'height': 8, 'width': 4, 'water_intensity': 0.5, 'cooling_effect': 0.8}
    ],
    40: [
        {'auto_target': True, 'target_edge': True, 'height': 10, 'width': 7, 'water_intensity': 0.7, 'cooling_effect': 0.8}
    ]
}

params.__post_init__()

T = 90
airdrop_fire = FireSpreadingAdvanced(params)
airdrop_fire.run_simulation(T, gif_name="results/test3_airdrop", scheduled_drops=drops, visualization=True)

#### Fourth Test: Firebreak (targets wind direction)
params.start_cells = [(50, 50)]
params.wind_velocity = 50
params.wind_direction = 100
params.wind_strength_factor = 0.6

# Firefighters calculate the wind and build a 80x10 diagonal-blocking box
firebreaks_dynamic = {
    15: [
        {'auto_target': True, 'wind_offset': 50, 'height': 40, 'width': 5}
    ]
}

params.__post_init__()
T = 80 
smart_firebreak_sim = FireSpreadingAdvanced(params)
smart_firebreak_sim.run_simulation(T, gif_name="results/test4_smart_firebreak", scheduled_firebreaks=firebreaks_dynamic, visualization=True)

#### Fifth Test: Simulation with moisture
# fire should spread slower in areas with higher moisture
params.topo_mask = None
params.moisture_mask = None

# create a "river" in the center (columns 60 to 65)
water = np.zeros((n, m))
water[:, 60:66] = 1.0  
params.water_mask = water

# with wind to the right
params.wind_velocity = 50
params.wind_direction = 70
params.wind_strength_factor = 0.5

params.__post_init__()

water_fire = FireSpreadingAdvanced(params)
water_fire.run_simulation(T, gif_name="results/test5_water_barrier", visualization=True)