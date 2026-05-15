'''
This code is for testing the creation of simulations with the FireSpreadingAdvanced class. 
It sets up a simple simulation with default parameters and runs it for a specified number of time steps, 
saving the results as a GIF. The parameters can be adjusted to create different types of simulations, such as varying
the fuel distribution, wind conditions, or ignition points.
'''
from src.fire_spreading_model import FireSpreadingAdvanced, Parameters
import numpy as np

### First Test: Simple simulation
# no wind, single ignition point in the center
n,m = 100,100
params = Parameters(
    n=n,
    m=m,
    mu_O=[0.52, 0.12, 0.12, 0.12, 0.12],
    mu_H=[0.52, 0.12, 0.12, 0.12, 0.12],
    dF=0.02,
    dO=0.05,
    dW=0.1,
    ignition_temp=0.3,
    ignition_oxy=0.76,
    ignition_fuel=0.3,
    extinction_fuel_ratio=0.1,
    extinction_oxy=0.1,
    wind=(0, 0),
    start_cells=[(50, 50)],
    random_F=True,
    fuel_mask=None,
    water_mask=None,
    moisture_mask=None,
    topo_mask=None,
    k_slope=0.1,
    wind_strength_factor=0
)

T = 50
simple_fire_spreading = FireSpreadingAdvanced(params)
simple_fire_spreading.run_simulation(T, gif_name = "results/test1")



#### Second Test: Simulation with wind
# wind blowing to the right and slightly downward, two ignition points in the center and right upper quadrant
# mu values are now given only with the first value, the rest will be calculated as (1 - first_value) / 4
# this also test if the fire does not wrap around the grid when wind pushes it towards the edge
params.start_cells = [(50, 50), (20, 80)]
params.wind = (0.5, 0.2)
params.mu_O = 0.4
params.mu_H = 0.4
params.wind_strength_factor = 0.6

wind_fire_spreading = FireSpreadingAdvanced(params)
wind_fire_spreading.run_simulation(T, gif_name = "results/test2_wind")



#### Third Test: Simulation with slope effect
# fire should spread faster uphill 
# no wind, single ignition point in the center, and a slope from left to right
params.start_cells = [(50, 50)]
params.wind = (0, 0)
params.wind_strength_factor = 0
params.k_slope = 0.4  # higher value --> stronger slope effect

# Steady ramp rising 1.5 units per pixel from left to right
topo = np.zeros((n, m))
for col in range(m):
    topo[:, col] = col * 1.5

params.topo_mask = topo
params.moisture_mask = None
params.water_mask = None

T = 70
topo_fire = FireSpreadingAdvanced(params)
topo_fire.run_simulation(T, gif_name="results/test3_slope")




#### Fourth Test: Simulation with moisture
# fire should spread slower in areas with higher moisture
params.topo_mask = None
params.moisture_mask = None

# create a "river" in the center (columns 60 to 65)
water = np.zeros((n, m))
water[:, 60:66] = 1.0  
params.water_mask = water

# with wind to the right
params.wind = (0.5, 0.0)
params.wind_strength_factor = 0.5

water_fire = FireSpreadingAdvanced(params)
water_fire.run_simulation(T, gif_name="results/test4_water_barrier")
