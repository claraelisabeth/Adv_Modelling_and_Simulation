from src.fire_spreading_model import FireSpreadingAdvanced, Parameters
'''
This code is for testing the creation of simulations with the FireSpreadingAdvanced class. 
It sets up a simple simulation with default parameters and runs it for a specified number of time steps, 
saving the results as a GIF. The parameters can be adjusted to create different types of simulations, such as varying
the fuel distribution, wind conditions, or ignition points.
'''

### First Test: Simple simulation
# no wind, single ignition point in the center
params = Parameters(
    n=100,
    m=100,
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
# wind blowing to the right, single ignition point in the center
# mu values are now given only with the first value, the rest will be calculated as (1 - first_value) / 4

params.wind = (0.5, 0.1)
params.mu_O = 0.4
params.mu_H = 0.4

simple_fire_spreading = FireSpreadingAdvanced(params)
simple_fire_spreading.run_simulation(T, gif_name = "results/test2")
