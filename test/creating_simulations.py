from src.fire_spreading_model import FireSpreadingAdvanced

n, m = 100,100
max_H, max_F, max_O = 1.0, 1.0, 1.0
mu_O = [0.52, 0.12, 0.12, 0.12, 0.12]
mu_H = [0.52, 0.12, 0.12, 0.12, 0.12]
dF, dO = 0.02, 0.05
ignition_temp = 0.3
ignition_oxy = 0.76
ignition_fuel = 0.07
start_cells=[(n//2,m//2)]
random_F = True
wind = [0.4,0.1]

simple_fire_spreading = FireSpreadingAdvanced(n, m, max_H, max_F, max_O, mu_H, mu_O, dF, dO, ignition_temp, ignition_oxy, ignition_fuel, wind, start_cells, random_F)

T = 50
simple_fire_spreading.run_simulation(T, gif_name = "results/simple_fire")




