from src.data_preprocessing import sentinel_client
from src.fire_spreading_model import FireSpreadingAdvanced

# Setup Authentication
client_id = 'sh-83b3baad-681c-49ff-967c-a1fe3eed19a4'
client_secret = 'L9cE8ASdvwyY64jWo8nUJQRIZ37A0XpS'
sc = sentinel_client(client_id=client_id, client_secret=client_secret)

fuel_data, water_data = sc.get_data(136.45, -36.1, 136.95, -35.75, "2019-05-01", "2019-05-15", 512, 442)
n = fuel_data.shape[0]
m = fuel_data.shape[1]
max_H, max_F, max_O = 1.0, 1.0, 1.0
mu_O = [0.52, 0.12, 0.12, 0.12, 0.12]
mu_H = [0.52, 0.12, 0.12, 0.12, 0.12]
dF, dO = 0.02, 0.05
ignition_temp = 0.3
ignition_oxy = 0.76
ignition_fuel = 0.0

wind = [0.4, 0.1]
start_cells = [(n // 2, m // 2)]
random_F = False
fuel_mask = fuel_data
water_mask = water_data

sim_australia = FireSpreadingAdvanced(
    n, m, 
    max_H, max_F, max_O, 
    mu_H, mu_O, 
    dF, dO, 
    ignition_temp, ignition_oxy, ignition_fuel, 
    wind, 
    start_cells, 
    random_F, 
    fuel_mask, 
    water_mask
)

T = 300
sim_australia.run_simulation(T, gif_name="results/australia_fire")