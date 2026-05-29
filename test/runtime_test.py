from src.fire_spreading_model import FireSpreadingAdvanced, Parameters

import numpy as np
import time

fuel_mask = np.loadtxt("../data/green_fire/fuelbefore_px=20m.csv", delimiter=";")
water_mask = np.loadtxt("../data/green_fire/waterbefore_px=20m.csv", delimiter=";")
moisture_mask = np.loadtxt("../data/green_fire/moisturebefore_px=20m.csv", delimiter=";")
topo_mask = np.loadtxt("../data/green_fire/topo_px=20m.csv", delimiter=";")

n, m = fuel_mask.shape

param = Parameters(n=n, m=m, mu_H=0.5, mu_O=0.5, dF=0.5, dO=0.5, dW=0.5)
sim = FireSpreadingAdvanced(param)

start = time.perf_counter()
sim.run_simulation(100)
end = time.perf_counter()
runtime = end - start

print(runtime)

# original: ≈10s
# gradient once: no significant change
# 
