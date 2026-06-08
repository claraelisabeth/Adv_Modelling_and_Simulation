import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass
from typing import Sequence, Union


H, F, O, B, W = 0, 1, 2, 3, 4
# H: heat level
# F: fuel level
# O: oxygen level
# B: burning state (0 or 1)
# W: water/moisture level (optional)


@dataclass
class Parameters:
    """
    A class holding all parameters for the simulation.

    Attributes
    ----
    n : int
        Number of grid rows.
    m : int
        Number of grid columns.
    mu_H : float or ArrayLike of length 5
        Heat diffusion coefficient(s). Can be a scalar for isotropic diffusion, or a 5-element vector specifying
        diffusion weights for [center, up, down, left, right].
    mu_O : float or ArrayLike of length 5
        Oxygen diffusion coefficient(s), with the same format as `mu_H`.
    dF : float
        Amount of fuel consumed per timestep during burning.
    dO : float
        Amount of oxygen consumed per timestep during burning.
    dW : float
        Amount of water/moisture evaporated per timestep.
    ignition_temp : float
        Minimum heat required for ignition.
    ignition_oxy : float
        Minimum oxygen level required for ignition.
    ignition_fuel : float
        Minimum fuel level required for ignition.
    extinction_fuel_ratio : float
        Fire extinguishes when fuel falls below this fraction of `max_F`.
    extinction_oxy : float
        Fire extinguishes when oxygen falls below this threshold.
    wind : tuple[float, float], optional
        Wind vector `(wx, wy)` influencing fire spread direction. Values between -1.0 and 1.0
    wind_velocity : float | list[float], optional
        Wind velocity in km/h. The Wind can either be static or it can be a list with the wind velocity for every hour.
        Default is 0.
    wind_direction : int | list[int], optional
        The direction of the wind. Can be static or it can be a list with a value for every hour. Default is 0.
    start_cells : list[tuple[int, int]], optional
        Grid coordinates where fire is initially ignited. Default is [(0,0)].
    random_F : bool
        If True, initialize fuel values randomly instead of uniformly using `max_F`. Default is False.
    fuel_mask : np.ndarray, optional
        Spatial distribution of fuel availability.
    water_mask : np.ndarray, optional
        Spatial distribution of water bodies or non-burnable regions.
    moisture_mask : np.ndarray, optional
        Spatial distribution of moisture levels affecting combustion.
    topo_mask : np.ndarray, optional
        Terrain elevation map used to compute slope effects.
    resolution : float
        Resolution of grid cells in meters. Default is 20.
    k_slope : float
        Scaling factor controlling the influence of terrain slope on diffusion. Default is 0.1.
    wind_strength_factor : float
        Scaling factor controlling the influence of wind on diffusion. Default is 0, keep between 0.0 and 1.0
    """
    n : int
    m : int
    mu_H : Union[float, list[float]]  # TODO: optimize, then set constant
    mu_O : Union[float, list[float]]  # TODO: optimize, then set constant
    dF : float  # TODO: optimize, then set constant
    dO : float  # TODO: optimize, then set constant
    dW : float  # TODO: optimize, then set constant
    ignition_temp : float = 0.3  # TODO: optimize, then set constant
    ignition_oxy : float = 0.76
    ignition_fuel : float = 0.3  # TODO: optimize, then set constant
    extinction_fuel_ratio : float = 0.15
    extinction_oxy : float = 0.05
    wind : Union[tuple[float, float], list[tuple[float, float]]] = (0.0, 0.0)
    wind_velocity : float | list[float] = 0
    wind_direction : int = 0
    start_cells : list[tuple[int, int]] = None
    random_F : bool = False
    fuel_mask : np.ndarray = None
    water_mask : np.ndarray = None
    moisture_mask : np.ndarray = None
    topo_mask : np.ndarray = None
    k_slope : float = 0.1  # TODO: optimize, then set constant
    resolution : float = 20
    wind_strength_factor : float = 0  # TODO: optimize, then set constant
    timesteps : int = 100


    def __post_init__(self):
        if self.start_cells is None:
            self.start_cells = [(0,0)]

        # shape the wind data
        if isinstance(self.wind_velocity, (int, float)):
            self.wind_velocity = np.full(self.timesteps, self.wind_velocity, dtype=float)
            self.wind_direction = np.full(self.timesteps, self.wind_direction, dtype=int)
        else:
            self.wind_velocity = np.array(self.wind_velocity)[:self.timesteps]
            self.wind_direction = np.array(self.wind_direction)[:self.timesteps]

        assert (len(self.wind_velocity) == len(self.wind_direction)) and (len(self.wind_velocity) >= self.timesteps), \
            "Not enough wind data!"

        # fill mask in case none are given
        if self.fuel_mask is None and self.random_F:
            self.fuel_mask = np.random.uniform(low=0, high=1, size=(self.n, self.m))
            for _ in range(2):
                self.fuel_mask = (self.fuel_mask + np.roll(self.fuel_mask, 1, axis=0)
                                  + np.roll(self.fuel_mask, -1, axis=0) + np.roll(self.fuel_mask, 1, axis=1)
                                  + np.roll(self.fuel_mask, -1, axis=1)) / 5
        elif self.fuel_mask is None and not self.random_F:
            self.fuel_mask = np.ones(shape=(self.n, self.m))
        if self.water_mask is None:
            self.water_mask = np.zeros(shape=(self.n, self.m))
        if self.moisture_mask is None:
            self.moisture_mask = np.zeros(shape=(self.n, self.m))
        if self.topo_mask is None:
            self.topo_mask = np.zeros(shape=(self.n, self.m))

        # check the shape of the masks
        assert (self.n, self.m) == self.fuel_mask.shape, f"Fuel mask shape different from grid!"
        assert (self.n, self.m) == self.water_mask.shape, f"Water mask shape different from grid!"
        assert (self.n, self.m) == self.moisture_mask.shape, f"Moisture mask shape different from grid!"
        assert (self.n, self.m) == self.topo_mask.shape, f"Topology mask shape different from grid!"


class FireSpreadingAdvanced:
    """
    Simulates wildfire spreading on a 2D grid using diffusion and combustion dynamics, including the effects of wind,
    terrain slope, and moisture.

    Attributes
    ----------
    param : Parameters
        An instance of the Parameters class containing all necessary parameters for the simulation.
    precompute_mu : bool, optional
        Whether to precompute µ for every timestep or update it during the simulation. Precomputing µ is faster but uses
        more memory. Default is False.
    """
    def __init__(self, param: Parameters, precompute_mu: bool = False):
        self.param = param
        self.timesteps = param.timesteps
        self.n = param.n
        self.m = param.m
        self.max_H = 1.0
        self.max_F = 1.0
        self.max_O = 1.0

        self.dW = param.dW
        self.water_mask = param.water_mask
        self.fuel_mask = param.fuel_mask
        self.moisture_mask = param.moisture_mask
        self.topo_mask = param.topo_mask
        self.k_slope = param.k_slope
        self.wind = param.wind
        self.dF = param.dF
        self.dO = param.dO
        self.ignition_temp = param.ignition_temp
        self.ignition_oxy = param.ignition_oxy
        self.ignition_fuel = param.ignition_fuel
        self.extinction_fuel_ratio = param.extinction_fuel_ratio
        self.extinction_oxy = param.extinction_oxy
        self.extinction_fuel = 0.1  # TODO: delete or change

        self.precompute_mu = precompute_mu

        # wind conversion and normalization
        wind_upper_limit = 150
        wind_vel_normal = np.array(param.wind_velocity) / wind_upper_limit
        self.wind_EW = wind_vel_normal * np.sin(np.radians(param.wind_direction)) * param.wind_strength_factor
        self.wind_NS = wind_vel_normal * np.cos(np.radians(param.wind_direction)) * param.wind_strength_factor

        # calculate transport vector
        grad_NS, grad_EW = np.gradient(param.topo_mask, param.resolution)
        self.trans_EW = self.wind_EW[:, np.newaxis, np.newaxis] + np.tanh(grad_EW) * param.k_slope
        self.trans_NS = self.wind_NS[:, np.newaxis, np.newaxis] + np.tanh(grad_NS) * param.k_slope

        # baseline µ or precompute for every timestep
        if precompute_mu:
            self.mu_H = self._precompute_mu(param.mu_H)
            self.mu_O = self._precompute_mu(param.mu_O)
        else:
            self.mu_H = [param.mu_H] + [(1 - param.mu_H) / 4 for _ in range(4)] \
                if np.isscalar(param.mu_H) else param.mu_H
            self.mu_O = [param.mu_O] + [(1 - param.mu_O) / 4 for _ in range(4)] \
                if np.isscalar(param.mu_O) else param.mu_O

        # 5-layer state setup: H, F, O, B, W
        self.state = np.zeros((self.n, self.m, 5))  
        self.state[:, :, O] = self.max_O   

        for cell in param.start_cells:
            self.state[cell[0], cell[1], H] = self.max_H
            self.state[cell[0], cell[1], B] = 1

        self.state[:, :, F] = np.maximum(0, param.fuel_mask).copy()

        if param.water_mask is not None:
            self.state[:, :, F][param.water_mask > 0.0] = 0.0

        self.initial_fuel = self.state[:, :, F].copy()
        self.diffused_state = np.copy(self.state)

        if param.moisture_mask is not None:
            self.state[:, :, W] = param.moisture_mask


    def _precompute_mu(self, mu: float | tuple[float, float]) -> np.ndarray:
        """ Precomputes µ for every timestep and grid position. This takes up more memory and may result in a crash. """
        mu_result = np.zeros(shape=(self.timesteps, self.n, self.m, 5), dtype=np.float32)

        if isinstance(mu, (tuple, list, np.ndarray)):
            mu_neighbour = mu[1]
        else:
            mu_neighbour = (1 - mu) / 4

        mu_result[:, :, :, 0] = mu[0] if isinstance(mu, (tuple, list, np.ndarray)) else mu
        mu_result[:, :, :, 1] = mu_neighbour * (1 + self.trans_NS[:, :, :])  # north
        mu_result[:, :, :, 2] = mu_neighbour * (1 - self.trans_NS[:, :, :])  # south
        mu_result[:, :, :, 3] = mu_neighbour * (1 - self.trans_EW[:, :, :])  # west
        mu_result[:, :, :, 4] = mu_neighbour * (1 + self.trans_EW[:, :, :])  # east

        mu_result = np.clip(mu_result, a_min=0, a_max=None)
        mu_sums = np.linalg.norm(mu_result, axis=3, keepdims=True)
        mu_result = mu_result / (mu_sums + 1e-6)

        return mu_result


    def _compute_mu_for_timestep(self, mu: np.ndarray, t: int) -> np.ndarray:
        """ Compute µ for every grid position for a specific timestep. """
        mu_result = np.zeros(shape=(self.n, self.m, 5))

        mu_result[:, :, 0] = mu[0]
        mu_result[:, :, 1] = mu[1] * (1 + self.trans_NS[t, :, :])  # north
        mu_result[:, :, 2] = mu[2] * (1 - self.trans_NS[t, :, :])  # south
        mu_result[:, :, 3] = mu[3] * (1 - self.trans_EW[t, :, :])  # west
        mu_result[:, :, 4] = mu[4] * (1 + self.trans_EW[t, :, :])  # east

        mu_result = np.clip(mu_result, a_min=0, a_max=None)
        mu_sums = np.linalg.norm(mu_result, axis=2, keepdims=True)
        mu_result = mu_result / (mu_sums + 1e-6)

        return mu_result


    def _diffuse(self, t: int) -> None:
        """ Compute diffusion for heat and oxygen based on the current state, wind, and slope. """
        diffused_state = np.copy(self.state)

        H_pad = np.pad(self.state[:, :, H], 1, mode='edge')
        O_pad = np.pad(self.state[:, :, O], 1, mode='edge')

        if self.precompute_mu:
            diffused_state[:, :, H] = (self.mu_H[t, :, :, 0] * self.state[:, :, H] +
                                   self.mu_H[t, :, :, 1] * H_pad[:-2, 1:-1] +
                                   self.mu_H[t, :, :, 2] * H_pad[2:, 1:-1] +
                                   self.mu_H[t, :, :, 3] * H_pad[1:-1, :-2] +
                                   self.mu_H[t, :, :, 4] * H_pad[1:-1, 2:])

            diffused_state[:, :, O] = (self.mu_O[t, :, :, 0] * self.state[:, :, O] +
                                   self.mu_O[t, :, :, 1] * O_pad[:-2, 1:-1] +
                                   self.mu_O[t, :, :, 2] * O_pad[2:, 1:-1] +
                                   self.mu_O[t, :, :, 3] * O_pad[1:-1, :-2] +
                                   self.mu_O[t, :, :, 4] * O_pad[1:-1, 2:])
        else:
            mu_H = self._compute_mu_for_timestep(self.mu_H, t)
            mu_O = self._compute_mu_for_timestep(self.mu_O, t)

            diffused_state[:, :, H] = (mu_H[:, :, 0] * self.state[:, :, H] +
                                   mu_H[:, :, 1] * H_pad[:-2, 1:-1] +
                                   mu_H[:, :, 2] * H_pad[2:, 1:-1] +
                                   mu_H[:, :, 3] * H_pad[1:-1, :-2] +
                                   mu_H[:, :, 4] * H_pad[1:-1, 2:])

            diffused_state[:, :, O] = (mu_O[:, :, 0] * self.state[:, :, O] +
                                   mu_O[:, :, 1] * O_pad[:-2, 1:-1] +
                                   mu_O[:, :, 2] * O_pad[2:, 1:-1] +
                                   mu_O[:, :, 3] * O_pad[1:-1, :-2] +
                                   mu_O[:, :, 4] * O_pad[1:-1, 2:])

        self.diffused_state = diffused_state


    def _burning(self):
        """
        Update the state of the grid based on burning dynamics, including fuel consumption, heat generation, and fire
        spread to neighbouring cells based on ignition conditions.
        """
        state_new = np.copy(self.state)
        fuel_state = self.state[:, :, F]
        fuel_start = self.initial_fuel
        fuel_ratio = fuel_state / (fuel_start + 1e-6) # avoid division by zero

        heat_diffused = self.diffused_state[:, :, H]
        oxy_diffused = self.diffused_state[:, :, O]

        burning = (self.state[:, :, B] == 1) # burning cells

        # 1. decrement the fuel and oxygen level
        state_new[:, :, F][burning] = np.maximum(0, fuel_state[burning] - self.dF)
        state_new[:, :, O][burning] = np.maximum(0, oxy_diffused[burning] - self.dO)

        # 2. set the heat level to maximum heat
        state_new[:, :, H][burning] = self.max_H

        # 3. check if the fire is extinguished
        extinguish = (fuel_ratio < self.extinction_fuel_ratio) | (state_new[:, :, O] < self.extinction_oxy)
        state_new[:, :, B][burning & extinguish] = 0
        
        not_burning = ~burning # non burning cells
        
        # water (inside of plants) has to eveporate = be zero to ignite
        evaporation_mask = (not_burning & (self.state[:, :, W] > 0) & (self.diffused_state[:, :, H] > 0))

        # decremnent the water level
        state_new[:, :, W][evaporation_mask] = np.maximum(0, self.state[:, :, W][evaporation_mask] - self.dW)

        # check conditions if the cell ignites, if the cell ignites set B to 1 and H to max_H
        # conditions:
        ignite = (
            (state_new[:, :, W] == 0) & 
            (heat_diffused > self.ignition_temp) &
            (fuel_state > self.ignition_fuel) &
            (oxy_diffused > self.ignition_oxy)
        )

        # optional randomness??
        # rand = np.random.rand(*H_val.shape)
        # ignite = ignite & (rand < some_probability)

        state_new[:, :, B][not_burning & ignite] = 1
        state_new[:, :, H][not_burning & ignite] = self.max_H

        # update remaining cells
        state_new[:, :, H][not_burning & ~ignite] = heat_diffused[not_burning & ~ignite]
        state_new[:, :, O][not_burning] = oxy_diffused[not_burning]

        self.state = state_new

    def _burning_exp(self):
        state_new = np.copy(self.state)
        fuel_state = self.state[:, :, F]
        heat_diffused = self.diffused_state[:, :, H]
        burning = (self.state[:, :, B] == 1)

        # deprecate fuel and check if cells extinguishes
        self.dF = 0.1  # TODO: delete
        state_new[:, :, F][burning] = fuel_state[burning] * (1 - self.dF)

        extinguish = state_new[:, :, F] < self.extinction_fuel
        state_new[:, :, B][burning & extinguish] = 0

        # evaporation
        state_new[:, :, W] = np.maximum(0, state_new[:, :, W] - (self.dW * heat_diffused))

        # ignition
        ignite = (state_new[:, :, W] == 0) & (heat_diffused > self.ignition_temp) & (fuel_state > self.ignition_fuel)
        state_new[:, :, B][~burning & ignite] = 1

        state_new[:, :, H][(~burning & ignite) | (burning & ~extinguish)] = self.max_H
        state_new[:, :, H][burning & extinguish] = heat_diffused[burning & extinguish]
        state_new[:, :, H][~burning & ~ignite] = heat_diffused[~burning & ~ignite]

        self.state = state_new

    def _make_rgb(self):
        """
        Create an RGB image representation of the current state for visualization.

        Color scheme:
        - Beige: rocks or non-burnable areas (default background)
        - Blue: water bodies or high moisture areas
        - Black: burnt areas (low fuel ratio)
        - Green: living fuel (intensity based on fuel level)
        - Red: active fire and heat glow (intensity based on heat level)
        """
        fire = self.state[:, :, B]
        fuel = self.state[:, :, F]
        heat = self.state[:, :, H]
        F_start = self.initial_fuel
        fuel_ratio = fuel / (F_start + 1e-6)

        # Initialize to rock baseline (Beige-ish dark yellow)
        rgb = np.full((self.state.shape[0], self.state.shape[1], 3), [0.3, 0.3, 0.0])

        if self.water_mask is not None:
            rgb[self.water_mask > 0.0] = [0.0, 0.0, 1.0]  # Blue for water

        # black = burnt areas
        burnt_mask = (fuel_ratio <= self.extinction_fuel_ratio) & (F_start > 0.2)
        rgb[burnt_mask] = [0, 0, 0]
        
        # green = living fuel
        living_mask = (fuel_ratio > self.extinction_fuel_ratio) & (F_start > 0.2)
        rgb[living_mask, 1] = fuel[living_mask]

        # red = fire + heat glow
        rgb[:, :, 0] += fire #+ 0.4 * heat
        rgb[:, :, 1] += 0.2 * fire

        return np.clip(rgb, 0, 1)
    

    def run_simulation(self, timesteps: int = None, gif_name: str = "fire", visualization: bool = False):
        """ Run the fire spreading simulation for a specific amount of timesteps and save the result as a GIF. """
        if timesteps > self.timesteps:
            print(f"Warning: timesteps {timesteps} exceeds fire spreading simulation, "
                  f"simulationg {self.timesteps} timesteps instead")
            timesteps = self.timesteps
        elif timesteps is None:
            timesteps = self.timesteps

        if visualization:
            fig, ax = plt.subplots()
            img = ax.imshow(self._make_rgb())
            frames = [[img]]

        for t in range(timesteps):
            self._diffuse(t)
            self._burning_exp()

            if visualization:
                img_data = ax.imshow(self._make_rgb(), animated=True)
                frames.append([img_data])
        
        if visualization:
            ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)
            ani.save(gif_name + ".gif", writer="pillow")
            plt.close(fig)


    def calculate_simulation_burned_mask(self):
        """
        check is a cell is water
        if fuel ratio is lower than extinction_fuel_ratio, the cell is considered burned
        """
        fuel = self.state[:, :, F]
        fuel_ratio = fuel / (self.initial_fuel + 1e-6)  # Avoid division by zero
        has_fuel = self.initial_fuel > 1e-6
        
        burned_mask = has_fuel & ((self.state[:, :, B] == 1) | (fuel_ratio <= self.extinction_fuel_ratio))
        return burned_mask
