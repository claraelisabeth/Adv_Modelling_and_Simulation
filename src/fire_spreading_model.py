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
    k_slope : float
        Scaling factor controlling the influence of terrain slope on diffusion. Default is 0.1.
    wind_strength_factor : float
        Scaling factor controlling the influence of wind on diffusion. Default is 0, keep between 0.0 and 1.0
    """
    n : int
    m : int
    mu_H : Union[float, Sequence[float]]
    mu_O : Union[float, Sequence[float]]
    dF : float
    dO : float
    dW : float
    ignition_temp : float = 0.3
    ignition_oxy : float = 0.76
    ignition_fuel : float = 0.3
    extinction_fuel_ratio : float = 0.15
    extinction_oxy : float = 0.05
    wind : tuple[float, float] = (0.0, 0.0)
    start_cells : list[tuple[int, int]] = None
    random_F : bool = False
    fuel_mask : np.ndarray = None
    water_mask : np.ndarray = None
    moisture_mask : np.ndarray = None
    topo_mask : np.ndarray = None
    k_slope : float = 0.1
    wind_strength_factor : float = 0

    def __post_init__(self):
        if self.start_cells is None:
            self.start_cells = [(0,0)]


class FireSpreadingAdvanced:
    """
    Simulates wildfire spreading on a 2D grid using diffusion and combustion dynamics, including the effects of wind,
    terrain slope, and moisture.
    """

    def __init__(self, param: Parameters):
        self.n = param.n
        self.m = param.m
        self.max_H = 1.0
        self.max_F = 1.0
        self.max_O = 1.0
        self.mu_H = self._build_mu(param.mu_H)
        self.mu_O = self._build_mu(param.mu_O)

        if (sum(self.mu_O) != 1) or (sum(self.mu_H) != 1):
            print("Warning: The sum over the vector entries must be equal to one.")

        self.dW = param.dW
        self.water_mask = param.water_mask
        self.fuel_mask = param.fuel_mask
        self.moisture_mask = param.moisture_mask
        self.topo_mask = param.topo_mask
        self.wind_strength_factor = param.wind_strength_factor
        self.k_slope = param.k_slope
        self.wind = param.wind
        self.dF = param.dF
        self.dO = param.dO
        self.ignition_temp = param.ignition_temp
        self.ignition_oxy = param.ignition_oxy
        self.ignition_fuel = param.ignition_fuel
        self.extinction_fuel_ratio = param.extinction_fuel_ratio
        self.extinction_oxy = param.extinction_oxy

        # 5-layer state setup: H, F, O, B, W
        self.state = np.zeros((self.n, self.m, 5))  
        self.state[:, :, O] = self.max_O   

        for cell in param.start_cells:
            self.state[cell[0], cell[1], H] = self.max_H
            self.state[cell[0], cell[1], B] = 1
        
        if param.fuel_mask is not None:
            self.state[:, :, F] = np.maximum(0, param.fuel_mask).copy()
        elif param.random_F:
            raw = np.random.uniform(0, 1, (param.n, param.m))
            for _ in range(2):
                raw = (raw + np.roll(raw, 1, axis=0) + np.roll(raw, -1, axis=0) +
                       np.roll(raw, 1, axis=1) + np.roll(raw, -1, axis=1)) / 5
            self.state[:, :, F] = self.max_F * raw
        else:
            self.state[:, :, F] = self.max_F

        if param.water_mask is not None:
            self.state[:, :, F][param.water_mask > 0.5] = 0.0

        self.initial_fuel = self.state[:, :, F].copy()
        self.diff_state = np.copy(self.state)

        if param.moisture_mask is not None:
            self.state[:, :, W] = param.moisture_mask


    def _build_mu(self, mu):
        """
        Build symmetric diffusion kernel.  If mu is a scalar, it is interpreted as the center retention
        coefficient and the remaining weight is distributed equally
        among the four neighboring cells.

        Parameters
        ----------
        mu : float or array-like of length 5

        Returns
        -------
        list or array
            A 5-element list or array of diffusion coefficients corresponding to
            [center, up, down, left, right].
        """
        if np.isscalar(mu):
            mu_neighbor = (1 - mu) / 4
            return [mu, mu_neighbor, mu_neighbor, mu_neighbor, mu_neighbor]
        return mu


    def _compute_mu_with_wind_slope(self, wx, wy, base_mu):
        '''
        Compute diffusion coefficients with wind and slope effects for each cell.

        Parameters
        ----------
        wx, wy: Wind vector
        sx, sy: Slope gradients from topo_mask (taken from self.topo_mask)

        Returns
        -------
        array
            A 5-element array of diffusion coefficients corresponding to
            [center, up, down, left, right].
        '''
        # Convert base wind velocities to full float arrays immediately
        # wind effects: positive wx pushes fire to the right (higher column index), positive wy pushes fire down (higher row index)
        wx_arr = np.full((self.n, self.m), wx * self.wind_strength_factor, dtype=float)
        wy_arr = np.full((self.n, self.m), wy * self.wind_strength_factor, dtype=float)

        # slope effect
        # sy is change along axis 0 (rows), sx is change along axis 1 (columns)
        if self.topo_mask is not None:
            sy, sx = np.gradient(self.topo_mask)
            wx_arr += np.tanh(sx) * self.k_slope
            wy_arr += np.tanh(sy) * self.k_slope
        
        base = np.array(base_mu, dtype=float)
        mu_arr = np.tile(base[:, None, None], (1, self.n, self.m))

        mu_arr[1] *= (1 + wy_arr)  # up (lower row index)
        mu_arr[2] *= (1 - wy_arr)  # down (higher row index)
        mu_arr[3] *= (1 + wx_arr)  # left (lower column index)
        mu_arr[4] *= (1 - wx_arr)  # right (higher column index)

        # normalization boundary protection
        mu_arr = np.clip(mu_arr, 0, None)
        sum_arr = mu_arr.sum(axis=0)
        sum_arr[sum_arr == 0] = 1.0
        mu_arr /= sum_arr

        return mu_arr
    


    def _diffuse(self):
        '''
        Compute diffusion for heat and oxygen based on the current state, wind, and slope.
        '''
        diff_state = np.copy(self.state)
        H_state = self.state[:, :, H]
        O_state = self.state[:, :, O]

        wx, wy = self.wind
        mu_H = self._compute_mu_with_wind_slope(wx, wy, self.mu_H)
        mu_O = self._compute_mu_with_wind_slope(wx, wy, self.mu_O)
        
        H_pad = np.pad(H_state, 1, mode='edge')
        O_pad = np.pad(O_state, 1, mode='edge')

        H_up, H_down, H_left, H_right = H_pad[:-2, 1:-1], H_pad[2:, 1:-1], H_pad[1:-1, :-2], H_pad[1:-1, 2:]
        O_up, O_down, O_left, O_right = O_pad[:-2, 1:-1], O_pad[2:, 1:-1], O_pad[1:-1, :-2], O_pad[1:-1, 2:]

        diff_state[:, :, H] = (mu_H[0] * H_state + mu_H[1] * H_up + mu_H[2] * H_down + mu_H[3] * H_left + mu_H[4] * H_right)
        diff_state[:, :, O] = (mu_O[0] * O_state + mu_O[1] * O_up + mu_O[2] * O_down + mu_O[3] * O_left + mu_O[4] * O_right)
        
        self.diff_state = diff_state



    def _burning(self):
        '''
        Update the state of the grid based on burning dynamics, including fuel consumption, 
        heat generation, and fire spread to neighboring cells based on ignition conditions.
        '''
        state_new = np.copy(self.state)
        F_state = self.state[:, :, F]
        F_start = self.initial_fuel
        fuel_ratio = F_state / (F_start + 1e-6) # avoid division by zero

        H_diff = self.diff_state[:, :, H]
        O_diff = self.diff_state[:, :, O]

        burning = (self.state[:, :, B] == 1) # burning cells

        # 1. decrement the fuel and oxygen level
        state_new[:, :, F][burning] = np.maximum(0, F_state[burning] - self.dF)
        state_new[:, :, O][burning] = np.maximum(0, O_diff[burning] - self.dO)

        # 2. set the heat level to maximum heat
        state_new[:, :, H][burning] = self.max_H

        # 3. check if the fire is extinguished
        extinguish = (fuel_ratio < self.extinction_fuel_ratio) | (state_new[:, :, O] < self.extinction_oxy)
        state_new[:, :, B][burning & extinguish] = 0

        
        not_burning = ~burning # non burning cells
        
        # water (inside of plants) has to eveporate = be zero to ignite
        evaporation_mask = (not_burning & (self.state[:, :, W] > 0) & (self.diff_state[:, :, H] > 0))

        # decremnent the water level
        state_new[:, :, W][evaporation_mask] = np.maximum(0, self.state[:, :, W][evaporation_mask] - self.dW)

        # check conditions if the cell ignites, if the cell ignites set B to 1 and H to max_H
        # conditions:
        ignite = (
            (state_new[:, :, W] == 0) & 
            (H_diff > self.ignition_temp) &
            (F_state > self.ignition_fuel) &
            (O_diff > self.ignition_oxy)
        )

        # optional randomness??
        # rand = np.random.rand(*H_val.shape)
        # ignite = ignite & (rand < some_probability)

        state_new[:, :, B][not_burning & ignite] = 1
        state_new[:, :, H][not_burning & ignite] = self.max_H

        # update remaining cells
        state_new[:, :, H][not_burning & ~ignite] = H_diff[not_burning & ~ignite]
        state_new[:, :, O][not_burning] = O_diff[not_burning]

        self.state = state_new


    def _make_rgb(self):
        '''
        Create an RGB image representation of the current state for visualization.

        Color scheme:
        - Beige: rocks or non-burnable areas (default background)
        - Blue: water bodies or high moisture areas
        - Black: burnt areas (low fuel ratio)
        - Green: living fuel (intensity based on fuel level)
        - Red: active fire and heat glow (intensity based on heat level)
        '''
        fire = self.state[:, :, B]
        fuel = self.state[:, :, F]
        heat = self.state[:, :, H]
        F_start = self.initial_fuel
        fuel_ratio = fuel / (F_start + 1e-6)

        # Initialize to rock baseline (Beige-ish dark yellow)
        rgb = np.full((self.state.shape[0], self.state.shape[1], 3), [0.3, 0.3, 0.0])

        if self.water_mask is not None:
            rgb[self.water_mask > 0.5] = [0.0, 0.0, 1.0]  # Blue for water

        # black = burnt areas
        burnt_mask = (fuel_ratio <= self.extinction_fuel_ratio) & (F_start > 0.2)
        rgb[burnt_mask] = [0, 0, 0]
        
        # green = living fuel
        living_mask = (fuel_ratio > self.extinction_fuel_ratio) & (F_start > 0.2)
        rgb[living_mask, 1] = fuel[living_mask]

        # red = fire + heat glow
        rgb[:, :, 0] += fire + 0.4 * heat
        rgb[:, :, 1] += 0.2 * fire

        return np.clip(rgb, 0, 1)
    

    def run_simulation(self, T, gif_name = "fire"):
        '''
        Run the fire spreading simulation for T timesteps and save the result as a GIF.
        '''
        fig, ax = plt.subplots()

        img = ax.imshow(self._make_rgb())
        frames = []

        for _ in range(T):
            self._diffuse()
            self._burning()

            img_data = ax.imshow(self._make_rgb(), animated=True)
            frames.append([img_data])

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
