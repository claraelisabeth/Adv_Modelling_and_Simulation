import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

H, F, O, B = 0, 1, 2, 3

class FireSpreadingAdvanced:
    '''
    Class for....
    '''
    def __init__(self, n, m, max_H, max_F, max_O, mu_H, mu_O, dF, dO, ignition_temp=0.3, ignition_oxy = 0.76, ignition_fuel = 0.01,  wind = [0.5,0.5], start_cells=[(0,0)], random_F=False, fuel_mask=None, water_mask=None):
        self.n = n
        self.m = m
        self.max_H = max_H
        self.max_F = max_F
        self.max_O = max_O
        self.mu_H = mu_H
        self.mu_O = mu_O
        self.water_mask = water_mask
        self.fuel_mask = fuel_mask

        if (sum(self.mu_O) != 1) or (sum(self.mu_H) != 1):
            print("Warning: The sum over the vector entries must be equal to one.")

        self.wind = wind

        self.dF = dF
        self.dO = dO
        self.ignition_temp = ignition_temp
        self.ignition_oxy = ignition_oxy
        self.ignition_fuel = ignition_fuel

        self.state = np.zeros((n, m, 4))
 
        self.state[:, :, O] = self.max_O   

        for cell in start_cells:
            self.state[cell[0], cell[1], H] = self.max_H
            self.state[cell[0], cell[1], B] = 1
        
        if fuel_mask is not None:
            self.state[:, :, F] = np.maximum(0, fuel_mask)
            if water_mask is not None:
                self.state[:, :, F][water_mask > 0.1] = 0
        elif random_F:
            raw = np.random.uniform(0, 1, (n, m))

            for _ in range(2):
                raw = (
                    raw +
                    np.roll(raw, 1, axis=0) +
                    np.roll(raw, -1, axis=0) +
                    np.roll(raw, 1, axis=1) +
                    np.roll(raw, -1, axis=1)
                ) / 5

            self.state[:, :, F] = self.max_F * raw
        else:
            self.state[:, :, F] = self.max_F

        self.diff_state= np.copy(self.state)



    def compute_mu_with_wind(self, wx, wy, base_mu):
        '''
        Diffusion Phase
        bla bla
        Output:....
        '''
        base = np.array(base_mu, dtype=float)

        # scalar wind -> return a 1D mu vector (length 5)
        if np.isscalar(wx) or np.isscalar(wy):
            mu = base.copy()
            mu[1] *= (1 - wy)  # up
            mu[2] *= (1 + wy)  # down
            mu[3] *= (1 + wx)  # left
            mu[4] *= (1 - wx)  # right

            mu = np.clip(mu, 0, None)
            total = mu.sum()
            if total > 0:
                mu /= total
            return mu

        # per-cell wind field: expect wx, wy arrays of shape (n, m)
        wx_arr = np.array(wx)
        wy_arr = np.array(wy)
        if wx_arr.shape != wy_arr.shape:
            raise ValueError("wx and wy must have same shape for vector wind field")

        n, m = wx_arr.shape
        mu_arr = np.tile(base[:, None, None], (1, n, m))

        mu_arr[1, :, :] *= (1 - wy_arr)
        mu_arr[2, :, :] *= (1 + wy_arr)
        mu_arr[3, :, :] *= (1 + wx_arr)
        mu_arr[4, :, :] *= (1 - wx_arr)

        mu_arr = np.clip(mu_arr, 0, None)
        sum_arr = mu_arr.sum(axis=0)
        zero_mask = (sum_arr == 0)
        sum_arr[zero_mask] = 1.0
        mu_arr /= sum_arr

        return mu_arr
    



    def diffuse(self):
        diff_state = np.copy(self.state)

        H_state = self.state[:, :, H]
        O_state = self.state[:, :, O]

        if isinstance(self.wind, (tuple, list)):
            wx, wy = self.wind
        else:
            wx = self.wind[:, :, 0]
            wy = self.wind[:, :, 1]

        mu_H = self.compute_mu_with_wind(wx, wy, self.mu_H)
        mu_O = self.compute_mu_with_wind(wx, wy, self.mu_O)
        
        H_pad = np.pad(H_state, 1, mode='edge')
        O_pad = np.pad(O_state, 1, mode='edge')

        H_up, H_down, H_left, H_right = (
            H_pad[:-2, 1:-1],
            H_pad[2:, 1:-1],
            H_pad[1:-1, :-2],
            H_pad[1:-1, 2:]
        )

        O_up, O_down, O_left, O_right = (
            O_pad[:-2, 1:-1],
            O_pad[2:, 1:-1],
            O_pad[1:-1, :-2],
            O_pad[1:-1, 2:]
        )

        H_center = H_state
        O_center = O_state

        # mu may be either shape (5,) or (5, n, m)
        if mu_H.ndim == 1:
            diff_state[:, :, H] = (
                mu_H[0] * H_center +
                mu_H[1] * H_up +
                mu_H[2] * H_down +
                mu_H[3] * H_left +
                mu_H[4] * H_right
            )
        else:
            diff_state[:, :, H] = (
                mu_H[0, :, :] * H_center +
                mu_H[1, :, :] * H_up +
                mu_H[2, :, :] * H_down +
                mu_H[3, :, :] * H_left +
                mu_H[4, :, :] * H_right
            )

        if mu_O.ndim == 1:
            diff_state[:, :, O] = (
                mu_O[0] * O_center +
                mu_O[1] * O_up +
                mu_O[2] * O_down +
                mu_O[3] * O_left +
                mu_O[4] * O_right
            )
        else:
            diff_state[:, :, O] = (
                mu_O[0, :, :] * O_center +
                mu_O[1, :, :] * O_up +
                mu_O[2, :, :] * O_down +
                mu_O[3, :, :] * O_left +
                mu_O[4, :, :] * O_right
            )
        self.diff_state = diff_state




    def burning(self):

        state_new = np.copy(self.state)

        F_state = self.state[:, :, F]

        H_diff = self.diff_state[:, :, H]
        O_diff = self.diff_state[:, :, O]

        burning = (self.state[:, :, B] == 1) # burning cells

        # 1. decrement the fuel and oxygen level
        state_new[:, :, F][burning] = np.maximum(0, F_state[burning] - self.dF)
        state_new[:, :, O][burning] = np.maximum(0, O_diff[burning] - self.dO)

        # 2. set the heat level to maximum heat
        state_new[:, :, H][burning] = self.max_H

        # 3. check if the fire is extinguished
        extinguish = (state_new[:, :, F] == 0) | (state_new[:, :, O] == 0)
        state_new[:, :, B][burning & extinguish] = 0

        
        not_burning = ~burning # non burning cells

        # check conditions if the cell ignites, if the cell ignites set B to 1 and H to max_H
        # conditions:
        if self.water_mask is not None:
            ndwi_val = self.water_mask
            dynamic_ignition_temp = self.ignition_temp * (1.0 + ndwi_val)
        else:
            dynamic_ignition_temp = self.ignition_temp
        ignite = (
            (H_diff > dynamic_ignition_temp) &
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
        state_new[:, :, O][not_burning & ~ignite] = O_diff[not_burning & ~ignite]

        self.state = state_new




    
    def make_rgb(self):
        fire = self.state[:, :, B]
        fuel = self.state[:, :, F]
        heat = self.state[:, :, H]

        rgb = np.zeros((self.state.shape[0], self.state.shape[1], 3))

        # red = fire + heat glow
        rgb[:, :, 0] = fire + 0.5 * heat

        # green = fuel
        rgb[:, :, 1] = fuel

        # blue = water or nothing
        if self.water_mask is not None:
            rgb[:, :, 2][self.water_mask > 0.5] = 1.0
        else:
            rgb[:, :, 2] = 0

        return np.clip(rgb, 0, 1)




    def run_simulation(self, T, gif_name = "fire"):

        fig, ax = plt.subplots()
        img = ax.imshow(self.make_rgb())

        frames = []

        for _ in range(T):
            #self.diffuse_loop()
            #self.burning_loop()

            self.diffuse()
            self.burning()

            frame = [ax.imshow(self.make_rgb(), animated=True)]
            frames.append(frame)

        ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)

        ani.save(gif_name+".gif", writer="pillow")
        



    