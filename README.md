# Advanced Modelling of Fire Spreading

Clara Pichler 1191769, 
Hannah Pernthaner 11920573, 
Hanno Bösch 11827161, and
Ambrus Toth 12450771


## Motivation

(this has to be changed)   
Fire only starts when three essential elements come together: sufficient **fuel**, sufficient **oxygen**, and a sufficiently high **temperature**. If one of these elements is missing, for example due to a lack of oxygen or insufficient heat, a fire will not survive or will not even start.

The goal is to simulate a fire spreading over a grid depending on these three elements and states of the neighbouring cells.

Key questions:
- How long does the raging fire burn for?
- Has it consumed all available material?
- How much damage did it cause?


## How to run

First one needs to make sure to have the neccessary libraries installed. These are listed in the `requirements.txt` file and can be downloaded with the following command:

```
pip install -r requirements.txt
```
With the following line one can run our simulation (example)
```
python3 -m test.creating_simulations
```


## Repository Structure



## Model

We consider a **2D cellular automaton** with rectangularly aligned cells.  
The index $(i, j)$ maps directly to a quadratic grid-cell in a 2D landscape.

The state of a cell $\vec{S}_{i,j}(t)$ is four-dimensional:

1. **Heat** $H_{i,j}(t) \in [0, \text{maxh}]$  
2. **Fuel** $F_{i,j}(t) \in [0, \text{maxf}]$  
3. **Oxygen** $O_{i,j}(t) \in [0, \text{maxo}]$  
4. **Fire state** $B_{i,j}(t) \in \{0,1\}$ (burning or not)


The model updates in two phases **diffusion phase** and **burning phase**.

### Diffusion Phase

In this phase, **heat** and **oxygen** diffuse across neighboring cells.

We define two vectors:

- $\vec{\mu}_O$ for oxygen
- $\vec{\mu}_H$ for heat  

Each contains 5 values (center + 4 neighbors), summing to 1.

$$
O_{i,j}(t') =
\mu^O_1 O_{i,j}(t) +
\mu^O_2 O_{i-1,j}(t) +
\mu^O_3 O_{i+1,j}(t) +
\mu^O_4 O_{i,j-1}(t) +
\mu^O_5 O_{i,j+1}(t)
$$

$$
H_{i,j}(t') =
\mu^H_1 H_{i,j}(t) +
\mu^H_2 H_{i-1,j}(t) +
\mu^H_3 H_{i+1,j}(t) +
\mu^H_4 H_{i,j-1}(t) +
\mu^H_5 H_{i,j+1}(t)
$$

- $\vec{\mu}_O$ and $\vec{\mu}_H$ can differ.
- $\mu_1$ controls how much stays in the cell vs spreads.
- $\mu_2 \dots \mu_5$ can:
  - be equal $\rightarrow$ uniform spread
  - be biased $\rightarrow$ simulate wind $\vec{w}$
- Parameters may vary over space and time.




### Burning Phase

This phase uses updated values $O(t')$ and $H(t')$.

**Case 1:** Burning Cell ($B_{i,j}(t) = 1$)

1. Consume fuel and oxygen

$$
F_{i,j}(t+1) = \max(0, F_{i,j}(t) - \Delta F)
$$

$$
O_{i,j}(t+1) = \max(0, O_{i,j}(t') - \Delta O)
$$

2. Set heat to maximum

$$
H_{i,j}(t+1) = \text{maxh}
$$

3. Check extinction

- If $F_{i,j}(t+1) = 0$ AND $O_{i,j}(t+1) = 0$:
  - $B_{i,j}(t+1) = 0$

**Case 2:** Non-burning Cell ($B_{i,j}(t) = 0$)

1. Check ignition condition (depends on heat, oxygen, fuel)
2. If ignition occurs:
   - $B_{i,j}(t+1) = 1$
   - $H_{i,j}(t+1) = \text{maxh}$



## Data














## Tasks

### Task 1
- Implement the model
- Test edge cases
- Optimize performance (e.g. matrix ops instead of loops)

### Task 2
- Add a **2D wind vector**
- Verify its influence on spreading

### Task 3
- Build interface for **real landscape data**
- Load & rasterize satellite data
- Estimate initial fuel per cell

### Task 5
- Compare with real wildfire videos/images
- Tune parameters to match behavior
- Possible extensions:
  - ember state
  - cooling effects (water, rain)
  - smoke modeling

### Task 6
- Implement **fire-fighting strategies**

### Task 7
- Validate using **historical wildfire data**
- Compare fire boundaries
- Use weather data (wind, rain)
- Optimize parameters

### Task 8
- Create **fictional case studies**
  - simulate different fire scenarios
  - test intervention strategies



## Notes

Minimize Methods https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

Forest-fire model https://en.wikipedia.org/wiki/Forest-fire_model   
Wikipedia article on forest fire modelling by using the Drossel–Schwabl forest fire model. 

Cellular automata (CA) are discrete, computational models consisting of a regular grid of cells, each in one of a finite number of states (e.g., on/off). They evolve synchronously in discrete time steps based on local rules, where a cell's next state depends on its current state and its neighborhood's states. Following YouTube video explains it well.
https://www.youtube.com/watch?v=DKGodqDs9sA&t=189 


### Questions that arose

1. In the buring phase when considering a cell that is not buring we have to check conditions if the cell ignites. For now we define thse conditions as 
    ```
    self.diff_state[i, j, H] > self.ignition_temp and
    self.state[i, j, F] > self.ignition_fuel and
    self.diff_state[i, j, O] > self.ignition_oxy
    ```
    Citing the fireprotection.net.nz (https://fireprotection.net.nz/Elements-of-the-Fire-Triangle#:~:text=Air%20contains%20about%2021%20percent,%2C%20embers%2C%20etc).
    They write,  
    *Air contains about 21 percent oxygen, and most fires require at least 16 percent oxygen content to burn. Oxygen supports the chemical processes that occur during fire. When fuel burns, it reacts with oxygen from the surrounding air, releasing heat and generating combustion products (gases, smoke, embers, etc.). This process is known as oxidation.*   
    So should we set `ignition_oxy` as $16/21=0.76$?  
    What about `ignition_temp` or `ignition_fuel`???

    Ignition fuel should probably be somewhere around here:
    *Values close to zero (-0.1 to 0.1) generally correspond to barren areas of rock, sand, or snow. Low, positive values represent shrub and grassland (approximately 0.2 to 0.4), while high values indicate temperate and tropical rainforests (values approaching 1). It is a good proxy for live green vegetation*
    https://custom-scripts.sentinel-hub.com/custom-scripts/hls/ndvi/

    Ignition temp dependent on type of fuel?
    Also should we consider random ignition? Randomness still depending on the fire triangle..

2. wind dynamics... influence on different parameters or only mu

3. extinguish if F == 0 AND or OR O == 0 ?

4. dF und dO how to set? (videos?? - task 5)
    multiplication factor... exponential decay - up to us

5. Parameter units? T in ..., cell size... zur zeit 88m pro pixel

6. 





## Sources

This can be written nicer later on

[1] https://arxiv.org/abs/cond-mat/0202022

[2] https://custom-scripts.sentinel-hub.com/custom-scripts/hls/ndvi/

[3] https://custom-scripts.sentinel-hub.com/custom-scripts/planet/planetscope/ndwi/

[4] 

[5]

[6]










