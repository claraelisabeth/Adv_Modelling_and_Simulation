# Advanced Modelling of Fire Spreading

Clara Pichler 1191769, 
Name Lastname MN, 
Name Lastname MN, and
Name Lastname MN


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
python3 main.py
```

## Data


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





## Sources

This can be written nicer later on

[1] https://arxiv.org/abs/cond-mat/0202022

[2]

[3]

[4]

[5]

[6]










