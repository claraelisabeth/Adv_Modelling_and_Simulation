# Advanced Modelling of Fire Spreading

Clara Pichler 1191769,   
Hannah Pernthaner 11920573,   
Hanno Bösch 11827161, and  
Ambrus Toth 12450771


## Motivation

Wildfire spread is driven by the interaction of fuel, oxygen, heat and the environment. This project implements a lightweight, grid-based wildfire simulator that extends a lecture-provided 2D cellular automaton with physically motivated diffusion and combustion rules. The model includes wind, terrain slope, moisture and water bodies as environmental drivers, while remaining simple enough to inspect, extend and apply to data-derived landscapes.

The goal is to provide a scalable and interpretable simulation framework for wildfire propagation, and to support experiments that compare unmitigated fire spread with scenarios initialized from real landscape data.

For a more indepth explaination please refer to our report `Adv_FireSpreading_Modelling.pdf`.


## How to run

First one needs to make sure to have the neccessary libraries installed. These are listed in the `requirements.txt` file and can be downloaded with the following command:

```
pip install -r requirements.txt
```
With the following line one can run our simulation (example)
```
python3 -m test.creating_simulations
```









