![logo](https://github.com/stavrosgreece/EVsSimulator/assets/17108978/4852f3c4-3371-4561-8e8f-661acbaf2a01)

# EVsSimulator
A V2X Simulation Environment for large scale EV charging optimization

## TODO short term

#### General
- [x] Implement results visualizer (potentiall using stacked plot charts maybe like in (https://ev.caltech.edu/assets/images/conventional_v_adaptive.svg))
- [x] Implement the replay_game feature using the saved data from the simulation for actual objective value (using the gurobi optimization model)
- [x] TODO debug the replay_game feature(the spawn of EVS asserion fails assert (self.n_evs_connected < self.n_ports))
- [x] Add more plot visualizations for the simulation about EV such as energy level and  charging cycles, etc.
- [ ] Minimize charging cycles too?

#### Debugging
Here, I will write down the debugging tasks I am currently working on:
- [x] Why is transformer total power always positive? (changed math model formulation)

#### Gym Environment
- [x] Implement step for ev charger and ev
- [x] Implement the get_observation and reward functions
- [x] Implement the Load_Scenario from replay file
- [x] Add power limit to the problem formulation and in the environment
- [x] Add rest period so every EV has time to leave the parking lot (stop spawning EVs after a while)
- [ ] Research about electricity prices and how to include them in the problem formulation


#### PyOmo Optimization
- [x] Implement the Gurobi optimization problem formulation
  - [x] Add constraints
  - [x] Add objective function  
- [x] Debug the Gurobi optimization problem formulation
- [x] Save the data from the gym simulation to a file so they can be read 
- [x] Solve optimally using gurbi based on saved track of simulation
- [x] Add power limit to the problem formulation and in the math model
- [x] Debug again using the simulation load
- [ ] Final debug with every plot and visualization
 
## TODO long term
- [ ] Add an battery charging model curve in the problem formulation equations (0-80 fast, 80-100 slow)
- [ ] Replace 'EV-Spawner' with realistic EV spawn rate using distributions for different times of the day and days of the week, and staying time
- [ ] Ask about charging and discharging speed of the EVs (how much power can they charge and discharge), from the perspective of EVs and EV chargers.
- [ ] Complete the simulate_grid functionality of the EVsSimulator package
- [ ] Develop a visual interface for charging stations and EVs (with small battery icons for each car showing the level. also showing the connectivity to transformers, etc.)
- [ ] Add a new class called **Buildings** for V2B and V2H scenarios

 

## Future Ideas
Here, I will write down abstract ideas about the V2X problem I am trying to solve:
- Use PandaPower to simulate the grid (use it with the power transformer class id characteristics, use the storage class to simulate the chargers fro each node **aggregate many chargers to one storage** per bus)
- Support any kind of  MV or LV network for the PandaPower grid
- Replace PandaPower with PowerFactory simulations for better accuracy
- Discretize action space (-100%, ...-10%,0,10%, ...., 100%) for better convergence
- Think of how to impose the constraints (e.g. the power for empty ev chargers should be zero ,the transformer should not exceed its maximum power)

### Improve Problem Formulation
- [ ] Add the **grid** as a part of the problem formulation
- [x] Add power limit as part fo the problem formulation: https://ev.caltech.edu/assets/images/conventional_v_adaptive.svg (the power limit is the maximum power that can be drawn from a group of ev chargers) https://ev.caltech.edu/info
- [x] Add **j number of ports** per charging station and include related constraints
- [ ] Add the battery behavior model 0-80 fast, 80-100 slow
- [ ] Add battery degradation model (simple -> just include charge cycles, more complex -> include temperature, SOC, etc.)
- [ ] Create a highly heterogeneous EV/ EV Chager environment -> Closer to realistic cases (different chargers, different EVs, different parking lots, different buildings, different transformers, different grids)
- [ ] Improve the user satisfaction term

## Limitations
- The transformer power limit makes the problem harder to solve (more constraints) for gurobi
- Real **time-series** grid data are required for the PandaPower grid simulator to work, for the following parts:
    - Power transformer
    - Loads
    - Generators
    - Renewable energy sources

## Datasets
 - EV charging transactions: https://platform.elaad.io/download-data/
 - EV charing prices and transactions [Not free]: https://www.chargeprice.app/

## URLs

#### Articles
 - PowerFlow Problem formulation: https://invenia.github.io/blog/2020/12/04/pf-intro/
 - PandaPower MV networks: https://pandapower.readthedocs.io/en/v2.1.0/networks/cigre.html
 - PowerFactory API for python: https://thesmartinsights.com/run-digsilent-powerfactory-via-the-python-api-jump-start-to-your-powerfactory-automatization/
 - EV battery and EV chargers characteristics: https://www.ovoenergy.com/guides/energy-guides/ev-charging-speeds-explained-slow-fast-and-rapid
 - EV battery characteristics: https://axlewise.com/ev-car-battery/#:~:text=The%20size%20of%20an%20electric%20car%20battery%20can,depending%20on%20the%20car%E2%80%99s%20make%2C%20model%2C%20and%20year.
 - EV charger manufacturer: https://new.abb.com/ev-charging
  
  #### Code repositories
  - EV simulator and datasets: https://github.com/zach401/acnportal


# Assumptions
Assumptions regarding the EVsSimulator environment:
- Power that is charged and discharged is tranferred in the:
    1. charger level 
    2. parking lot/ building level
    3. transformer level
    4. grid level (PandaPower)
- The charging and discharging speed is related to the charger and the EV type (battery size). Otherwise, each EV can have different maximum charging and discharging speeds.
