![logo](https://github.com/stavrosgreece/EVsSimulator/assets/17108978/4852f3c4-3371-4561-8e8f-661acbaf2a01)

# EVsSimulator
A V2X Simulation Environment for large scale EV charging optimization

## Use-Cases
- [ ] Public residential chargers Power Management and Transformer capacity management (follow setpoint)
- [ ] Fast chargers capacity management
- [ ] Office building chargers capacity management 100+ EVSEs

### Insights for RL in EVsSimulator
- The state should be normalized [0,1] for better convergence
- The reward should be either positive or negative so that policy loss converges
- ...

### Development of RL algorithms

- [x] Linearize the User Satisfaction constraint as a negative infinite reward.
- [x] Experiment with DDPG for this problem
- [ ] Experiment with different use-cases.
  - [x] Create a pipeline that quickly compares the Developed algorithm(DDPG in this case) with the optimal solution and baseline solution (add new percentage metric for comparison "profits/opt_profits")
  - [ ] Vizuallize the results in the same figures, get statistics, etc.
  - [x] Try to improve as much as possible the 1 cs 150 timesteps problem
  - [x] Develop statistics that can average the performance of the algorithm for multiple evaluation scenarios
  - [~] Then, enable dynamic prices (tested:works for 1 cs but makes learning slower)
  - [x] Then, enable dynamic EV spawn rate (works perfectly--tested for cs<4 )
  - [ ] Then, enable transformer aggregation (new reward should be designed)
  - [ ] Then, enable dynamic transformer power level (because of grid constraints or because of very expensive energy)
  - [~] Then, enable different spawn rates in each test case (under testing now, much harder to have very good solutions with DDPG)
  - [ ] Then, enable heterogeneous EVs and EV chargers 
- [ ] Normalize reward based on the number of EVs in the parking lot (e.g., if there are 100 EVs, the reward should be divided by 100)
- [ ] !!! Apply DT transformer on simple cases 
- [ ] Create a business as usual baseline solution for comparison (check literature)

##### Open Questions for RL
- What is a good metric to evaluate the performance of an online vs an offline policy, e.g., when comparing the RL online actions with the optimal gurobi (complete information) methods?
- How can we associate rewards for specific actions (e.g. when a specific ev leaves without completely charging, it is a fault of a series of actions for charger **i** port **j**)?
  - Maybe, run the algorithm once per charging station and then metaheuristically combine them on the transformer level to solve the problem without power level constraints.

#### General
- [x] Implement results visualizer (potentiall using stacked plot charts maybe like in (https://ev.caltech.edu/assets/images/conventional_v_adaptive.svg))
- [x] Implement the replay_game feature using the saved data from the simulation for actual objective value (using the gurobi optimization model)
- [x] TODO debug the replay_game feature(the spawn of EVS asserion fails assert (self.n_evs_connected < self.n_ports))
- [x] Add more plot visualizations for the simulation about EV such as energy level and  charging cycles, etc.
- [x] Implement seed in the main environment

#### Debugging
Here, I will write down the debugging tasks I am currently working on:
- [x] Why is transformer total power always positive? (changed math model formulation)
- [x] Why does it terminate earlier even if trheshold is 0 (Fixed)
- [x] Reduce the required storage for each run
  - [x] Save only the best model of each run (not all of them)

#### Gym Environment
- [x] Implement step for ev charger and ev
- [x] Implement the get_observation and reward functions
- [x] Implement the Load_Scenario from replay file
- [x] Add power limit to the problem formulation and in the environment
- [x] Add rest period so every EV has time to leave the parking lot (stop spawning EVs after a while)
- [ ] Research about electricity prices and how to include them in the problem formulation
- [ ] Get distributions about EV characteristics (battery size, max charging and discharging power per level--max_current--, etc.)
  - [ ] Each EV has unique battery capacitance (50-100 kWh) and max charging and discharging acceptance ratio (max amps) 
  - [ ] Each charger has unique charging and discharging power depending on the level (1,2, fast charging) and an efficiency factor 95%+ to model power losses.
  - [ ] TODO - Implement both DC and AC chargers and specifications for both.


#### PyOmo/ Gurobi Optimization
- [x] Implement the Gurobi optimization problem formulation
  - [x] Add constraints
  - [x] Add objective function  
- [x] Debug the Gurobi optimization problem formulation
- [x] Save the data from the gym simulation to a file so they can be read 
- [x] Solve optimally using gurbi based on saved track of simulation
- [x] Add power limit to the problem formulation and in the math model
- [x] Debug again using the simulation load
- [x] Final debug with every plot and visualization
 
## TODO long term
- [ ] Add an battery charging model curve in the problem formulation equations (0-80 fast, 80-100 slow)
- [ ] Replace 'EV-Spawner' with realistic EV spawn rate using distributions for different times of the day and days of the week, and staying time
- [ ] Ask about charging and discharging speed of the EVs (how much power can they charge and discharge), from the perspective of EVs and EV chargers.
- [ ] Complete the simulate_grid functionality of the EVsSimulator package
- [ ] Develop a visual interface for charging stations and EVs (with small battery icons for each car showing the level. also showing the connectivity to transformers, etc.)
- [ ] Add a new class called **Buildings** for V2B and V2H scenarios
- [ ] Minimize charging cycles too?
 

## Future Ideas
Here, I will write down abstract ideas about the V2X problem I am trying to solve:
- Use PandaPower to simulate the grid (use it with the power transformer class id characteristics, use the storage class to simulate the chargers fro each node **aggregate many chargers to one storage** per bus)
- Support any kind of  MV or LV network for the PandaPower grid
- Replace PandaPower with PowerFactory simulations for better accuracy
- Discretize action space (-100%, ...-10%,0,10%, ...., 100%) for better convergence
- Think of how to impose the constraints (e.g. the power for empty ev chargers should be zero ,the transformer should not exceed its maximum power)

### Improve Problem Formulation
- [ ] Add the **grid** as a part of the problem formulation
- [x] Add power limit as part of the problem formulation: https://ev.caltech.edu/assets/images/conventional_v_adaptive.svg (the power limit is the maximum power that can be drawn from a group of ev chargers) https://ev.caltech.edu/info
- [x] Add **j number of ports** per charging station and include related constraints
- [x] Add the battery behavior model 0-80 fast, 80-100 slow
- [x] Add battery degradation model (simple -> just include charge cycles, more complex -> include temperature, SOC, etc.)
- [x] Create a highly heterogeneous EV/ EV Charger environment -> Closer to realistic cases (different chargers, different EVs, different parking lots, different buildings, different transformers, different grids) https://site.ieee.org/pes-iss/data-sets/#elec (check the 34 ev types dataset)
- [x] Improve the user satisfaction term

## Limitations
- The transformer power limit makes the problem harder to solve (more constraints) for gurobi
- Real **time-series** grid data are required for the PandaPower grid simulator to work, for the following parts:
    - Power transformer
    - Loads
    - Generators
    - Renewable energy sources

## Datasets
 - EV charging transactions: https://platform.elaad.io/download-data/
 - EV charging prices and transactions [Not free]: https://www.chargeprice.app/
 - Day-ahaed prices 2015-2023 hourly https://ember-climate.org/data/data-tools/europe-power-prices/

## RL Benchmarks
 - DDPG: https://github.com/schneimo/ddpg-pytorch

## URLs

#### Articles
 - PowerFlow Problem formulation: https://invenia.github.io/blog/2020/12/04/pf-intro/
 - PandaPower MV networks: https://pandapower.readthedocs.io/en/v2.1.0/networks/cigre.html
 - PowerFactory API for python: https://thesmartinsights.com/run-digsilent-powerfactory-via-the-python-api-jump-start-to-your-powerfactory-automatization/
 - EV battery and EV chargers characteristics: https://www.ovoenergy.com/guides/energy-guides/ev-charging-speeds-explained-slow-fast-and-rapid
 - EV battery characteristics: https://axlewise.com/ev-car-battery/#:~:text=The%20size%20of%20an%20electric%20car%20battery%20can,depending%20on%20the%20car%E2%80%99s%20make%2C%20model%2C%20and%20year.
 - EV charger manufacturer: https://new.abb.com/ev-charging
 - Useful V2G info: https://www.virta.global/vehicle-to-grid-v2g#:~:text=With%20V2G%20technology%2C%20an%20EV,back%20to%20the%20power%20grid.
 - EV charger 2 port specs: https://cyberswitching.com/product/dual-chargers-on-pedestal/
  #### Code repositories
  - EV simulator and datasets: https://github.com/zach401/acnportal


# Assumptions
Assumptions regarding the EVsSimulator environment:
- Power that is charged and discharged is transferred in the:
    1. charger level 
    2. parking lot/ building level
    3. transformer level
    4. grid level (PandaPower)
- The charging and discharging speed is related to the charger and the EV type (battery size). Otherwise, each EV can have different maximum charging and discharging speeds.
