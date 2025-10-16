
# EV2Gym: A Realistic EV-V2G-Gym Simulator for EV Smart Charging

<div align="center">
<img align="center" src="https://github.com/StavrosOrf/EV2Gym/assets/17108978/86e921ad-d711-4dbb-b7b9-c69dee20da11" width="55%"/>
</div>

[![Python 3.6](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://www.python.org/downloads/release/python-360/) [![PyPI](https://img.shields.io/pypi/v/ev2gym.svg)](https://pypi.org/project/ev2gym/) ![License](https://img.shields.io/github/license/AI4Finance-Foundation/finrl.svg?color=brightgreen)
---

Develop and evaluate **any type of smart charging algorithm**: from simple heuristics, Model Predictive Control, Mathematical Programming, to Reinforcement Learning!

EV2Gym is **fully customizable** and easily **configurable**! Now can also **simulate the grid**, thanks to [RL-ADN](https://github.com/EnergyQuantResearch/RL-ADN)!

The EV2Gym **Paper** can be found at: [link](https://arxiv.org/abs/2404.01849) and [link](https://dl.acm.org/doi/abs/10.1109/TITS.2024.3510945).

**Highly recommended** related works and repositories:
- Physics-Informed Reinforcement Learning for Large-Scale EV Smart Charging Considering Distribution Network Voltage Constraints [link](https://arxiv.org/abs/2510.12335) repository: https://github.com/StavrosOrf/EV2Gym_PI-TD3
- GNN-DT: Graph Neural Network Enhanced Decision Transformer for Efficient Optimization in Dynamic Environments [Paper link](https://arxiv.org/abs/2502.01778) https://github.com/StavrosOrf/DT4EVs
- Scalable Reinforcement Learning for Dynamic Electric Vehicle Charging Optimization Using Graph Neural Networks [Paper link](https://github.com/StavrosOrf/EV-GNN) repo https://github.com/StavrosOrf/EV-GNN
- Open source algorithms for maximizing V2G flexibility based on modelpredictive contro [Paper link](https://www.sciencedirect.com/science/article/pii/S0378779625006704) repo https://github.com/CesarDiazLondono/MPC-G2V-V2G



## Installation

Install the package using pip:
```bash
pip install ev2gym
```

 Run the example code below to get started ...
```python
from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.mpc.V2GProfitMax import V2GProfitMaxOracle
from ev2gym.baselines.heuristics import ChargeAsFastAsPossible

config_file = "ev2gym/example_config_files/V2GProfitPlusLoads.yaml"

# Initialize the environment
env = EV2Gym(config_file=config_file,
              save_replay=True,
              save_plots=True)
state, _ = env.reset()
agent = V2GProfitMaxOracle(env,verbose=True) # optimal solution
#        or 
agent = ChargeAsFastAsPossible() # heuristic
for t in range(env.simulation_length):
    actions = agent.get_action(env) # get action from the agent/ algorithm
    new_state, reward, done, truncated, stats = env.step(actions)  # takes action

```
- ### For Reinforcement Learning:
To train an RL agent, using the [StableBaselines3](https://stable-baselines3.readthedocs.io/en/master/) library, you can use the following code:
```python
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from sb3_contrib import TQC, TRPO, ARS, RecurrentPPO

from ev2gym.models.ev2gym_env import EV2Gym
# Choose a default reward function and state function or create your own!!!
from ev2gym.rl_agent.reward import profit_maximization, SquaredTrackingErrorReward, ProfitMax_TrPenalty_UserIncentives
from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, V2G_profit_max_loads

config_file = "ev2gym/example_config_files/V2GProfitPlusLoads.yaml"
env = gym.make('EV2Gym-v1',
                config_file=config_file,
                reward_function=reward_function,
                state_function=state_function)
# Initialize the RL agent
model = DDPG("MlpPolicy", env)
# Train the agent
model.learn(total_timesteps=1_000_000,
            progress_bar=True)
# Evaluate the agent
env = model.get_env()
obs = env.reset()
stats = []
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    if done:
        stats.append(info)
```
!!! You can develop your own reward and state functions and use them in the environment.


## Table of Contents

- [Installation](#Installation)
- [Overview](#Overview)
- [Configuration File](#Configuration-File)
- [File Structure](#File-Structure)
- [Citing](#Citing-EV2Gym)
- [License](#License)
- [Contributing](#Contributing)

<!-- Bullet points with all the benefits -->
## Overview

![EV2Gym](https://github.com/StavrosOrf/EV2Gym/assets/17108978/4695efa7-5c92-4118-9470-4cd16d262cf9)

- The simulator can be used to evaluate any type of algorithm to gain insights into its efficiency.
- The “gym environment” can readily support the development of RL algorithms.
- Replays of simulations are saved and can be solved optimally using the Gurobi Solver.
- Easy to incorporate additional functionality for any use-case.
- Now, also simulates grid for voltage magnitude!
- The number and the topology of Transformers, Charging stations, and Electric Vehicles are parameterizable.
- The user can import custom data.
- Uses only open-source data:
  - EV spawn rate, time of stay, and energy required are based on realistic probability distributions *ElaadNL* conditioned on time, day, month and year.
  - *Pecan Street* data is used for the load profiles.
  - *Renewables Ninja* data is used for the PV generation profiles.
  - EV and Charger characteristics are based on real EVs and chargers existing in NL (*RVO Survey*).
  - Charging/ Discharging prices are based on historical day-ahead prices from *ENTSO-e*.

Focused on **realistic** parameters and **fully customizable**:

- **Power Transformer** model:
  - Max Power Limit
  - Inflexible Loads, PV, Capacity Reduction events
- **Charging Stations** model:
  - Min and Max charge/discharge power/ Current
  - Voltage and phases, AC or DC
  - List of connected transformers
- **Electric Vehicle** model:
  - Connected charging station and port
  - Min and Max battery energy level
  - Time of arrival and departure
  - Energy at arrival/ desired energy at departure
  - Min and Max power levels
  - Charge and discharge efficiency
  - Constant-Current/ Constant-Voltage load-curve option
- **Battery Degradation** model:
  - Cyclic aging
  - Calendar aging


<div align="center">
<img align="center" src="https://github.com/StavrosOrf/EV2Gym/assets/17108978/d15d258c-b454-498c-ba7f-634d858df3a6" width="90%"/>
</div>

An EV2Gym simulation comprises three phases: the configuration phase, which initializes the models; the simulation phase, which spans $T$ steps, during which the state of models like EVs and charging stations is updated according to the decision-making algorithm; and finally, in the last phase, the simulator generates evaluation metrics for comparisons, produces replay files for reproducibility, and generates real-time renders for evaluation.

## Configuration File

The configuration file is used to set the parameters of the simulation. The configuration file is a YAML file that contains the following parameters:
```yaml
# This yml file is used to configure the evsim simulation

##############################################################################
# Simulation Parameters
##############################################################################
timescale: 15 # in minutes per step
simulation_length: 96 #90 # in steps per simulation

##############################################################################
# Date and Time
##############################################################################
# Year, month, 
year: 2022 # 2015-2023
month: 1 # 1-12
day: 17 # 1-31
# Whether to get a random date every time the environment is reset
random_day: True # True or False
random_hour: False # True or False

# Simulation Starting Time
# Hour and minute do not change after the environment has been reset
hour: 5 # Simulation starting hour (24 hour format)
minute: 0 # Simulation starting minute (0-59)

# Simulate weekdays, weekends, or both
simulation_days: weekdays # weekdays, weekends, or both

# EV Spawn Behavior
scenario: public # public, private, or workplace
spawn_multiplier: 5 # 1 is default, the higher the number the more EVs spawn. Play somewhere between 3-7. With 1 often only 1/5 EVs show up.

##############################################################################
# Prices
##############################################################################
discharge_price_factor: 1 # how many times more abs(expensive/cheaper) it is to discharge than to charge. Are similar, discharging cost could be a bit higher.

##############################################################################
# Distribution Network Settings
##############################################################################
v2g_enabled: True # True or False
number_of_charging_stations: 150 # Play somewhere 5-20
number_of_transformers: -1
number_of_ports_per_cs: 1
# Provide path if you want to load a specific charging topology
# *(where chargers are located and what are their characteristics),
# else write None for a randomized one with the above parameters
charging_network_topology: None #./config_files/charging_topology_10.json

simulate_grid: True # True or False
network_info: 
  vm_pu: 1.0 # Voltage magnitude in per unit
  s_base: 1000 # Base power in kVA
  load_multiplier: 1 # Load multiplier for the grid
  pv_scale: 80 # Percentage% of maximum PV power with respect to the total load of the bus
  bus_info_file: './ev2gym/data/network_data/node_34/Nodes_34.csv'
  branch_info_file: './ev2gym/data/network_data/node_34/Lines_34.csv'
  # bus_info_file: './ev2gym/data/network_data/node_123/Nodes_123.csv'
  # branch_info_file: './ev2gym/data/network_data/node_123/Lines_123.csv'

pf_solver: 'Laurent' # 'Laurent' or 'PandaPower'

##############################################################################
# Power Setpoints Settings
##############################################################################
# How much the power setpoints can vary in percentage compared to the nominal power
# The higher the number the easier it is to meet the power setpoints, the opposite for negative numbers
power_setpoint_enabled: True # True or False
power_setpoint_flexiblity: 80 # (in percentage +/- %)

##############################################################################
# Inflexible Loads, Solar Generation, and Demand Response (Not compatible with simulate_grid = True)
##############################################################################
# Whether to include inflexible loads in the transformer power limit, such as residential loads
tr_seed: -1 # seed for the random number generator of transformer loads(-1 for random seed)

inflexible_loads: # Offices, homes
  include: False # True or False
  inflexible_loads_capacity_multiplier_mean: 1 # 1 is default, the higher the number the more inflexible loads
  forecast_mean: 30 # in percentage of load at time t%
  forecast_std: 5 # in percentage of load at time t%

# PV solar Power
solar_power:
  include: False # True or False
  solar_power_capacity_multiplier_mean: 1 # 1 is default, the higher the number the more solar power
  forecast_mean: 20 # in percentage of load at time t%
  forecast_std: 5 # in percentage of load at time t%

# Whether to include demand response in the transformer power limit
demand_response:
  include: False # True or False
  events_per_day: 1
  #How much of the transformer power limit can be used for demand response
  event_capacity_percentage_mean: 35 # (in percentage +/- %) reduction
  event_capacity_percentage_std: 5 # (in percentage +/- %)
  event_length_minutes_min: 60
  event_length_minutes_max: 60
  event_start_hour_mean: 12
  event_start_hour_std: 2
  # How many minutes ahead we know the event is going to happen
  notification_of_event_minutes: 60

##############################################################################
# EV Specifications
##############################################################################
heterogeneous_ev_specs: False #if False, each EV has the same specifications
# such as battery capacity, charging rate, etc.
ev_specs_file: ./ev2gym/data/ev_specs_v2g_enabled2024.json # path to the file with the EV specifications

##############################################################################
# Default Model values
##############################################################################
# These values are used if not using a charging network topology file or 
# if the EV specifications are not provided

# Default Transformer model
transformer:
  max_power: 200 # in kW

# Default Charging Station model
charging_station:  
  min_charge_current: 0 # Amperes
  max_charge_current: 32 # Amperes
  min_discharge_current: 0 # Amperes (actual value <=0)
  max_discharge_current: -32 # Amperes (actual value <=0)
  voltage: 400 # Volts
  phases: 3 # 1,2, or 3

# Default EV model
ev:
  #The following values are used if "heterogeneous_ev_specs = False"
  battery_capacity: 70 # in kWh
  max_ac_charge_power: 22 # in kW
  min_ac_charge_power: 0 # in kW
  max_dc_charge_power: 50 # in kW
  max_discharge_power: -22 # in kW
  min_discharge_power: 0 # in kW
  ev_phases: 3  
  charge_efficiency: 1 # 0-1 (0% - 100%)
  discharge_efficiency: 1 # 0-1 (0% - 100%)
  transition_soc: 1 # 0-1 (0% - 100%)

  #The following values are also used if "heterogeneous_ev_specs = True"
  min_battery_capacity: 15 # in kWh
  min_time_of_stay: 200 # in minutes
  min_emergency_battery_capacity: 15 # in kWh
  desired_capacity: 1 # in (0-1) (0% - 100%) #Keep at 100% for now
  #if trasition_soc is < 1, the curve of the line is affected by:
  transition_soc_multiplier: 50 # default 1 (the higher the number the shorter the effect of CCCV region)
```

## File Structure
The file structure of the EV2Gym package is as follows:
```bash
├── ev2gym
│   ├── baselines
│   │   ├── gurobi_models/
│   │   ├── mpc/
│   │   ├── heuristics.py
│   ├── data/
│   ├── models
│   │   ├── ev2gym_env.py
│   │   ├── ev.py
│   │   ├── transformer.py
│   │   ├── ev_charger.py
│   │   ├── replay.py
│   │   ├── grid.py
│   ├── rl_agent
│   │   ├── reward.py
│   │   ├── state.py
│   ├── utilities
│   │   ├── loaders.py
│   │   ├── utils.py
│   │   ├── arg_parser.py
│   ├── example_config_files
│   │   ├── BusinessPST.yaml
│   │   ├── PublicPST.yaml
│   │   ├── V2GProfitPlusLoads.yaml
│   ├── visuals
│   │   ├── plots.py
│   │   ├── renderer.py
│   ├── scripts/
```

Class Diagram of the EV2Gym Environment:
<div align="center">
<img align="center" src="https://github.com/StavrosOrf/EV2Gym/assets/17108978/8ca5bf11-6ed4-44f6-9faf-386382609af1" width="55%"/>
</div>

## Citing EV2Gym

If you use this code in your research, please cite as:
```bibtex
@ARTICLE{10803908,
  author={Orfanoudakis, Stavros and Diaz-Londono, Cesar and Emre Yılmaz, Yunus and Palensky, Peter and Vergara, Pedro P.},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={EV2Gym: A Flexible V2G Simulator for EV Smart Charging Research and Benchmarking}, 
  year={2025},
  volume={26},
  number={2},
  pages={2410-2421},
  keywords={Vehicle-to-grid;Smart charging;Optimization;Benchmark testing;Batteries;Data models;Schedules;Reinforcement learning;Prediction algorithms;Power transformers;Electric vehicle optimization;gym environment;reinforcement learning;mathematical programming;model predictive control (MPC)},
  doi={10.1109/TITS.2024.3510945}}
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.


## Contributing

EV2Gym is an open-source project and welcomes contributions! Please get in contact with us if you would like to discuss about the simulator.


