
# EV2Gym: A realistic V2G Simulator for EV Smart Charging Research

<div align="center">
<img align="center" src="https://github.com/StavrosOrf/EV2Gym/assets/17108978/86e921ad-d711-4dbb-b7b9-c69dee20da11" width="55%"/>
</div>

[![Downloads](https://static.pepy.tech/badge/ev2gym)](https://pepy.tech/project/finrl)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI](https://img.shields.io/pypi/v/ev2gym.svg)](https://pypi.org/project/finrl/)
[![Documentation Status](https://readthedocs.org/projects/finrl/badge/?version=latest)](https://finrl.readthedocs.io/en/latest/?badge=latest)
![License](https://img.shields.io/github/license/AI4Finance-Foundation/finrl.svg?color=brightgreen)
---

##### A realistic V2G Simulation Environment for large scale EV charging optimization! 

Develop and evaluate any type of smart charging algorithm: from simple heuristics, Model Predictive Control, Mathematical Programming, to Reinforcement Learning!



Install the package using pip:
```bash
pip install ev2gym
```

Documentation can be found at: [link](https://ev2gym.readthedocs.io/en/latest/). Run the example code below to get started ...
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

To train an RL agent, you can use the following code:
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
!!! note
    Notice that you can develop your own reward and state functions and use them in the environment. EV2Gym is fully customizable and can be used to evaluate any type of algorithm to gain insights into its efficiency!


## Table of Contents

- [Introduction](#introduction)



![EV2Gym](https://github.com/StavrosOrf/EV2Gym/assets/17108978/4695efa7-5c92-4118-9470-4cd16d262cf9)

<!-- Bullet points with all the benefits -->
## Important Features

- The simulator can be used to evaluate any type of algorithm to gain insights into its efficiency.
- The “gym environment” can readily support the development of RL algorithms.
- Uses only open-source data.
- Replays of simulations are saved and can be solved optimally using the Gurobi Solver.
- Easy to incorporate additional functionality for any use-case.
- Does not simulate the grid yet, but groups EV chargers at the level of the transformer/ parking lot, etc, so extra functionality can be easily added.

![Screenshot_2](https://github.com/StavrosOrf/EV2Gym/assets/17108978/d15d258c-b454-498c-ba7f-634d858df3a6)

Focused on **realistic** parameters and **fully customizable**:

- Transformer models
  - Max Current
- Charging Stations models
  - Min and Max charge/discharge power/ Current
  - Voltage and phases, AC or DC
  - Charge and discharge efficiency
  - List of connected transformers
- Electric Vehicles models
  - Connected charging station and port
  - Min and Max battery energy level
  - Time of arrival and departure
  - Energy at arrival/ desired energy at departure
  - Min and Max current /power levels
  - Constant-Current/ Constant-Voltage load-curve option

<img width="405" alt="EV2Gym Class Diagram (1)" src="https://github.com/StavrosOrf/EV2Gym/assets/17108978/8ca5bf11-6ed4-44f6-9faf-386382609af1">

## Data sources

- The number and the topology of Transformers, Charging stations, and Electric Vehicles are parameterizable.
- Charging/ Discharging prices are based on historical day-ahead prices.
- EV spawn rate, time of stay, and energy required are based on realistic distributions ElaadNL,time, day, month and year.
- EV and Charger characteristics are based on real EVs and chargers existing in NL.

## File Structure

```bash
├── ev2gym
│   ├── __init__.py
│   ├── baselines
│   │   ├── __init__.py
│   │   ├── gurobi_models


```

## Citing EV2Gym

If you use this code in your research, please cite as:
```bibtex
@misc{ev2gym,

}
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.


## Contributing

EV2Gym is an open-source project and welcomes contributions! Please get in contact with us if you would like to discuss about the simulator.


