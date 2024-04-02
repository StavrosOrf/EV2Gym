.. ev2gym documentation master file, created by
   sphinx-quickstart on Tue Mar 26 00:22:23 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EV2Gym's documentation!
==================================

Develop and evaluate **any type of smart charging algorithm**: from simple heuristics, Model Predictive Control, Mathematical Programming, to Reinforcement Learning!

EV2Gym is **fully customizable** and easily **configurable**!

Run the example code below to get started ...

.. code-block:: Python

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

To train an RL agent, using the [StableBaselines3](https://stable-baselines3.readthedocs.io/en/master/) library, you can use the following code:

.. code-block:: Python   

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

.. toctree::
   :maxdepth: 1
   :hidden:

   Home <self>

.. toctree::
   :maxdepth: 1
   :caption: Models:

   autoapi/ev2gym/models/ev2gym_env/index.rst
   autoapi/ev2gym/models/ev/index.rst
   autoapi/ev2gym/models/ev_charger/index.rst
   autoapi/ev2gym/models/transformer/index.rst
   autoapi/ev2gym/models/replay/index.rst

.. toctree::
   :maxdepth: 1
   :caption: Baselines:

   autoapi/ev2gym/baselines/heuristics/index.rst
   autoapi/ev2gym/baselines/gurobi_models/profit_max/index.rst
   autoapi/ev2gym/baselines/gurobi_models/tracking_error/index.rst
   autoapi/ev2gym/baselines/mpc/mpc/index.rst
   autoapi/ev2gym/baselines/mpc/eMPC/index.rst
   autoapi/ev2gym/baselines/mpc/ocmf_mpc/index.rst


.. toctree::
   :maxdepth: 1
   :caption: RL-Agent:

   autoapi/ev2gym/rl_agent/reward/index.rst
   autoapi/ev2gym/rl_agent/state/index.rst

.. toctree::
   :maxdepth: 1
   :caption: Visuals:

   autoapi/ev2gym/visuals/plots/index.rst
   autoapi/ev2gym/visuals/render/index.rst
   autoapi/ev2gym/visuals/evaluator_plot/index.rst

.. toctree::
   :maxdepth: 1
   :caption: Utils:

   autoapi/ev2gym/utilities/loaders/index.rst
   autoapi/ev2gym/utilities/utils/index.rst

