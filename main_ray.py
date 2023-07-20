import ray
ray.init(address='auto', runtime_env={"working_dir": "./"})
import ray.rllib.agents.ppo as ppo
from gym_example.envs.example_env import Example_v0
from ray.tune.registry import register_env
import ray
from gym_env import ev_city

import numpy as np
import matplotlib.pyplot as plt

import os
import shutil
chkpt_root = "tmp/exa"
shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
ray_results = "{}/ray_results/".format(os.getenv("HOME"))
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

ray.init(ignore_reinit_error=True)

verbose = False
number_of_charging_stations = 200
steps = 100
select_env = "ev-city"
register_env(select_env, lambda config: ev_city.EVCity(cs=number_of_charging_stations,
                                                       timescale=5,
                                                       verbose=verbose,
                                                       simulation_length=steps))

config = ppo.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"
agent = ppo.PPOTrainer(config, env=select_env)

status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
n_iter = 5
for n in range(n_iter):
    result = agent.train()
    chkpt_file = agent.save(chkpt_root)
    print(status.format(
        n + 1,
        result["episode_reward_min"],
        result["episode_reward_mean"],
        result["episode_reward_max"],
        result["episode_len_mean"],
        chkpt_file
    ))


# # main funtion for testing
# if __name__ == "__main__":

#     verbose = False
#     number_of_charging_stations = 200
#     steps = 100

#     env = ev_city.EVCity(cs=number_of_charging_stations,
#                             timescale=5,
#                             verbose=verbose,
#                             simulation_length=steps)
#     state = env.reset()

#     env.visualize()
#     rewards = []

#     for i in range(steps):
#         print("-"*80)
#         # actions = env.action_space.sample()   # sample random actions
#         actions = np.random.uniform(-1, 1, env.number_of_ports)
#         # all ports are charging instantly
#         # actions = np.ones(env.number_of_ports)
#         if verbose:
#             print(f'Actions: {actions}')

#         new_state, reward, done = env.step(actions)  # takes action
#         rewards.append(reward)
#         env.visualize()

#         if verbose:
#             print(f'Reward: {reward} \t Done: {done}')

#         if done:
#             break

#     if verbose:
#         env.print_statistics()

#     # Plot the commulative reward in subplot 1
#     plt.subplot(2, 1, 1)
#     plt.plot(np.cumsum(rewards))
#     plt.ylabel('Cumulative reward')
#     plt.title('Cumulative reward')

#     # Plot the reward per step in subplot 2
#     plt.subplot(2, 1, 2)
#     plt.plot(rewards)
#     plt.xlabel('Time step')
#     plt.ylabel('Reward')
#     plt.title('Reward per time step')
#     plt.tight_layout()
#     plt.show()
