"""
This script is used to evaluate the baselines in the EVsSimulator environment.
"""

from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from sb3_contrib import TQC, TRPO, ARS, RecurrentPPO
from tqdm import tqdm
from EVsSimulator.ev_city import EVsSimulator
import gymnasium as gym
import os
from EVsSimulator.rl_agent.reward import SquaredTrackingErrorReward

import pickle

algorithms = ['ddpg', 'td3', 'sac', 'a2c', 'ppo', 'tqc', 'trpo', 'ars', 'rppo']
algorithms = ['tqc']
# algorithms = ['rr']
device = "cuda"

config_file = "EVsSimulator/example_config_files/PublicPST.yaml"

gym.envs.register(id='evs-v0', entry_point='EVsSimulator.ev_city:EVsSimulator',
                  kwargs={'config_file': config_file,
                          'verbose': False,
                            'save_plots': True,
                          'generate_rnd_game': True,
                          #   'reward_function': MinimizeTrackerSurplusWithChargeRewards,
                          })

env = gym.make('evs-v0')


if not os.path.exists("./results"):
    os.makedirs("./results")

for algorithm in algorithms:
    # load_path = "./saved_models/" + algorithm + "_20cs_1_port_best_reward.zip"
    load_path = "./saved_models/" + algorithm + \
        "_15cs_1_port_SquaredTrackingErrorReward.zip"
    # load_path = "./saved_models/" + algorithm + "_20cs_1_port.zip"

    if algorithm == "ddpg":
        model = DDPG.load(load_path, env=env, device=device)
    elif algorithm == "td3":
        model = TD3.load(load_path, env=env, device=device)
    elif algorithm == "sac":
        model = SAC.load(load_path, env=env, device=device)
    elif algorithm == "a2c":
        model = A2C.load(load_path, env=env, device=device)
    elif algorithm == "ppo":
        model = PPO.load(load_path, env=env, device=device)
    elif algorithm == "tqc":
        model = TQC.load(load_path, env=env, device=device)
    elif algorithm == "trpo":
        model = TRPO.load(load_path, env=env, device=device)
    elif algorithm == "ars":
        model = ARS.load(load_path, env=env, device=device)
    elif algorithm == "rppo":
        model = RecurrentPPO.load(load_path, env=env, device=device)
    elif algorithm == "rr":
        model = RoundRobin(env)
    else:
        raise ValueError("Algorithm not supported")

    env = model.get_env()
    obs = env.reset()

    stats = []
    for i in tqdm(range(96*1)):               

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if done:
            stats.append(info)
            obs = env.reset()

    # make directory if it does not exist

    # save stats to file
    pickle.dump(stats, open("./results/"+algorithm +
                "_15cs_1_port_SquaredTrackingErrorRewardWithPenalty.pkl", "wb"))

    # print average stats for
    print("=====================================================")
    print(f' Average stats for {algorithm} algorithm, {len(stats)} episodes')
    print("total_ev_served: ", sum(
        [i[0]['total_ev_served'] for i in stats])/len(stats))
    print("total_profits: ", sum(
        [i[0]['total_profits'] for i in stats])/len(stats))
    print("total_energy_charged: ", sum(
        [i[0]['total_energy_charged'] for i in stats])/len(stats))
    print("total_energy_discharged: ", sum(
        [i[0]['total_energy_discharged'] for i in stats])/len(stats))
    print("average_user_satisfaction: ", sum(
        [i[0]['average_user_satisfaction'] for i in stats])/len(stats))
    print("power_tracker_violation: ", sum(
        [i[0]['power_tracker_violation'] for i in stats])/len(stats))
    print("tracking_error: ", sum(
        [i[0]['tracking_error'] for i in stats])/len(stats))
    print("energy_user_satisfaction: ", sum(
        [i[0]['energy_user_satisfaction'] for i in stats])/len(stats))
    print("total_transformer_overload: ", sum(
        [i[0]['total_transformer_overload'] for i in stats])/len(stats))
    print("reward: ", sum([i[0]['episode']['r'] for i in stats])/len(stats))
