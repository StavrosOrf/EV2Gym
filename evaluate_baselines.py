"""
This script is used to evaluate the baselines in the ev2gym environment.
"""

from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from sb3_contrib import TQC, TRPO, ARS, RecurrentPPO
from tqdm import tqdm
from ev2gym.models.ev2gym_env import EV2Gym
import gymnasium as gym
import os
from ev2gym.rl_agent.reward import SquaredTrackingErrorReward, SqTrError_TrPenalty_UserIncentives
from ev2gym.rl_agent.reward import profit_maximization

from ev2gym.rl_agent.state import V2G_profit_max
import pickle

algorithms = ['ddpg', 'td3', 'sac', 'a2c', 'ppo', 'tqc', 'trpo', 'ars', 'rppo']
algorithms = ['ars']
device = "cuda"

config_file = "ev2gym/example_config_files/PublicPST.yaml"

config_file = "ev2gym/example_config_files/V2GProfitMax.yaml"

gym.envs.register(id='evs-v0', entry_point='ev2gym.ev_city:ev2gym',
                  kwargs={'config_file': config_file,
                          'verbose': False,
                          'save_plots': True,
                          'generate_rnd_game': True,
                          'state_function': V2G_profit_max,
                          'reward_function': profit_maximization,
                          #   'reward_function': MinimizeTrackerSurplusWithChargeRewards,
                          })

env = gym.make('evs-v0')

if not os.path.exists("./results"):
    os.makedirs("./results")

for algorithm in algorithms:    
    load_path = "./saved_models/" + algorithm 
    # + "_50cs_1_port_SqTrError_TrPenalty_UserIncentives.zip"    
    load_path += "_40cs_1_port_V2G_profit_max.zip"    

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
    else:
        raise ValueError("Algorithm not supported")

    env = model.get_env()
    obs = env.reset()

    stats = []
    for i in tqdm(range(85*1000)):

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if done:
            stats.append(info)
            obs = env.reset()

    # save stats to file
    pickle.dump(stats, open("./results/"+algorithm +
                "_50cs_1_port_SqTrError_TrPenalty_UserIncentives.pkl", "wb"))

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
