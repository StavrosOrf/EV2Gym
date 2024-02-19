# This script reads the replay files and evaluates the performance.

import yaml
import os
import pickle
from EVsSimulator.utilities.arg_parser import arg_parser
import numpy as np
from EVsSimulator import ev_city

from EVsSimulator.baselines.heuristics import RoundRobin, ChargeAsLateAsPossible, ChargeAsFastAsPossible
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from sb3_contrib import TQC, TRPO, ARS, RecurrentPPO

from EVsSimulator.rl_agent.reward import SquaredTrackingErrorReward, SqTrError_TrPenalty_UserIncentives
from EVsSimulator.rl_agent.reward import profit_maximization

from EVsSimulator.rl_agent.state import V2G_profit_max, PublicPST

import gymnasium as gym
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

args = arg_parser()
config = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)

number_of_charging_stations = config["number_of_charging_stations"]
n_transformers = config["number_of_transformers"]
steps = config["simulation_length"]
timescale = config["timescale"]

n_test_cycles = args.n_test_cycles

scenario = args.config_file.split("/")[-1].split(".")[0]
eval_replay_path = f'./replay/{number_of_charging_stations}cs_{n_transformers}tr_{scenario}/'

eval_replay_files = [f for f in os.listdir(
    eval_replay_path) if os.path.isfile(os.path.join(eval_replay_path, f))]
assert len(
    eval_replay_files) > 0, "No replay files found in evaluation replay folder"

print(f'Found {len(eval_replay_files)} replay files in {eval_replay_path}')
if n_test_cycles > len(eval_replay_files):
    n_test_cycles = len(eval_replay_files)

print(f'Number of test cycles: {n_test_cycles}')

if args.config_file == "EVsSimulator/example_config_files/V2GProfitMax.yaml":
    reward_function = profit_maximization
    state_function = V2G_profit_max

elif args.config_file == "EVsSimulator/example_config_files/PublicPST.yaml":
    reward_function = SquaredTrackingErrorReward
    state_function = PublicPST

# Load the replay files and aggregate the statistics
opt_energy_charged = []
opt_tracking_errors = []
opt_tracking_surpluses = []
opt_energy_user_satisfaction = []
opt_transformer_overloads = []

unst_energy_charged = []
unst_tracking_errors = []
unst_tracking_surpluses = []
unst_energy_user_satisfaction = []
uns_transformer_overloads = []

for i in range(n_test_cycles):
    replay = pickle.load(open(eval_replay_path + eval_replay_files[i], 'rb'))

    opt_energy_charged.append(replay.optimal_stats["total_energy_charged"])
    opt_tracking_errors.append(replay.optimal_stats["tracking_error"])
    opt_tracking_surpluses.append(
        replay.optimal_stats["power_tracker_violation"])
    opt_energy_user_satisfaction.append(
        replay.optimal_stats["energy_user_satisfaction"])
    opt_transformer_overloads.append(
        replay.optimal_stats["total_transformer_overload"])

    unst_energy_charged.append(replay.unstirred_stats["total_energy_charged"])
    unst_tracking_errors.append(replay.unstirred_stats["tracking_error"])
    unst_tracking_surpluses.append(
        replay.unstirred_stats["power_tracker_violation"])
    unst_energy_user_satisfaction.append(
        replay.unstirred_stats["energy_user_satisfaction"])
    uns_transformer_overloads.append(
        replay.unstirred_stats["total_transformer_overload"])


energy_charged = []
tracking_errors = []
tracking_surpluses = []
energy_user_satisfaction = []
transformer_overloads = []

# Evaluate the performance of your algorithms
# Algorithms to compare:
algorithms = [RoundRobin, PPO, ChargeAsLateAsPossible, ChargeAsFastAsPossible]

for algorithm in algorithms:
    print('Evaluating', algorithm.__name__)
    for i in range(n_test_cycles):

        replay_path = eval_replay_path + eval_replay_files[i]
        if algorithm in [RoundRobin, ChargeAsLateAsPossible, ChargeAsFastAsPossible]:
            env = ev_city.EVsSimulator(
                config_file=args.config_file,
                load_from_replay_path=replay_path,
                generate_rnd_game=True,
            )
            model = algorithm(env)

        elif algorithm in [PPO, A2C, DDPG, SAC, TD3, TQC, TRPO, ARS, RecurrentPPO]:
            gym.envs.register(id='evs-v0', entry_point='EVsSimulator.ev_city:EVsSimulator',
                              kwargs={'config_file': args.config_file,
                                      'generate_rnd_game': True,
                                      'state_function': state_function,
                                      'reward_function': reward_function,
                                      'load_from_replay_path': replay_path,
                                      })
            env = gym.make('evs-v0')

            load_path = f'./saved_models/{number_of_charging_stations}cs_{scenario}/' + \
                f"{algorithm.__name__}_SquaredTrackingErrorReward_PublicPST"
            model = algorithm.load(load_path, env, device=device)
            env = model.get_env()
        else:
            raise ValueError("Invalid algorithm")

        state = env.reset()

        rewards = []

        for i in range(config['simulation_length']):

            ################# Your algorithm goes here #################
            # actions is a vector of size number of ports and it takes values from -1 to 1
            # 0 means no charging, 1 means charging at the maximum rate, -1 means discharging at the maximum rate
            # discharging might not be supported by the charging station, so negative values might be clipped to 0

            if algorithm in [RoundRobin, ChargeAsLateAsPossible, ChargeAsFastAsPossible]:
                actions = model.get_action(env)
                new_state, reward, done, _, stats = env.step(actions)  # takes action
                  
            elif algorithm in [PPO, A2C, DDPG, SAC, TD3, TQC, TRPO, ARS, RecurrentPPO]:
                  action, _ = model.predict(state, deterministic=True)
                  obs, reward, done, info = env.step(action)
            ############################################################

            
            rewards.append(reward)

            # input("Press Enter to continue...")
            if done:
                # print(f'End of simulation at step {env.current_step}')
                break
        energy_charged.append(stats["total_energy_charged"])
        tracking_errors.append(stats["tracking_error"])
        tracking_surpluses.append(stats["power_tracker_violation"])
        energy_user_satisfaction.append(stats["energy_user_satisfaction"])
        transformer_overloads.append(stats["total_transformer_overload"])


# Print the average and the standard deviation of the statistics
print(f'--------- Optimal -------------')
print(f'Average energy charged: {sum(opt_energy_charged)/len(opt_energy_charged)}',
      f'Standard deviation: {np.std(opt_energy_charged)}')
print(f'Average tracking error: {sum(opt_tracking_errors)/len(opt_tracking_errors)}',
      f'Standard deviation: {np.std(opt_tracking_errors)}')
print(f'Average power tracker violation: {sum(opt_tracking_surpluses)/len(opt_tracking_surpluses)}',
      f'Standard deviation: {np.std(opt_tracking_surpluses)}')
print(f'Average energy user satisfaction: {sum(opt_energy_user_satisfaction)/len(opt_energy_user_satisfaction)}',
      f'Standard deviation: {np.std(opt_energy_user_satisfaction)}')
print(f'Average transformer overload: {sum(opt_transformer_overloads)/len(opt_transformer_overloads)}',
      f'Standard deviation: {np.std(opt_transformer_overloads)}')

print(f'\n--------- Unstirred -------------')
print(f'Average energy charged: {sum(unst_energy_charged)/len(unst_energy_charged)}',
      f'Standard deviation: {np.std(unst_energy_charged)}')
print(f'Average tracking error: {sum(unst_tracking_errors)/len(unst_tracking_errors)}',
      f'Standard deviation: {np.std(unst_tracking_errors)}')
print(f'Average power tracker violation: {sum(unst_tracking_surpluses)/len(unst_tracking_surpluses)}',
      f'Standard deviation: {np.std(unst_tracking_surpluses)}')
print(f'Average energy user satisfaction: {sum(unst_energy_user_satisfaction)/len(unst_energy_user_satisfaction)}',
      f'Standard deviation: {np.std(unst_energy_user_satisfaction)}')
print(f'Average transformer overload: {sum(uns_transformer_overloads)/len(uns_transformer_overloads)}',
      f'Standard deviation: {np.std(uns_transformer_overloads)}')

# Print the average and the standard deviation of the statistics
print(f'\n--------- Your Algorithm -------------')
print(f'Avg. energy charged: {sum(energy_charged)/len(energy_charged)}',
      f'Std: {np.std(energy_charged)}')
print(f'Avg. tracking error: {sum(tracking_errors)/len(tracking_errors)}',
      f'Std: {np.std(tracking_errors)}')
print(f'Avg. power tracker violation: {sum(tracking_surpluses)/len(tracking_surpluses)}',
      f'Std: {np.std(tracking_surpluses)}')
print(f'Avg. energy user satisfaction: {sum(energy_user_satisfaction)/len(energy_user_satisfaction)}',
      f'Std: {np.std(energy_user_satisfaction)}')
print(f'Avg. transformer overload: {sum(transformer_overloads)/len(transformer_overloads)}',
      f'Std: {np.std(transformer_overloads)}')
