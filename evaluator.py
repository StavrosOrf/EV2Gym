# This script reads the replay files and evaluates the performance.

import yaml
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import datetime
import math

from EVsSimulator.utilities.arg_parser import arg_parser
from EVsSimulator import ev_city

from EVsSimulator.baselines.heuristics import RoundRobin, ChargeAsLateAsPossible, ChargeAsFastAsPossible
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from sb3_contrib import TQC, TRPO, ARS, RecurrentPPO

from EVsSimulator.baselines.gurobi_models.ev_city_power_tracker_model import PowerTrackingErrorrMin

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
simulation_length = config["simulation_length"]

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

replay_to_print = 5
replay_to_print = min(replay_to_print, len(eval_replay_files)-1)

print(f'Number of test cycles: {n_test_cycles}')

if args.config_file == "EVsSimulator/example_config_files/V2GProfitMax.yaml":
    reward_function = profit_maximization
    state_function = V2G_profit_max

elif args.config_file == "EVsSimulator/example_config_files/PublicPST.yaml":
    reward_function = SquaredTrackingErrorReward
    state_function = PublicPST


# Algorithms to compare:
algorithms = [ChargeAsLateAsPossible,
              ChargeAsFastAsPossible,
              RoundRobin,
              PPO,
              ARS,
              DDPG,
              TD3,
              PowerTrackingErrorrMin,]

counter = 0
for algorithm in algorithms:

    print(' +------- Evaluating', algorithm.__name__, " -------+")
    for k in range(n_test_cycles):
        counter += 1
        replay_path = eval_replay_path + eval_replay_files[k]

        if algorithm in [PPO, A2C, DDPG, SAC, TD3, TQC, TRPO, ARS, RecurrentPPO]:
            gym.envs.register(id='evs-v0', entry_point='EVsSimulator.ev_city:EVsSimulator',
                              kwargs={'config_file': args.config_file,
                                      'generate_rnd_game': True,
                                      'state_function': state_function,
                                      'reward_function': reward_function,
                                      'load_from_replay_path': replay_path,
                                      })
            env = gym.make('evs-v0')

            load_path = f'./saved_models/{number_of_charging_stations}cs_{scenario}/' + \
                f"{algorithm.__name__.lower()}_SquaredTrackingErrorReward_PublicPST"
                                 
            model = algorithm.load(load_path, env, device=device)
            env = model.get_env()
        else:
            env = ev_city.EVsSimulator(
                config_file=args.config_file,
                load_from_replay_path=replay_path,
                generate_rnd_game=True,
            )

            model = algorithm(env=env, replay_path=replay_path)

        state = env.reset()

        rewards = []

        for _ in range(simulation_length):

            ################# Your algorithm goes here #################
            # actions is a vector of size number of ports and it takes values from -1 to 1
            # 0 means no charging, 1 means charging at the maximum rate, -1 means discharging at the maximum rate
            # discharging might not be supported by the charging station, so negative values might be clipped to 0

            if algorithm in [PPO, A2C, DDPG, SAC, TD3, TQC, TRPO, ARS, RecurrentPPO]:
                action, _ = model.predict(state, deterministic=True)
                obs, reward, done, info = env.step(action)
            else:
                actions = model.get_action(env)
                new_state, reward, done, _, stats = env.step(
                    actions)  # takes action
            ############################################################

            rewards.append(reward)

            if done:
                results_i = pd.DataFrame({'run': k,
                                          'Algorithm': algorithm.__name__,
                                          'total_ev_served': stats['total_ev_served'],
                                          'total_profits': stats['total_profits'],
                                          'total_energy_charged': stats['total_energy_charged'],
                                          'total_energy_discharged': stats['total_energy_discharged'],
                                          'average_user_satisfaction': stats['average_user_satisfaction'],
                                          'power_tracker_violation': stats['power_tracker_violation'],
                                          'tracking_error': stats['tracking_error'],
                                          'energy_user_satisfaction': stats['energy_user_satisfaction'],
                                          'total_transformer_overload': stats['total_transformer_overload'],
                                          'battery_degradation': stats['battery_degradation'],
                                          'battery_degradation_calendar': stats['battery_degradation_calendar'],
                                          'battery_degradation_cycling': stats['battery_degradation_cycling'],
                                          }, index=[counter])

                if counter == 1:
                    results = results_i
                else:
                    results = pd.concat([results, results_i])

                if algorithm in [PPO, A2C, DDPG, SAC, TD3, TQC, TRPO, ARS, RecurrentPPO]:
                    power_usage = env.get_attr('previous_power_usage')[0]
                    power_setpoints = env.get_attr('power_setpoints')[0]                    
                else:
                    power_usage = env.current_power_usage
                    power_setpoints = env.power_setpoints

                results_i = {f'Step_{j}': power_usage[j]
                             for j in range(simulation_length)}
                results_i['run'] = k
                results_i['Algorithm'] = algorithm.__name__
                results_i = pd.DataFrame(results_i, index=[counter])

                if counter == 1:
                    results_to_draw = results_i
                else:
                    results_to_draw = pd.concat([results_to_draw, results_i])

                results_i = {
                    f'Step_{j}': power_setpoints[j] for j in range(simulation_length)}
                results_i['run'] = k

                results_i = pd.DataFrame(results_i, index=[counter])

                if counter == 1:
                    results_to_draw_setpoint = results_i
                else:
                    results_to_draw_setpoint = pd.concat(
                        [results_to_draw_setpoint, results_i])

                break

# Group the results by algorithm and print the average and the standard deviation of the statistics
results_grouped = results.groupby('Algorithm').agg(['mean', 'std'])
# replace Nan with 0
results_grouped = results_grouped.fillna(0)

#print the main statistics in a latex table
# print(results_grouped.to_latex())


print(results_grouped)
results_grouped.to_csv('results_grouped.csv')

# plot the power usage and the power_setpoints vs time in vetical subplots for the last replay for every algorithm for one run

fig, ax = plt.subplots(len(algorithms), 1, figsize=(10, 5))
plt.grid(True, which='major', axis='both')
plt.rcParams['font.family'] = ['serif']
# print(results_to_draw.head(15))
fontsize = 28


date_range_print = pd.date_range(start=env.sim_starting_date,
                                 end=env.sim_date,
                                 periods=8)

run = 0
for i, algorithm in enumerate(algorithms):
    actual_power = results_to_draw[results_to_draw.Algorithm ==
                                   algorithm.__name__]
    actual_power = actual_power[actual_power.run == run]
    actual_power = actual_power.drop(columns=['run', 'Algorithm']).values

    setpoints = results_to_draw_setpoint[results_to_draw_setpoint.run == run]
    setpoints = setpoints.drop(columns=['run']).values

    x = np.arange(0, simulation_length, 1)
    ax[i].step(x, actual_power.T, alpha=0.5, color='blue')
    ax[i].step(x, setpoints.T, alpha=0.5, color='red')

    # plot a black line at y=0
    ax[i].axhline(0, color='black', lw=2)

    ax[i].set_title(f'{algorithms[i].__name__}', fontsize=fontsize-2)

    if i == len(algorithms) - 1:

        ax[i].set_xticks(ticks=[j for j in range(0, simulation_length,
                                                 math.ceil(simulation_length / len(date_range_print)))],
                         labels=[
                             f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print],
                         rotation=45,
                         fontsize=22)

        ax[i].set_xlabel('Time', fontsize=fontsize-2)
    else:
        ax[i].set_xticks([j for j in range(0, simulation_length,
                                           math.ceil(simulation_length / len(date_range_print)))],
                         labels=[' ' for d in date_range_print])

    ax[i].set_ylabel('Power (kW)', fontsize=fontsize-2)
    ax[i].legend(['Actual Power', 'Setpoint'], fontsize=fontsize-2)
    ax[i].set_xlim([0, simulation_length-1])
    ax[i].grid(True, which='major', axis='both')
    ax[i].set_ylim([0, max(actual_power.max(), setpoints.max())+10])

    # ax[i].set_yticks(ticks = ax[i].get_yticks,#get the yticks from the data)
    #                     size=fontsize-2)

    # ax[i].set_grid(True)

plt.show()
