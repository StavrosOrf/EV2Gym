import gymnasium as gym

from ev2gym.visuals.evaluator_plot import plot_comparable_EV_SoC_single, plot_prices
from ev2gym.visuals.evaluator_plot import plot_total_power_V2G, plot_actual_power_vs_setpoint
from ev2gym.visuals.evaluator_plot import plot_total_power, plot_comparable_EV_SoC
from ev2gym.rl_agent.reward import SquaredTrackingErrorReward, SimpleReward
from ev2gym.rl_agent.action_wrappers import Rescale_RepairLayer

from ev2gym.baselines.gurobi_models.profit_max import V2GProfitMaxOracleGB
from ev2gym.baselines.gurobi_models.tracking_error import PowerTrackingErrorrMin

from sb3_contrib import TQC, TRPO, ARS, RecurrentPPO
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.noise import  OrnsteinUhlenbeckActionNoise
import torch as th



from ev2gym.baselines.mpc.eMPC import eMPC_V2G, eMPC_G2V
from ev2gym.baselines.mpc.ocmf_mpc import OCMF_V2G, OCMF_G2V
from ev2gym.baselines.heuristics import RoundRobin, ChargeAsLateAsPossible, ChargeAsFastAsPossible, RoundRobin_GF_off_allowed

from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, BusinessPSTwithMoreKnowledge
from ev2gym.rl_agent.reward import SquaredTrackingErrorReward, SqTrError_TrPenalty_UserIncentives

from ev2gym.models.ev2gym_env import EV2Gym
import yaml
import os
import pickle
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import datetime
import time
import random
import gzip

import warnings

# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)


# GNN-based models evaluations


# from DT.evaluation.evaluate_episodes import evaluate_episode_rtg_from_replays
# from DT.models.decision_transformer import DecisionTransformer
# from DT.load_model import load_DT_model

# set seeds
# seed = 6
# np.random.seed(seed)
# torch.manual_seed(seed)
# random.seed(seed)

def evaluator():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ############# Simulation Parameters #################
    n_test_cycles = 100 # try 50+ for better results
    SAVE_EV_PROFILES = False
    seed = 9

    # seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    config_file = "./ev2gym/example_config_files/BusinessPST.yaml"  # 25

    if "PST" in config_file:
        state_function_Normal = BusinessPSTwithMoreKnowledge
        reward_function = SquaredTrackingErrorReward
    else:
        raise ValueError(f'Unknown config file {config_file}')

    # Algorithms to compare:
    # Use algorithm name or the saved RL model path as string
    algorithms = [
        ChargeAsFastAsPossible,
        # ChargeAsLateAsPossible,
        
        ####
        "./saved_models/10cs_APEN_PST/td3_RepairL_USER_BPST_seed=9",
        "./saved_models/10cs_APEN_PST/td3_RepairL_USER_BPST_seed=9_SL",
        './saved_models/10cs_APEN_PST/td3_RepairL_STER_BPST_seed=9',
        './saved_models/10cs_APEN_PST/td3_RepairL_STER_BPST_seed=9_SL',
        ###

        # "./saved_models/10cs_APEN_PST/ddpg_RepairL_STER_BPST_seed=9",
        # "./saved_models/10cs_APEN_PST/ddpg_RepairL_STER_BPST_seed=9_SL",
        #"./saved_models/10cs_APEN_PST/ddpg_RepairL_S_BPST_seed=9",
        #"./saved_models/10cs_APEN_PST/td3_RepairL_S_BPST_seed=9",
        #"./saved_models/10cs_APEN_PST/td3_RepairL_S_BPST_seed=9",
        # "./saved_models/10cs_APEN_PST/td3_RepairL_STER_BPST_seed=9",
        # "./saved_models/10cs_APEN_PST/td3_RepairL_STER_BPST_seed=9_SL",
        #  here put the paths to the saved RL models
        # "TD3-114002",

        # adding the _SL suffix to the algorithm name
        # "TD3-114002_SL",

        RoundRobin_GF_off_allowed,
        PowerTrackingErrorrMin
    ]

    # create a AnalysisReplayBuffer object for each algorithm

    env = EV2Gym(config_file=config_file,
                 generate_rnd_game=True,
                 state_function=state_function_Normal,
                 reward_function=reward_function,
                 )

    #####################################################

    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)

    number_of_charging_stations = config["number_of_charging_stations"]
    n_transformers = config["number_of_transformers"]
    timescale = config["timescale"]
    simulation_length = config["simulation_length"]

    scenario = config_file.split("/")[-1].split(".")[0]

    eval_replay_path = f'./replay/{config["number_of_charging_stations"]}cs_APEN_PST/'

    print(f'Looking for replay files in {eval_replay_path}')
    try:
        eval_replay_files = [f for f in os.listdir(
            eval_replay_path) if os.path.isfile(os.path.join(eval_replay_path, f))]

        print(
            f'Found {len(eval_replay_files)} replay files in {eval_replay_path}')
        if n_test_cycles > len(eval_replay_files):
            # n_test_cycles = len(eval_replay_files)
            replays_exist = False
        else:

            replay_to_print = 1
            replay_to_print = min(replay_to_print, len(eval_replay_files)-1)
            replays_exist = True

    except:
        n_test_cycles = n_test_cycles
        replays_exist = False

    print(f'Number of test cycles: {n_test_cycles}')

    if SAVE_EV_PROFILES:
        ev_profiles = []

    def generate_replay(evaluation_name):
        env = EV2Gym(config_file=config_file,
                     save_replay=True,
                     replay_save_path=f"{evaluation_name}/",
                     )
        replay_path = f"{evaluation_name}/replay_{env.sim_name}.pkl"

        for _ in range(env.simulation_length):
            actions = np.ones(env.cs)

            new_state, reward, done, truncated, _ = env.step(
                actions, visualize=False)  # takes action

            if done:
                break

        if SAVE_EV_PROFILES:
            ev_profiles.append(env.EVs_profiles)
        return replay_path

    evaluation_name = f'eval_{number_of_charging_stations}cs_{n_transformers}tr_{scenario}_{len(algorithms)}_algos' +\
        f'_{n_test_cycles}_exp_' +\
        f'{datetime.datetime.now().strftime("%Y_%m_%d_%f")}'

    # make a directory for the evaluation
    save_path = f'./results/{evaluation_name}/'
    os.makedirs(save_path, exist_ok=True)
    os.system(f'cp {config_file} {save_path}')

    if not replays_exist:
        eval_replay_files = [generate_replay(
            eval_replay_path) for _ in range(n_test_cycles)]

    # save the list of EV profiles to a pickle file
    if SAVE_EV_PROFILES:
        with open(save_path + 'ev_profiles.pkl', 'wb') as f:
            print(f'Saving EV profiles to {save_path}ev_profiles.pkl')
            pickle.dump(ev_profiles, f)

        exit()

    plot_results_dict = {}
    counter = 0

    for i_a, algorithm in enumerate(algorithms):
        print(' +------- Evaluating', algorithm, " -------+")
        for k in range(n_test_cycles):
            print(f' Test cycle {k+1}/{n_test_cycles} -- {algorithm}')
            counter += 1
            h = -1

            if replays_exist:
                replay_path = eval_replay_path + eval_replay_files[k]
            else:
                replay_path = eval_replay_files[k]

            state_function = state_function_Normal

            env = EV2Gym(config_file=config_file,
                         load_from_replay_path=replay_path,
                         state_function=state_function,
                         reward_function=reward_function,
                         )

            if isinstance(algorithm, str) and 'SL' in algorithm:                
                env = Rescale_RepairLayer(env=env)

            # initialize the timer
            timer = time.time()
            state, _ = env.reset()
            # try:
            if type(algorithm) == str:
                if any(algo in algorithm for algo in ['ppo', 'a2c', 'ddpg', 'tqc', 'trpo', 'ars', 'rppo', 'td3', 'sac']):

                    gym.envs.register(id='evs-v0', entry_point='ev2gym.models.ev2gym_env:EV2Gym',
                                      kwargs={'config_file': config_file,
                                              'state_function': state_function_Normal,
                                              'reward_function': reward_function,
                                              'load_from_replay_path': replay_path,
                                              })
                    env = gym.make('evs-v0')
                    
                    # Apply Rescale_RepairLayer before loading the model if needed
                    if isinstance(algorithm, str) and 'SL' in algorithm:
                        print(f'Applying SL wrapper before loading model for {algorithm}')
                        env = Rescale_RepairLayer(env=env)


                    # load_path = f'./eval_models/{algorithm}/td3oldparam.zip'
                    # if 'SL' in algorithm: remove this
                    if 'SL' in algorithm:
                        load_path = algorithm.replace('_SL', '')
                        load_path = f'{load_path}/best_model.zip'
                    else:
                        load_path = f'{algorithm}/best_model.zip'

                    # initialize the timer
                    timer = time.time()
                    # algorithm_name = algorithm.split('_')[0]

                    if 'rppo' in algorithm:
                        sb3_algo = RecurrentPPO
                    elif 'ppo' in algorithm:
                        sb3_algo = PPO
                        algorithm_name = 'PPO'
                    elif 'a2c' in algorithm:
                        sb3_algo = A2C
                        algorithm_name = 'A2C'
                    elif 'ddpg' in algorithm:
                        sb3_algo = DDPG("MlpPolicy", env, verbose=1,
                    learning_rate = 1e-3,
                    buffer_size = 1_000_00,  # 1e6
                    learning_starts = 100,
                    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(config['number_of_charging_stations']),
                                            sigma=0.2 * np.ones(config['number_of_charging_stations'])),
                    policy_kwargs = dict(activation_fn=th.nn.Sigmoid, net_arch=dict(pi=[128, 128], qf=[64, 64])),
                    batch_size = 64, #100
                    tau = 0.0005, #0.005
                    gamma = 0.99,                     
                     device=device, tensorboard_log="./logs/")
                        algorithm_name = 'DDPG'

                    elif 'tqc' in algorithm:
                        sb3_algo = TQC
                        algorithm_name = 'TQC'
                    elif 'trpo' in algorithm:
                        sb3_algo = TRPO
                        algorithm_name = 'TRPO'
                    elif 'ars' in algorithm:
                        sb3_algo = ARS
                        algorithm_name = 'ARS'
                    elif 'sac' in algorithm:
                        sb3_algo = SAC
                        algorithm_name = 'SAC'
                    elif 'td3' in algorithm:
                        sb3_algo = TD3("MlpPolicy", env, verbose=1,
                    learning_rate = 1e-3,
                    buffer_size = 1_000_00,  # 1e6
                    learning_starts = 100,
                    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(config['number_of_charging_stations']),
                                            sigma=0.2 * np.ones(config['number_of_charging_stations'])),
                    policy_kwargs = dict(activation_fn=th.nn.Sigmoid, net_arch=dict(pi=[128, 128], qf=[64, 64])),
                    batch_size = 64, #100
                    tau = 0.0005, #0.005
                    gamma = 0.99,
                    device=device, tensorboard_log="./logs/")
                        algorithm_name = 'TD3'
                    else:
                        raise ValueError(
                            f'Unknown algorithm {algorithm}')

                    if 'SL' in algorithm:
                        algorithm_name += '_SL'

                    if 'USER' in algorithm:
                        algorithm_name += '_USER'

                    model = sb3_algo.load(load_path,
                                          env,
                                          device=device)
                    # set replay buffer to None

                    # if 'tqc' in algorithm or 'ddpg' in algorithm:
                    #     model.replay_buffer = model.replay_buffer.__class__(1,
                    #                                                         model.observation_space,
                    #                                                         model.action_space,
                    #                                                         device=model.device,
                    #                                                         optimize_memory_usage=model.replay_buffer.optimize_memory_usage) 

                    env = model.get_env()
                    
                    state = env.reset()

                else:
                    raise ValueError(
                        f'Unknown algorithm {algorithm}')

            else:
                model = algorithm(env=env,
                                  replay_path=replay_path,
                                  verbose=False)
                algorithm_name = algorithm.__name__
            # except Exception as error:
            #     print(error)
            #     print(
            #         f'!!!!!!!!!! Error in {algorithm} with replay {replay_path}')
            #     continue

            rewards = []
            saved_env = None  # Initialize saved_env

            for i in range(simulation_length):

                if type(algorithm) == str:
                    if any(algo in algorithm for algo in ['ppo', 'a2c', 'ddpg', 'tqc', 'trpo', 'ars', 'rppo', 'td3', 'sac']):
                        action, _ = model.predict(
                            state, deterministic=True)
                        state, reward, done, stats = env.step(action)

                        if i == simulation_length - 2:
                            saved_env = deepcopy(
                                env.get_attr('env')[0])

                        stats = stats[0]

                    else:
                        raise ValueError(
                            f'Unknown algorithm {algorithm}')

                else:
                    action = model.get_action(env=env)
                    new_state, reward, done, _, stats = env.step(
                        action)

                ############################################################

                rewards.append(reward)

                if done:

                    results_i = pd.DataFrame({'run': k,
                                              'Algorithm': algorithm_name,
                                              'algorithm_version': algorithm,
                                              'control_horizon': h,
                                              'discharge_price_factor': config['discharge_price_factor'],
                                              'total_ev_served': stats['total_ev_served'],
                                              'total_profits': stats['total_profits'],
                                              'profits_from_customers': stats['profits_from_customers'],
                                              'total_energy_charged': stats['total_energy_charged'],
                                              'total_energy_discharged': stats['total_energy_discharged'],
                                              'average_user_satisfaction': stats['average_user_satisfaction'],
                                              'power_tracker_violation': stats['power_tracker_violation'],
                                              'tracking_error': stats['tracking_error'],
                                              'energy_tracking_error': stats['energy_tracking_error'],
                                              'energy_user_satisfaction': stats['energy_user_satisfaction'],
                                              'min_energy_user_satisfaction': stats['min_energy_user_satisfaction'],
                                              'std_energy_user_satisfaction': stats['std_energy_user_satisfaction'],
                                              'total_transformer_overload': stats['total_transformer_overload'],
                                              'battery_degradation': stats['battery_degradation'],
                                              'battery_degradation_calendar': stats['battery_degradation_calendar'],
                                              'battery_degradation_cycling': stats['battery_degradation_cycling'],
                                              'total_reward': sum(rewards),
                                              'time': time.time() - timer,
                                              }, index=[counter])

                    if counter == 1:
                        results = results_i
                    else:
                        results = pd.concat([results, results_i])

                    if algorithm in [PPO, A2C, DDPG, SAC, TD3, TQC, TRPO, ARS, RecurrentPPO]:
                        env = saved_env

                    if k == 0:
                        # For string-based RL algorithms, use saved_env if it exists
                        if type(algorithm) == str and any(algo in algorithm for algo in ['ppo', 'a2c', 'ddpg', 'tqc', 'trpo', 'ars', 'rppo', 'td3', 'sac']):
                            if saved_env is not None:
                                plot_results_dict[str(algorithm)] = deepcopy(
                                    saved_env)
                            else:
                                plot_results_dict[str(
                                    algorithm)] = deepcopy(env)
                        else:
                            plot_results_dict[str(algorithm)] = deepcopy(env)

                    break
    # save the plot_results_dict to a pickle file
    # with open(save_path + 'plot_results_dict.pkl', 'wb') as f:
    #     pickle.dump(plot_results_dict, f)

        # replace some algorithm_version to other names:
    # change from PowerTrackingErrorrMin -> PowerTrackingError

    # print unique algorithm versions

    results['algorithm_version'] = results['algorithm_version'].astype(str).replace(
        "<class 'ev2gym.baselines.heuristics.ChargeAsFastAsPossible'>", 'ChargeAsFastAsPossible')
    results['algorithm_version'] = results['algorithm_version'].astype(str).replace(
        "<class 'ev2gym.baselines.heuristics.RoundRobin_GF_off_allowed'>", 'RoundRobin')
    results['algorithm_version'] = results['algorithm_version'].astype(str).replace(
        "<class 'ev2gym.baselines.heuristics.RoundRobin_GF'>", 'RoundRobin_GF')
    results['algorithm_version'] = results['algorithm_version'].astype(str).replace(
        "<class 'ev2gym.baselines.heuristics.RoundRobin'>", 'RoundRobin')
    results['algorithm_version'] = results['algorithm_version'].astype(str).replace(
        "<class 'ev2gym.baselines.gurobi_models.tracking_error.PowerTrackingErrorrMin'>",
        'Oracle'
    )
    print(results['algorithm_version'].unique())

    # save the results to a csv file
    results.to_csv(save_path + 'data.csv')

    # drop_columns = ['algorithm_version']
    # drop_columns = ['Algorithm']

    # results = results.drop(columns=drop_columns)

    # Select only numeric columns for aggregation (exclude categorical columns)
    numeric_columns = results.select_dtypes(
        include=[np.number]).columns.tolist()
    # Remove 'run' from numeric columns as it's just an index
    if 'run' in numeric_columns:
        numeric_columns.remove('run')

    results_grouped = results.groupby(
        'Algorithm')[numeric_columns].agg(['mean', 'std'])

    # print columns of the results
    # print(results_grouped.columns)

    # savethe latex results in a txt file
    # with open(save_path + 'results_grouped.txt', 'w') as f:
    #     f.write(results_grouped.to_latex())

    # results_grouped.to_csv('results_grouped.csv')
    # print(results_grouped[['tracking_error', 'energy_tracking_error']])
    # print(results_grouped[['tracking_error',
    #       'total_transformer_overload', 'time']])

    # sort results by tracking error
    results_grouped = results_grouped.sort_values(
        by=('tracking_error', 'mean'), ascending=True)

    print(results_grouped[['total_reward',
                           'tracking_error',
                            'energy_tracking_error',
                            'average_user_satisfaction',
                            #'profits_from_customers',
                            #'total_profits',
                           ]])

    with gzip.open(save_path + 'plot_results_dict.pkl.gz', 'wb') as f:
        pickle.dump(plot_results_dict, f)

    algorithm_names = []
    for algorithm in algorithms:
        # if class has attribute .name, use it
        if hasattr(algorithm, 'algo_name'):
            algorithm_names.append(algorithm.algo_name)
        elif type(algorithm) == str:
            #  algorithm_names.append(algorithm.split('_')[0])
            algorithm_names.append(algorithm.split('/')[-1])
        else:
            algorithm_names.append(algorithm.__name__)

    # save algorithm names to a txt file
    with open(save_path + 'algorithm_names.txt', 'w') as f:
        for item in algorithm_names:
            f.write("%s\n" % item)

    print(f'Plottting results at {save_path}')

    plot_actual_power_vs_setpoint(results_path=save_path + 'plot_results_dict.pkl.gz',
                                  save_path=save_path,
                                  algorithm_names=algorithm_names)

    # plot_total_power(results_path=save_path + 'plot_results_dict.pkl',
    #                  save_path=save_path,
    #                  algorithm_names=algorithm_names)

    # plot_comparable_EV_SoC(results_path=save_path + 'plot_results_dict.pkl',
    #                        save_path=save_path,
    #                        algorithm_names=algorithm_names)

    # plot_total_power_V2G(results_path=save_path + 'plot_results_dict.pkl',
    #                      save_path=save_path,
    #                      algorithm_names=algorithm_names)

    # plot_comparable_EV_SoC_single(results_path=save_path + 'plot_results_dict.pkl',
    #                               save_path=save_path,
    #                               algorithm_names=algorithm_names)

    # plot_prices(results_path=save_path + 'plot_results_dict.pkl',
    #             save_path=save_path,
    #             algorithm_names=algorithm_names)


if __name__ == "__main__":
    evaluator()
