

"""
This script is used to evaluate the performance of the ev2gym environment.
"""
from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.gurobi_models.tracking_error import PowerTrackingErrorrMin
from ev2gym.baselines.gurobi_models.profit_max import V2GProfitMaxOracleGB
from ev2gym.baselines.mpc.ocmf_mpc import OCMF_V2G, OCMF_G2V
from ev2gym.baselines.mpc.eMPC import eMPC_V2G, eMPC_G2V

from ev2gym.baselines.mpc.eMPC_v2 import eMPC_V2G_v2, eMPC_G2V_v2

from ev2gym.baselines.mpc.V2GProfitMax import V2GProfitMaxOracle

from ev2gym.baselines.heuristics import RoundRobin, ChargeAsLateAsPossible, ChargeAsFastAsPossible
from ev2gym.baselines.heuristics import ChargeAsFastAsPossibleToDesiredCapacity

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import time
import datetime
from typing import List, Dict
import concurrent.futures
# Save the load profiles to a csv file
import pandas as pd
import os

from ev2gym.models.ev import EV

def eval(number_of_scenarios: int = 1) -> None:
    """
    Runs an evaluation of the ev2gym environment.
    """

    config_file = "ev2gym/example_config_files/residential.yaml"

    env = EV2Gym(config_file=config_file)
    
    load_profiles_list = []

    counter = 1
    while counter <= number_of_scenarios:
        
        timer_start = time.time()
        state, _ = env.reset()

        ev_profiles = env.EVs_profiles
        max_time_of_stay = max([ev.time_of_departure - ev.time_of_arrival
                                for ev in ev_profiles])
        min_time_of_stay = min([ev.time_of_departure - ev.time_of_arrival
                                for ev in ev_profiles])

        # print(f'Number of EVs: {len(ev_profiles)}')
        # print(f'Max time of stay: {max_time_of_stay}')
        # print(f'Min time of stay: {min_time_of_stay}')

        agent = ChargeAsFastAsPossible()

        rewards = []

        # for t in range(env.simulation_length):
        while True:
            actions = agent.get_action(env)

            new_state, reward, done, truncated, stats = env.step(
                actions)  # takes action
            rewards.append(reward)

            if done:
                # print(stats)
                print(f'End of simulation at step {env.current_step}')
                break

        # print(env.port_energy_level)
        print(f'shape of port_energy_level: {env.port_energy_level.shape}')

        print(f'Time taken: {time.time() - timer_start} seconds')
        load_profiles_list.append(env.port_energy_level)
        print(f'List of load profiles: {len(load_profiles_list)}')
    
    return

import time
import concurrent.futures
import copy

def run_simulation(args):
    simulation_id, config_file = args
    
    print(f'Running simulation {simulation_id}...')
    env = EV2Gym(config_file=config_file)
    env.reset()
    agent = ChargeAsFastAsPossible()
    
    while True:
        actions = agent.get_action(env)
        _, _, done, _, _ = env.step(actions)
        if done:
            break
    print(f'Simulation {simulation_id} completed.')
    return env.cs_power

def run_parallel_simulations(number_of_scenarios):
    timer_start = time.time()
    config_file = "ev2gym/example_config_files/residential.yaml"

    print(f'Starting parallel simulations with {number_of_scenarios} scenarios...')

    total_cores = os.cpu_count()
    print(f'Total cores available: {total_cores}')
    workers = max(1, total_cores - 1)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        args = [(i, config_file) for i in range(1, number_of_scenarios + 1)]
        results = executor.map(run_simulation, args)
        load_profiles_list = list(results)

    print(f'Total scenarios run: {len(load_profiles_list)}')
    print(f'Total time taken: {time.time() - timer_start:.2f} seconds')
    return load_profiles_list


if __name__ == "__main__":
    # eval(number_of_scenarios=1)
    # exit
    
    if True:
        number_of_scenarios = 100  # Number of scenarios to run in parallel
        
        load_profiles_list = run_parallel_simulations(number_of_scenarios)
        
        print(f'List of load profiles: {len(load_profiles_list)}')    
        #make into 2d array (-1, last dimension)
        load_profiles_list = np.array(load_profiles_list)
        #reduce floating point accuracy
        load_profiles_list = np.round(load_profiles_list, 2)
        # make into float8
        load_profiles_list = load_profiles_list.astype(np.float16)    
        load_profiles_list = np.reshape(load_profiles_list, (-1, load_profiles_list.shape[-1]))
        
        #save as zipped pickle file
        dt_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        np.savez_compressed(f'load_profiles_{dt_string}.npz', load_profiles_list=load_profiles_list)
            
        print(f'List of load profiles shape: {load_profiles_list.shape}')
        
        # # Save the load profiles to a csv file    
        # df = pd.DataFrame(load_profiles_list)
        # df.to_csv('load_profiles.csv', index=False)
        
    # Load the load profiles from the npz file
    #get datetime string
    
    load_profiles = np.load(f'load_profiles_{dt_string}.npz')['load_profiles_list']
    print(f'Loaded load profiles shape: {load_profiles.shape}')
    
    #plot the load profiles
    plt.figure(figsize=(10, 6))
    plt.plot(load_profiles[:1,:].T, alpha=0.5)
    plt.show()