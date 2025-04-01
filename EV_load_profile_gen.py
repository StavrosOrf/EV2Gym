

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

import time
import concurrent.futures
import copy

def run_simulation(config_file):
    env = EV2Gym(config_file=config_file)
    env.reset()
    agent = ChargeAsFastAsPossible()

    print(f'Running simulation...')
    while True:
        actions = agent.get_action(env)
        _, _, done, _, _ = env.step(actions)
        if done:
            break
    return env.port_energy_level

def run_parallel_simulations(number_of_scenarios):
    timer_start = time.time()
    config_file = "ev2gym/example_config_files/residential.yaml"

    print(f'Starting parallel simulations with {number_of_scenarios} scenarios...')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(run_simulation,[config_file] * number_of_scenarios)
        load_profiles_list = list(results)

    print(f'Total scenarios run: {len(load_profiles_list)}')
    print(f'Total time taken: {time.time() - timer_start} seconds')
    return load_profiles_list


if __name__ == "__main__":
    # eval(number_of_scenarios=1)
    # exit
    
    
    number_of_scenarios = 2  # Number of scenarios to run in parallel
    
    load_profiles_list = run_parallel_simulations(number_of_scenarios)
    
    print(f'List of load profiles: {len(load_profiles_list)}')
    print(f'Load profiles: {load_profiles_list}')
