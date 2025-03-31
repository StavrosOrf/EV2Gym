

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


def eval():
    """
    Runs an evaluation of the ev2gym environment.
    """

    save_plots = False
    replay_path = None

    config_file = "ev2gym/example_config_files/residential.yaml"

    env = EV2Gym(config_file=config_file,
                 load_from_replay_path=replay_path,
                 verbose=False,
                 save_replay=True,
                 save_plots=save_plots,
                 )


    new_replay_path = f"replay/replay_{env.sim_name}.pkl"

    state, _ = env.reset()

    ev_profiles = env.EVs_profiles
    max_time_of_stay = max([ev.time_of_departure - ev.time_of_arrival
                            for ev in ev_profiles])
    min_time_of_stay = min([ev.time_of_departure - ev.time_of_arrival
                            for ev in ev_profiles])

    print(f'Number of EVs: {len(ev_profiles)}')
    print(f'Max time of stay: {max_time_of_stay}')
    print(f'Min time of stay: {min_time_of_stay}')
    
    agent = ChargeAsFastAsPossible()

    rewards = []

    for t in range(env.simulation_length):
        actions = agent.get_action(env)

        new_state, reward, done, truncated, stats = env.step(
            actions)  # takes action
        rewards.append(reward)

        if done:
            print(stats)
            print(f'End of simulation at step {env.current_step}')
            break
    
    print(env.port_energy_level)
    print(f'shape of port_energy_level: {env.port_energy_level.shape}')
    
    return 



if __name__ == "__main__":
    # while True:
        eval()    
