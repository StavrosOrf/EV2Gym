"""
This script is used to evaluate the performance of the EVsSimulator environment.
"""
from EVsSimulator.ev_city import EVsSimulator
from EVsSimulator.baselines.gurobi_models.tracking_error import PowerTrackingErrorrMin
from EVsSimulator.baselines.gurobi_models.profit_max import V2GProfitMaxOracleGB
from EVsSimulator.baselines.mpc.occf_mpc import OCCF_V2G, OCCF_G2V
from EVsSimulator.baselines.mpc.eMPC import eMPC_V2G, eMPC_G2V
from EVsSimulator.baselines.mpc.V2GProfitMax import V2GProfitMaxOracle

from EVsSimulator.baselines.heuristics import RoundRobin, ChargeAsLateAsPossible, ChargeAsFastAsPossible
from EVsSimulator.baselines.heuristics import ChargeAsFastAsPossibleToDesiredCapacity

import numpy as np
import matplotlib.pyplot as plt
import pkg_resources


def eval():
    """
    Runs an evaluation of the EVsSimulator environment.
    """

    verbose = True
    save_plots = True
    
    replay_path = "./replay/replay_sim_2024_02_21_056441.pkl"
    replay_path = None

    config_file = "EVsSimulator/example_config_files/V2G_MPC.yaml"
    # config_file = "EVsSimulator/example_config_files/PublicPST.yaml"
    # config_file = "EVsSimulator/example_config_files/BusinessPST.yaml"
    # config_file = "EVsSimulator/example_config_files/V2GProfitPlusLoads.yaml"
    

    env = EVsSimulator(config_file=config_file,
                       load_from_replay_path=replay_path,
                       verbose=False,
                       save_replay=True,
                       empty_ports_at_end_of_simulation=True,
                       save_plots=save_plots,
                    #    seed=42,
                       #    render_mode = True,
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
    # exit()
    # agent = OCCF_V2G(env, control_horizon=30, verbose=True)
    # agent = OCCF_G2V(env, control_horizon=25, verbose=True)
    # agent = eMPC_V2G(env, control_horizon=25, verbose=True)
    # agent = V2GProfitMaxOracle(env,verbose=True)
    # agent = eMPC_G2V(env, control_horizon=15, verbose=True)
    agent = RoundRobin(env, verbose=False)
    # agent = ChargeAsLateAsPossible(verbose=False)
    # agent = ChargeAsFastAsPossible()
    # agent = ChargeAsFastAsPossibleToDesiredCapacity()
    rewards = []

    for t in range(env.simulation_length):        
        actions = agent.get_action(env)

        new_state, reward, done, truncated, _ = env.step(
            actions, visualize=False)  # takes action
        rewards.append(reward)

        if done:
            print(f'End of simulation at step {env.current_step}')
            break
    
    # exit()
    # Solve optimally
    # Power tracker optimizer
    # agent = PowerTrackingErrorrMin(replay_path=new_replay_path)
    # # Profit maximization optimizer
    # # agent = V2GProfitMaxOracleGB(replay_path=new_replay_path)    
    # # Simulate in the gym environment and get the rewards

    # env = EVsSimulator(config_file=config_file,
    #                    load_from_replay_path=new_replay_path,
    #                    verbose=True,
    #                    save_plots=True,
    #                    )
    # state, _ = env.reset()
    # rewards_opt = []

    # for t in range(env.simulation_length):
    #     actions = agent.get_action(env)
    #     if verbose:
    #         print(f' OptimalActions: {actions}')

    #     new_state, reward, done, truncated, _ = env.step(
    #         actions, visualize=True)  # takes action
    #     rewards_opt.append(reward)

    #     if verbose:
    #         print(f'Reward: {reward} \t Done: {done}')

    #     if done:
    #         break

if __name__ == "__main__":
    while True:
        eval()
        # exit()
