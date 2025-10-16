"""
This script is used to evaluate the performance of the ev2gym environment.
"""
from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.gurobi_models.tracking_error import PowerTrackingErrorrMin
from ev2gym.baselines.gurobi_models.profit_max import V2GProfitMaxOracleGB
from ev2gym.baselines.mpc.ocmf_mpc import OCMF_V2G, OCMF_G2V
from ev2gym.baselines.mpc.eMPC_v2 import eMPC_V2G_v2, eMPC_G2V_v2

from ev2gym.rl_agent.action_wrappers import BinaryAction, ThreeStep_Action, Rescale_RepairLayer

from ev2gym.baselines.mpc.V2GProfitMax import V2GProfitMaxOracle

from ev2gym.baselines.heuristics import RoundRobin, ChargeAsLateAsPossible, ChargeAsFastAsPossible
from ev2gym.baselines.heuristics import ChargeAsFastAsPossibleToDesiredCapacity, RandomAgent

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


def eval():
    """
    Runs an evaluation of the ev2gym environment.
    """

    save_plots = True

    replay_path = "./replay/replay_sim_2024_07_05_106720.pkl"
    replay_path = None

    config_file = "ev2gym/example_config_files/PublicPST.yaml"
    config_file = "ev2gym/example_config_files/V2GProfitMax.yaml"
    config_file = "ev2gym/example_config_files/V2GProfitPlusLoads.yaml"
    config_file = "ev2gym/example_config_files/V2Ggrid.yaml"

    env = EV2Gym(config_file=config_file,
                 load_from_replay_path=replay_path,
                 verbose=False,
                 save_replay=True,
                 save_plots=save_plots,      
                 )

    new_replay_path = f"replay/replay_{env.sim_name}.pkl"

    state, _ = env.reset()

    # agent = OCMF_V2G(env, control_horizon=30, verbose=True)
    # agent = OCMF_G2V(env, control_horizon=25, verbose=True)
    # agent = eMPC_V2G(env, control_horizon=15, verbose=False)
    # agent = V2GProfitMaxOracle(env,verbose=True)
    # agent = PowerTrackingErrorrMin(new_replay_path)
    # agent = eMPC_G2V(env, control_horizon=15, verbose=False)
    # agent = eMPC_V2G_v2(env, control_horizon=10, verbose=False)        
    # agent = RoundRobin(env, verbose=False)
    # agent = ChargeAsLateAsPossible(verbose=False)
    # agent = ChargeAsFastAsPossible()
    agent = RandomAgent(verbose=False)
    # agent = ChargeAsFastAsPossibleToDesiredCapacity()
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

    return
    # Solve optimally
    # Power tracker optimizer
    agent = PowerTrackingErrorrMin(replay_path=new_replay_path)
    # # Profit maximization optimizer
    # agent = V2GProfitMaxOracleGB(replay_path=new_replay_path)
    # # Simulate in the gym environment and get the rewards

    env = EV2Gym(config_file=config_file,
                       load_from_replay_path=new_replay_path,
                       verbose=False,
                       save_plots=True,
                       )
    state, _ = env.reset()
    rewards_opt = []

    for t in range(env.simulation_length):
        actions = agent.get_action(env)
        # if verbose:
        #     print(f' OptimalActions: {actions}')

        new_state, reward, done, truncated, stats = env.step(
            actions, visualize=False)  # takes action
        rewards_opt.append(reward)
        
        # if verbose:
        #     print(f'Reward: {reward} \t Done: {done}')

        if done:
            print(stats)
            break


if __name__ == "__main__":
    # while True:
        eval()    
