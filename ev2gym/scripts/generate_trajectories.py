
import os
import time
import numpy as np
import pickle
import yaml
from tqdm import tqdm 

from ev2gym.models import ev2gym_env
from ev2gym.utilities.arg_parser import arg_parser
from ev2gym.rl_agent.reward import SquaredTrackingErrorReward, ProfitMax_TrPenalty_UserIncentives, profit_maximization
from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, V2G_profit_max_loads
from ev2gym.baselines.heuristics import RoundRobin, ChargeAsLateAsPossible, ChargeAsFastAsPossible

if __name__ == "__main__":

    args = arg_parser()

    # Define the directory where to save and load models
    checkpoint_dir = args.save_dir + args.env

    reward_function = SquaredTrackingErrorReward
    state_function = PublicPST
    problem = "PublicPST"

    env = ev2gym_env.EV2Gym(config_file=args.config_file,
                            generate_rnd_game=True,
                            state_function=state_function,
                            reward_function=reward_function)
    
    n_trajectories = args.n_trajectories
    config = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)
    number_of_charging_stations = config["number_of_charging_stations"]
    n_transformers = config["number_of_transformers"]
    steps = config["simulation_length"]
    timescale = config["timescale"]
    
    trajectories = []

    # if args.dataset not in ["ddpg", "random", "optimal"]:
    #     raise ValueError("Dataset not supported")

    trajecotries_type = "mixed-RR-Asap" #args.dataset

    file_name = f"{problem}_{trajecotries_type}_{number_of_charging_stations}_cs_{n_transformers}_tr_{steps}_steps_{timescale}_timescale_{n_trajectories}_trajectories2.pkl"
    save_folder_path = f"./trajectories/"
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    epoch = 0
    # use tqdm with a fancy bar
    for i in tqdm(range(n_trajectories)):

        trajectory_i = {"observations": [],
                        "actions": [],
                        "rewards": [],
                        "dones": []}

        epoch_return = 0

        # print(f'Trajectory: {i}')
        state, _ = env.reset()
        if i % 2 == 0:
            agent = ChargeAsFastAsPossible(env)
        else:
            agent = RoundRobin(env)
        
        test_reward = 0
        while True:

            actions = agent.get_action(env)

            new_state, reward, done, truncated, _ = env.step(actions)
            test_reward += reward

            trajectory_i["observations"].append(state)
            trajectory_i["actions"].append(actions)
            trajectory_i["rewards"].append(reward)
            trajectory_i["dones"].append(done)            
            state = new_state

            if done:
                break

        trajectory_i["observations"] = np.array(trajectory_i["observations"])
        trajectory_i["actions"] = np.array(trajectory_i["actions"])
        trajectory_i["rewards"] = np.array(trajectory_i["rewards"])
        trajectory_i["dones"] = np.array(trajectory_i["dones"])

        trajectories.append(trajectory_i)

        if i % 100000 == 0:
            print(f'Saving trajectories to {save_folder_path+file_name}')
            f = open(save_folder_path+file_name, 'wb')
            # source, destination
            pickle.dump(trajectories, f)

    env.close()
    print(trajectories[:1])

    print(f'Saving trajectories to {save_folder_path+file_name}')
    f = open(save_folder_path+file_name, 'wb')
    # source, destination
    pickle.dump(trajectories, f)
    f.close()
