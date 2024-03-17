import argparse
import logging
import os
import random
import time

import gym
import numpy as np
from ev2gym.models import ev2gym_env

import torch
import pickle

from ev2gym.utilities.arg_parser import arg_parser

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    args = arg_parser()

    # Define the directory where to save and load models
    checkpoint_dir = args.save_dir + args.env
    # name the run accordign to time
    if args.name:
        run_name = args.name
    else:
        run_name = 'r_' + time.strftime("%Y%m%d-%H%M%S")

    log_to_wandb = args.wandb
    verbose = False
    n_transformers = args.transformers
    number_of_charging_stations = args.cs
    steps = args.steps  # 288 steps = 1 day with 5 minutes per step
    timescale = args.timescale  # (5 minutes per step)    
    n_trajectories = args.n_trajectories

    gym.register(id='evcity-v0', entry_point='gym_env.ev_city:EVCity')

    env = ev2gym_env.EV2Gym(cs=number_of_charging_stations,
                         number_of_transformers=n_transformers,
                         load_from_replay_path=None,
                         generate_rnd_game=True,
                         simulation_length=steps,
                         timescale=timescale,
                         save_plots=False,
                         save_replay=False,)

    trajectories = []

    # Set random seed for all used libraries where possible
    # env.seed(args.seed)
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Define and build DDPG agent
    hidden_size = tuple(args.hidden_size)

    if args.dataset not in ["ddpg","random","optimal"]:
        raise ValueError("Dataset not supported")
    
    trajecotries_type = args.dataset



    file_name = f"{trajecotries_type}_{number_of_charging_stations}_cs_{n_transformers}_tr_{steps}_steps_{timescale}_timescale_{n_trajectories}_trajectories.pkl"
    save_folder_path = f"./trajectories/"
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # Initialize OU-Noise
    nb_actions = env.action_space.shape[-1]

    # Define counters and other variables
    start_step = 0
    # timestep = start_step


        

    timestep = start_step // 10000 + 1
    epoch = 0

    for i in range(n_trajectories):

        trajectory_i = {"observations": [],
                        "actions": [],
                        "rewards": [],
                        "dones": [] }

        epoch_return = 0

        print(f'Trajectory: {i}')
        state = torch.Tensor([env.reset()]).to(device)       

        state = torch.Tensor([env.reset()]).to(device)
        test_reward = 0
        while True:
            
            if args.dataset == "random":
                action = torch.rand(1,env.number_of_ports) #*2-1

            next_state, reward, done, stats = env.step(
                action.cpu().numpy()[0])
            test_reward += reward

            trajectory_i["observations"].append(state.cpu().numpy()[0])
            trajectory_i["actions"].append(action.cpu().numpy()[0])
            trajectory_i["rewards"].append(reward)
            trajectory_i["dones"].append(done)

            next_state = torch.Tensor([next_state]).to(device)
            state = next_state

            if done:
                break
        
        trajectory_i["observations"] = np.array(trajectory_i["observations"])
        trajectory_i["actions"] = np.array(trajectory_i["actions"])
        trajectory_i["rewards"] = np.array(trajectory_i["rewards"])
        trajectory_i["dones"] = np.array(trajectory_i["dones"])

        trajectories.append(trajectory_i)

        if i % 10000 == 0:
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
