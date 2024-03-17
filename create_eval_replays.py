from ev2gym.models import ev2gym_env
from ev2gym.baselines.gurobi_models.tracking_error import PowerTrackingErrorrMin

import numpy as np
import os
import pickle
from ev2gym.utilities.arg_parser import arg_parser
import yaml

""""
This file is used to create replay files with optimal profits for evaluation purposes.
The replay files are saved in the replay folder and can be used to evaluate the performance of the RL agent."""


def evalreplay(config_file,
               save_opt_trajectories,
               save_replay):

    verbose = False

    env = ev2gym_env.EV2Gym(config_file=config_file,
                               load_from_replay_path=None,
                               generate_rnd_game=True,
                               save_plots=False,
                               eval_mode="unstirred",
                               save_replay=True)

    new_replay_path = f"replay/replay_{env.sim_name}.pkl"
    # new_replay_path = replay_path

    _ = env.reset()
    rewards = []

    for i in range(steps):
        # all ports are charging instantly
        actions = np.ones(env.number_of_ports)
        if verbose:
            print(f'Actions: {actions}')

        new_state, reward, done, _ , _ = env.step(
            actions, visualize=False)  # takes action
        rewards.append(reward)

        if verbose:
            print(f'Reward: {reward} \t Done: {done}')

        if done and i < steps - 1:
            # print(f'End of simulation at step {i}')
            exit()

    # Solve optimally
    math_model = PowerTrackingErrorrMin(replay_path=new_replay_path)
    opt_actions = math_model.get_actions()

    scenario = config_file.split("/")[-1].split(".")[0]
    group_name = f'{number_of_charging_stations}cs_{n_transformers}tr_{scenario}'

    # Simulate in the gym environment and get the rewards
    # save replay in the replay folder for evaluating pther algorithms
    env = ev2gym_env.EV2Gym(config_file=config_file,
                               load_from_replay_path=new_replay_path,
                               replay_save_path="./replay/"+group_name+"/",
                               generate_rnd_game=False,
                               save_plots=False,
                               save_replay=save_replay,
                               eval_mode="optimal"
                               )
    _ = env.reset()
    rewards_opt = []

    trajectory_i = {"observations": [],
                    "actions": [],
                    "rewards": [],
                    "dones": []}

    for i in range(steps):
        actions = opt_actions[:, :, i].T.reshape(-1)
        if verbose:
            print(f' OptimalActions: {actions}')

        new_state, reward, done, _, _ = env.step(
            actions, visualize=False)  # takes action
        rewards_opt.append(reward)

        if save_opt_trajectories:
            trajectory_i["observations"].append(new_state)
            trajectory_i["actions"].append(actions)
            trajectory_i["rewards"].append(reward)
            trajectory_i["dones"].append(done)

        if verbose:
            print(f'Reward: {reward} \t Done: {done}')

        if done:
            break

    if save_opt_trajectories:
        trajectory_i["observations"] = np.array(trajectory_i["observations"])
        trajectory_i["actions"] = np.array(trajectory_i["actions"])
        trajectory_i["rewards"] = np.array(trajectory_i["rewards"])
        trajectory_i["dones"] = np.array(trajectory_i["dones"])

    # delete the replay file
    os.remove(new_replay_path)

    return trajectory_i


if __name__ == "__main__":

    args = arg_parser()

    trajectories = []

    config = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)

    number_of_charging_stations = config["number_of_charging_stations"]
    n_transformers = config["number_of_transformers"]
    steps = config["simulation_length"]
    timescale = config["timescale"]

    n_trajectories = args.n_trajectories
    save_opt_trajectories = args.save_opt_trajectories

    file_name = f"optimal_{number_of_charging_stations}_cs_{n_transformers}_tr_{steps}_steps_{timescale}_timescale_{n_trajectories}_trajectories.pkl"

    save_folder_path = f"./trajectories/"
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    for i in range(n_trajectories):
        print(f'Trajectory: {i}')
        trajectory = evalreplay(config_file=args.config_file,
                                save_opt_trajectories=save_opt_trajectories,
                                save_replay=True)
        trajectories.append(trajectory)

        if i % 1000 == 0 and save_opt_trajectories:
            print(f'Saving trajectories to {save_folder_path+file_name}')
            f = open(save_folder_path+file_name, 'wb')
            # source, destination
            pickle.dump(trajectories, f)
            f.close()
