# this file is used to evalaute the performance of the ev2gym environment with various stable baselines algorithms.

from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib import TQC, TRPO, ARS, RecurrentPPO

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.rl_agent.reward import SquaredTrackingErrorReward, ProfitMax_TrPenalty_UserIncentives
from ev2gym.rl_agent.reward import profit_maximization

from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, V2G_profit_max_loads

import gymnasium as gym
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
import os
import yaml

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default="ppo")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--train_steps', type=int, default=20_000) 
    parser.add_argument('--run_name', type=str, default="")
    parser.add_argument('--config_file', type=str,
                        # default="ev2gym/example_config_files/V2GProfitMax.yaml")
    default="ev2gym/example_config_files/V2GProfitPlusLoads.yaml")

    algorithm = parser.parse_args().algorithm
    device = parser.parse_args().device
    run_name = parser.parse_args().run_name
    config_file = parser.parse_args().config_file

    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)

    if config_file == "ev2gym/example_config_files/V2GProfitMax.yaml":
        reward_function = profit_maximization
        state_function = V2G_profit_max
        group_name = f'{config["number_of_charging_stations"]}cs_V2GProfitMax'

    elif config_file == "ev2gym/example_config_files/PublicPST.yaml":
        reward_function = SquaredTrackingErrorReward
        state_function = PublicPST
        group_name = f'{config["number_of_charging_stations"]}cs_PublicPST'
    elif config_file == "ev2gym/example_config_files/V2GProfitPlusLoads.yaml":
        reward_function = ProfitMax_TrPenalty_UserIncentives
        state_function = V2G_profit_max_loads
        group_name = f'{config["number_of_charging_stations"]}cs_V2GProfitPlusLoads'
                
    run_name += f'{algorithm}_{reward_function.__name__}_{state_function.__name__}'

    run = wandb.init(project='ev2gym',
                     sync_tensorboard=True,
                     group=group_name,
                     name=run_name,
                     save_code=True,
                     )

    gym.envs.register(id='evs-v0', entry_point='ev2gym.models.ev2gym_env:EV2Gym',
                      kwargs={'config_file': config_file,
                              'verbose': False,
                              'save_plots': False,
                              'generate_rnd_game': True,
                              'reward_function': reward_function,
                              'state_function': state_function,
                              })

    env = gym.make('evs-v0')

    eval_log_dir = "./eval_logs/" + group_name + "_" + run_name + "/"
    save_path = f"./saved_models/{group_name}/{run_name}/"
    
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(f"./saved_models/{group_name}", exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    
    print(f'Model will be saved at: {save_path}')

    eval_callback = EvalCallback(env,
                                 best_model_save_path=save_path,
                                 log_path=eval_log_dir,
                                 eval_freq=config['simulation_length']*30,
                                 n_eval_episodes=50,
                                 deterministic=True)

    if algorithm == "ddpg":
        model = DDPG("MlpPolicy", env, verbose=1,
                    learning_rate = 1e-3,
                    buffer_size = 1_000_000,  # 1e6
                    learning_starts = 100,
                    batch_size = 100,
                    tau = 0.005,
                    gamma = 0.99,                     
                     device=device, tensorboard_log="./logs/")
    elif algorithm == "td3":
        model = TD3("MlpPolicy", env, verbose=1,
                    device=device, tensorboard_log="./logs/")
    elif algorithm == "sac":
        model = SAC("MlpPolicy", env, verbose=1,
                    device=device, tensorboard_log="./logs/")
    elif algorithm == "a2c":
        model = A2C("MlpPolicy", env, verbose=1,
                    device=device, tensorboard_log="./logs/")
    elif algorithm == "ppo":
        model = PPO("MlpPolicy", env, verbose=1,
                    device=device, tensorboard_log="./logs/")
    elif algorithm == "tqc":
        model = TQC("MlpPolicy", env, verbose=1,
                    device=device, tensorboard_log="./logs/")
    elif algorithm == "trpo":
        model = TRPO("MlpPolicy", env, verbose=1,
                     device=device, tensorboard_log="./logs/")
    elif algorithm == "ars":
        model = ARS("MlpPolicy", env, verbose=1,
                    device=device, tensorboard_log="./logs/")
    elif algorithm == "rppo":
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=1,
                             device=device, tensorboard_log="./logs/")
    else:
        raise ValueError("Unknown algorithm")

    model.learn(total_timesteps=parser.parse_args().train_steps,
                progress_bar=True,
                callback=[
                    WandbCallback(
                        verbose=2),
                    eval_callback])
    # model.save(f"./saved_models/{group_name}/{run_name}.last")
    print(f'Finished training {algorithm} algorithm, {run_name} saving model at {save_path}_last.pt')

    model.save(f"{save_path}/last_model.zip")    
    
    #load the best model
    model = model.load(f"{save_path}/best_model.zip", env=env)

    env = model.get_env()
    obs = env.reset()

    stats = []
    for i in range(96*100):

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # env.render()
        # VecEnv resets automatically
        if done:
            stats.append(info)
            obs = env.reset()

    # print average stats
    print("=====================================================")
    print(f' Average stats for {algorithm} algorithm, {len(stats)} episodes')
    print("total_ev_served: ", sum(
        [i[0]['total_ev_served'] for i in stats])/len(stats))
    print("total_profits: ", sum(
        [i[0]['total_profits'] for i in stats])/len(stats))
    print("total_energy_charged: ", sum(
        [i[0]['total_energy_charged'] for i in stats])/len(stats))
    print("total_energy_discharged: ", sum(
        [i[0]['total_energy_discharged'] for i in stats])/len(stats))
    print("average_user_satisfaction: ", sum(
        [i[0]['average_user_satisfaction'] for i in stats])/len(stats))
    print("power_tracker_violation: ", sum(
        [i[0]['power_tracker_violation'] for i in stats])/len(stats))
    print("tracking_error: ", sum(
        [i[0]['tracking_error'] for i in stats])/len(stats))
    print("energy_user_satisfaction: ", sum(
        [i[0]['energy_user_satisfaction'] for i in stats])/len(stats))
    print("total_transformer_overload: ", sum(
        [i[0]['total_transformer_overload'] for i in stats])/len(stats))
    print("reward: ", sum([i[0]['episode']['r'] for i in stats])/len(stats))

    run.log({
        "test/total_ev_served": sum([i[0]['total_ev_served'] for i in stats])/len(stats),
        "test/total_profits": sum([i[0]['total_profits'] for i in stats])/len(stats),
        "test/total_energy_charged": sum([i[0]['total_energy_charged'] for i in stats])/len(stats),
        "test/total_energy_discharged": sum([i[0]['total_energy_discharged'] for i in stats])/len(stats),
        "test/average_user_satisfaction": sum([i[0]['average_user_satisfaction'] for i in stats])/len
        (stats),
        "test/power_tracker_violation": sum([i[0]['power_tracker_violation'] for i in stats])/len(stats),
        "test/tracking_error": sum([i[0]['tracking_error'] for i in stats])/len(stats),
        "test/energy_user_satisfaction": sum([i[0]['energy_user_satisfaction'] for i in stats])/len
        (stats),
        "test/total_transformer_overload": sum([i[0]['total_transformer_overload'] for i in stats])/len
        (stats),
        "test/reward": sum([i[0]['episode']['r'] for i in stats])/len(stats),
    })

    run.finish()
