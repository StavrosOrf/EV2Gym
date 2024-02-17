# this file is used to evalaute the performance of the EVsSimulator environment with various stable baselines algorithms.

from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib import TQC, TRPO, ARS, RecurrentPPO

from EVsSimulator.ev_city import EVsSimulator
from EVsSimulator.rl_agent.reward import SquaredTrackingErrorReward

import gymnasium as gym
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
import os
import yaml


class ExtraLoggingCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    # use the _on_rollout_end method for logging end-of-epoch metrics, rather than _on_step
    def _on_rollout_end(self) -> None:
        net_change = self.training_env.envs[0].list_networth[-1] - \
            self.training_env.envs[0].list_networth[0]
        self.logger.record("net_change", net_change)

        print(f'Net Change: {net_change}')

    def _on_step(self) -> bool:
        return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default="ddpg")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--run_name', type=str, default="")
    parser.add_argument('--config_file', type=str,
                        default="EVsSimulator/example_config_files/PublicPST.yaml")

    algorithm = parser.parse_args().algorithm
    device = parser.parse_args().device
    run_name = parser.parse_args().run_name
    config_file = parser.parse_args().config_file

    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)

    run = wandb.init(project='EVsSimulator',
                     sync_tensorboard=True,
                     group=f'30cs_PST',
                     name=f'{algorithm}_{run_name}',
                     save_code=True,
                     )

    gym.envs.register(id='evs-v0', entry_point='EVsSimulator.ev_city:EVsSimulator',
                      kwargs={'config_file': config_file,
                              'verbose': False,
                              'save_plots': False,
                              'generate_rnd_game': True,
                              'reward_function': SquaredTrackingErrorReward,
                              })

    env = gym.make('evs-v0')

    eval_log_dir = "./eval_logs/"
    os.makedirs(eval_log_dir, exist_ok=True)

    eval_callback = EvalCallback(env, best_model_save_path=eval_log_dir,
                                 log_path=eval_log_dir,
                                 eval_freq=config['simulation_length']*50,
                                 n_eval_episodes=10, deterministic=True,
                                 render=False)

    if algorithm == "ddpg":
        model = DDPG("MlpPolicy", env, verbose=1,
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

    model.learn(total_timesteps=2_000_000,
                progress_bar=True,
                callback=[
                    WandbCallback(
                        gradient_save_freq=100000,
                        model_save_path=f"models/{run.id}",
                        verbose=2),
                    eval_callback])

    model.save("./saved_models/"+algorithm +
               "_30cs_1_port_SquaredTrackingErrorReward")

    env = model.get_env()
    obs = env.reset()

    stats = []
    for i in range(96*2):

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

    # print last stats

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