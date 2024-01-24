# this file is used to evalaute the performance of the EVsSimulator environment with various stable baselines algorithms.

from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC, HER, DQN, TD3
from EVsSimulator.ev_city import EVsSimulator

import pkg_resources
import gymnasium as gym

# config_file = "/example_config_files/simple_config.yaml"
# config_file = pkg_resources.resource_filename('EVsSimulator', config_file)
config_file = "EVsSimulator/example_config_files/simple_config.yaml"

gym.envs.register(id='evs-v0', entry_point='EVsSimulator.ev_city:EVsSimulator',
                  kwargs={'config_file': config_file,
                          'verbose': False})

if __name__ == "__main__":
    env = gym.make('evs-v0')

    # model = TD3("MlpPolicy", env, verbose=1,device='cuda',tensorboard_log="./logs/")
    model = PPO("MlpPolicy", env, verbose=1,device='cuda',tensorboard_log="./logs/")
    model.learn(total_timesteps=1_000_000,progress_bar=True)
    
    model.save("./saved_models/ppo_evs")
    del model  # delete trained model to demonstrate loading
    
    model = PPO.load("./saved_models/ppo_evs",env=env,device='cuda')
    env = model.get_env()
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # env.render()
        # VecEnv resets automatically
        if done:
            print(info)
            obs = env.reset()