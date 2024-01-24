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
                          'verbose': False,
                          'save_plots': False,})

if __name__ == "__main__":
    env = gym.make('evs-v0')

    model = TD3("MlpPolicy", env, verbose=1,device='cuda',tensorboard_log="./logs/")
    # # model = PPO("MlpPolicy", env, verbose=1,device='cuda',tensorboard_log="./logs/")
    
    model.learn(total_timesteps=1_000_000,progress_bar=True,tb_log_name="td3_evs")
    
    model.save("./saved_models/td3_evs")
    # exit()
    # del model  # delete trained model to demonstrate loading
    
    model = TD3.load("./saved_models/td3_evs",env=env,device='cuda')
    env = model.get_env()
    obs = env.reset()
    
    stats = []
    for i in range(34*3):
            
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # env.render()
            # VecEnv resets automatically
            if done:
                stats.append(info)
                obs = env.reset()   
    
#     print(stats)
#     info = [{'total_ev_served': 0, 'total_profits': -2.7930474555562457, 'total_energy_charged': 14.000000000000002, 'total_energy_discharged': 0, 'average_user_satisfaction': 0.0, 'power_tracker_violation': 7.7601508515419635, 'tracking_error': 34.25415943340212, 'energy_user_satisfaction': 100, 'total_transformer_overload': 0.0, 'episode': {'r': -34.254159, 'l': 34, 't': 1.253219}, 'TimeLimit.truncated': 
# False, 'terminal_observation': array([ 1.        ,  0.14285714, -0.8660254 , -0.5       ,  0.        ,
#         0.        ,  0.035     ,  0.27612404,  0.05020437,  0.88235294])}]
    # print average stats
    print("Average stats:")
    print("total_ev_served: ", sum([i[0]['total_ev_served'] for i in stats])/len(stats))
    print("total_profits: ", sum([i[0]['total_profits'] for i in stats])/len(stats))
    print("total_energy_charged: ", sum([i[0]['total_energy_charged'] for i in stats])/len(stats))
    print("total_energy_discharged: ", sum([i[0]['total_energy_discharged'] for i in stats])/len(stats))
    print("average_user_satisfaction: ", sum([i[0]['average_user_satisfaction'] for i in stats])/len(stats))
    print("power_tracker_violation: ", sum([i[0]['power_tracker_violation'] for i in stats])/len(stats))
    print("tracking_error: ", sum([i[0]['tracking_error'] for i in stats])/len(stats))
    print("energy_user_satisfaction: ", sum([i[0]['energy_user_satisfaction'] for i in stats])/len(stats))
    print("total_transformer_overload: ", sum([i[0]['total_transformer_overload'] for i in stats])/len(stats))
    print("reward: ", sum([i[0]['episode']['r'] for i in stats])/len(stats))
    
    

    
    # print last stats
    

    