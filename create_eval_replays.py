from gym_env import ev_city
from math_model import ev_city_model

import numpy as np
import os

""""
This file is used to create replay files with optimal profits for evaluation purposes.
The replay files are saved in the replay folder and can be used to evaluate the performance of the RL agent."""


def evalreplay(number_of_charging_stations,n_transformers,steps,timescale,static_prices):
    verbose = False

    env = ev_city.EVCity(cs=number_of_charging_stations,
                         number_of_ports_per_cs=2,
                         number_of_transformers=n_transformers,
                         static_ev_spawn_rate = True,
                         load_ev_from_replay=True,
                         load_prices_from_replay=True,
                         static_prices=static_prices,
                         load_from_replay_path=None,
                         empty_ports_at_end_of_simulation=True,
                         generate_rnd_game=True,
                         simulation_length=steps,
                         timescale=timescale,
                         score_threshold=1,                         
                         save_plots=False,
                         save_replay=True,
                         verbose=verbose,)

    new_replay_path = f"replay/replay_{env.sim_name}.pkl"
    # new_replay_path = replay_path
    
    _ = env.reset()
    rewards = []    

    for i in range(steps):        
        # all ports are charging instantly
        actions = np.ones(env.number_of_ports)
        if verbose:
            print(f'Actions: {actions}')

        new_state, reward, done, _ = env.step(
            actions, visualize=True)  # takes action
        rewards.append(reward)

        if verbose:
            print(f'Reward: {reward} \t Done: {done}')

        if done and i < steps - 1:
            # print(f'End of simulation at step {i}')
            exit()

    # Solve optimally
    math_model = ev_city_model.EV_City_Math_Model(sim_file_path=new_replay_path)    
    opt_actions = math_model.get_actions()    

    if static_prices:
        prices = "static"
    else:
        prices = "dynamic"
        
    group_name = f'{number_of_charging_stations}cs_{n_transformers}tr_{prices}_prices'

    # Simulate in the gym environment and get the rewards
    env = ev_city.EVCity(cs=number_of_charging_stations,
                         number_of_transformers=n_transformers,
                         load_ev_from_replay=True,
                         load_prices_from_replay=True,
                         load_from_replay_path=new_replay_path,
                         empty_ports_at_end_of_simulation=True,
                         replay_path = "./replay/"+group_name+"/",
                         generate_rnd_game=False,
                         simulation_length=steps,
                         timescale=timescale,
                         save_plots=False,
                         save_replay=True,
                         verbose=verbose,)
    _ = env.reset()    
    rewards_opt = []
    
    for i in range(steps):        
        # all ports are charging instantly
        # print(f'Optimal actions: {opt_actions[:,:,i]}')
        # print(f'Optimal actions: {opt_actions[:,:,i].T.reshape(-1)}')
        actions = opt_actions[:,:,i].T.reshape(-1)        
        if verbose:
            print(f' OptimalActions: {actions}')

        new_state, reward, done, _ = env.step(
            actions, visualize=True)  # takes action
        rewards_opt.append(reward)

        if verbose:
            print(f'Reward: {reward} \t Done: {done}')

        if done:
            break    
    
    #delete the replay file
    os.remove(new_replay_path)

if __name__ == "__main__":

    eval_samples = 1000

    for i in range(eval_samples):
        evalreplay(number_of_charging_stations=1,
                   n_transformers=1,
                   steps=150,
                   timescale=5,
                   static_prices=True,)
        