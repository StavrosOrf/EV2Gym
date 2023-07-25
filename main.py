from gym_env import ev_city
from pyomo_model import ev_city_model

import numpy as np
import matplotlib.pyplot as plt

# main funtion for testing
if __name__ == "__main__":

    verbose = False
    number_of_charging_stations = 2000
    steps = 288 # 288 steps = 1 day with 5 minutes per step
    timescale = 5 #(5 minutes per step)

    env = ev_city.EVCity(cs=number_of_charging_stations,
                            timescale=timescale,
                            verbose=verbose,
                            simulation_length=steps)
    state = env.reset()
    
    env.visualize()
    rewards = []

    for i in range(steps):
        print("-"*80)
        # actions = env.action_space.sample()   # sample random actions
        # actions = np.random.uniform(-1, 1, env.number_of_ports)
        # all ports are charging instantly
        actions = np.ones(env.number_of_ports)
        if verbose:
            print(f'Actions: {actions}')

        new_state, reward, done = env.step(actions)  # takes action
        rewards.append(reward)
        env.visualize()

        if verbose:
            print(f'Reward: {reward} \t Done: {done}')

        if done:
            break
    
    if verbose:
        env.print_statistics()
    
    math_model = ev_city_model.EV_City_Math_Model(sim_file_path=f"replay/replay_{env.sim_name}.pkl")    
    #Example with t_departure
    # math_model = ev_city_model.EV_City_Math_Model(sim_file_path=f"replay/replay_ev_city_5_2023-07-25_20-23.pkl")    
    # a 5 steps case that was solved wrong
    # math_model = ev_city_model.EV_City_Math_Model(sim_file_path=f"replay/replay_ev_city_5_2023-07-25_15-08.pkl")    
    #a normal 5 steps case
    # math_model = ev_city_model.EV_City_Math_Model(sim_file_path=f"replay/replay_ev_city_10_2023-07-25_14-47.pkl")    
    
    exit()

    # Plot the commulative reward in subplot 1
    plt.subplot(2, 1, 1)
    plt.plot(np.cumsum(rewards))
    plt.ylabel('Cumulative reward')
    plt.title('Cumulative reward')

    # Plot the reward per step in subplot 2
    plt.subplot(2, 1, 2)
    plt.plot(rewards)
    plt.xlabel('Time step')
    plt.ylabel('Reward')
    plt.title('Reward per time step')
    plt.tight_layout()
    plt.show()
