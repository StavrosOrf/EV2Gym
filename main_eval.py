from gym_env import ev_city
from math_model import ev_city_model

import numpy as np
import matplotlib.pyplot as plt

# main funtion for testing
if __name__ == "__main__":

    verbose = False
    n_transformers = 1
    number_of_charging_stations = 1
    steps = 150  # 288 steps = 1 day with 5 minutes per step
    timescale = 5  # (5 minutes per step)
    save_plots = True

    replay_path = "replay/replay_ev_city_50_2023-07-27_10-12.pkl"
    replay_path = "replay/replay_ev_city_250_2023-07-27_16-48.pkl"
    replay_path = None

    env = ev_city.EVCity(cs=number_of_charging_stations,
                         number_of_ports_per_cs=2,
                         number_of_transformers=n_transformers,
                         load_ev_from_replay=True,
                         load_prices_from_replay=True,
                         load_from_replay_path=replay_path,
                         empty_ports_at_end_of_simulation=True,
                         generate_rnd_game=True,
                         simulation_length=steps,
                         timescale=timescale,
                         save_plots=False,
                         verbose=verbose,)

    new_replay_path = f"replay/replay_{env.sim_name}.pkl"
    # new_replay_path = replay_path
    
    state = env.reset()
    env.visualize()
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
            print(f'End of simulation at step {i}')
            exit()

    # env.plot()
    
    # Solve optimally
    math_model = ev_city_model.EV_City_Math_Model(sim_file_path=new_replay_path)
    # math_model = ev_city_model.EV_City_Math_Model(sim_file_path=f"replay/replay_ev_city_100_2023-07-26_14-19.pkl")
    opt_actions = math_model.get_actions()
    print(f'Optimal actions: {opt_actions.shape}')

    # Simulate in the gym environment and get the rewards
    env = ev_city.EVCity(cs=number_of_charging_stations,
                         number_of_transformers=n_transformers,
                         load_ev_from_replay=True,
                         load_prices_from_replay=True,
                         load_from_replay_path=new_replay_path,
                         empty_ports_at_end_of_simulation=True,
                         generate_rnd_game=False,
                         simulation_length=steps,
                         timescale=timescale,
                         verbose=verbose,)
    state = env.reset()
    env.visualize()
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

    if save_plots:
        plt.figure(figsize=(10, 10))
        # Plot the commulative reward in subplot 1
        plt.subplot(2, 1, 1)
        plt.plot(np.cumsum(rewards))
        plt.plot(np.cumsum(rewards_opt))
        plt.legend(['Charge Immediately', 'Optimal'])
        plt.ylabel('Cumulative reward')
        # plt.xticks(np.arange(0, steps, 1))
        plt.title('Cumulative reward')

        # Plot the reward per step in subplot 2
        plt.subplot(2, 1, 2)
        plt.plot(rewards)
        plt.plot(rewards_opt)
        plt.legend(['Charge Immediately', 'Optimal'])
        plt.xlabel('Time step')
        plt.ylabel('Reward')
        # plt.xticks(np.arange(0, steps, 1))
        plt.title('Reward per time step')

        plt.tight_layout()    
        plt.savefig(f'plots/{env.sim_name}/RewardsComparison.html',
                    format='svg', dpi=600, bbox_inches='tight')
        
