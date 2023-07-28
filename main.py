from gym_env import ev_city
from math_model import ev_city_model

import numpy as np
import matplotlib.pyplot as plt

# main funtion for testing
if __name__ == "__main__":

    verbose = False
    n_transformers = 3
    number_of_charging_stations = 10
    steps = 20  # 288 steps = 1 day with 5 minutes per step
    timescale = 5  # (5 minutes per step)
    save_plots = False

    # steps = 100
    replay_path = "replay/replay_ev_city_100_2023-07-26_15-33.pkl"
    replay_path = None

    env = ev_city.EVCity(cs=number_of_charging_stations,
                         number_of_transformers=n_transformers,
                         load_ev_from_replay=True,
                         load_prices_from_replay=True,
                         load_from_replay_path=replay_path,
                         empty_ports_at_end_of_simulation=True,
                         generate_rnd_game=False,
                         simulation_length=steps,
                         timescale=timescale,
                         verbose=verbose,)
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

        new_state, reward, done = env.step(
            actions, visualize=True)  # takes action
        rewards.append(reward)

        if verbose:
            print(f'Reward: {reward} \t Done: {done}')

        if done:
            break

    if verbose:
        env.print_statistics()

    math_model = ev_city_model.EV_City_Math_Model(
        sim_file_path=f"replay/replay_{env.sim_name}.pkl")
    # math_model = ev_city_model.EV_City_Math_Model(sim_file_path=f"replay/replay_ev_city_100_2023-07-26_14-19.pkl")
    actions = math_model.get_actions()
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
