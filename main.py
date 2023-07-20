from gym_env import *

import numpy as np
import matplotlib.pyplot as plt

# main funtion for testing
if __name__ == "__main__":

    verbose = False
    number_of_charging_stations = 2
    steps = 1000

    env = CityEVEnvironment(cs=number_of_charging_stations,
                            timescale=5,
                            verbose=verbose,
                            simulation_length=steps)
    state = env.reset()

    env.visualize()
    rewards = []

    for i in range(steps):
        print("-"*80)
        # actions = env.action_space.sample()   # sample random actions
        actions = np.random.uniform(-1, 1, env.number_of_ports)
        # all ports are charging instantly
        actions = np.ones(env.number_of_ports)
        if verbose:
            print(f'Actions: {actions}')

        new_state, reward, done = env.step(actions)  # takes action
        rewards.append(reward)
        env.visualize()

        if verbose:
            print(f'Reward: {reward} \t Done: {done}')

    env.print_statistics()

    # Plot the commulative reward in subplot 1
    plt.subplot(2, 1, 1)
    plt.plot(np.cumsum(rewards))
    plt.xlabel('Time step')
    plt.ylabel('Cumulative reward')
    plt.show()

    # Plot the reward per step in subplot 2
    plt.subplot(2, 1, 2)
    plt.plot(rewards)
    plt.xlabel('Time step')
    plt.ylabel('Reward')
    plt.show()
