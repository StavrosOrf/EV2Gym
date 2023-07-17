# Write code for an gym environment acor that simulates an electric vehicle charging station.

# The environment should have the following properties:
# - The environment has a fixed number of charging stations c
# - The environment has a fixed number of electric vehicles n
# - The environment has a fixed number of time steps t

# The environment should have the following methods:
# - reset(): resets the environment to its initial state
# - step(): takes an action as input and returns the next state, reward, and whether the episode is done
# - render(): renders the current state of the environment

from typing import Any
import gym
from gym import spaces
import numpy as np

# Transformer class


class PowerTransformer:
    def __init__(self, id, capacity, max_charging_cap, max_discharging_cap, current_capacity, location = None):
        
    
    def charge(self, power):       
        self.current_capacity += power

    def discharge(self, power):
        self.current_capacity -= power

    def load_percentage(self):
        return self.current_capacity / self.max_capacity

    def overload(self):
        return self.current_capacity > self.max_capacity

# EV charger class

class EV_charger:
    def __init__(self, id, location, connected_transformer, n_ports=1):

    def charge(self, ev_id):
        if self.current_capacity > 0:
            self.current_capacity -= 1
            return True
        else:
            return False

    def recharge(self):
        self.current_capacity = self.max_capacity


# EV class
class EV():
    def __init__(self, id, location, max_capacity, current_capacity):
        self.id = id
        self.location = location
        self.max_capacity = max_capacity
        self.current_capacity = current_capacity

        self.charging_cycles = 0

    def charge(self, ev_id):
        if self.current_capacity > 0:
            self.current_capacity -= 1
            return True
        else:
            return False

    def recharge(self):
        self.current_capacity = self.max_capacity

# main environment class


class CityEVEnvironment(gym.Env):
    def __init__(self, n, c):
        super(CityEVEnvironment, self).__init__()

        self.n = n  # Number of EVs
        self.c = c  # Number of charging stations

        # Action space: Charging station selection
        self.action_space = spaces.Discrete(c)
        # Observation space: (Selected charging station, EV charging status)
        self.observation_space = spaces.MultiDiscrete([c, n])

        # Current charging status of each EV
        self.ev_charge_status = np.zeros(n)
        self.current_step = 0

    def reset(self):
        self.ev_charge_status = np.zeros(self.n)
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action!"

        # Update charging status based on action and environment dynamics
        for i in range(self.n):
            if self.ev_charge_status[i] == action:
                self.ev_charge_status[i] = 0

        # Update environment dynamics (e.g., charging time, EV movement, etc.)
        # ...

        self.current_step += 1

        reward = self._calculate_reward()
        done = self.current_step >= MAX_STEPS  # Define your own termination condition

        return self._get_observation(), reward, done, {}

    def visualize(self):
        # Define your own visualization function
        # ...
        pass

    def _get_observation(self):
        return (self.ev_charge_status, self.current_step)

    def _calculate_reward(self):
        # Define your own reward function based on the current state and action
        # ...
        reward = 0
        return reward


# main funtion for testing
if __name__ == "__main__":
    env = CityEVEnvironment(n=10, c=5)
    env.reset()

    for i in range(5):
        action = env.action_space.sample()
        new_state, reward = env.step(action)
        env.visualize()
