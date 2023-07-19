# Write code for an gym environment acor that simulates an electric vehicle charging station.

# The environment should have the following properties:
# - The environment has a fixed number of charging stations c
# - The environment has a fixed number of electric vehicles n
# - The environment has a fixed number of time steps t

# The environment should have the following methods:
# - reset(): resets the environment to its initial state
# - step(): takes an action as input and returns the next state, reward, and whether the episode is done
# - render(): renders the current state of the environment

import gym
from gym import spaces
import numpy as np

from grid import Grid
from ev_charger import EV_Charger
from ev import EV


class CityEVEnvironment(gym.Env):
    def __init__(self,
                 evs,
                 cs,
                 ev_profiles_path=None,
                 charger_profiles_path=None,
                 simulate_grid=True,
                 case='default',):

        super(CityEVEnvironment, self).__init__()

        self.evs = evs  # Number of EVs
        self.cs = cs  # Number of charging stations

        # Action space: Charging station selection
        self.action_space = spaces.Discrete(cs)
        # Observation space: (Selected charging station, EV charging status)
        self.observation_space = spaces.MultiDiscrete([cs, evs])

        self.current_step = 0

        # Simulate grid
        if simulate_grid:
            self.grid = Grid(charging_stations=cs, case=case)
            self.cs_buses = self.grid.get_charging_stations_buses()
            self.cs_transformers = self.grid.get_charging_stations_transformers()
        else:
            self.cs_buses = [None] * cs
            self.cs_transformers = [None] * cs

        # Instatiate Charging Stations
        self.charging_stations = []
        if charger_profiles_path is None:
            for i in range(self.cs):
                ev_charger = EV_Charger(id=i,
                                        connected_bus=self.cs_buses[i],
                                        connected_transformer=self.cs_transformers)

                self.charging_stations.append(ev_charger)
        else:
            self.charging_stations = self._load_ev_charger_profiles(
                charger_profiles_path)

        if ev_profiles_path is None:
            self.evs = []
            for i in range(self.evs):
                ev = EV(id=i, connected_bus=None, connected_transformer=None)
                self.evs.append(ev)            

    def _load_ev_charger_profiles(self, path):
        # TODO: Load EV charger profiles from a csv file
        # ...
        #
        pass

    def _load_ev_profiles(self, path):
        # TODO: Load EV profiles from a csv file
        # ...
        #
        pass

    def reset(self):

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

    env = CityEVEnvironment(n=10, c=3)
    env.reset()

    for i in range(5):
        action = env.action_space.sample()
        new_state, reward = env.step(action)
        env.visualize()
