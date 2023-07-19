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

from grid import Grid

class EV_Charger:
    def __init__(self,
                 id,
                 connected_bus,
                 connected_transformer,
                 geo_location,
                 max_power=300, #kW
                 min_power=0,
                 n_ports=2,
                 charger_type="DC",
                 bi_directional=True):
        
        self.id = id

        #EV Charger location and grid characteristics
        self.connected_bus = connected_bus
        self.connected_transformer = connected_transformer
        self.geo_location = geo_location
        
        #EV Charger technical characteristics
        self.max_power = max_power
        self.min_power = min_power
        self.n_ports = n_ports
        self.charger_type = charger_type
        self.bi_directional = bi_directional

        #EV Charger status
        self.current_power_output = 0        
        self.evs_connected = []


    def step(self,actions):
        #actions are in the format of (power,n_ports) positive for charging negative for discharging
        #default to 0 if no ev is connected

        #TODO: check if the power requested is within the limits of the charger, AND NORMALIZE ELSEWISE
        #TODO: update the information of the connected EVs according to actions

        return       
    
    def reset(self):
        self.current_power_output = 0        
        self.evs_connected = []


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
    def __init__(self,
                 evs,
                 cs,
                 simulate_grid=True,
                 case='default',
                 charger_profiles_path=None):

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
                                        location=None,
                                        connected_bus=self.cs_buses[i],
                                        connected_transformer=None,
                                        n_ports=2,
                                        bi_directional=True)
                self.charging_stations.append(ev_charger)
        else:
            self.charging_stations = self._load_ev_charger_profiles(
                charger_profiles_path)

    def _load_ev_charger_profiles(self, path):
        # TODO: Load EV charger profiles from a csv file
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
