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
                 electricity_prices_path=None,
                 simulate_grid=False,
                 case='default',
                 timescale=5,
                 simulation_length=1000):

        super(CityEVEnvironment, self).__init__()

        self.evs = evs  # Number of EVs
        self.cs = cs  # Number of charging stations
        self.timescale = timescale  # Timescale of the simulation (in minutes)
        self.simulation_length = simulation_length

        # Simulation time
        self.minute = 0  # Starting minute of the simulation
        self.hour = 8  # Starting hour of the simulation
        self.day = 1  # Starting day of the simulation
        self.week = 1  # Starting week of the simulation
        self.month = 1  # Starting month of the simulation
        self.year = 2023  # Starting year of the simulation

        self.simulate_grid = simulate_grid  # Whether to simulate the grid or not

        # Simulate grid
        if self.simulate_grid:
            self.grid = Grid(charging_stations=cs, case=case)
            self.cs_buses = self.grid.get_charging_stations_buses()
            self.cs_transformers = self.grid.get_bus_transformers()
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

        if ev_profiles_path is not None:
            self.ev_profiles = self._load_ev_profiles(ev_profiles_path)
        else:

            self.ev_profiles = None

        # Load Electricity prices for every charging station
        if electricity_prices_path is not None:
            self.charge_prices, self.discharge_prices = self._load_electricity_prices(
                electricity_prices_path)
        else:
            self.charge_prices = np.random.normal(
                -0.05, 0.05, size=(self.cs, self.simulation_length))
            self.discharge_prices = np.random.normal(
                0.1, 0.05, size=(self.cs, self.simulation_length))

        # Action space: is a vector of size "Sum of all ports of all charging stations"
        self.number_of_ports = np.array(
            [cs.n_ports for cs in self.charging_stations]).sum()
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.number_of_ports, 1), dtype=np.float32)

        # Observation space: is a matrix of size ("Sum of all ports of all charging stations",n_features)
        n_features = 5
        self.observation_space = (self.number_of_ports, n_features)
        # TODO: Observation space is different when simulating the grid

        # Observation mask: is a vector of size ("Sum of all ports of all charging stations") showing in which ports an EV is connected
        self.observation_mask = np.zeros(self.number_of_ports)

        self.current_step = 0
        self.total_evs_spawned = 0

        self.current_ev_departed = 0
        self.current_ev_arrived = 0
        self.current_evs_parked = 0

    def _load_ev_charger_profiles(self, path):
        # TODO: Load EV charger profiles from a csv file
        # ...
        #
        pass

    def _load_ev_profiles(self, path):
        # TODO: Load predefined EV profiles from a csv file
        # ...
        #
        pass

    def _load_electricity_prices(self, electricity_prices_path):
        # TODO: Load electricity prices from a csv file
        # ...
        #
        pass

    def reset(self):
        '''Resets the environment to its initial state'''
        self.current_step = 0
        # Reset all charging stations
        for cs in self.charging_stations:
            cs.reset()

        return self._get_observation()

    def step(self, actions):
        ''''
        Takes an action as input and returns the next state, reward, and whether the episode is done
        Inputs:
            - actions: is a vector of size "Sum of all ports of all charging stations taking values in [-1,1]"
        Returns:
            - observation: is a matrix with the complete observation space
            - reward: is a scalar value representing the reward of the current step
            - done: is a boolean value indicating whether the episode is done or not
        '''
        total_costs = 0
        user_satisfaction_list = []

        self.current_ev_departed = 0
        self.current_ev_arrived = 0

        port_counter = 0
        # Call step for each charging station and spawn EVs where necessary
        for cs in self.charging_stations:
            n_ports = cs.n_ports
            costs, user_satisfaction = cs.step(
                actions[port_counter:port_counter + n_ports],
                self.charge_prices[cs.id, self.current_step],
                self.discharge_prices[cs.id, self.current_step])
            
            for u in user_satisfaction:
                user_satisfaction_list.append(u)
                
            total_costs += costs
            self.current_ev_departed += len(user_satisfaction_list)

            port_counter += n_ports

            # Spawn EVs
            if self.ev_profiles is None:

                if n_ports > cs.n_evs_connected:
                    # get a random float in [0,1] to decide if spawn an EV
                    # TODO: Replace with realistic EV spawn rate using distributions for different times of the day and days of the week, and staying time
                    self.spawn_rate = 0.2
                    if np.random.rand() < self.spawn_rate:
                        ev = EV(id=None,
                                location=cs.id,
                                battery_capacity_at_arrival=np.random.uniform(
                                    1, 49),
                                time_of_arrival=self.current_step+1,
                                earlier_time_of_departure=self.current_step+1 + np.random.randint(5, 10),)
                        cs.spawn_ev(ev)

                        self.total_evs_spawned += 1
                        self.current_ev_arrived += 1

            # TODO: record the spawn history of EVs for reproducible results, so the evs_profiles can be loaded again

        # Spawn EVs
        if self.ev_profiles is not None:
            # TODO: Spawn EVs based on the EV profiles onspecific chargers with fixed time of departure, and soc
            raise NotImplementedError

        self.current_step += 1
        self.current_evs_parked += self.current_ev_arrived - self.current_ev_departed

        # Call step for the grid
        if self.simulate_grid:
            # TODO: transform actions -> grid_actions
            raise NotImplementedError
            grid_report = self.grid.step(actions=actions)
            reward = self._calculate_reward(grid_report)
        else:
            reward = self._calculate_reward()

        done = False

        return self._get_observation(), reward, done

    def visualize(self):
        '''Renders the current state of the environment in the terminal'''
        print(f"\n Current step: {self.current_step} ===========" +
              f"EVs +{self.current_ev_arrived}/-{self.current_ev_departed}" +
              f"| fullness: {self.current_evs_parked}/{self.number_of_ports}")

        for cs in self.charging_stations:
            print(f'  - Charging station {cs.id}:')            
            print(f'\t Power: {cs.current_power_output:4.1f} kWh |' +
                  f' \u2197 {self.charge_prices[cs.id, self.current_step]:4.2f} €/kWh ' +
                  f' \u2198 {self.discharge_prices[cs.id, self.current_step]:4.2f} €/kWh |' +
                  f' EVs served: {cs.total_evs_served:3d} ' +
                  f' {cs.total_profits:4.2f} €') 

            for port in range(cs.n_ports):
                ev = cs.evs_connected[port]
                if ev is not None:
                    print(f'\t\tPort {port}: {ev}')
                else:
                    print(f'\t\tPort {port}: -')

    def print_statistics(self):
        '''Prints the statistics of the simulation'''
        print("\n\Simulation statistics:")
        print(f'  - Total EVs spawned: {self.total_evs_spawned}\n')

        for cs in self.charging_stations:
            print(cs)

    def _get_observation(self):
        # Define your own observation function based on the current state
        # ...

        return 0

    def _calculate_reward(self, grid_report=None):
        # Define your own reward function based on the current state and action
        # ...
        reward = 0
        return reward


# main funtion for testing
if __name__ == "__main__":

    env = CityEVEnvironment(evs=10, cs=3, timescale=5)
    state = env.reset()

    env.visualize()

    for i in range(20):
        print("-"*80)
        # actions = env.action_space.sample()   # sample random actions
        # actions = [a for a in actions]
        actions = np.random.uniform(-1, 1, 6)        
        print(f'Actions: {actions}')
        new_state, reward, done = env.step(actions)  # takes action
        env.visualize()

    env.print_statistics()
