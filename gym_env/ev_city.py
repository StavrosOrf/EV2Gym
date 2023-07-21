'''
===================================
Author: Stavros Orfanoudakis 2023
===================================
'''

import gym
from gym import spaces
import numpy as np
import datetime

from .grid import Grid
from .ev_charger import EV_Charger
from .ev import EV


class EVCity(gym.Env):
    '''
    This file contains the EVCity class, which is used to represent the environment of the city.    
    '''

    def __init__(self,
                 cs,
                 ev_profiles_path=None,
                 charger_profiles_path=None,
                 electricity_prices_path=None,
                 simulate_grid=False,
                 case='default',
                 score_threshold=1,
                 timescale=5,
                 date=(2023, 7, 21),  # (year, month, day)
                 hour=(18, 0),  # (hour, minute) 24 hour format
                 verbose=False,
                 simulation_length=1000):

        super(EVCity, self).__init__()

        print(f'Initializing EVCity environment...')

        self.cs = cs  # Number of charging stations
        # Threshold for the user satisfaction score
        self.score_threshold = score_threshold
        self.timescale = timescale  # Timescale of the simulation (in minutes)
        self.simulation_length = simulation_length
        self.verbose = verbose  # Whether to print the simulation progress or not

        # Simulation time
        self.sim_date = datetime.datetime(date[0],
                                          date[1],
                                          date[2],
                                          hour[0],
                                          hour[1])

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
                                        connected_transformer=self.cs_transformers,
                                        timescale=self.timescale,
                                        verbose=self.verbose,)

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
            self.current_ev_departed += len(user_satisfaction)

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
                                earlier_time_of_departure=self.current_step+1 + np.random.randint(10, 40),)
                        cs.spawn_ev(ev)

                        self.total_evs_spawned += 1
                        self.current_ev_arrived += 1

            # TODO: record the spawn history of EVs for reproducible results, so the evs_profiles can be loaded again

        # Spawn EVs
        if self.ev_profiles is not None:
            # TODO: Spawn EVs based on the EV profiles onspecific chargers with fixed time of departure, and soc
            raise NotImplementedError

        self.current_step += 1
        self._step_date()
        self.current_evs_parked += self.current_ev_arrived - self.current_ev_departed

        # Call step for the grid
        if self.simulate_grid:
            # TODO: transform actions -> grid_actions
            raise NotImplementedError
            grid_report = self.grid.step(actions=actions)
            reward = self._calculate_reward(grid_report)
        else:
            reward = self._calculate_reward(total_costs,
                                            user_satisfaction_list)

        # Check if the episode is done
        if self.current_step >= self.simulation_length or \
                any(score < self.score_threshold for score in user_satisfaction_list):

            print(f"\n Episode finished after {self.current_step} timesteps")

            return self._get_observation(), reward, True
        else:
            return self._get_observation(), reward, False

    def visualize(self):
        '''Renders the current state of the environment in the terminal'''
        print(f"\n Step: {self.current_step}" +
              f" | {self.sim_date.hour}:{self.sim_date.minute}:{self.sim_date.second} |" +
              f" \tEVs +{self.current_ev_arrived}/-{self.current_ev_departed}" +
              f"| fullness: {self.current_evs_parked}/{self.number_of_ports}")

        if self.verbose:
            for cs in self.charging_stations:
                print(f'  - Charging station {cs.id}:')
                print(f'\t Power: {cs.current_power_output:4.1f} kWh |' +
                      f' \u2197 {self.charge_prices[cs.id, self.current_step -1 ]:4.2f} €/kWh ' +
                      f' \u2198 {self.discharge_prices[cs.id, self.current_step - 1]:4.2f} €/kWh |' +
                      f' EVs served: {cs.total_evs_served:3d} ' +
                      f' {cs.total_profits:4.2f} €')

                for port in range(cs.n_ports):
                    ev = cs.evs_connected[port]
                    if ev is not None:
                        print(f'\t\tPort {port}: {ev}')
                    else:
                        print(f'\t\tPort {port}:')

    def print_statistics(self):
        '''Prints the statistics of the simulation'''
        total_ev_served = np.array(
            [cs.total_evs_served for cs in self.charging_stations]).sum()
        total_profits = np.array(
            [cs.total_profits for cs in self.charging_stations]).sum()
        toal_energy_charged = np.array(
            [cs.total_energy_charged for cs in self.charging_stations]).sum()
        total_energy_discharged = np.array(
            [cs.total_energy_discharged for cs in self.charging_stations]).sum()
        average_user_satisfaction = np.average(np.array(
            [cs.get_avg_user_satisfaction() for cs in self.charging_stations]))

        print("\n\n==============================================================")
        print("Simulation statistics:")
        print(f'  - Total EVs spawned: {self.total_evs_spawned}')
        print(f'  - Total EVs served: {total_ev_served}')
        print(f'  - Total profits: {total_profits:.2f} €')
        print(
            f'  - Average user satisfaction: {average_user_satisfaction:.2f} %')

        print(f'  - Total energy charged: {toal_energy_charged:.1f} kWh')
        print(
            f'  - Total energy discharged: {total_energy_discharged:.1f} kWh\n')

        for cs in self.charging_stations:
            print(cs)

    def _step_date(self):
        '''Steps the simulation date by one timestep'''
        self.sim_date = self.sim_date + \
            datetime.timedelta(minutes=self.timescale)

    def _get_observation(self, include_grid=False):
        '''Returns the current state of the environment'''
        state = [self.current_step,
                 self.timescale,
                 self.cs,]

        for cs in self.charging_stations:
            state.append(cs.get_state())

        if include_grid:
            state.append(self.grid.get_grid_state())

        return np.hstack(state)

    def _calculate_reward(self, total_costs, user_satisfaction_list):
        '''Calculates the reward for the current step'''
        reward = total_costs
        return reward
