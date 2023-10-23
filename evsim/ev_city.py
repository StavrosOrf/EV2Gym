'''    
This file contains the EVCity class, which is used to represent the environment of the city.    
The environment is a gym environment and can be used with the OpenAI baselines.

===================================
Author: Stavros Orfanoudakis 2023
===================================
'''

import gym
from gym import spaces
import numpy as np
import datetime
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

from .grid import Grid
from .ev_charger import EV_Charger
from .ev import EV
from .transformer import Transformer
from .replay import EvCityReplay
from .utils import ev_city_plot, get_statistics, print_statistics, visualize, spawn_EV

class EVCity(gym.Env):

    def __init__(self,
                 cs=None,
                 static_prices=False,
                 load_prices_from_replay=False,
                 static_ev_spawn_rate=False,
                 ev_spawn_rate = 0.85,
                 load_ev_from_replay=False,
                 load_from_replay_path=None,
                 empty_ports_at_end_of_simulation=True,
                 simulate_grid=False,
                 scenario="public",
                 heterogeneous_specs=False,
                 replay_path = './replay/',
                 generate_rnd_game=False,  # generate a random game without terminating conditions
                 case='default',
                 number_of_ports_per_cs=2,
                 number_of_transformers=1,
                 score_threshold=1,
                 timescale=5,
                 date=(2023, 7, 17),  # (year, month, day)
                 hour=(5, 0),  # (hour, minute) 24 hour format
                 seed=42,  # TODO: add seed
                 save_replay=True,
                 save_plots=True,
                 extra_sim_name=None,
                 verbose=False,
                 simulation_length=1000):

        super(EVCity, self).__init__()

        print(f'Initializing EVCity environment...')

        self.generate_rnd_game = generate_rnd_game
        self.load_from_replay_path = load_from_replay_path
        self.load_ev_from_replay = load_ev_from_replay
        self.load_prices_from_replay = load_prices_from_replay
        self.empty_ports_at_end_of_simulation = empty_ports_at_end_of_simulation
        self.save_replay = save_replay
        self.save_plots = save_plots
        self.verbose = verbose  # Whether to print the simulation progress or not
        self.simulation_length = simulation_length
        self.replay_path = replay_path        
    
        self.score_threshold = score_threshold

        self.seed = seed

        #set random seed
        # np.random.seed(self.seed)
        
        if load_from_replay_path is not None:
            with open(load_from_replay_path, 'rb') as file:
                self.replay = pickle.load(file)

            sim_name = self.replay.replay_path.split('replay_')[-1].split('.')[0]
            self.sim_name = sim_name + '_replay'
            # self.save_replay = False
            self.sim_date = self.replay.sim_date
            self.simulate_grid = self.replay.simulate_grid
            self.timescale = self.replay.timescale
            # self.simulation_length = self.replay.sim_length
            self.cs = self.replay.n_cs
            self.number_of_transformers = self.replay.n_transformers
            # self.score_threshold = self.replay.score_threshold
            self.number_of_ports_per_cs = self.replay.max_n_ports
            self.spawn_rate = -1
            self.scenario = self.replay.scenario
            self.heterogeneous_specs = self.replay.heterogeneous_specs

        else:
            assert cs is not None, "Please provide the number of charging stations"
            self.cs = cs  # Number of charging stations
            # Threshold for the user satisfaction score

            self.number_of_ports_per_cs = number_of_ports_per_cs
            self.number_of_transformers = number_of_transformers
            # Timescale of the simulation (in minutes)
            self.timescale = timescale
            self.simulation_length = simulation_length
            # Simulation time
            self.sim_date = datetime.datetime(date[0],
                                              date[1],
                                              date[2],
                                              hour[0],
                                              hour[1])
            self.static_prices = static_prices
            self.spawn_rate = ev_spawn_rate
            self.static_ev_spawn_rate = static_ev_spawn_rate
            self.replay = None
            self.sim_name = f'ev_city_{self.simulation_length}_' + \
                f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")}'

            self.simulate_grid = simulate_grid  # Whether to simulate the grid or not
            self.scenario = scenario
            self.heterogeneous_specs = heterogeneous_specs

        self.sim_starting_date = self.sim_date

        self.sim_name = extra_sim_name + self.sim_name if extra_sim_name is not None else self.sim_name
        # Simulate grid
        if self.simulate_grid:
            self.grid = Grid(charging_stations=self.cs, case=case)
            self.cs_buses = self.grid.get_charging_stations_buses()
            self.cs_transformers = self.grid.get_bus_transformers()
        else:
            self.cs_buses = [None] * self.cs
            self.cs_transformers = np.random.randint(
                self.number_of_transformers, size=self.cs)

        # Instatiate Transformers
        self.transformers = self._load_transformers()

        # Instatiate Charging Stations
        self.charging_stations = self._load_ev_charger_profiles()

        # Instatiate EV profiles if they exist
        self.ev_profiles = self._load_ev_profiles()

        # Load Electricity prices for every charging station
        self.charge_prices, self.discharge_prices = self._load_electricity_prices()

        # Load power setpoint of simulation
        self.power_setpoints = self._load_power_setpoints()
        self.current_power_setpoints = np.zeros(self.simulation_length)

        # Action space: is a vector of size "Sum of all ports of all charging stations"
        self.number_of_ports = np.array(
            [cs.n_ports for cs in self.charging_stations]).sum()

        high = np.ones([self.number_of_ports])
        self.action_space = spaces.Box(low=-high, high=high, dtype=np.float64)

        # Observation space: is a matrix of size ("Sum of all ports of all charging stations",n_features)
        obs_dim = 1 + \
            self.cs * (2 + 2*number_of_ports_per_cs)
        # + number_of_transformers * 3

        # print(f'Observation space dimension: {obs_dim}')

        high = np.inf*np.ones([obs_dim])
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float64)

        # Observation mask: is a vector of size ("Sum of all ports of all charging stations") showing in which ports an EV is connected
        self.observation_mask = np.zeros(self.number_of_ports)

        self.init_statistic_variables()

        if self.ev_profiles is None:
            self._load_ev_spawn_scenarios()

        self.done = False

        # Make folders for results
        if self.save_replay:
            os.makedirs(self.replay_path, exist_ok=True)

        if self.save_plots:
            os.makedirs("./plots", exist_ok=True)
            print(f"Creating directory: ./plots/{self.sim_name}")
            os.makedirs(f"./plots/{self.sim_name}", exist_ok=True)

        if self.save_replay:
            self.EVs = []  # Store all of the EVs in the simulation that arrived        

    def _load_ev_spawn_scenarios(self):
        '''Loads the EV spawn scenarios of the simulation'''

        self.df_arrival_week = pd.read_csv('.\data\distribution-of-arrival.csv') #weekdays
        self.df_arrival_weekend = pd.read_csv('.\data\distribution-of-arrival-weekend.csv') #weekends
        self.df_connection_time = pd.read_csv('.\data\distribution-of-connection-time.csv') #connection time
        self.df_energy_demand = pd.read_csv('.\data\distribution-of-energy-demand.csv') #energy demand                        
        self.time_of_connection_vs_hour = np.load('.\data\Time_of_connection_vs_hour.npy')
        
    def _load_power_setpoints(self):
        if self.load_from_replay_path is None:
            return np.ones(self.simulation_length) * 20
        
        return self.replay.power_setpoints

    def _load_transformers(self):
        '''Loads the transformers of the simulation
        If load_from_replay_path is None, then the transformers are created randomly

        Returns:
            - transformers: a list of transformer objects'''

        transformers = []
        if self.load_from_replay_path is None:
            for i in range(self.number_of_transformers):
                transformer = Transformer(id=i,
                                          cs_ids=np.where(
                                              self.cs_transformers == i)[0],
                                          timescale=self.timescale,)
                transformers.append(transformer)
        else:
            transformers = self.replay.transformers

        return transformers

    def _load_ev_charger_profiles(self):
        '''Loads the EV charger profiles of the simulation
        If load_from_replay_path is None, then the EV charger profiles are created randomly

        Returns:
            - ev_charger_profiles: a list of ev_charger_profile objects'''

        charging_stations = []
        if self.load_from_replay_path is None:
            for i in range(self.cs):
                ev_charger = EV_Charger(id=i,
                                        connected_bus=self.cs_buses[i],
                                        connected_transformer=self.cs_transformers[i],
                                        n_ports=self.number_of_ports_per_cs,
                                        timescale=self.timescale,
                                        verbose=self.verbose,)

                charging_stations.append(ev_charger)
            return charging_stations

        return self.replay.charging_stations

    def _load_ev_profiles(self):
        '''Loads the EV profiles of the simulation
        If load_from_replay_path is None, then the EV profiles are created randomly

        Returns:
            - ev_profiles: a list of ev_profile objects'''

        if self.load_from_replay_path is None:
            return None
        elif self.load_ev_from_replay:
            return self.replay.EVs

    def _load_electricity_prices(self):
        '''Loads the electricity prices of the simulation
        If load_from_replay_path is None, then the electricity prices are created randomly

        Returns:
            - charge_prices: a matrix of size (number of charging stations, simulation length) with the charge prices
            - discharge_prices: a matrix of size (number of charging stations, simulation length) with the discharge prices'''
        if not self.load_prices_from_replay:
            if self.static_prices:
                return np.ones((self.cs, self.simulation_length)) * -0.01, \
                np.ones((self.cs, self.simulation_length)) * 0.1

        if self.load_from_replay_path is None or not self.load_prices_from_replay:
            charge_prices = np.random.normal(
                -0.05, 0.05, size=(self.cs, self.simulation_length))
            charge_prices = -1 * np.abs(charge_prices)
            discharge_prices = np.random.normal(
                0.1, 0.05, size=(self.cs, self.simulation_length))
            discharge_prices = np.abs(discharge_prices)
            return charge_prices, discharge_prices

        return self.replay.charge_prices, self.replay.discharge_prices

    def reset(self):
        '''Resets the environment to its initial state'''
        self.current_step = 0
        # Reset all charging stations
        for cs in self.charging_stations:
            cs.reset()

        self.sim_date = self.sim_starting_date

        self.EVs = []

        self.current_power_setpoints = np.zeros(self.simulation_length)

        if self.load_from_replay_path is None:
            if not self.static_ev_spawn_rate:
                self.spawn_rate = np.random.uniform(0.3, 0.85)
            
        # self.sim_name = f'ev_city_{self.simulation_length}_' + \
        # f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}'

        # TODO reset grid if implemented
        self.init_statistic_variables()

        return self._get_observation()

    def init_statistic_variables(self):
        self.current_step = 0
        self.total_evs_spawned = 0

        self.current_ev_departed = 0
        self.current_ev_arrived = 0
        self.current_evs_parked = 0

        self.transformer_amps = np.zeros([self.number_of_transformers,
                                           self.simulation_length])

        self.cs_power = np.zeros([self.cs, self.simulation_length])
        self.cs_current = np.zeros([self.cs, self.simulation_length])
        self.port_power = np.zeros([self.number_of_ports,   
                                    self.cs,
                                    self.simulation_length])
        self.port_current = np.zeros([self.number_of_ports,
                                        self.cs,
                                        self.simulation_length])
        self.port_energy_level = np.zeros([self.number_of_ports,
                                           self.cs,
                                           self.simulation_length])
        self.port_charging_cycles = np.zeros([self.number_of_ports,
                                              self.cs,
                                              self.simulation_length])
        self.port_arrival = dict({f'{j}.{i}': []
                                  for i in range(self.number_of_ports)
                                  for j in range(self.cs)})

        self.done = False

    def step(self, actions, visualize=False):
        ''''
        Takes an action as input and returns the next state, reward, and whether the episode is done
        Inputs:
            - actions: is a vector of size "Sum of all ports of all charging stations taking values in [-1,1]"
        Returns:
            - observation: is a matrix with the complete observation space
            - reward: is a scalar value representing the reward of the current step
            - done: is a boolean value indicating whether the episode is done or not
        '''
        assert not self.done, "Episode is done, please reset the environment"

        if self.verbose:
            print("-"*80)

        total_costs = 0
        total_invalid_action_punishment = 0
        user_satisfaction_list = []

        self.current_ev_departed = 0
        self.current_ev_arrived = 0

        port_counter = 0

        # Reset current power of all transformers
        for tr in self.transformers:
            tr.current_amps = 0

        # Call step for each charging station and spawn EVs where necessary
        for i, cs in enumerate(self.charging_stations):
            n_ports = cs.n_ports
            costs, user_satisfaction, invalid_action_punishment = cs.step(
                actions[port_counter:port_counter + n_ports],
                self.charge_prices[cs.id, self.current_step],
                self.discharge_prices[cs.id, self.current_step])

            for u in user_satisfaction:
                user_satisfaction_list.append(u)

            self.transformers[cs.connected_transformer].step(
                cs.current_total_amps)
            
            self.current_power_setpoints[self.current_step] += cs.current_power_output * self.timescale /60            

            total_costs += costs
            total_invalid_action_punishment += invalid_action_punishment
            self.current_ev_departed += len(user_satisfaction)

            port_counter += n_ports

            # Spawn EVs
            if self.ev_profiles is None:
                # Spawn EVs based on the spawn rate on specific chargers with random time of departure, and soc
                for _ in range (n_ports - cs.n_evs_connected):
                    ev = spawn_EV(self, cs.id)
                    if ev is None:
                        continue

                    index = cs.spawn_ev(ev)
                    self.port_arrival[f'{cs.id}.{index}'].append(
                        (self.current_step + 1, ev.earlier_time_of_departure))

                    if self.save_replay:
                        self.EVs.append(ev)

                    self.total_evs_spawned += 1
                    self.current_ev_arrived += 1

        # Spawn EVs
        if self.ev_profiles is not None:
            # Spawn EVs based on the EV profiles onspecific chargers with fixed time of departure, and soc

            counter = self.total_evs_spawned
            for i, ev in enumerate(self.ev_profiles[counter:]):
                if ev.time_of_arrival == self.current_step + 1:
                    ev.reset()
                    ev.simulation_length = self.simulation_length
                    index = self.charging_stations[ev.location].spawn_ev(ev)
                    self.port_arrival[f'{ev.location}.{index}'].append(
                        (self.current_step + 1, ev.earlier_time_of_departure))

                    self.total_evs_spawned += 1
                    self.current_ev_arrived += 1
                    if self.save_replay:
                        self.EVs.append(ev)

                elif ev.time_of_arrival > self.current_step + 1:
                    break

        self.update_power_statistics()

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
                                            user_satisfaction_list,
                                            total_invalid_action_punishment)

        if visualize:
            self.visualize()

        return self.check_termination(user_satisfaction_list, reward)
    
    def visualize(self):
        visualize(self)

    def check_termination(self, user_satisfaction_list, reward):
        # Check if the episode is done or any constraint is violated
        if self.current_step >= self.simulation_length or \
                any(score < self.score_threshold for score in user_satisfaction_list) or \
            (any(tr.is_overloaded() for tr in self.transformers)
                    and not self.generate_rnd_game):
            """Terminate if:
                - The simulation length is reached
                - Any user satisfaction score is below the threshold
                - Any charging station is overloaded 
                Dont terminate when overloading if :
                - generate_rnd_game is True
                Carefull: if generate_rnd_game is True, 
                the simulation might end up in infeasible problem
                """
            if self.verbose:
                print_statistics(self)

            if any(score < self.score_threshold for score in user_satisfaction_list):
                print(f"User satisfaction score below threshold of {self.score_threshold}, {self.current_step} timesteps\n") 
            elif any(tr.is_overloaded() for tr in self.transformers):
                print(f"Transformer overloaded, {self.current_step} timesteps\n")
            else:
                print(f"Episode finished after {self.current_step} timesteps\n")

            if self.save_replay:
                self.save_sim_replay()

            if self.save_plots:                
                ev_city_plot(self)

            self.done = True

            # create an objext with statistics about the simulation for vizualization



            return self._get_observation(), reward, True, get_statistics(self)
        else:
            return self._get_observation(), reward, False, None
        

    def save_sim_replay(self):
        '''Saves the simulation data in a pickle file'''
        replay = EvCityReplay(self)
        print(f"Saving replay file at {replay.replay_path}")
        with open(replay.replay_path, 'wb') as f:
            pickle.dump(replay, f)

    def update_power_statistics(self):
        '''Updates the power statistics of the simulation'''
        for tr in self.transformers:
            self.transformer_amps[tr.id, self.current_step] = tr.current_amps

        for cs in self.charging_stations:
            self.cs_power[cs.id, self.current_step] = cs.current_power_output
            self.cs_current[cs.id,self.current_step] = cs.current_total_amps

            for port in range(cs.n_ports):
                ev = cs.evs_connected[port]
                if ev is not None:
                    self.port_power[port, cs.id,
                                    self.current_step] = ev.current_power
                    self.port_current[port, cs.id,
                                    self.current_step] = ev.actual_current
                    self.port_energy_level[port, cs.id,
                                           self.current_step] = ev.current_capacity

    def _step_date(self):
        '''Steps the simulation date by one timestep'''
        self.sim_date = self.sim_date + \
            datetime.timedelta(minutes=self.timescale)

    def _get_observation(self, include_grid=False):
        '''Returns the current state of the environment'''
        state = [self.current_step / self.simulation_length,
                 #  self.timescale,
                 #  self.cs,
                 #  self.number_of_transformers,
                 ]

        # for tr in self.transformers:
        #     state.append(tr.get_state())

        for cs in self.charging_stations:
            state.append(cs.get_state())

        if include_grid:
            state.append(self.grid.get_grid_state())

        state = np.array(np.hstack(state))

        np.set_printoptions(suppress=True)

        # print(f'state: {state}')
        return state  # .reshape(-1)

    def _calculate_reward(self, total_costs, user_satisfaction_list, invalid_action_punishment):
        '''Calculates the reward for the current step'''
        reward = total_costs# - 0.5 
        # print(f'total_costs: {total_costs}')
        # print(f'user_satisfaction_list: {user_satisfaction_list}')
        for score in user_satisfaction_list:
            reward -= 100 * (1 - score) 

        # Punish invalid actions (actions that try to charge or discharge when there is no EV connected)
        reward -= 2 * (invalid_action_punishment/self.number_of_ports)

        return reward
