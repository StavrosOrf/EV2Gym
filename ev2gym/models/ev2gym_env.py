'''
This file contains the EVCity class, which is used to represent the environment of the city.
The environment is a gym environment and can be also used with the OpenAI gym standards and baselines.
The environment an also be used for standalone simulations without the gym environment.
'''

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import datetime
import pickle
import os
import random
from copy import deepcopy
import yaml
import json

# from .grid import Grid
from ev2gym.models.replay import EvCityReplay
from ev2gym.visuals.plots import ev_city_plot, visualize_step
from ev2gym.utilities.utils import get_statistics, print_statistics, calculate_charge_power_potential
from ev2gym.utilities.loaders import load_ev_spawn_scenarios, load_power_setpoints, load_transformers, load_ev_charger_profiles, load_ev_profiles, load_electricity_prices
from ev2gym.visuals.render import Renderer

from ev2gym.rl_agent.reward import SquaredTrackingErrorReward
from ev2gym.rl_agent.state import PublicPST


class EV2Gym(gym.Env):

    def __init__(self,
                 config_file=None,
                 load_from_replay_path=None,  # path of replay file to load
                 replay_save_path='./replay/',  # where to save the replay file
                 generate_rnd_game=True,  # generate a random game without terminating conditions
                 seed=None,
                 save_replay=False,
                 save_plots=False,
                 state_function=PublicPST,
                 reward_function=SquaredTrackingErrorReward,
                 eval_mode="Normal",  # eval mode can be "Normal", "Unstirred" or "Optimal" in order to save the correct statistics in the replay file
                 lightweight_plots=False,
                 # whether to empty the ports at the end of the simulation or not
                 empty_ports_at_end_of_simulation=True,
                 extra_sim_name=None,
                 verbose=False,
                 render_mode=None,
                 ):

        super(EV2Gym, self).__init__()

        if verbose:
            print(f'Initializing EVs-Simulator environment...')

        # read yaml config file
        assert config_file is not None, "Please provide a config file!!!"
        self.config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
        
        self.generate_rnd_game = generate_rnd_game
        self.load_from_replay_path = load_from_replay_path
        self.empty_ports_at_end_of_simulation = empty_ports_at_end_of_simulation
        self.save_replay = save_replay
        self.save_plots = save_plots
        self.lightweight_plots = lightweight_plots
        self.eval_mode = eval_mode
        self.verbose = verbose  # Whether to print the simulation progress or not
        # Whether to render the simulation in real-time or not
        self.render_mode = render_mode

        self.simulation_length = self.config['simulation_length']

        self.replay_path = replay_save_path

        cs = self.config['number_of_charging_stations']

        self.reward_function = reward_function
        self.state_function = state_function

        if seed is None:
            self.seed = np.random.randint(0, 1000000)
            # print(f"Random seed: {self.seed}")
        else:
            self.seed = seed
        # set random seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        self.tr_seed = self.config['tr_seed']        
        if self.tr_seed == -1:
            self.tr_seed = self.seed            
        self.tr_rng = np.random.default_rng(seed=self.tr_seed)
                

        if load_from_replay_path is not None:
            with open(load_from_replay_path, 'rb') as file:
                self.replay = pickle.load(file)

            sim_name = self.replay.replay_path.split(
                'replay_')[-1].split('.')[0]
            self.sim_name = sim_name + '_replay'
            self.sim_date = self.replay.sim_date
            self.timescale = self.replay.timescale
            self.cs = self.replay.n_cs
            self.number_of_transformers = self.replay.n_transformers
            self.number_of_ports_per_cs = self.replay.max_n_ports
            self.scenario = self.replay.scenario
            self.heterogeneous_specs = self.replay.heterogeneous_specs

        else:
            assert cs is not None, "Please provide the number of charging stations"
            self.cs = cs  # Number of charging stations
            # Threshold for the user satisfaction score

            self.number_of_ports_per_cs = self.config['number_of_ports_per_cs']
            self.number_of_transformers = self.config['number_of_transformers']
            self.timescale = self.config['timescale']
            self.simulation_length = int(self.config['simulation_length'])
            # Simulation time

            self.sim_date = datetime.datetime(self.config['year'],
                                              self.config['month'],
                                              self.config['day'],
                                              self.config['hour'],
                                              self.config['minute'])
            self.replay = None
            self.sim_name = f'sim_' + \
                f'{datetime.datetime.now().strftime("%Y_%m_%d_%f")}'

            self.scenario = self.config['scenario']
            self.heterogeneous_specs = self.config['heterogeneous_ev_specs']

        # Whether to simulate the grid or not (Future feature...)
        self.simulate_grid = False

        if self.cs > 100:
            self.lightweight_plots = True
        self.sim_starting_date = self.sim_date

        # Read the config.charging_network_topology json file and read the topology
        try:
            with open(self.config['charging_network_topology']) as json_file:
                self.charging_network_topology = json.load(json_file)

        except FileNotFoundError:
            if not self.config['charging_network_topology'] == 'None':
                print(
                    f'Did not find file {self.config["charging_network_topology"]}')
            self.charging_network_topology = None

        self.sim_name = extra_sim_name + \
            self.sim_name if extra_sim_name is not None else self.sim_name

        # Simulate grid
        if self.simulate_grid:
            pass
            # self.grid = Grid(charging_stations=self.cs, case=case)
            # self.cs_buses = self.grid.get_charging_stations_buses()
            # self.cs_transformers = self.grid.get_bus_transformers()
        else:
            # self.cs_buses = [None] * self.cs
            if self.charging_network_topology is None:
                self.cs_transformers = [
                    *np.arange(self.number_of_transformers)] * (self.cs // self.number_of_transformers)
                self.cs_transformers += random.sample(
                    [*np.arange(self.number_of_transformers)], self.cs % self.number_of_transformers)
                random.shuffle(self.cs_transformers)

        # Instatiate Transformers
        self.transformers = load_transformers(self)
        for tr in self.transformers:
            tr.reset(step=0)

        # Instatiate Charging Stations
        self.charging_stations = load_ev_charger_profiles(self)
        for cs in self.charging_stations:
            cs.reset()

        # Calculate the total number of ports in the simulation
        self.number_of_ports = np.array(
            [cs.n_ports for cs in self.charging_stations]).sum()

        # Load EV spawn scenarios
        if self.load_from_replay_path is None:
            load_ev_spawn_scenarios(self)

        # Spawn EVs
        self.EVs_profiles = load_ev_profiles(self)
        self.EVs = []

        # Load Electricity prices for every charging station
        self.charge_prices, self.discharge_prices = load_electricity_prices(
            self)

        # Load power setpoint of simulation
        self.power_setpoints = load_power_setpoints(self)
        self.current_power_usage = np.zeros(self.simulation_length)
        self.charge_power_potential = np.zeros(self.simulation_length)

        self.init_statistic_variables()

        # Variable showing whether the simulation is done or not
        self.done = False

        # Make folders for results
        if self.save_replay:
            os.makedirs(self.replay_path, exist_ok=True)

        if self.render_mode:
            # Initialize the rendering of the simulation
            self.renderer = Renderer(self)

        if self.save_plots:
            os.makedirs("./results", exist_ok=True)
            print(f"Creating directory: ./results/{self.sim_name}")
            os.makedirs(f"./results/{self.sim_name}", exist_ok=True)

        # Action space: is a vector of size "Sum of all ports of all charging stations"
        high = np.ones([self.number_of_ports])
        if self.config['v2g_enabled']:
            lows = -1 * np.ones([self.number_of_ports])
        else:
            lows = np.zeros([self.number_of_ports])
        self.action_space = spaces.Box(low=lows, high=high, dtype=np.float64)

        # Observation space: is a matrix of size ("Sum of all ports of all charging stations",n_features)
        obs_dim = len(self._get_observation())

        high = np.inf*np.ones([obs_dim])
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float64)

        # Observation mask: is a vector of size ("Sum of all ports of all charging stations") showing in which ports an EV is connected
        self.observation_mask = np.zeros(self.number_of_ports)

    def reset(self, seed=None, options=None, **kwargs):
        '''Resets the environment to its initial state'''

        if seed is None:
            self.seed = np.random.randint(0, 1000000)
        else:
            self.seed = seed

        # set random seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        if self.tr_seed == -1:
            self.tr_seed = self.seed            
        self.tr_rng = np.random.default_rng(seed=self.tr_seed)

        self.current_step = 0
        # Reset all charging stations
        for cs in self.charging_stations:
            cs.reset()
            
        for tr in self.transformers:
            tr.reset(step=self.current_step)

        if self.load_from_replay_path is not None or not self.config['random_day']:
            self.sim_date = self.sim_starting_date
        else:
            # select random date in range

            self.sim_date = datetime.datetime(2022,
                                              1,
                                              1,
                                              self.config['hour'],
                                              self.config['minute'],
                                              ) + datetime.timedelta(days=random.randint(0, int(1.5*365)))

            if self.scenario == 'workplace':
                # dont simulate weekends
                while self.sim_date.weekday() > 4:
                    self.sim_date += datetime.timedelta(days=1)

            if self.config['simulation_days'] == "weekdays":
                # dont simulate weekends
                while self.sim_date.weekday() > 4:
                    self.sim_date += datetime.timedelta(days=1)
            elif self.config['simulation_days'] == "weekends" and self.scenario != 'workplace':
                # simulate only weekends
                while self.sim_date.weekday() < 5:
                    self.sim_date += datetime.timedelta(days=1)

        self.sim_starting_date = self.sim_date
        self.EVs_profiles = load_ev_profiles(self)
        self.power_setpoints = load_power_setpoints(self)
        self.EVs = []

        # print(f'Simulation starting date: {self.sim_date}')

        # self.sim_name = f'ev_city_{self.simulation_length}_' + \
        # f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}'

        self.init_statistic_variables()

        return self._get_observation(), {}

    def init_statistic_variables(self):
        '''
        Initializes the variables used for keeping simulation statistics
        '''
        self.current_step = 0
        self.total_evs_spawned = 0
        self.total_reward = 0

        self.current_ev_departed = 0
        self.current_ev_arrived = 0
        self.current_evs_parked = 0

        self.previous_power_usage = self.current_power_usage
        self.current_power_usage = np.zeros(self.simulation_length)

        # self.transformer_amps = np.zeros([self.number_of_transformers,
        #                                   self.simulation_length])

        self.cs_power = np.zeros([self.cs, self.simulation_length])
        self.cs_current = np.zeros([self.cs, self.simulation_length])

        self.tr_overload = np.zeros(
            [self.number_of_transformers, self.simulation_length])

        self.tr_inflexible_loads = np.zeros(
            [self.number_of_transformers, self.simulation_length])

        self.tr_solar_power = np.zeros(
            [self.number_of_transformers, self.simulation_length])

        # self.port_power = np.zeros([self.number_of_ports,
        #                             self.cs,
        #                             self.simulation_length],
        #                            dtype=np.float16)
        if not self.lightweight_plots:
            self.port_current = np.zeros([self.number_of_ports,
                                          self.cs,
                                          self.simulation_length],
                                         dtype=np.float16,
                                         )
            self.port_current_signal = np.zeros([self.number_of_ports,
                                                 self.cs,
                                                 self.simulation_length],
                                                dtype=np.float16,
                                                )

            self.port_energy_level = np.zeros([self.number_of_ports,
                                               self.cs,
                                               self.simulation_length],
                                              dtype=np.float16)
            # self.port_charging_cycles = np.zeros([self.number_of_ports,
            #                                       self.cs,
            #                                       self.simulation_length],
            #                                      dtype=np.float16)
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
        departing_evs = []

        self.current_ev_departed = 0
        self.current_ev_arrived = 0

        port_counter = 0

        # Reset current power of all transformers
        for tr in self.transformers:
            tr.reset(step=self.current_step)

        # Call step for each charging station and spawn EVs where necessary
        for i, cs in enumerate(self.charging_stations):
            n_ports = cs.n_ports
            costs, user_satisfaction, invalid_action_punishment, ev = cs.step(
                actions[port_counter:port_counter + n_ports],
                self.charge_prices[cs.id, self.current_step],
                self.discharge_prices[cs.id, self.current_step])

            departing_evs += ev

            for u in user_satisfaction:
                user_satisfaction_list.append(u)

            self.current_power_usage[self.current_step] += cs.current_power_output

            # Update transformer variables for this timestep
            self.transformers[cs.connected_transformer].step(
                cs.current_total_amps, cs.current_power_output)

            total_costs += costs
            total_invalid_action_punishment += invalid_action_punishment
            self.current_ev_departed += len(user_satisfaction)

            port_counter += n_ports

        # Spawn EVs
        counter = self.total_evs_spawned
        for i, ev in enumerate(self.EVs_profiles[counter:]):
            if ev.time_of_arrival == self.current_step + 1:
                ev = deepcopy(ev)
                ev.reset()
                ev.simulation_length = self.simulation_length
                index = self.charging_stations[ev.location].spawn_ev(ev)

                if not self.lightweight_plots:
                    self.port_arrival[f'{ev.location}.{index}'].append(
                        (self.current_step+1, ev.time_of_departure+1))

                self.total_evs_spawned += 1
                self.current_ev_arrived += 1
                self.EVs.append(ev)

            elif ev.time_of_arrival > self.current_step + 1:
                break

        self._update_power_statistics(departing_evs)

        self.current_step += 1
        self._step_date()

        if self.current_step < self.simulation_length:
            self.charge_power_potential[self.current_step] = calculate_charge_power_potential(
                self)

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
            visualize_step(self)

        self.render()

        return self._check_termination(user_satisfaction_list, reward)

    def _check_termination(self, user_satisfaction_list, reward):
        '''Checks if the episode is done or any constraint is violated'''
        truncated = False
        # Check if the episode is done or any constraint is violated
        if self.current_step >= self.simulation_length or \
            (any(tr.is_overloaded() > 0 for tr in self.transformers)
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

                if any(tr.is_overloaded() for tr in self.transformers):
                    print(
                        f"Transformer overloaded, {self.current_step} timesteps\n")
                else:
                    print(
                        f"Episode finished after {self.current_step} timesteps\n")

            if self.save_replay:
                self._save_sim_replay()

            if self.save_plots:
                #save the env as a pickle file
                with open(f"./results/{self.sim_name}/env.pkl", 'wb') as f:
                    self.renderer = None
                    pickle.dump(self, f)
                ev_city_plot(self)

            self.done = True
            return self._get_observation(), reward, True, truncated, get_statistics(self)
        else:
            return self._get_observation(), reward, False, truncated, {'None': None}

    def render(self):
        '''Renders the simulation'''
        if self.render_mode:
            self.renderer.render()

    def _save_sim_replay(self):
        '''Saves the simulation data in a pickle file'''
        replay = EvCityReplay(self)
        print(f"Saving replay file at {replay.replay_path}")
        with open(replay.replay_path, 'wb') as f:
            pickle.dump(replay, f)

        return replay.replay_path
    
    def set_save_plots(self, save_plots):
        if save_plots:
            os.makedirs("./results", exist_ok=True)
            print(f"Creating directory: ./results/{self.sim_name}")
            os.makedirs(f"./results/{self.sim_name}", exist_ok=True)
            
        self.save_plots = save_plots

    def _update_power_statistics(self, departing_evs):
        '''Updates the power statistics of the simulation'''

        # if not self.lightweight_plots:
        for tr in self.transformers:
            # self.transformer_amps[tr.id, self.current_step] = tr.current_amps
            self.tr_overload[tr.id,
                             self.current_step] = tr.get_how_overloaded()
            self.tr_inflexible_loads[tr.id,
                                     self.current_step] = tr.inflexible_load[self.current_step]
            self.tr_solar_power[tr.id,
                                self.current_step] = tr.solar_power[self.current_step]

        for cs in self.charging_stations:
            self.cs_power[cs.id, self.current_step] = cs.current_power_output
            self.cs_current[cs.id, self.current_step] = cs.current_total_amps

            for port in range(cs.n_ports):
                if not self.lightweight_plots:
                    self.port_current_signal[port, cs.id,
                                             self.current_step] = cs.current_signal[port]
                ev = cs.evs_connected[port]
                if ev is not None and not self.lightweight_plots:
                    # self.port_power[port, cs.id,
                    #                 self.current_step] = ev.current_energy
                    self.port_current[port, cs.id,
                                      self.current_step] = ev.actual_current

                    self.port_energy_level[port, cs.id,
                                           self.current_step] = ev.current_capacity/ev.battery_capacity

            for ev in departing_evs:
                if not self.lightweight_plots:
                    self.port_energy_level[ev.id, ev.location, self.current_step] = \
                        ev.current_capacity/ev.battery_capacity
                    self.port_current[ev.id, ev.location,
                                      self.current_step] = ev.actual_current

    def _step_date(self):
        '''Steps the simulation date by one timestep'''
        self.sim_date = self.sim_date + \
            datetime.timedelta(minutes=self.timescale)

    def _get_observation(self):

        return self.state_function(self)

    def set_reward_function(self, reward_function):
        '''
        This function sets the reward function of the environment
        '''
        self.reward_function = reward_function

    def _calculate_reward(self, total_costs, user_satisfaction_list, invalid_action_punishment):
        '''Calculates the reward for the current step'''

        reward = self.reward_function(self, total_costs, user_satisfaction_list, invalid_action_punishment)
        self.total_reward += reward
        
        return reward
