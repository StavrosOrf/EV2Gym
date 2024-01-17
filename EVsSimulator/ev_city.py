'''    
This file contains the EVCity class, which is used to represent the environment of the city.    
The environment is a gym environment and can be also used with the OpenAI gym standards and baselines.
The environment an also be used for standalone simulations without the gym environment.

===================================
Author: Stavros Orfanoudakis 2023
===================================
'''

import gym
from gym import spaces
import numpy as np
import math
import datetime
import pickle
import os
import random
from copy import deepcopy
import yaml
import json

# from .grid import Grid
from .replay import EvCityReplay
from .utils import ev_city_plot, get_statistics, print_statistics, visualize_step, spawn_EV
from .loaders import load_ev_spawn_scenarios, load_power_setpoints, load_transformers, load_ev_charger_profiles, load_ev_profiles, load_electricity_prices
from .render import Renderer


class EVCity(gym.Env):

    def __init__(self,
                 config_file=None,                 
                 load_from_replay_path=None,  # path of replay file to load
                 replay_path='./replay/', # where to save the replay file
                 generate_rnd_game=False,  # generate a random game without terminating conditions
                 seed=42,
                 save_replay=True,
                 save_plots=True,
                 eval_mode = "Normal", #eval mode can be "Normal", "Unstirred" or "Optimal" in order to save the correct statistics in the replay file
                 lightweight_plots=False,
                 extra_sim_name=None,
                 verbose=False,
                 render_mode=None,
                 ):

        super(EVCity, self).__init__()

        if verbose:
            print(f'Initializing EVCity environment...')

        # read yaml config file
        self.config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)

        self.generate_rnd_game = generate_rnd_game
        self.load_from_replay_path = load_from_replay_path        
        # self.empty_ports_at_end_of_simulation = empty_ports_at_end_of_simulation
        self.save_replay = save_replay
        self.save_plots = save_plots
        self.lightweight_plots = lightweight_plots
        self.eval_mode = eval_mode
        self.verbose = verbose  # Whether to print the simulation progress or not
        # Whether to render the simulation in real-time or not
        self.render_mode = render_mode

        self.simulation_length = self.config['simulation_length']
        
        
        self.replay_path = replay_path

        self.score_threshold = self.config['score_threshold']
        cs = self.config['number_of_charging_stations']

        self.seed = seed

        # set random seed
        # np.random.seed(self.seed)

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
            self.spawn_rate = -1
            self.scenario = self.replay.scenario
            self.heterogeneous_specs = self.replay.heterogeneous_specs

        else:
            assert cs is not None, "Please provide the number of charging stations"
            self.cs = cs  # Number of charging stations
            # Threshold for the user satisfaction score

            self.number_of_ports_per_cs = self.config['number_of_ports_per_cs']
            self.number_of_transformers = self.config['number_of_transformers']            
            self.timescale = self.config['timescale']
            self.simulation_length = self.config['simulation_length']
            # Simulation time

            self.sim_date = datetime.datetime(self.config['year'],
                                              self.config['month'],
                                              self.config['day'],
                                              self.config['hour'],
                                              self.config['minute'])
            self.replay = None
            self.sim_name = f'ev_city_{self.simulation_length}_' + \
                f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")}'

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
                print(f'Did not find file {self.config["charging_network_topology"]}')
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
            if self.charging_network_topology is None :
                self.cs_transformers = [
                    *np.arange(self.number_of_transformers)] * (self.cs // self.number_of_transformers)
                self.cs_transformers += random.sample(
                    [*np.arange(self.number_of_transformers)], self.cs % self.number_of_transformers)
                random.shuffle(self.cs_transformers)

        # Instatiate Transformers
        self.transformers = load_transformers(self)

        # Instatiate Charging Stations
        self.charging_stations = load_ev_charger_profiles(self)

        # Instatiate EV profiles if they exist
        self.ev_profiles = load_ev_profiles(self)

        # Load Electricity prices for every charging station
        self.charge_prices, self.discharge_prices = load_electricity_prices(
            self)

        # Load power setpoint of simulation
        self.power_setpoints = load_power_setpoints(self, randomly=True)
        self.current_power_setpoints = np.zeros(self.simulation_length)
        self.charge_power_potential = np.zeros(self.simulation_length)

        self.number_of_ports = np.array(
            [cs.n_ports for cs in self.charging_stations]).sum()

        self.init_statistic_variables()

        if self.ev_profiles is None:
            load_ev_spawn_scenarios(self)

        # Variable showing whether the simulation is done or not
        self.done = False
        
        # Make folders for results
        if self.save_replay:
            os.makedirs(self.replay_path, exist_ok=True)

        if self.render_mode:
            # Initialize the rendering of the simulation
            self.renderer = Renderer(self)



        if self.save_plots:
            os.makedirs("./plots", exist_ok=True)
            print(f"Creating directory: ./plots/{self.sim_name}")
            os.makedirs(f"./plots/{self.sim_name}", exist_ok=True)

        # if self.save_replay:
        self.EVs = []  # Store all of the EVs in the simulation that arrived

        # Action space: is a vector of size "Sum of all ports of all charging stations"
        high = np.ones([self.number_of_ports])
        self.action_space = spaces.Box(low=-high, high=high, dtype=np.float64)

        # Observation space: is a matrix of size ("Sum of all ports of all charging stations",n_features)
        obs_dim = len(self._get_observation())

        high = np.inf*np.ones([obs_dim])
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float64)

        # Observation mask: is a vector of size ("Sum of all ports of all charging stations") showing in which ports an EV is connected
        self.observation_mask = np.zeros(self.number_of_ports)

    def reset(self):
        '''Resets the environment to its initial state'''
        self.current_step = 0
        # Reset all charging stations
        for cs in self.charging_stations:
            cs.reset()

        if self.load_from_replay_path is not None:
            self.sim_date = self.sim_starting_date
        else:
            # select random date in range
            self.sim_date = datetime.datetime(2022,
                                              1,
                                              1,
                                              5,
                                              0) + datetime.timedelta(days=random.randint(0, int(1.5*365)))
            self.sim_starting_date = self.sim_date

        # print(f'Simulation starting date: {self.sim_date}')

        self.EVs = []

        # self.sim_name = f'ev_city_{self.simulation_length}_' + \
        # f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}'

        # TODO reset grid if implemented
        self.init_statistic_variables()

        return self._get_observation()

    def _calculate_ev_load_curve(self):
        '''This function calculates the load curve of the EVs in the simulation'''
        for i in range(self.simulation_length):
            # all ports are charging instantly
            actions = np.ones(self.number_of_ports)

            new_state, reward, done, _ = self.step(
                actions, visualize=False)  # takes action

            # input("Press Enter to continue...")

            if done and i < self.simulation_length-1:
                return

    def init_statistic_variables(self):
        self.current_step = 0
        self.total_evs_spawned = 0

        self.current_ev_departed = 0
        self.current_ev_arrived = 0
        self.current_evs_parked = 0

        self.current_power_setpoints = np.zeros(self.simulation_length)

        # self.transformer_amps = np.zeros([self.number_of_transformers,
        #                                   self.simulation_length])

        self.cs_power = np.zeros([self.cs, self.simulation_length])
        self.cs_current = np.zeros([self.cs, self.simulation_length])
        
        self.tr_overload = np.zeros([self.number_of_transformers, self.simulation_length])
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

            self.current_power_setpoints[self.current_step] += cs.current_power_output * \
                60/self.timescale  # transform from energy to power

            total_costs += costs
            total_invalid_action_punishment += invalid_action_punishment
            self.current_ev_departed += len(user_satisfaction)

            port_counter += n_ports

            # Spawn EVs
            if self.ev_profiles is None:
                # Spawn EVs based on the spawn rate on specific chargers with random time of departure, and soc
                for _ in range(n_ports - cs.n_evs_connected):
                    ev = spawn_EV(self, cs.id)
                    if ev is None:
                        continue

                    index = cs.spawn_ev(ev)

                    if not self.lightweight_plots:
                        self.port_arrival[f'{cs.id}.{index}'].append(
                            (self.current_step + 1, ev.earlier_time_of_departure))

                    # if self.save_replay:
                    self.EVs.append(ev)

                    self.total_evs_spawned += 1
                    self.current_ev_arrived += 1

        # Spawn EVs
        if self.ev_profiles is not None:
            # Spawn EVs based on the EV profiles onspecific chargers with fixed time of departure, and soc

            counter = self.total_evs_spawned
            for i, ev in enumerate(self.ev_profiles[counter:]):
                if ev.time_of_arrival == self.current_step + 1:
                    ev = deepcopy(ev)
                    ev.reset()
                    ev.simulation_length = self.simulation_length
                    index = self.charging_stations[ev.location].spawn_ev(ev)
                    if not self.lightweight_plots:
                        self.port_arrival[f'{ev.location}.{index}'].append(
                            (self.current_step+1, ev.earlier_time_of_departure))

                    self.total_evs_spawned += 1
                    self.current_ev_arrived += 1
                    # if self.save_replay:
                    self.EVs.append(ev)

                elif ev.time_of_arrival > self.current_step + 1:
                    break

        self.update_power_statistics()

        self.current_step += 1
        if self.current_step < self.simulation_length:
            self.charge_power_potential[self.current_step] = self._calculate_charge_power_potential(
            )

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
            visualize_step(self)

        if self.render_mode:
            self.renderer.render()

        return self._check_termination(user_satisfaction_list, reward)

    def _check_termination(self, user_satisfaction_list, reward):
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
                    print(
                        f"User satisfaction score below threshold of {self.score_threshold}, {self.current_step} timesteps\n")
                elif any(tr.is_overloaded() for tr in self.transformers):
                    print(
                        f"Transformer overloaded, {self.current_step} timesteps\n")
                else:
                    print(
                        f"Episode finished after {self.current_step} timesteps\n")

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

        return replay.replay_path

    def update_power_statistics(self):
        '''Updates the power statistics of the simulation'''

        # if not self.lightweight_plots:
        for tr in self.transformers:
            # self.transformer_amps[tr.id, self.current_step] = tr.current_amps
            self.tr_overload[tr.id, self.current_step] = tr.get_how_overloaded()

        for cs in self.charging_stations:
            self.cs_power[cs.id, self.current_step] = cs.current_power_output
            self.cs_current[cs.id, self.current_step] = cs.current_total_amps

            for port in range(cs.n_ports):
                ev = cs.evs_connected[port]
                if ev is not None and not self.lightweight_plots:
                    # self.port_power[port, cs.id,
                    #                 self.current_step] = ev.current_power
                    self.port_current[port, cs.id,
                                      self.current_step] = ev.actual_current
                    self.port_current_signal[port, cs.id,
                                             self.current_step] = cs.current_signal[port]

                    self.port_energy_level[port, cs.id,
                                           self.current_step] = ev.current_capacity

    def _calculate_charge_power_potential(self):
        '''
        This function calculates the total charge power potential of all currently parked EVs        
        '''

        power_potential = 0
        for cs in self.charging_stations:
            cs_power_potential = 0
            for port in range(cs.n_ports):
                ev = cs.evs_connected[port]
                if ev is not None:
                    if ev.time_of_arrival-1 != self.current_step:
                        phases = min(cs.phases, ev.ev_phases)
                        ev_current = ev.max_ac_charge_power * \
                            1000/(math.sqrt(phases)*cs.voltage)
                        current = min(cs.max_charge_current, ev_current)
                        cs_power_potential += math.sqrt(phases) * \
                            cs.voltage*current/1000

            max_cs_power = math.sqrt(cs.phases) * \
                cs.voltage*cs.max_charge_current/1000
            min_cs_power = math.sqrt(cs.phases) * \
                cs.voltage*cs.min_charge_current/1000

            if cs_power_potential > max_cs_power:
                power_potential += max_cs_power
            elif cs_power_potential < min_cs_power:
                power_potential += 0
            else:
                power_potential += cs_power_potential

        return power_potential
    
    # def _calculate_unstirred_energy_level(self):

    def _step_date(self):
        '''Steps the simulation date by one timestep'''
        self.sim_date = self.sim_date + \
            datetime.timedelta(minutes=self.timescale)

    def _get_observation(self):
        '''Returns the current state of the environment'''

        scenario = self.scenario.split('_')[1]

        if scenario == 'PowerSetpointTracking':

            state = [
                (self.current_step-1) / self.simulation_length,
                # self.sim_date.year / 2025,
                # self.sim_date.month / 12,
                self.sim_date.weekday() / 7,
                # turn hour and minutes in sin and cos
                math.sin(self.sim_date.hour/24*2*math.pi),
                math.cos(self.sim_date.hour/24*2*math.pi),
                self.power_setpoints[self.current_step-1]/100,  # /self.cs,
            ]

            for tr in self.transformers:
                state.append(tr.get_state(scenario=scenario))
                for cs in self.charging_stations:
                    if cs.connected_transformer == tr.id:
                        state.append(cs.get_state(scenario=scenario))

            # if include_grid:
            #     state.append(self.grid.get_grid_state())

            state = np.array(np.hstack(state))

            np.set_printoptions(suppress=True)

            # print(f'state: {state}')
            return state  # .reshape(-1)
        else:
            raise NotImplementedError

    def _calculate_reward(self, total_costs, user_satisfaction_list, invalid_action_punishment):
        '''Calculates the reward for the current step'''
        reward = 0
        # reward = total_costs  # - 0.5
        # print(f'total_costs: {total_costs}')
        # print(f'user_satisfaction_list: {user_satisfaction_list}')
        # for score in user_satisfaction_list:
        #     reward -= 100 * (1 - score)

        # Punish invalid actions (actions that try to charge or discharge when there is no EV connected)
        # reward -= 2 * (invalid_action_punishment/self.number_of_ports)

        scenario = self.scenario.split('_')[1]
        if scenario == "PowerSetpointTracking":

            # reward = min(2, 1 * 4 * self.cs / (0.00001 + (
            #     self.power_setpoints[self.current_step-1] - self.current_power_setpoints[self.current_step-1])**2))

            # this is the new reward function
            # reward = min(2, 1/((min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
            #           self.current_power_setpoints[self.current_step-1])**2 + 0.000001))

            # new_*10*charging
            # if self.power_setpoints[self.current_step-1] < self.current_power_setpoints[self.current_step-1]:
            #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
            #             self.current_power_setpoints[self.current_step-1])
            # else:
            #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
            #             self.current_power_setpoints[self.current_step-1])*10

            # new_1_equal
            # if self.power_setpoints[self.current_step-1] < self.current_power_setpoints[self.current_step-1]:
            #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
            #             self.current_power_setpoints[self.current_step-1])
            # else:
            #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
            #             self.current_power_setpoints[self.current_step-1])

            # new_0.1
            # if self.power_setpoints[self.current_step-1] < self.current_power_setpoints[self.current_step-1]:
            #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
            #             self.current_power_setpoints[self.current_step-1])**2
            # else:
            #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
            #             self.current_power_setpoints[self.current_step-1])*0.1

            # new_reward squared
            # if self.power_setpoints[self.current_step-1] < self.current_power_setpoints[self.current_step-1]:
            #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
            #           self.current_power_setpoints[self.current_step-1])**2
            # else:
            #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
            #           self.current_power_setpoints[self.current_step-1])

            # for score in user_satisfaction_list:
            #     reward -= 100 * (1 - score)

            # for tr in self.transformers:
            #     if tr.current_amps > tr.max_current:
            #         reward -= 1000 * abs(tr.current_amps - tr.max_current)
            #     elif tr.current_amps < tr.min_current:
            #         reward -= 1000 * abs(tr.current_amps - tr.min_current)

            # reward -= 100 * (tr.current_amps < tr.min_amps)
            #######################################################################################################
            # squared tracking error
            reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
                       self.current_power_setpoints[self.current_step-1])**2

            # best reward so far
            ############################################################################################################
            # if self.power_setpoints[self.current_step-1] < self.current_power_setpoints[self.current_step-1]:
            #     reward -= (self.current_power_setpoints[self.current_step-1]-self.power_setpoints[self.current_step-1])

            # reward += self.current_power_setpoints[self.current_step-1]/75
            ############################################################################################################
            # normalize reward to -1 1
            # reward = reward/1000
            # reward = (100 +reward) / 1000
            # print(f'reward: {reward}')

            # reward -= 2 * (invalid_action_punishment/self.number_of_ports)
            # reward /= 100
            # reward = (100 +reward) / 1000
            # print(f'current_power_setpoints: {self.current_power_setpoints[self.current_step-1]}')

            return reward
