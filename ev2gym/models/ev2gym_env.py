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

from ev2gym.models.replay import EvCityReplay
from ev2gym.visuals.plots import ev_city_plot, visualize_step
from ev2gym.utilities.utils import (get_statistics, print_statistics,
                                    calculate_charge_power_potential,
                                    init_statistic_variables)
from ev2gym.utilities.loaders import (load_ev_spawn_scenarios,
                                      load_power_setpoints,
                                      load_transformers,
                                      load_ev_charger_profiles,
                                      load_ev_profiles,
                                      load_electricity_prices,
                                      load_grid)
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
                 cost_function=None,  # cost function to use in the simulation
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
            print(f'Initializing EV2Gym environment...')

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
        self.cost_function = cost_function

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
            self.simulate_grid = self.replay.simulate_grid

        else:
            assert cs is not None, "Please provide the number of charging stations"
            self.cs = cs  # Number of charging stations
            # Threshold for the user satisfaction score

            self.number_of_ports_per_cs = self.config['number_of_ports_per_cs']
            self.number_of_transformers = self.config['number_of_transformers']
            self.timescale = self.config['timescale']
            self.scenario = self.config['scenario']
            self.simulation_length = int(self.config['simulation_length'])
            self.simulate_grid = self.config['simulate_grid']

            if self.config['random_day']:
                if "random_hour" in self.config:
                    if self.config["random_hour"]:
                        self.config['hour'] = random.randint(5, 15)

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
            else:

                self.sim_date = datetime.datetime(self.config['year'],
                                                  self.config['month'],
                                                  self.config['day'],
                                                  self.config['hour'],
                                                  self.config['minute'])
            self.replay = None
            self.sim_name = f'sim_' + \
                f'{datetime.datetime.now().strftime("%Y_%m_%d_%f")}'

            self.heterogeneous_specs = self.config['heterogeneous_ev_specs']

        self.stats = None
        
        # self.lightweight_plots = False
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

        # Load grid data
        self.grid = load_grid(self)

        # Instatiate Transformers
        self.transformers = load_transformers(self)

        # Instatiate Charging Stations
        self.charging_stations = load_ev_charger_profiles(self)

        # Calculate the total number of ports in the simulation
        self.number_of_ports = np.array(
            [cs.n_ports for cs in self.charging_stations]).sum()

        # Load EV spawn scenarios
        if self.load_from_replay_path is None:
            load_ev_spawn_scenarios(self)

        self.price_data = None
        self.reset(seed=seed)
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
        self.stats = None
        # Reset all charging stations
        for cs in self.charging_stations:
            cs.reset()

        for tr in self.transformers:
            tr.reset(step=self.current_step)

        if self.load_from_replay_path is not None or not self.config['random_day']:
            self.sim_date = self.sim_starting_date
        else:
            # select random date in range

            if "random_hour" in self.config:
                if self.config["random_hour"]:
                    self.config['hour'] = random.randint(5, 15)

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
        self.charge_prices, self.discharge_prices = load_electricity_prices(
            self)
        self.power_setpoints = load_power_setpoints(self)

        self.EVs = []
        init_statistic_variables(self)

        if self.simulate_grid:
            if self.load_from_replay_path is not None:
                load_data = self.replay.load_data
                pv_data = self.replay.pv_data
            else:
                load_data = None
                pv_data = None
                
            active_power, reactive_power = self.grid.reset(
                self.sim_date,
                load_data=load_data,
                pv_data=pv_data)

            self.node_active_power[1:, self.current_step] = active_power
            self.node_reactive_power[1:, self.current_step] = reactive_power
        else:
            self.node_active_power = np.zeros(
                (34, self.simulation_length))
            self.node_reactive_power = np.zeros(
                (34, self.simulation_length))            

        state = self._get_observation()

        return state, {}

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
        self.departing_evs = []

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

            self.departing_evs += ev

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

        if self.simulate_grid:
            for tr in self.transformers:
                self.node_ev_power[tr.id + 1:,
                                   self.current_step] = tr.current_power

            active_power, reactive_power, vm = self.grid.step(
                self.node_ev_power[1:, self.current_step])

            self.node_active_power[1:, self.current_step] = active_power
            self.node_reactive_power[1:, self.current_step] = reactive_power
            self.node_voltage[:, self.current_step] = vm

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

        self._update_power_statistics(self.departing_evs)

        self.current_step += 1
        self._step_date()

        if self.current_step < self.simulation_length:
            self.charge_power_potential[self.current_step] = calculate_charge_power_potential(
                self)

        self.current_evs_parked += self.current_ev_arrived - self.current_ev_departed

        reward = self._calculate_reward(total_costs,
                                        user_satisfaction_list,
                                        total_invalid_action_punishment)

        if self.cost_function is not None:
            cost = self.cost_function(self,
                                      total_costs,
                                      user_satisfaction_list,
                                      total_invalid_action_punishment)
        else:
            cost = None

        if visualize:
            visualize_step(self)

        self.render()

        return self._check_termination(reward, cost)

    def _check_termination(self, reward, cost):
        '''Checks if the episode is done or any constraint is violated'''
        truncated = False
        action_mask = np.zeros(self.number_of_ports)
        # action mask is 1 if an EV is connected to the port
        for i, cs in enumerate(self.charging_stations):
            for j in range(cs.n_ports):
                if cs.evs_connected[j] is not None:
                    action_mask[i*cs.n_ports + j] = 1

        # Check if the episode is done or any constraint is violated
        if self.current_step >= self.simulation_length:

            self.done = True
            self.stats = get_statistics(self)

            self.stats['action_mask'] = action_mask
            self.cost = cost

            if self.verbose:
                print_statistics(self)
                print(
                    f"Episode finished after {self.current_step} timesteps\n")

            if self.save_replay:
                self._save_sim_replay()

            if self.save_plots:
                # save the env as a pickle file
                with open(f"./results/{self.sim_name}/env.pkl", 'wb') as f:
                    self.renderer = None
                    pickle.dump(self, f)
                ev_city_plot(self)

            if self.cost_function is not None:
                return self._get_observation(), reward, True, truncated, self.stats
            else:
                return self._get_observation(), reward, True, truncated, self.stats
        else:
            stats = {
                'cost': cost,
                'action_mask': action_mask,
            }

            if self.cost_function is not None:
                return self._get_observation(), reward, False, truncated, stats
            else:
                return self._get_observation(), reward, False, truncated, stats

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

            for ev in self.departing_evs:
                if not self.lightweight_plots:
                    self.port_energy_level[ev.id, ev.location,
                                           self.current_step] = ev.current_capacity/ev.battery_capacity
                    self.port_current[ev.id, ev.location,
                                      self.current_step] = ev.actual_current

    def _step_date(self):
        '''Steps the simulation date by one timestep'''
        self.sim_date = self.sim_date + \
            datetime.timedelta(minutes=self.timescale)

    def _get_observation(self):

        return self.state_function(self)

    def set_cost_function(self, cost_function):
        '''
        This function sets the cost function of the environment
        '''
        self.cost_function = cost_function

    def set_reward_function(self, reward_function):
        '''
        This function sets the reward function of the environment
        '''
        self.reward_function = reward_function

    def _calculate_reward(self, total_costs, user_satisfaction_list, invalid_action_punishment):
        '''Calculates the reward for the current step'''

        reward = self.reward_function(
            self, total_costs, user_satisfaction_list, invalid_action_punishment)
        self.total_reward += reward

        return reward