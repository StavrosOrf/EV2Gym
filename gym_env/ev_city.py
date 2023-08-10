'''
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


class EVCity(gym.Env):
    '''
    This file contains the EVCity class, which is used to represent the environment of the city.
    '''

    def __init__(self,
                 cs=None,
                 load_prices_from_replay=False,
                 load_ev_from_replay=False,
                 load_from_replay_path=None,
                 empty_ports_at_end_of_simulation=True,
                 simulate_grid=False,
                 generate_rnd_game=False,  # generate a random game without terminating conditions
                 case='default',
                 number_of_ports_per_cs=2,
                 number_of_transformers=1,
                 score_threshold=1,
                 timescale=5,
                 date=(2023, 7, 21),  # (year, month, day)
                 hour=(10, 0),  # (hour, minute) 24 hour format
                 seed = 42, #TODO: add seed
                 save_replay=True,
                 save_plots=True,
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

        self.seed = seed

        if load_from_replay_path is not None:
            with open(load_from_replay_path, 'rb') as file:
                self.replay = pickle.load(file)

            # self.save_replay = False
            self.sim_date = self.replay.sim_date
            self.simulate_grid = self.replay.simulate_grid
            self.timescale = self.replay.timescale
            # self.simulation_length = self.replay.sim_length
            self.cs = self.replay.n_cs
            self.number_of_transformers = self.replay.n_transformers
            self.score_threshold = self.replay.score_threshold
            self.number_of_ports_per_cs = self.replay.max_n_ports

        else:
            assert cs is not None, "Please provide the number of charging stations"
            self.cs = cs  # Number of charging stations
            # Threshold for the user satisfaction score
            self.score_threshold = score_threshold
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

            self.simulate_grid = simulate_grid  # Whether to simulate the grid or not

        self.sim_starting_date = self.sim_date
        self.sim_name = f'ev_city_{self.simulation_length}_' + \
            f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}'

        # Simulate grid
        if self.simulate_grid:
            self.grid = Grid(charging_stations=cs, case=case)
            self.cs_buses = self.grid.get_charging_stations_buses()
            self.cs_transformers = self.grid.get_bus_transformers()
        else:
            self.cs_buses = [None] * cs
            self.cs_transformers = np.random.randint(
                self.number_of_transformers, size=cs)

        # Instatiate Transformers
        self.transformers = self._load_transformers()

        # Instatiate Charging Stations
        self.charging_stations = self._load_ev_charger_profiles()

        # Instatiate EV profiles if they exist
        self.ev_profiles = self._load_ev_profiles()

        # Load Electricity prices for every charging station
        self.charge_prices, self.discharge_prices = self._load_electricity_prices()

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

        self.transformer_power = np.zeros([self.number_of_transformers,
                                           self.simulation_length])

        self.cs_power = np.zeros([self.cs, self.simulation_length])
        self.port_power = np.zeros([self.number_of_ports,
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

        os.makedirs("./plots", exist_ok=True)
        os.makedirs(f"./plots/{self.sim_name}", exist_ok=True)

        if self.save_replay:
            self.EVs = []  # Store all of the EVs in the simulation that arrived

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

        return self._get_observation()

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
        user_satisfaction_list = []

        self.current_ev_departed = 0
        self.current_ev_arrived = 0

        port_counter = 0

        # Reset current power of all transformers
        for tr in self.transformers:
            tr.current_power = 0

        # Call step for each charging station and spawn EVs where necessary
        for cs in self.charging_stations:
            n_ports = cs.n_ports
            costs, user_satisfaction = cs.step(
                actions[port_counter:port_counter + n_ports],
                self.charge_prices[cs.id, self.current_step],
                self.discharge_prices[cs.id, self.current_step])

            for u in user_satisfaction:
                user_satisfaction_list.append(u)

            self.transformers[cs.connected_transformer].step(
                cs.current_power_output)

            total_costs += costs
            self.current_ev_departed += len(user_satisfaction)

            port_counter += n_ports

            # Spawn EVs
            if self.ev_profiles is None:
                min_stay_of_ev = int(
                    20 * 5 / self.timescale)  # from 50 minutes
                max_stay_of_ev = int(40 * 5 / self.timescale)  # to 100 minutes
                if max_stay_of_ev > self.simulation_length:
                    self.empty_ports_at_end_of_simulation = False
                    raise ValueError(
                        "The maximum stay of an EV is greater than the simulation length! \n" +
                        "Please increase the simulation length or disable the empty_ports_at_end_of_simulation option")

                if not (self.empty_ports_at_end_of_simulation and
                        self.current_step + 1 + max_stay_of_ev >= self.simulation_length) and \
                        n_ports > cs.n_evs_connected:

                    # get a random float in [0,1] to decide if spawn an EV
                    self.spawn_rate = 0.85
                    if np.random.rand() < self.spawn_rate:
                        ev = EV(id=None,
                                location=cs.id,
                                battery_capacity_at_arrival=np.random.uniform(
                                    1, 49),
                                time_of_arrival=self.current_step+1,
                                earlier_time_of_departure=self.current_step+1
                                + np.random.randint(min_stay_of_ev, max_stay_of_ev),
                                timescale=self.timescale,)
                        # earlier_time_of_departure=self.current_step+1 + np.random.randint(10, 40),)
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
                                            user_satisfaction_list)

        if visualize:
            self.visualize()

        # Check if the episode is done
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
                self.print_statistics()

            print(f"\nEpisode finished after {self.current_step} timesteps")

            if self.save_replay:
                self.save_sim_replay()

            if self.save_plots:
                self.plot()

            self.done = True

            return self._get_observation(), reward, True
        else:
            return self._get_observation(), reward, False

    def save_sim_replay(self):
        '''Saves the simulation data in a pickle file'''
        replay = EvCityReplay(self)
        print(f"Saving replay file at {replay.replay_path}")
        with open(replay.replay_path, 'wb') as f:
            pickle.dump(replay, f)

    def update_power_statistics(self):
        '''Updates the power statistics of the simulation'''
        for tr in self.transformers:
            self.transformer_power[tr.id, self.current_step] = tr.current_power

        for cs in self.charging_stations:
            self.cs_power[cs.id, self.current_step] = cs.current_power_output

            for port in range(cs.n_ports):
                ev = cs.evs_connected[port]
                if ev is not None:
                    self.port_power[port, cs.id,
                                    self.current_step] = ev.current_power
                    self.port_energy_level[port, cs.id,
                                           self.current_step] = ev.current_capacity

    def visualize(self):
        '''Renders the current state of the environment in the terminal'''

        print(f"\n Step: {self.current_step}" +
              f" | {self.sim_date.hour}:{self.sim_date.minute}:{self.sim_date.second} |" +
              f" \tEVs +{self.current_ev_arrived} / -{self.current_ev_departed}" +
              f" | Total: {self.current_evs_parked} / {self.number_of_ports}")

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
            print("")
            for tr in self.transformers:
                print(tr)

    def plot(self):
        '''Plots the simulation data

        Plots:
            - The total power of each transformer
            - The power of each charging station
            - The energy level of each EV in charging stations
        '''
        print("Plotting simulation data at ./plots/" + self.sim_name + "/")
        # date_range = pd.date_range(start=self.sim_starting_date,
        #                            end=self.sim_date -
        #                            datetime.timedelta(
        #                                minutes=self.timescale),
        #                            freq=f'{self.timescale}min')
        date_range = pd.date_range(start=self.sim_starting_date,
                                   end=self.sim_starting_date +
                                   (self.simulation_length - 1) *
                                   datetime.timedelta(
                                       minutes=self.timescale),
                                   freq=f'{self.timescale}min')
        date_range_print = pd.date_range(start=self.sim_starting_date,
                                         end=self.sim_date,
                                         periods=10)

        # Plot the energy level of each EV for each charging station
        plt.figure(figsize=(20, 17))
        plt.style.use('seaborn-darkgrid')
        plt.rcParams.update({'font.size': 16})
        counter = 1
        dim_x = int(np.ceil(np.sqrt(self.cs)))
        dim_y = int(np.ceil(self.cs/dim_x))
        for cs in self.charging_stations:

            plt.subplot(dim_x, dim_y, counter)
            df = pd.DataFrame([], index=date_range)

            for port in range(cs.n_ports):
                df[port] = self.port_energy_level[port, cs.id, :]

            # Add another row with one datetime step to make the plot look better
            df.loc[df.index[-1] +
                   datetime.timedelta(minutes=self.timescale)] = df.iloc[-1]

            for port in range(cs.n_ports):
                for i, (t_arr, t_dep) in enumerate(self.port_arrival[f'{cs.id}.{port}']):
                    # x = df.index[t_arr:t_dep]
                    y = df[port].values.T[t_arr:t_dep]
                    # fill y with 0 before and after to match the length of df
                    y = np.concatenate(
                        [np.zeros(t_arr), y, np.zeros(len(df) - t_dep)])

                    plt.step(df.index, y, where='post')
                    plt.fill_between(df.index,
                                     y,
                                     step='post',
                                     alpha=0.7,
                                     label=f'EV {i}, Port {port}')

            plt.title(f'Charging Station {cs.id}')
            plt.xlabel(f'Time')
            plt.ylabel('Energy Level (kW)')
            plt.xlim([self.sim_starting_date, self.sim_date])
            plt.xticks(ticks=date_range_print,
                       labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print], rotation=45)
            if len(self.port_arrival[f'{cs.id}.{port}']) < 6:
                plt.legend()
            plt.grid(True, which='minor', axis='both')
            counter += 1

        plt.tight_layout()
        # Save plt to html
        fig_name = f'plots/{self.sim_name}/EV_Energy_Level.html'
        plt.savefig(fig_name, format='svg',
                    dpi=600, bbox_inches='tight')

        # Plot the total power of each transformer
        plt.figure(figsize=(20, 17))
        counter = 1
        dim_x = int(np.ceil(np.sqrt(self.number_of_transformers)))
        dim_y = int(np.ceil(self.number_of_transformers/dim_x))
        for tr in self.transformers:

            plt.subplot(dim_x, dim_y, counter)
            df = pd.DataFrame([],
                              index=date_range)

            for cs in tr.cs_ids:
                df[cs] = self.cs_power[cs, :]

            # create 2 dfs, one for positive power and one for negative
            df_pos = df.copy()
            df_pos[df_pos < 0] = 0
            df_neg = df.copy()
            df_neg[df_neg > 0] = 0
            colors = plt.cm.gist_earth(np.linspace(0.1, 0.8, len(tr.cs_ids)))

            # Add another row with one datetime step to make the plot look better
            df_pos.loc[df_pos.index[-1] +
                       datetime.timedelta(minutes=self.timescale)] = df_pos.iloc[-1]
            df_neg.loc[df_neg.index[-1] +
                       datetime.timedelta(minutes=self.timescale)] = df_neg.iloc[-1]

            # plot the positive power
            plt.stackplot(df_pos.index, df_pos.values.T,
                          interpolate=True,
                          step='post',
                          alpha=0.7,
                          colors=colors,
                          linestyle='--')

            df['total'] = df.sum(axis=1)
            # print(df)
            max_power = tr.max_power * self.timescale / 60
            min_power = tr.min_power * self.timescale / 60
            plt.plot([self.sim_starting_date, self.sim_date],
                     [max_power, max_power], 'r--')
            plt.step(df.index, df['total'], 'darkgreen',
                     where='post', linestyle='--')
            plt.plot([self.sim_starting_date, self.sim_date],
                     [min_power, min_power], 'r--')
            plt.stackplot(df_neg.index, df_neg.values.T,
                          interpolate=True,
                          step='post',
                          colors=colors,
                          alpha=0.7,
                          linestyle='--')
            plt.plot([self.sim_starting_date, self.sim_date], [0, 0], 'black')

            # for cs in tr.cs_ids:
            #     plt.step(df.index, df[cs], 'white', where='post', linestyle='--')
            plt.title(f'Transformer {tr.id}')
            plt.xlabel(f'Time')
            plt.ylabel(f'Power (kWh per {self.timescale} min)')
            plt.xlim([self.sim_starting_date, self.sim_date])
            plt.xticks(ticks=date_range_print,
                       labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print], rotation=45)
            plt.legend([f'CS {i}' for i in tr.cs_ids] +
                       ['Total Power Limit', 'Total Power'])
            plt.grid(True, which='minor', axis='both')
            counter += 1

        plt.tight_layout()
        # plt.show()
        fig_name = f'plots/{self.sim_name}/Transformer_Power.html'
        plt.savefig(fig_name, format='svg',
                    dpi=600, bbox_inches='tight')

        # Plot the power of each charging station
        counter = 1
        plt.figure(figsize=(20, 17))
        dim_x = int(np.ceil(np.sqrt(self.cs)))
        dim_y = int(np.ceil(self.cs/dim_x))
        for cs in self.charging_stations:

            plt.subplot(dim_x, dim_y, counter)
            df = pd.DataFrame([], index=date_range)

            for port in range(cs.n_ports):
                df[port] = self.port_power[port, cs.id, :]
                       # create 2 dfs, one for positive power and one for negative

            df_pos = df.copy()
            df_pos[df_pos < 0] = 0
            df_neg = df.copy()
            df_neg[df_neg > 0] = 0
            colors = plt.cm.gist_earth(np.linspace(0.1, 0.8, cs.n_ports))

            # Add another row with one datetime step to make the plot look better
            df_pos.loc[df_pos.index[-1] +
                       datetime.timedelta(minutes=self.timescale)] = df_pos.iloc[-1]
            df_neg.loc[df_neg.index[-1] +
                       datetime.timedelta(minutes=self.timescale)] = df_neg.iloc[-1]

            plt.stackplot(df_pos.index, df_pos.values.T,
                          interpolate=True,
                          step='post',
                          alpha=0.7,
                          colors=colors)
            df['total'] = df.sum(axis=1)

            # plot the power limit
            max_power = cs.max_charge_power * self.timescale / 60
            min_power = -cs.max_discharge_power * self.timescale / 60
            plt.plot([self.sim_starting_date, self.sim_date],
                     [max_power, max_power], 'r--')
            plt.step(df.index, df['total'], 'darkgreen',
                     where='post', linestyle='--')
            plt.plot([self.sim_starting_date, self.sim_date],
                     [min_power, min_power], 'r--')
            plt.stackplot(df_neg.index, df_neg.values.T,
                          interpolate=True,
                          step='post',
                          colors=colors,
                          alpha=0.7)
            plt.plot([self.sim_starting_date, self.sim_date], [0, 0], 'black')

            # for i in range(cs.n_ports):
            #     plt.step(df.index, df[i], 'grey', where='post', linestyle='--')

            plt.title(f'Charging Station {cs.id}')
            plt.xlabel(f'Time')
            plt.ylabel(f'Power (kWh per {self.timescale} min)')
            plt.ylim([min_power*1.1, max_power*1.1])
            plt.xlim([self.sim_starting_date, self.sim_date])
            plt.xticks(ticks=date_range_print,
                       labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print], rotation=45)
            # place the legend under each plot

            plt.legend([f'Port {i}' for i in range(
                cs.n_ports)] + ['Total Power Limit',
                                'Total Power'])
            plt.grid(True, which='minor', axis='both')
            counter += 1

        plt.tight_layout()
        # Save plt to html
        fig_name = f'plots/{self.sim_name}/CS_Power.html'
        plt.savefig(fig_name, format='svg', dpi=600, bbox_inches='tight')

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
        print(f'  - Total profits: {total_profits*100:.2f} €')
        print(
            f'  - Average user satisfaction: {average_user_satisfaction:.2f} %')

        print(f'  - Total energy charged: {toal_energy_charged:.1f} kW')
        print(
            f'  - Total energy discharged: {total_energy_discharged:.1f} kW\n')

        for cs in self.charging_stations:
            print(cs)
        print("==============================================================\n\n")

    def _step_date(self):
        '''Steps the simulation date by one timestep'''
        self.sim_date = self.sim_date + \
            datetime.timedelta(minutes=self.timescale)

    def _get_observation(self, include_grid=False):
        '''Returns the current state of the environment'''
        state = [self.current_step,
                 self.timescale,
                 self.cs,]

        for tr in self.transformers:
            state.append(tr.get_state())

        for cs in self.charging_stations:
            state.append(cs.get_state())

        if include_grid:
            state.append(self.grid.get_grid_state())

        return np.hstack(state)

    def _calculate_reward(self, total_costs, user_satisfaction_list):
        '''Calculates the reward for the current step'''
        reward = total_costs
        return reward
