'''
This file contains the loaders for the EV City environment.
'''

import numpy as np
import pandas as pd

from .ev_charger import EV_Charger
from .ev import EV
from .transformer import Transformer

def load_ev_spawn_scenarios(env):
    '''Loads the EV spawn scenarios of the simulation'''

    env.df_arrival_week = pd.read_csv(
        '.\data\distribution-of-arrival.csv')  # weekdays
    env.df_arrival_weekend = pd.read_csv(
        '.\data\distribution-of-arrival-weekend.csv')  # weekends
    env.df_connection_time = pd.read_csv(
        '.\data\distribution-of-connection-time.csv')  # connection time
    env.df_energy_demand = pd.read_csv(
        '.\data\distribution-of-energy-demand.csv')  # energy demand
    env.time_of_connection_vs_hour = np.load(
        '.\data\Time_of_connection_vs_hour.npy')

def load_power_setpoints(env):
    if env.load_from_replay_path is None:
        return np.ones(env.simulation_length) * 20  # kW

    return env.replay.power_setpoints

def load_transformers(env):
    '''Loads the transformers of the simulation
    If load_from_replay_path is None, then the transformers are created randomly

    Returns:
        - transformers: a list of transformer objects'''

    transformers = []
    if env.load_from_replay_path is None:
        for i in range(env.number_of_transformers):
            transformer = Transformer(id=i,
                                        cs_ids=np.where(
                                            env.cs_transformers == i)[0],
                                        timescale=env.timescale,)
            transformers.append(transformer)
    else:
        transformers = env.replay.transformers

    return transformers

def load_ev_charger_profiles(env):
    '''Loads the EV charger profiles of the simulation
    If load_from_replay_path is None, then the EV charger profiles are created randomly

    Returns:
        - ev_charger_profiles: a list of ev_charger_profile objects'''

    charging_stations = []
    if env.load_from_replay_path is None:
        for i in range(env.cs):
            ev_charger = EV_Charger(id=i,
                                    connected_bus=env.cs_buses[i],
                                    connected_transformer=env.cs_transformers[i],
                                    n_ports=env.number_of_ports_per_cs,
                                    timescale=env.timescale,
                                    verbose=env.verbose,)

            charging_stations.append(ev_charger)
        return charging_stations

    return env.replay.charging_stations

def load_ev_profiles(env):
    '''Loads the EV profiles of the simulation
    If load_from_replay_path is None, then the EV profiles are created randomly

    Returns:
        - ev_profiles: a list of ev_profile objects'''

    if env.load_from_replay_path is None:
        return None
    elif env.load_ev_from_replay:
        return env.replay.EVs

def load_electricity_prices(env):
    '''Loads the electricity prices of the simulation
    If load_from_replay_path is None, then the electricity prices are created randomly

    Returns:
        - charge_prices: a matrix of size (number of charging stations, simulation length) with the charge prices
        - discharge_prices: a matrix of size (number of charging stations, simulation length) with the discharge prices'''
    if not env.load_prices_from_replay:
        if env.static_prices:
            return np.ones((env.cs, env.simulation_length)) * -0.01, \
                np.ones((env.cs, env.simulation_length)) * 0.1

    if env.load_from_replay_path is None or not env.load_prices_from_replay:
        charge_prices = np.random.normal(
            -0.05, 0.05, size=(env.cs, env.simulation_length))
        charge_prices = -1 * np.abs(charge_prices)
        discharge_prices = np.random.normal(
            0.1, 0.05, size=(env.cs, env.simulation_length))
        discharge_prices = np.abs(discharge_prices)
        return charge_prices, discharge_prices

    return env.replay.charge_prices, env.replay.discharge_prices
