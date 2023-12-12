'''
This file contains the loaders for the EV City environment.
'''

import numpy as np
import pandas as pd
import datetime
import random

from .ev_charger import EV_Charger
from .ev import EV
from .transformer import Transformer

def load_ev_spawn_scenarios(env):
    '''Loads the EV spawn scenarios of the simulation'''

    env.df_arrival_week = pd.read_csv(
        './data/distribution-of-arrival.csv')  # weekdays
    env.df_arrival_weekend = pd.read_csv(
        './data/distribution-of-arrival-weekend.csv')  # weekends
    env.df_connection_time = pd.read_csv(
        './data/distribution-of-connection-time.csv')  # connection time
    env.df_energy_demand = pd.read_csv(
        './data/distribution-of-energy-demand.csv')  # energy demand
    env.time_of_connection_vs_hour = np.load(
        './data/time_of_connection_vs_hour.npy')

def load_power_setpoints(env,randomly):
    '''
    Loads the power setpoints of the simulation based on the day-ahead prices'''
    
    #It is necessary to run the simulation first in order to get the ev_load_potential
    if not randomly and env.load_from_replay_path is None:
        raise ValueError('Cannot load power setpoints from day-ahead prices if load_from_replay_path is None')
    
    power_setpoints = np.ones(env.simulation_length)
    
    if env.load_from_replay_path:    
        return env.replay.power_setpoints

    if randomly:
        inverse_prices = 1/abs(env.charge_prices[0,:]+0.001)
        # return power_setpoints*(inverse_prices*env.cs)*np.random.uniform(0.25,0.35,1)
        return power_setpoints*(inverse_prices*env.cs)*np.random.uniform(0.08,0.09,1)
    else:
        raise NotImplementedError('Loading power setpoints from is not implemented yet')

    

def load_transformers(env):
    '''Loads the transformers of the simulation
    If load_from_replay_path is None, then the transformers are created randomly

    Returns:
        - transformers: a list of transformer objects'''

    transformers = []
    if env.load_from_replay_path is None:
        if env.number_of_transformers > env.cs:
            raise ValueError('The number of transformers cannot be greater than the number of charging stations')                
        for i in range(env.number_of_transformers):            
            #get indexes where the transformer is connected
            transformer = Transformer(id=i,
                                        cs_ids=np.where(
                                            np.array(env.cs_transformers) == i)[0],
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
    else:
        return None

def load_electricity_prices(env):
    '''Loads the electricity prices of the simulation
    If load_from_replay_path is None, then the electricity prices are created randomly

    Returns:
        - charge_prices: a matrix of size (number of charging stations, simulation length) with the charge prices
        - discharge_prices: a matrix of size (number of charging stations, simulation length) with the discharge prices'''

    if env.load_from_replay_path is not None and env.load_prices_from_replay:     
        return env.replay.charge_prices, env.replay.discharge_prices
    
    #else load historical prices
    data = pd.read_csv('./data/Netherlands_day-ahead-2015-2023.csv', sep=',', header=0)
    drop_columns = ['Country','Datetime (UTC)']
    data.drop(drop_columns, inplace=True, axis=1)
    data['year'] = pd.DatetimeIndex(data['Datetime (Local)']).year
    data['month'] = pd.DatetimeIndex(data['Datetime (Local)']).month
    data['day'] = pd.DatetimeIndex(data['Datetime (Local)']).day
    data['hour'] = pd.DatetimeIndex(data['Datetime (Local)']).hour

    #assume charge and discharge prices are the same
    #assume prices are the same for all charging stations
    
    charge_prices = np.zeros((env.cs, env.simulation_length))
    discharge_prices = np.zeros((env.cs, env.simulation_length))
    #for every simulation step, take the price of the corresponding hour
    sim_temp_date = env.sim_starting_date
    for i in range(env.simulation_length):
        
        year = sim_temp_date.year
        month = sim_temp_date.month
        day = sim_temp_date.day
        hour = sim_temp_date.hour
        #find the corresponding price
        try: 
            charge_prices[:,i] = -data.loc[(data['year'] == year) & (data['month'] == month) & (data['day'] == day) & (data['hour'] == hour),
                                    'Price (EUR/MWhe)'].iloc[0]/1000 #€/kWh
            discharge_prices[:,i] = data.loc[(data['year'] == year) & (data['month'] == month) & (data['day'] == day) & (data['hour'] == hour),
                                    'Price (EUR/MWhe)'].iloc[0]/1000 #€/kWh
        except IndexError:
            print('Error: no price found for the given date and hour. Using 2022 prices instead.')
            
            year = 2022
            charge_prices[:,i] = -data.loc[(data['year'] == year) & (data['month'] == month) & (data['day'] == day) & (data['hour'] == hour),
                                    'Price (EUR/MWhe)'].iloc[0]/1000 #€/kWh
            discharge_prices[:,i] = data.loc[(data['year'] == year) & (data['month'] == month) & (data['day'] == day) & (data['hour'] == hour),
                                    'Price (EUR/MWhe)'].iloc[0]/1000 #€/kWh
        
        #step to next 
        sim_temp_date = sim_temp_date + datetime.timedelta(minutes=env.timescale)
    return charge_prices, discharge_prices