'''
This file contains the loaders for the EV City environment.
'''

import numpy as np
import pandas as pd
import math
import datetime
import pkg_resources
import json
from typing import List, Tuple

from ev2gym.models.ev_charger import EV_Charger
from ev2gym.models.ev import EV
from ev2gym.models.transformer import Transformer

from ev2gym.utilities.utils import EV_spawner, generate_power_setpoints


def load_ev_spawn_scenarios(env) -> None:
    '''Loads the EV spawn scenarios of the simulation'''

    df_arrival_week_file = pkg_resources.resource_filename(
        'ev2gym', 'data/distribution-of-arrival.csv')
    df_arrival_weekend_file = pkg_resources.resource_filename(
        'ev2gym', 'data/distribution-of-arrival-weekend.csv')
    df_connection_time_file = pkg_resources.resource_filename(
        'ev2gym', 'data/distribution-of-connection-time.csv')
    df_energy_demand_file = pkg_resources.resource_filename(
        'ev2gym', 'data/distribution-of-energy-demand.csv')
    time_of_connection_vs_hour_file = pkg_resources.resource_filename(
        'ev2gym', 'data/time_of_connection_vs_hour.npy')

    df_req_energy_file = pkg_resources.resource_filename(
        'ev2gym', 'data/mean-demand-per-arrival.csv')
    df_time_of_stay_vs_arrival_file = pkg_resources.resource_filename(
        'ev2gym', 'data/mean-session-length-per.csv')

    ev_specs_file = pkg_resources.resource_filename(
        'ev2gym', 'data/ev_specs.json')

    env.df_arrival_week = pd.read_csv(df_arrival_week_file)  # weekdays
    env.df_arrival_weekend = pd.read_csv(df_arrival_weekend_file)  # weekends
    env.df_connection_time = pd.read_csv(
        df_connection_time_file)  # connection time
    env.df_energy_demand = pd.read_csv(df_energy_demand_file)  # energy demand
    env.time_of_connection_vs_hour = np.load(
        time_of_connection_vs_hour_file)  # time of connection vs hour

    env.df_req_energy = pd.read_csv(
        df_req_energy_file)  # energy demand per arrival
    # replace column work with workplace
    env.df_req_energy = env.df_req_energy.rename(columns={'work': 'workplace',
                                                          'home': 'private'})
    env.df_req_energy = env.df_req_energy.fillna(0)

    env.df_time_of_stay_vs_arrival = pd.read_csv(
        df_time_of_stay_vs_arrival_file)  # time of stay vs arrival
    env.df_time_of_stay_vs_arrival = env.df_time_of_stay_vs_arrival.fillna(0)
    env.df_time_of_stay_vs_arrival = env.df_time_of_stay_vs_arrival.rename(columns={'work': 'workplace',
                                                                                    'home': 'private'})

    # Load the EV specs
    if env.config['heterogeneous_ev_specs']:
        with open(ev_specs_file) as f:
            env.ev_specs = json.load(f)

        registrations = np.zeros(len(env.ev_specs.keys()))
        for i, ev_name in enumerate(env.ev_specs.keys()):
            # sum the total number of registrations
            registrations[i] = env.ev_specs[ev_name]['number_of_registrations_2023_nl']

        env.normalized_ev_registrations = registrations/registrations.sum()


def load_power_setpoints(env) -> np.ndarray:
    '''
    Loads the power setpoints of the simulation based on the day-ahead prices
    '''

    if env.load_from_replay_path:
        return env.replay.power_setpoints
    else:
        return generate_power_setpoints(env)


def generate_residential_inflexible_loads(env) -> np.ndarray:
    '''
    This function loads the inflexible loads of each transformer
    in the simulation.
    '''

    # Load the data
    data_path = pkg_resources.resource_filename(
        'ev2gym', 'data/residential_loads.csv')
    data = pd.read_csv(data_path, header=None)

    desired_timescale = env.timescale
    simulation_length = env.simulation_length
    simulation_date = env.sim_starting_date.strftime('%Y-%m-%d %H:%M:%S')
    number_of_transformers = env.number_of_transformers

    dataset_timescale = 15
    dataset_starting_date = '2022-01-01 00:00:00'

    if desired_timescale > dataset_timescale:
        data = data.groupby(
            data.index // (desired_timescale/dataset_timescale)).max()
    elif desired_timescale < dataset_timescale:
        # extend the dataset to data.shape[0] * (dataset_timescale/desired_timescale)
        # by repeating the data every (dataset_timescale/desired_timescale) rows
        data = data.loc[data.index.repeat(
            dataset_timescale/desired_timescale)].reset_index(drop=True)

    # duplicate the data to have two years of data
    data = pd.concat([data, data], ignore_index=True)

    # add a date column to the dataframe
    data['date'] = pd.date_range(
        start=dataset_starting_date, periods=data.shape[0], freq=f'{desired_timescale}min')

    # find year of the data
    year = int(dataset_starting_date.split('-')[0])
    # replace the year of the simulation date with the year of the data
    simulation_date = f'{year}-{simulation_date.split("-")[1]}-{simulation_date.split("-")[2]}'

    simulation_index = data[data['date'] == simulation_date].index[0]

    # select the data for the simulation date
    data = data[simulation_index:simulation_index+simulation_length]

    # drop the date column
    data = data.drop(columns=['date'])
    new_data = pd.DataFrame()

    for i in range(number_of_transformers):
        new_data['tr_'+str(i)] = data.sample(10, axis=1,
                                             random_state=env.tr_seed).sum(axis=1)

    # return the "tr_" columns
    return new_data.to_numpy().T


def generate_pv_generation(env) -> np.ndarray:
    '''
    This function loads the PV generation of each transformer by loading the data from a file
    and then adding minor variations to the data
    '''

    # Load the data
    data_path = pkg_resources.resource_filename(
        'ev2gym', 'data/pv_netherlands.csv')
    data = pd.read_csv(data_path, sep=',', header=0)
    data.drop(['time', 'local_time'], inplace=True, axis=1)

    desired_timescale = env.timescale
    simulation_length = env.simulation_length
    simulation_date = env.sim_starting_date.strftime('%Y-%m-%d %H:%M:%S')
    number_of_transformers = env.number_of_transformers

    dataset_timescale = 60
    dataset_starting_date = '2019-01-01 00:00:00'

    if desired_timescale > dataset_timescale:
        data = data.groupby(
            data.index // (desired_timescale/dataset_timescale)).max()
    elif desired_timescale < dataset_timescale:
        # extend the dataset to data.shape[0] * (dataset_timescale/desired_timescale)
        # by repeating the data every (dataset_timescale/desired_timescale) rows
        data = data.loc[data.index.repeat(
            dataset_timescale/desired_timescale)].reset_index(drop=True)
        # data = data/ (dataset_timescale/desired_timescale)

    # smooth data by taking the mean of every 5 rows
    data['electricity'] = data['electricity'].rolling(
        window=60//desired_timescale, min_periods=1).mean()
    # use other type of smoothing
    data['electricity'] = data['electricity'].ewm(
        span=60//desired_timescale, adjust=True).mean()

    # duplicate the data to have two years of data
    data = pd.concat([data, data], ignore_index=True)

    # add a date column to the dataframe
    data['date'] = pd.date_range(
        start=dataset_starting_date, periods=data.shape[0], freq=f'{desired_timescale}min')

    # find year of the data
    year = int(dataset_starting_date.split('-')[0])
    # replace the year of the simulation date with the year of the data
    simulation_date = f'{year}-{simulation_date.split("-")[1]}-{simulation_date.split("-")[2]}'

    simulation_index = data[data['date'] == simulation_date].index[0]

    # select the data for the simulation date
    data = data[simulation_index:simulation_index+simulation_length]

    # drop the date column
    data = data.drop(columns=['date'])
    new_data = pd.DataFrame()

    for i in range(number_of_transformers):
        new_data['tr_'+str(i)] = data * env.tr_rng.uniform(0.9, 1.1)

    return new_data.to_numpy().T


def load_transformers(env) -> List[Transformer]:
    '''Loads the transformers of the simulation
    If load_from_replay_path is None, then the transformers are created randomly

    Returns:
        - transformers: a list of transformer objects
    '''

    if env.load_from_replay_path is not None:
        return env.replay.transformers

    transformers = []

    if env.config['inflexible_loads']['include']:

        if env.scenario == 'private':
            inflexible_loads = generate_residential_inflexible_loads(env)

        # TODO add inflexible loads for public and workplace scenarios
        else:
            inflexible_loads = generate_residential_inflexible_loads(env)

    else:
        inflexible_loads = np.zeros((env.number_of_transformers,
                                    env.simulation_length))

    if env.config['solar_power']['include']:
        solar_power = generate_pv_generation(env)
    else:
        solar_power = np.zeros((env.number_of_transformers,
                                env.simulation_length))

    if env.charging_network_topology:
        # parse the topology file and create the transformers
        cs_counter = 0
        for i, tr in enumerate(env.charging_network_topology):
            cs_ids = []
            for cs in env.charging_network_topology[tr]['charging_stations']:
                cs_ids.append(cs_counter)
                cs_counter += 1
            transformer = Transformer(id=i,
                                      env=env,
                                      cs_ids=cs_ids,
                                      max_power=env.charging_network_topology[tr]['max_power'],
                                      inflexible_load=inflexible_loads[i, :],
                                      solar_power=solar_power[i, :],
                                      simulation_length=env.simulation_length
                                      )

            transformers.append(transformer)

    else:
        if env.number_of_transformers > env.cs:
            raise ValueError(
                'The number of transformers cannot be greater than the number of charging stations')
        for i in range(env.number_of_transformers):
            # get indexes where the transformer is connected
            transformer = Transformer(id=i,
                                      env=env,
                                      cs_ids=np.where(
                                          np.array(env.cs_transformers) == i)[0],
                                      max_power=env.config['transformer']['max_power'],
                                      inflexible_load=inflexible_loads[i, :],
                                      solar_power=solar_power[i, :],
                                      simulation_length=env.simulation_length
                                      )

            transformers.append(transformer)
    env.n_transformers = len(transformers)
    return transformers


def load_ev_charger_profiles(env) -> List[EV_Charger]:
    '''Loads the EV charger profiles of the simulation
    If load_from_replay_path is None, then the EV charger profiles are created randomly

    Returns:
        - ev_charger_profiles: a list of ev_charger_profile objects'''

    charging_stations = []
    if env.load_from_replay_path is not None:
        return env.replay.charging_stations

    v2g_enabled = env.config['v2g_enabled']

    if env.charging_network_topology:
        # parse the topology file and create the charging stations
        cs_counter = 0
        for i, tr in enumerate(env.charging_network_topology):
            for cs in env.charging_network_topology[tr]['charging_stations']:
                ev_charger = EV_Charger(id=cs_counter,
                                        connected_bus=0,
                                        connected_transformer=i,
                                        min_charge_current=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['min_charge_current'],
                                        max_charge_current=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['max_charge_current'],
                                        min_discharge_current=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['min_discharge_current'],
                                        max_discharge_current=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['max_discharge_current'],
                                        voltage=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['voltage'],
                                        n_ports=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['n_ports'],
                                        charger_type=env.charging_network_topology[tr][
                                            'charging_stations'][cs]['charger_type'],
                                        phases=env.charging_network_topology[tr]['charging_stations'][cs]['phases'],
                                        timescale=env.timescale,
                                        verbose=env.verbose,)
                cs_counter += 1
                charging_stations.append(ev_charger)
        env.cs = len(charging_stations)
        return charging_stations

    else:
        if v2g_enabled:
            max_discharge_current = env.config['charging_station']['max_discharge_current']
            min_discharge_current = env.config['charging_station']['min_discharge_current']
        else:
            max_discharge_current = 0
            min_discharge_current = 0

        for i in range(env.cs):
            ev_charger = EV_Charger(id=i,
                                    connected_bus=0,  # env.cs_buses[i],
                                    connected_transformer=env.cs_transformers[i],
                                    n_ports=env.number_of_ports_per_cs,
                                    max_charge_current=env.config['charging_station']['max_charge_current'],
                                    min_charge_current=env.config['charging_station']['min_charge_current'],
                                    max_discharge_current=max_discharge_current,
                                    min_discharge_current=min_discharge_current,
                                    phases=env.config['charging_station']['phases'],
                                    voltage=env.config['charging_station']['voltage'],
                                    timescale=env.timescale,
                                    verbose=env.verbose,)

            charging_stations.append(ev_charger)
        return charging_stations


def load_ev_profiles(env) -> List[EV]:
    '''Loads the EV profiles of the simulation
    If load_from_replay_path is None, then the EV profiles are created randomly

    Returns:
        - ev_profiles: a list of ev_profile objects'''

    if env.load_from_replay_path is None:
        
        ev_profiles = EV_spawner(env)
        while len(ev_profiles) == 0:
            ev_profiles = EV_spawner(env)
            
        return ev_profiles
    else:
        return env.replay.EVs


def load_electricity_prices(env) -> Tuple[np.ndarray, np.ndarray]:
    '''Loads the electricity prices of the simulation
    If load_from_replay_path is None, then the electricity prices are created randomly

    Returns:
        - charge_prices: a matrix of size (number of charging stations, simulation length) with the charge prices
        - discharge_prices: a matrix of size (number of charging stations, simulation length) with the discharge prices'''

    if env.load_from_replay_path is not None:
        return env.replay.charge_prices, env.replay.discharge_prices

    # else load historical prices
    file_path = pkg_resources.resource_filename(
        'ev2gym', 'data/Netherlands_day-ahead-2015-2023.csv')
    data = pd.read_csv(file_path, sep=',', header=0)
    drop_columns = ['Country', 'Datetime (Local)']
    data.drop(drop_columns, inplace=True, axis=1)
    data['year'] = pd.DatetimeIndex(data['Datetime (UTC)']).year
    data['month'] = pd.DatetimeIndex(data['Datetime (UTC)']).month
    data['day'] = pd.DatetimeIndex(data['Datetime (UTC)']).day
    data['hour'] = pd.DatetimeIndex(data['Datetime (UTC)']).hour

    # assume charge and discharge prices are the same
    # assume prices are the same for all charging stations

    charge_prices = np.zeros((env.cs, env.simulation_length))
    discharge_prices = np.zeros((env.cs, env.simulation_length))
    # for every simulation step, take the price of the corresponding hour
    sim_temp_date = env.sim_starting_date
    for i in range(env.simulation_length):

        year = sim_temp_date.year
        month = sim_temp_date.month
        day = sim_temp_date.day
        hour = sim_temp_date.hour
        # find the corresponding price
        try:
            charge_prices[:, i] = -data.loc[(data['year'] == year) & (data['month'] == month) & (data['day'] == day) & (data['hour'] == hour),
                                            'Price (EUR/MWhe)'].iloc[0]/1000  # €/kWh
            discharge_prices[:, i] = data.loc[(data['year'] == year) & (data['month'] == month) & (data['day'] == day) & (data['hour'] == hour),
                                              'Price (EUR/MWhe)'].iloc[0]/1000  # €/kWh
        except:
            print(
                'Error: no price found for the given date and hour. Using 2022 prices instead.')

            year = 2022
            if day > 28:
                day -= 1
            print("Debug:", year, month, day, hour)
            charge_prices[:, i] = -data.loc[(data['year'] == year) & (data['month'] == month) & (data['day'] == day) & (data['hour'] == hour),
                                            'Price (EUR/MWhe)'].iloc[0]/1000  # €/kWh
            discharge_prices[:, i] = data.loc[(data['year'] == year) & (data['month'] == month) & (data['day'] == day) & (data['hour'] == hour),
                                              'Price (EUR/MWhe)'].iloc[0]/1000  # €/kWh

        # step to next
        sim_temp_date = sim_temp_date + \
            datetime.timedelta(minutes=env.timescale)

    discharge_prices = discharge_prices * env.config['discharge_price_factor']
    return charge_prices, discharge_prices
