'''
This script is used to load the EVs of the simulation and plot the behavior of the EVs.
'''

import os
import sys
import datetime
sys.path.append(os.path.realpath('../'))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from EVsSimulator.ev_city import EVsSimulator
from EVsSimulator.models.ev import EV
import pandas as pd

import tqdm as tqdm

def plot_soc_vs_hour():
    
    df_private = pd.read_csv('./results/evs_data_private.csv')
    df_public = pd.read_csv('./results/evs_data_public.csv')
    df_workplace = pd.read_csv('./results/evs_data_workplace.csv')

    num_bins = 7*96
    font_size = 25

    # plt.figure(figsize=(20, 10))
    # plt.style.use('seaborn-darkgrid')
    plt.grid(True, which='minor', axis='both')
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams['font.family'] = ['serif']

    plt.close()

    #################### #################### #################### ####################
    fig, ax = plt.subplots()
    # sort data by arrival hour
    data = data.sort_values(by='arrival_hour')

    # plot 2d cmap of the arrival soc vs the arrival hour where the color is the probability

    # make a 2d histogram of the arrival soc vs the arrival hour
    # 2d histogram

    arrival_hour = np.arange(0, 24, 1)
    soc = np.arange(0, 1, 0.01)

    cmap = plt.cm.viridis
    ax.pcolormesh(arrival_hour,
                  soc,
                  np.histogram2d(data['arrival_hour'],
                                 data['arrival_soc'],
                                 bins=(len(arrival_hour), len(soc)))[0].T / len(data),
                  norm='linear',
                  cmap=cmap,
                  shading='auto')

    # add colorbar
    cbar = plt.colorbar(ax.pcolormesh(arrival_hour,
                                      soc,
                                      np.histogram2d(data['arrival_hour'],
                                                     data['arrival_soc'],
                                                     bins=(len(arrival_hour), len(soc)))[0].T / len(data),
                                      cmap=cmap,
                                      norm='linear',
                                      shading='auto'))

    ax.set_ylabel('Arrival SOC', fontsize=font_size)
    ax.set_xlabel('Arrival Time', fontsize=font_size)

    plt.show()
    #################### #################### #################### ####################

    # bin = data['arrival_hour'].hist(bins=24,
    #                                 density=True,
    #                                 label='Public',
    #                                 ax=ax,
    #                                 weights=data['arrival_soc'],
    #                                 color='blue',
    #                                 histtype='step')

    # ax.set_ylabel('Probability', fontsize=font_size)
    # ax.set_xlabel('Time (h)', fontsize=font_size)

    # ax.legend()
    # plt.show()

    #################### #################### #################### ####################
    # plot histogram of the normalized arrival time and departure time in the same plot
    # sort data by arrival time
    data = data.sort_values(by='arrival')

    bin = data['arrival'].hist(bins=num_bins, density=True, label='Arrival Time', ax=ax,
                               color='blue',
                               histtype='step')
    # #print bin edges

    data = data.sort_values(by='departure')
    # bin = data['departure'].hist(bins=num_bins,density=True,label='Departure Time',ax=ax)
    # plot only a line for the departure time
    bin = data['departure'].hist(bins=num_bins, density=True, ax=ax,
                                 label='Departure Time',
                                 color='red',
                                 histtype='step')

    # xtixks mention the full name of the day

    ticks = []
    counter = 2
    for i in range(0, 7*96, 32):
        arrival = datetime.datetime(
            2024, 2, 12, 5, 0) + datetime.timedelta(minutes=replay.timescale) * i
        if counter == 3:
            ticks.append(arrival.strftime('%a %H:%M'))
            counter = 0
        else:
            ticks.append(arrival.strftime('%H:%M'))
        counter += 1

    ax.set_xticks([i for i in range(0, 7*96, 32)],
                  ticks, rotation=45,
                  fontsize=font_size-8)

    # plt.xticks(rotation=45)
    ax.set_ylabel('Probability', fontsize=font_size)
    ax.set_xlabel('Time', fontsize=font_size)

    ax.legend()
    plt.tight_layout()
    plt.show()

#################### #################### #################### ####################


def create_ev_replay_files(num_bins=7*96):
    config_file = "../EVsSimulator/example_config_files/config.yaml"
    env = EVsSimulator(config_file=config_file,
                       empty_ports_at_end_of_simulation=False,
                       )
    evs = env.EVs_profiles
    name = env.scenario
    print(len(evs))

    # gather the arrival and departure times of the EVs

    data = pd.DataFrame(columns=['id', 'arrival', 'departure',
                        'required_energy', 'arrival_hour', 'time_of_stay', 'arrival_soc'])

    sim_date = env.sim_date
    starting_hour = 5
    print(sim_date)

    # use tqdm to show the progress of the loop
    for i, ev in enumerate(tqdm.tqdm(evs)):    
        required_energy = ev.battery_capacity - ev.battery_capacity_at_arrival
        # turn time into minutes using the timescale and the sim_date
        arrival = sim_date + \
            datetime.timedelta(minutes=env.timescale) * ev.time_of_arrival
        hour = arrival.hour
        day = arrival.weekday()
        if hour < starting_hour and day == 0:
            arrival = str(7) + arrival.strftime('%H:%M')
        else:
            arrival = str(arrival.weekday()) + arrival.strftime('%H:%M')

        departure = sim_date + \
            datetime.timedelta(minutes=env.timescale) * ev.time_of_departure
        hour = departure.hour
        day = departure.weekday()

        if hour < starting_hour and day == 0:
            departure = str(7) + departure.strftime('%H:%M')
        else:
            departure = str(departure.weekday()) + departure.strftime('%H:%M')

        arrival_hour = sim_date + \
            datetime.timedelta(minutes=env.timescale) * ev.time_of_arrival
        arrival_hour = arrival_hour.hour

        time_of_stay = ev.time_of_departure - ev.time_of_arrival
        time_of_stay = time_of_stay * env.timescale / 60

        arrival_soc = ev.battery_capacity_at_arrival / ev.battery_capacity

        new_row = pd.DataFrame({'id': [i], 'arrival': [arrival],
                                'departure': [departure],
                                'required_energy': [required_energy],
                                'arrival_hour': [arrival_hour],
                                'time_of_stay': [time_of_stay],
                                'arrival_soc': [arrival_soc]})

        data = data._append(new_row, ignore_index=True)

    for i in range(num_bins):
        arrival = sim_date + datetime.timedelta(minutes=env.timescale) * i
        hour = arrival.hour
        day = arrival.weekday()

        if hour < starting_hour and day == 0:
            arrival = str(7) + arrival.strftime('%H:%M')
        else:
            arrival = str(arrival.weekday()) + arrival.strftime('%H:%M')

        departure = sim_date + datetime.timedelta(minutes=env.timescale) * i
        hour = departure.hour
        day = departure.weekday()

        if hour < starting_hour and day == 0:
            departure = str(7) + departure.strftime('%H:%M')
        else:
            departure = str(departure.weekday()) + departure.strftime('%H:%M')

        new_row = pd.DataFrame({'id': [i],'arrival': [arrival],
                                'departure': [departure],
                                'required_energy': [0],
                                'arrival_hour': [hour],
                                'time_of_stay': [0],
                                'arrival_soc': [0.8]})
        
        data = data._append(new_row, ignore_index=True)
    
    data.reset_index(drop=True, inplace=True)    
    print(f' data lenght: {len(data)}')
    # save the data to a csv file
    data.to_csv(f'./results/evs_data_{name}.csv', index=False)


if __name__ == "__main__":
    # parse_files()
    # create_ev_replay_files(num_bins=7*96)
    pass
