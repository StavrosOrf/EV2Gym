'''
This script is used to load the EVs of the simulation and plot the behavior of the EVs.
'''
import os
import sys
sys.path.append(os.path.realpath('../'))

import datetime
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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

    names = ['Public', 'Work','Residential']

    num_bins = 7*96
    font_size = 25

    plt.rcParams.update({'font.size': font_size})
    plt.rcParams['font.family'] = ['serif']

    #################### #################### #################### ####################

    plt.figure(figsize=(12, 4))
    # sort data by arrival hour
    df_private = df_private.sort_values(by='arrival_hour')
    df_public = df_public.sort_values(by='arrival_hour')
    df_workplace = df_workplace.sort_values(by='arrival_hour')

    # plot 2d cmap of the arrival soc vs the arrival hour where the color is the probability

    # make a 2d histogram of the arrival soc vs the arrival hour
    # 2d histogram

    arrival_hour = np.arange(0, 24, 1)
    soc = np.arange(0, 1.001, 0.025)

    cmap = plt.cm.viridis
    
    prob = np.zeros((len(soc)-1, len(arrival_hour), 3))

    for i, df in enumerate([df_public, df_workplace, df_private]):
        # subplot
        ax = plt.subplot(1, 3, i+1)
        
        for j in range(len(arrival_hour)):
            hist, bins = np.histogram(df[df.arrival_hour == j]['arrival_soc'],
                                      bins=soc,
                                      density=True)
            prob[:,j, i] = hist/np.sum(hist)


        # ax.pcolormesh(arrival_hour,
        #               soc,
        #               np.histogram2d(df['arrival_hour'],
        #                              df['arrival_soc'],
        #                              density=True,
        #                              bins=(len(arrival_hour),
        #                                    len(soc)))[0].T / len(df),
        #               norm='linear',                         
        #               snap=True,                   
        #               cmap=cmap,
        #               shading='auto')
        
        plt.imshow(prob[:,:,i], cmap=cmap,
                  aspect='auto',
                  origin='lower',
                  interpolation='nearest',
                  norm=mcolors.PowerNorm(0.6),
                  extent=[0, 24, 0, 1])

        if i == 0:
            ax.set_ylabel('Arrival SOC', fontsize=font_size)
            ax.set_yticks([i for i in np.arange(0, 1.1, 0.2)],
                          [f'{i:.1f}' for i in np.arange(0, 1.1, 0.2)],
                          fontsize=font_size-4)
        else:
            ax.set_yticks([i for i in np.arange(0, 1.1, 0.2)],
                          ["" for i in np.arange(0, 1.1, 0.2)])

            ax.set_ylabel('')

        ax.set_xticks([i for i in range(0, 24, 4)],
                      [str(i) for i in range(0, 24, 4)],
                      fontsize=font_size-4)
        
        if i == 1:
            ax.set_xlabel(f'Arrival Time (h)', fontsize=font_size)
        # ax.set_title(f'{["Public", "Workplace", "Private"][i]}')

        ax.set_xlim(0, 23)
        ax.set_ylim(0, 1)
        plt.title(f'{names[i]}',fontsize=font_size)
        if i == 2:
            # add colorbar on the right side of the plot
            # left, bottom, width, height
            axins = plt.axes([0.88, 0.2, 0.02, 0.7])
            # cbar = plt.colorbar(ax.pcolormesh(arrival_hour,
            #                                   soc,
            #                                 #   np.histogram2d(df['arrival_hour'],
            #                                 #                  df['arrival_soc'],
            #                                 #                  bins=(len(arrival_hour),
            #                                 #                        len(soc)))[0].T / len(df),                                                
            #                                   cmap=cmap,
            #                                   norm='linear',
            #                                 #   shading='auto'
            #                                   ),
            
            #                     cax=axins,
            #                     label='Probability',
            #                     orientation='vertical',
            #                     location='right')  
            
            cbar = plt.colorbar( plt.imshow(prob[:,:,i], cmap=cmap,
                  aspect='auto',
                  origin='lower',
                  interpolation='nearest',
                  norm=mcolors.PowerNorm(0.6),
                  extent=[0, 24, 0, 1]),
                     cax=axins,
                                label='Probability',
                                orientation='vertical',
                                location='right'          )                      

            # roate the colorbar ticks
            cbar.ax.tick_params(labelsize=font_size-8)
            # cbar.ax.set_yticklabels([f'{i:.3f}' for i in np.arange(0, 0.002, 0.0001)])
            cbar.ax.yaxis.label.set_size(font_size)
            cbar.ax.yaxis.set_tick_params(rotation=35)

    plt.show()


def plot_time_of_stay_vs_hour():

    df_private = pd.read_csv('./results/evs_data_private.csv')
    df_public = pd.read_csv('./results/evs_data_public.csv')
    df_workplace = pd.read_csv('./results/evs_data_workplace.csv')

    names = ['Public', 'Work','Residential']
    print(df_private.head())
    num_bins = 7*96
    font_size = 25
    
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams['font.family'] = ['serif']

    # plot the time of stay vs the arrival hour
    
    plt.figure(figsize=(12, 4))
    # sort data by arrival hour
    df_private = df_private.sort_values(by='arrival_hour')
    df_public = df_public.sort_values(by='arrival_hour')
    df_workplace = df_workplace.sort_values(by='arrival_hour')

    cmap = plt.cm.viridis
    # cmap = plt.cm.cividis
    # cmap = plt.cm.gist_yarg
    # cmap = plt.cm.PuBuGn
    # cmap = plt.cm.Greys
    arrival_hour = np.arange(0, 24, 1)
    y_axis = np.arange(0, 20, 0.25)
    
    prob = np.zeros((len(y_axis)-1, len(arrival_hour), 3))

    for i, df in enumerate([df_public, df_workplace, df_private]):
        # subplot
        ax = plt.subplot(1, 3, i+1)
        
        for j in range(len(arrival_hour)):
            hist, bins = np.histogram(df[df.arrival_hour == j]['time_of_stay'],
                                      bins=y_axis,
                                      density=False)
            prob[:,j, i] = hist/np.sum(hist)            
            
        ax = plt.subplot(1, 3, i+1)
        # ax.pcolormesh(np.arange(0, 24, 1),
        #              y_axis,
        #               np.histogram2d(df['arrival_hour'],
        #                              df['time_of_stay'],
        #                              bins=(24, len(y_axis)))[0].T / len(df),
        #               norm='linear',
        #               cmap=cmap,
        #               shading='auto')
        
        plt.imshow(prob[:,:,i], cmap=cmap,
                  aspect='auto',
                  origin='lower',
                  interpolation='nearest',
                #   norm='linear',
                   norm=mcolors.PowerNorm(0.6),
                  extent=[0, 24, 0, 24])
        if i == 0:
            ax.set_ylabel('Time of stay (h)', fontsize=font_size)
            ax.set_yticks([i for i in range(0, 25, 4)],
                          [f'{i}' for i in range(0, 25, 4)],
                          fontsize=font_size-4)
        else:
            ax.set_yticks([i for i in range(0, 25, 4)],
                          ["" for i in range(0, 25, 4)])
            ax.set_ylabel('')

        ax.set_xticks([i for i in range(0, 24, 4)],
                      [str(i) for i in range(0, 24, 4)],
                      fontsize=font_size-4)
        
        if i == 1:
            ax.set_xlabel(f'Arrival Time (h)', fontsize=font_size)
            
        # ax.set_title(f'{["Public", "Workplace", "Private"][i]}')

        # ax.set_xlim(0, 23)
        # ax.set_ylim(0, 24)
        plt.title(f'{names[i]}',fontsize=font_size)

        if i == 2:
            # add colorbar on the right side of the plot
            axins = plt.axes([0.88, 0.2, 0.02, 0.7]) # left, bottom, width, height
            # cbar = plt.colorbar(ax.pcolormesh(np.arange(0, 24, 1),
            #                                   y_axis,
            #                                   np.histogram2d(df['arrival_hour'],
            #                                                  df['time_of_stay'],
            #                                                  bins=(24, len(y_axis)))[0].T / len(df),
            #                                   cmap=cmap,
            #                                   norm='linear',
            #                                   shading='auto'),
            #                     cax=axins,
                                
            #                     label='Probability',
            #                     orientation='vertical',
            #                     location='right')
            cbar = plt.colorbar( plt.imshow(prob[:,:,i], cmap=cmap,
                  aspect='auto',
                  origin='lower',
                  interpolation='nearest',
                #   norm='linear',
                norm=mcolors.PowerNorm(0.6),
                  extent=[0, 24, 0, 24]),
                     cax=axins,
                                label='Probability',
                                orientation='vertical',
                                location='right'          )   
            # # roate the colorbar ticks
            cbar.ax.tick_params(labelsize=font_size-8)
            cbar.ax.yaxis.label.set_size(font_size)
            cbar.ax.yaxis.set_tick_params(rotation=35)

    plt.show()

def plot_arrival_and_departure_time(num_bins=7*96, timescale=15):
    df_private = pd.read_csv('./results/evs_data_private.csv')
    df_public = pd.read_csv('./results/evs_data_public.csv')
    df_workplace = pd.read_csv('./results/evs_data_workplace.csv')

    names = ['Public', 'Work','Residential']
    print(df_private.head())
    num_bins = 7*96
    font_size = 25
    
    # plt.grid(True, which='minor', axis='both')
    plt.rcParams.update({'font.size': font_size})
    plt.rcParams['font.family'] = ['serif']

    data = df_private
    # fig, ax = plt.subplots()
    plt.figure(figsize=(10, 5))
    #################### #################### #################### ####################
    # plot histogram of the normalized arrival time and departure time in the same plot
    # sort data by arrival time
    for i, data in enumerate([df_public, df_workplace, df_private]):
        ax = plt.subplot(3,1, i+1)
        
        data = data.sort_values(by='arrival')

        bin = data['arrival'].hist(bins=num_bins,
                                   density=True,
                                   label=f'Arrival Time:',
                                   ax=ax,                                   
                                   color='#00429d',
                                    # alpha=0.7,
                                   histtype='barstacked'
                                   )

        data = data.sort_values(by='departure')
        # bin = data['departure'].hist(bins=num_bins,density=True,label='Departure Time',ax=ax)
        # plot only a line for the departure time
        bin = data['departure'].hist(bins=num_bins,
                                     density=True,
                                     ax=ax,
                                     label=f'Departure Time',
                                     color='#93003a',
                                     histtype='barstacked',
                                    alpha=0.9
                                     )
        plt.title(f'{names[i]}',fontsize=font_size)
        
        plt.xlim(0, num_bins)
        
        # xtixks mention the full name of the day

        ticks = []
        counter = 2
        for j in range(0, 7*96, 32):
            arrival = datetime.datetime(
                2024, 2, 12, 5, 0) + datetime.timedelta(minutes=timescale) * j
            if counter == 3:
                ticks.append(arrival.strftime('%a %H:%M'))
                counter = 0
            else:
                ticks.append(arrival.strftime('%a %H:%M'))
            counter += 1

        if i == 2:
            ax.set_xticks([k for k in range(0, 7*96, 32)],
                        ticks, rotation=35,
                        fontsize=font_size-8)
        else:
            ax.set_xticks([k for k in range(0, 7*96, 32)],
                          ["" for k in range(0, 7*96, 32)],)
        
        # ax.set_yticks(rotation=45)
        ax.yaxis.set_tick_params(rotation=45)
        
        # set font size of the ticks
        ax.tick_params(axis='both', which='major', labelsize=font_size-8)
        
        # ax.set_ylim(0.001, 0.02)

        # plt.xticks(rotation=45)
        ax.set_ylabel('Probability', fontsize=font_size)
        
        # ax.set_xlabel('Time', fontsize=font_size)
        if i == 1:
            ax.legend()
        else:
            ax.legend().remove()
        
    # plt.tight_layout()
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

        new_row = pd.DataFrame({'id': [i], 'arrival': [arrival],
                                'departure': [departure],
                                'required_energy': [0],
                                'arrival_hour': [hour],
                                'time_of_stay': [0],
                                # 'time_of_stay': [data[data.arrival_hour == hour]['time_of_stay'].mean()+0.01
                                #                  if data[data.arrival_hour == hour]['time_of_stay'].mean() is not None else 0.1],
                                'arrival_soc': [0]})

        data = data._append(new_row, ignore_index=True)

    data.reset_index(drop=True, inplace=True)
    print(f' data lenght: {len(data)}')
    # save the data to a csv file
    data.to_csv(f'./results/evs_data_{name}.csv', index=False)


if __name__ == "__main__":

    # create_ev_replay_files()
    # plot_time_of_stay_vs_hour()
    # plot_arrival_and_departure_time()
    plot_soc_vs_hour()
