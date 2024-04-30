'''
This file is used to plot the comparatigve results of the different algorithms.
'''

import pickle
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import datetime
import os
from EVsSimulator.ev_city import EVsSimulator


marker_list = ['.', 'x', 'o', 'v', 's', 'p',
               'P', '*', 'h', 'H', '+', 'X', 'D', 'd', '|', '_']

# color_list = ['#00429d', '#5681b9', '#93c4d2', '#ffa59e', '#dd4c65', '#93003a']

color_list = ['#00429d', '#5681b9', '#93c4d2', '#ffa59e',
              '#dd4c65', '#93003a', 'b', 'g', 'r', 'c', 'm', 'y', 'k']

algorithm_names = [
    'Charge As Fast As Possible',
    'DDPG_STER',
    'DDPG',
    'PowerTrackingErrorrMin'
    # 'Charge As Late As Possible',
    # 'Round Robin',
    #'OCCF V2G',
    #'OCCF G2V',
    #'eMPC V2G',
    #'eMPC G2V',
]


def plot_total_power(results_path, save_path=None, algorithm_names=None):

    # Load the env pickle files
    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.figure(figsize=(12, 4))
    plt.rc('font', family='serif')
    light_blue = np.array([0.529, 0.808, 0.922, 1])
    gold = np.array([1, 0.843, 0, 1])

    for index, key in enumerate(replay.keys()):        
        env = replay[key]

        date_range = pd.date_range(start=env.sim_starting_date,
                                   end=env.sim_starting_date +
                                   (env.simulation_length - 1) *
                                   datetime.timedelta(
                                       minutes=env.timescale),
                                   freq=f'{env.timescale}min')
        date_range_print = pd.date_range(start=env.sim_starting_date,
                                         end=env.sim_date,
                                         periods=10)

        counter = 1
        dim_x = int(np.ceil(np.sqrt(env.number_of_transformers)))
        dim_y = int(np.ceil(env.number_of_transformers/dim_x))
        for tr in env.transformers:

            plt.subplot(dim_x, dim_y, counter)
            df = pd.DataFrame([], index=date_range)

            if env.config['inflexible_loads']['include']:
                df['inflexible'] = env.tr_inflexible_loads[tr.id, :]
            if env.config['solar_power']['include']:
                df['solar'] = env.tr_solar_power[tr.id, :]

            for cs in tr.cs_ids:
                df[cs] = env.cs_power[cs, :]

            if index == 0:
                # plot the inflexible loads as a fill between
                if env.config['inflexible_loads']['include']:
                    plt.fill_between(df.index,
                                     np.array([0]*len(df.index)),
                                     df['inflexible'],
                                     step='post',
                                     alpha=0.7,
                                     color=light_blue,
                                     linestyle='--',
                                     linewidth=2,
                                     label='Inflexible Loads')

                # plot the solar power as a fill between the inflexible loads and the solar power
                if env.config['solar_power']['include']:
                    plt.fill_between(df.index,
                                     df['inflexible'],
                                     df['solar'] + df['inflexible'],
                                     step='post',
                                     alpha=0.7,
                                     color=gold,
                                     linestyle='--',
                                     linewidth=2,
                                     label='Solar Power')

                if env.config['demand_response']['include']:
                    plt.fill_between(df.index,
                                     np.array([tr.max_power.max()]
                                              * len(df.index)),
                                     tr.max_power,
                                     step='post',
                                     alpha=0.7,
                                     color='r',
                                     hatch='xx',
                                     linestyle='--',
                                     linewidth=2,
                                     label='Demand Response Event')

                plt.step(df.index,
                         #  tr.max_power
                         [tr.max_power.max()] * len(df.index),
                         where='post',
                         color='r',
                         linestyle='--',
                         linewidth=2,
                         label='Transformer Max Power')
                plt.plot([env.sim_starting_date, env.sim_date],
                         [0, 0], 'black')

            df['total'] = df.sum(axis=1)

            # plot total and use different color and linestyle for each algorithm
            plt.step(df.index, df['total'],
                     color=color_list[index],
                     where='post',
                     linestyle='-',
                     linewidth=1,
                     marker=marker_list[index],
                     label=algorithm_names[index])

            counter += 1

    plt.title(f'Transformer Limit', fontsize=20)
    plt.xlabel(f'Time', fontsize=14)
    plt.ylabel(f'Power (kW)', fontsize=14)
    plt.xlim([env.sim_starting_date, env.sim_date])
    plt.xticks(ticks=date_range_print,
               labels=[
                   f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print],
               rotation=45,
               fontsize=12)
    plt.yticks(fontsize=12)
    # put legend under the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               fancybox=True, shadow=True, ncol=3, fontsize=12)

    plt.grid(True, which='minor', axis='both')
    plt.tight_layout()

    fig_name = f'{save_path}/Transformer_Aggregated_Power.svg'
    plt.savefig(fig_name, format='svg',
                dpi=60, bbox_inches='tight')


def plot_total_power_V2G(results_path, save_path=None, algorithm_names=None):

    # Load the env pickle files
    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.close('all')
    fig, ax = plt.subplots()
    plt.grid(True, which='major', axis='both')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    ax.spines['left'].set_linewidth(2)
    # ax.spines['bottom'].set_linewidth(2)           
    
    plt.rc('font', family='serif')
   
    light_blue = np.array([0.529, 0.808, 0.922, 1])
    gold = np.array([1, 0.843, 0, 1])
    
    color_list_map = plt.cm.get_cmap('Set1', len(replay.keys()))
    color_list = color_list_map(np.linspace(0, 1, len(replay.keys())))

    for index, key in enumerate(replay.keys()):        
        env = replay[key]

        date_range = pd.date_range(start=env.sim_starting_date,
                                   end=env.sim_starting_date +
                                   (env.simulation_length - 1) *
                                   datetime.timedelta(
                                       minutes=env.timescale),
                                   freq=f'{env.timescale}min')
        date_range_print = pd.date_range(start=env.sim_starting_date,
                                         end=env.sim_date,
                                         periods=7)

        counter = 1
        dim_x = int(np.ceil(np.sqrt(env.number_of_transformers)))
        dim_y = int(np.ceil(env.number_of_transformers/dim_x))
        for tr in env.transformers:

            plt.subplot(dim_x, dim_y, counter)
            df = pd.DataFrame([], index=date_range)

            if env.config['inflexible_loads']['include']:
                df['inflexible'] = env.tr_inflexible_loads[tr.id, :]
            if env.config['solar_power']['include']:
                df['solar'] = env.tr_solar_power[tr.id, :]

            for cs in tr.cs_ids:
                df[cs] = env.cs_power[cs, :]

            if index == 0:
                # plot the inflexible loads as a fill between
                if env.config['inflexible_loads']['include']:
                    plt.fill_between(df.index,
                                     np.array([0]*len(df.index)),
                                     df['inflexible'],
                                     step='post',
                                     alpha=0.3,
                                     color=light_blue,
                                     linestyle='--',
                                     linewidth=2,
                                     label='Inflexible Loads')

                # plot the solar power as a fill between the inflexible loads and the solar power
                if env.config['solar_power']['include']:
                    plt.fill_between(df.index,
                                     df['inflexible'],
                                     df['solar'] + df['inflexible'],
                                     step='post',
                                     alpha=0.8,
                                     color=gold,
                                     linestyle='--',
                                     linewidth=2,
                                     label='Solar Power')

                if env.config['demand_response']['include']:
                    plt.fill_between(df.index,
                                     np.array([tr.max_power.max()]
                                              * len(df.index)),
                                     tr.max_power,
                                     step='post',
                                     alpha=0.7,
                                     color='r',
                                     hatch='xx',
                                     linestyle='--',
                                     linewidth=2,
                                     label='Demand Response Event')

                plt.step(df.index,
                         #  tr.max_power
                         [-tr.max_power.max()] * len(df.index),
                         where='post',
                         color='r',
                         linestyle='--',
                         linewidth=2,
                         alpha=0.7,
                        #  label='Transf. Limit'
                         )
                
                plt.step(df.index,
                         #  tr.max_power
                         [tr.max_power.max()] * len(df.index),
                         where='post',
                         color='r',
                         linestyle='--',
                         linewidth=2,
                         alpha=0.7,
                         label='Transf. Limit')
                plt.plot([env.sim_starting_date, env.sim_date],
                         [0, 0], 'black')

            df['total'] = df.sum(axis=1)

            # plot total and use different color and linestyle for each algorithm
            plt.step(df.index, df['total'],
                     color=color_list[index],
                     where='post',
                     linestyle='-',
                     linewidth=1,
                     marker=marker_list[index],
                     label=algorithm_names[index])

            counter += 1
    
    plt.ylabel(f'Power (kW)', fontsize=28)
    plt.xlim([env.sim_starting_date, env.sim_date])
    plt.xticks(ticks=date_range_print,
               labels=[
                   f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print],
            #    rotation=45,
               fontsize=28)
    plt.yticks(fontsize=28)
    # put legend under the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
               fancybox=True, shadow=True, ncol=3, fontsize=24)    

    fig_name = f'{save_path}/Transformer_Aggregated_Power_Prices.png'
    
    plt.savefig(fig_name, format='png',
                dpi=60, bbox_inches='tight')

    #plt.show()

def plot_comparable_EV_SoC(results_path, save_path=None, algorithm_names=None):
    '''
    This function is used to plot the SoC of the EVs in the same plot
    '''

    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.close('all')
    plt.figure(figsize=(10, 7))
    plt.rc('font', family='serif')

    for index, key in enumerate(replay.keys()):        
        env = replay[key]

        date_range = pd.date_range(start=env.sim_starting_date,
                                   end=env.sim_starting_date +
                                   (env.simulation_length - 1) *
                                   datetime.timedelta(
                                       minutes=env.timescale),
                                   freq=f'{env.timescale}min')
        date_range_print = pd.date_range(start=env.sim_starting_date,
                                         end=env.sim_date,
                                         periods=10)

        counter = 1
        dim_x = int(np.ceil(np.sqrt(env.cs)))
        dim_y = int(np.ceil(env.cs/dim_x))
        for cs in env.charging_stations:            
            
            plt.subplot(dim_x, dim_y, counter)
            df = pd.DataFrame([], index=date_range)

            for port in range(cs.n_ports):
                df[port] = env.port_energy_level[port, cs.id, :]

            # Add another row with one datetime step to make the plot look better
            df.loc[df.index[-1] +
                   datetime.timedelta(minutes=env.timescale)] = df.iloc[-1]

            for port in range(cs.n_ports):
                for i, (t_arr, t_dep) in enumerate(env.port_arrival[f'{cs.id}.{port}']):
                    t_dep = t_dep + 1
                    if t_dep > len(df):
                        t_dep = len(df)
                    # x = df.index[t_arr:t_dep]
                    y = df[port].values.T[t_arr:t_dep]
                    # fill y with 0 before and after to match the length of df
                    y = np.concatenate(
                        [np.zeros(t_arr), y, np.zeros(len(df) - t_dep)])

                    plt.step(df.index,
                             y,
                             where='post',
                             color=color_list[index],
                             marker=marker_list[index],
                             label=algorithm_names[index])

            plt.title(f'Charging Station {cs.id + 1}', fontsize=24)
            plt.ylabel('SoC', fontsize=24)
            plt.ylim([0.1, 1])
            plt.xlim([env.sim_starting_date, env.sim_date])
            plt.xticks(ticks=date_range_print,
                       labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print], rotation=45,
                       fontsize=22)
            counter += 1

    plt.legend(loc='upper center', bbox_to_anchor=(1.1, -0.15),
               fancybox=True, shadow=True, ncol=5, fontsize=24)

    plt.grid(True, which='minor', axis='both')
    plt.tight_layout()

    fig_name = f'{save_path}/EV_Energy_Level.png'
    plt.savefig(fig_name, format='png',
                dpi=60, bbox_inches='tight')
    

def plot_comparable_EV_SoC_single(results_path, save_path=None, algorithm_names=None):
    '''
    This function is used to plot the SoC of the EVs in the same plot
    '''

    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.close('all')
    fig, ax = plt.subplots()
    plt.rc('font', family='serif')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    plt.grid(True, which='major', axis='both')
    for index, key in enumerate(replay.keys()):        
        env = replay[key]

        date_range = pd.date_range(start=env.sim_starting_date,
                                   end=env.sim_starting_date +
                                   (env.simulation_length - 1) *
                                   datetime.timedelta(
                                       minutes=env.timescale),
                                   freq=f'{env.timescale}min')
        date_range_print = pd.date_range(start=env.sim_starting_date,
                                         end=env.sim_date,
                                         periods=7)

        color_list_map = plt.cm.get_cmap('Set1', len(replay.keys()))
        color_list = color_list_map(np.linspace(0, 1, len(replay.keys())))
        
        counter = 1
        for cs in env.charging_stations:   
            if counter != 1:
                counter += 1
                continue
            
            # plt.subplot(1, 2, counter)
            df = pd.DataFrame([], index=date_range)

            for port in range(cs.n_ports):
                df[port] = env.port_energy_level[port, cs.id, :]

            # Add another row with one datetime step to make the plot look better
            df.loc[df.index[-1] +
                   datetime.timedelta(minutes=env.timescale)] = df.iloc[-1]

            for port in range(cs.n_ports):
                for i, (t_arr, t_dep) in enumerate(env.port_arrival[f'{cs.id}.{port}']):
                    t_dep = t_dep + 1
                    if t_dep > len(df):
                        t_dep = len(df)
                    # x = df.index[t_arr:t_dep]
                    y = df[port].values.T[t_arr:t_dep]
                    # fill y with 0 before and after to match the length of df
                    y = np.concatenate(
                        [np.zeros(t_arr), y, np.zeros(len(df) - t_dep)])

                    plt.step(df.index,
                             y,
                             where='post',
                             color=color_list[index],
                             marker=marker_list[index],
                             alpha=0.8,
                             label=algorithm_names[index])

            # plt.title(f'Charging Station {cs.id + 1}', fontsize=24)
            
            if counter == 1:
                plt.ylabel('SoC', fontsize=28)
                plt.yticks(np.arange(0, 1.1, 0.2),
                           fontsize=28)
                    
            else:
                plt.yticks(fontsize=28)
                plt.yticks(np.arange(0, 1.1, 0.1),
                            labels=[' ' for d in np.arange(0, 1.1, 0.1)])            
            
            plt.ylim([0.1, 1.09])
            plt.xlim([env.sim_starting_date, env.sim_date])
            plt.xticks(ticks=date_range_print,
                       labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print],
                    #    rotation=45,
                       fontsize=28)
            counter += 1

    # plt.legend(loc='upper center', bbox_to_anchor=(0, -0.15),
    #            fancybox=True, shadow=True, ncol=, fontsize=24)

    plt.tight_layout()

    fig_name = f'{save_path}/EV_Energy_Level_single.png'
    plt.savefig(fig_name, format='png',
                dpi=60, bbox_inches='tight')

    #plt.show() 
    

def plot_comparable_CS_Power(results_path, save_path=None, algorithm_names=None):
    '''
    This function is used to plot the SoC of the EVs in the same plot
    '''

    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.close('all')
    fig, ax = plt.subplots()
    plt.rc('font', family='serif')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    plt.grid(True, which='major', axis='both')
    for index, key in enumerate(replay.keys()):        
        env = replay[key]

        date_range = pd.date_range(start=env.sim_starting_date,
                                   end=env.sim_starting_date +
                                   (env.simulation_length - 1) *
                                   datetime.timedelta(
                                       minutes=env.timescale),
                                   freq=f'{env.timescale}min')
        date_range_print = pd.date_range(start=env.sim_starting_date,
                                         end=env.sim_date,
                                         periods=7)

        color_list_map = plt.cm.get_cmap('Set1', len(replay.keys()))
        color_list = color_list_map(np.linspace(0, 1, len(replay.keys())))
        
        counter = 1
        for cs in env.charging_stations:   
            if counter != 1:
                counter += 1
                continue
            
            # plt.subplot(1, 2, counter)
            df = pd.DataFrame([], index=date_range)

            for port in range(cs.n_ports):
                df[port] = env.port_current[port, cs.id, :]
            
            #multiply df[port] by the voltage to get the power
            df = df * cs.voltage * math.sqrt(cs.phases) / 1000
            
            # Add another row with one datetime step to make the plot look better
            df.loc[df.index[-1] +
                   datetime.timedelta(minutes=env.timescale)] = df.iloc[-1]

            for port in range(cs.n_ports):
                for i, (t_arr, t_dep) in enumerate(env.port_arrival[f'{cs.id}.{port}']):
                    t_dep = t_dep + 1
                    if t_dep > len(df):
                        t_dep = len(df)
                    # x = df.index[t_arr:t_dep]
                    y = df[port].values.T[t_arr:t_dep]
                    # fill y with 0 before and after to match the length of df
                    y = np.concatenate(
                        [np.zeros(t_arr), y, np.zeros(len(df) - t_dep)])

                    plt.step(df.index,
                             y,
                             where='post',
                             color=color_list[index],
                             marker=marker_list[index],
                             alpha=0.8,
                             label=algorithm_names[index])

            # plt.title(f'Charging Station {cs.id + 1}', fontsize=24)
            
            if counter == 1:
                plt.ylabel('Power (kW)', fontsize=28)
                plt.yticks([-22,-11,0,11,22],
                           fontsize=28)
                    
            else:
                plt.yticks(fontsize=28)
                # plt.yticks(np.arange(0, 1.1, 0.1),
                #             labels=[' ' for d in np.arange(0, 1.1, 0.1)])            
            
            # plt.ylim([0.1, 1.09])
            plt.xlim([env.sim_starting_date, env.sim_date])
            plt.xticks(ticks=date_range_print,
                       labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print],
                    #    rotation=45,
                       fontsize=28)
            counter += 1

    # plt.legend(loc='upper center', bbox_to_anchor=(0, -0.15),
    #            fancybox=True, shadow=True, ncol=, fontsize=24)

    plt.tight_layout()

    fig_name = f'{save_path}/CS_Power_single.png'
    plt.savefig(fig_name, format='png',
                dpi=60, bbox_inches='tight')

    #plt.show()   

'''
def find_min_error_run(replay, algorithm_name='DDPG'):
    min_error = float('inf')
    min_error_run = None
    for key, env in replay.items():
        if algorithm_name in key:
            error = sum(abs(env.power_setpoints - env.current_power_usage))
            if error < min_error:
                min_error = error
                min_error_run = key
    return min_error_run.split(',')[1].strip()  # return the run number
'''

'''
def find_min_difference_replay(results_path, save_path=None, algorithm_name="DDPG"):
    results_path = os.path.join(save_path + "plot_results_dict.pkl")

    min_difference = float('inf')
    min_difference_replay_file = None

    # Find the replay with the minimum abs tracking error
    with open(results_path, 'rb') as f:
        replay = pickle.load(f)
        if algorithm_name in replay.keys():
            env = replay[algorithm_name]
            abs_tracking_error = [np.sum(np.abs(env.current_power_usage[i] - env.power_setpoints[i])) for i in range(env.simulation_length)]
            current_difference = min(abs_tracking_error)
            if current_difference < min_difference:
                min_difference = current_difference
                min_difference_replay_file = results_path

    return min_difference_replay_file
'''


def plot_actual_power_vs_setpoint(results_path, save_path=None, algorithm_names=None):
    
    
    #This function is used to plot the actual power vs the setpoint power.
    #It plots the behavior of each algorithm in subplots vertically.
    
    
    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.close('all')
    plt.figure(figsize=(17, 11))
    plt.rc('font', family='serif')    

    for index, key in enumerate(replay.keys()):        
        env = replay[key]

        date_range = pd.date_range(start=env.sim_starting_date,
                                   end=env.sim_starting_date +
                                   (env.simulation_length - 1) *
                                   datetime.timedelta(
                                       minutes=env.timescale),
                                   freq=f'{env.timescale}min')
        date_range_print = pd.date_range(start=env.sim_starting_date,
                                         end=env.sim_date,
                                         periods=7)

        #plot the actual power vs the setpoint power for each algorithm in subplots                
        plt.subplot(len(replay), 1, index+1)
        plt.grid(True, which='major', axis='both')
        
        actual_power = env.current_power_usage        
        setpoints = env.power_setpoints                

        plt.step(date_range, actual_power.T, alpha=0.9, color='#00429d')
        plt.step(date_range, setpoints.T, alpha=1, color='#93003a')
        
        # Define the mapping from algorithm names to special names
        special_names = {
        "DDPG": "DDPG",
        "ChargeAsLateAsPossible": "ChargeAsLateAsPossible",
        "PowerTrackingErrorrMin": "Optimal",
        "ChargeAsFastAsPossible": "CAFAP",
        "RoundRobin": "RoundRobin",
    # Add more mappings as needed
            }

# Use the special name in the title
        plt.title(f'{special_names.get(algorithm_names[index], algorithm_names[index])}', fontsize=22)

        plt.axhline(0, color='black', lw=2)
        #plt.title(f'{algorithm_names[index]}', fontsize=22)
        
        plt.yticks(fontsize=22)
        
        if index == len(replay) - 1:
            plt.xticks(ticks=date_range_print,
                       labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print],
                    #    rotation=45,
                       fontsize=22)
            # plt.xlabel('Time', fontsize=28)
        else:
            plt.xticks(ticks=date_range_print,
                       labels=[' ' for d in date_range_print])
        
        if index == len(replay) // 2:
            plt.ylabel('Power (kW)', fontsize=22)               

        plt.xlabel('Time', fontsize=22)    
        plt.xlim([env.sim_starting_date, env.sim_date])
        plt.ylim([0, 1.1*(max(env.current_power_usage.max(), env.power_setpoints.max()))])
        
    # Put the legend under the plot in a separate axis           
    plt.legend(['Actual Power', 'Setpoint'], loc='upper center',
               bbox_to_anchor=(0.5, -0.5),
               fancybox=True, shadow=True, ncol=2, fontsize=22)
        
    plt.tight_layout()
    fig_name = f'{save_path}/Actual_vs_Setpoint_Power.svg'
    plt.savefig(fig_name, format='svg',
                dpi=60, bbox_inches='tight')    


def plot_prices(results_path, save_path=None, algorithm_names=None):
    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.close('all')
    plt.figure(figsize=(7, 11))
    plt.rc('font', family='serif')
    
    keys = list(replay.keys())
    env = replay[keys[0]]
    
    date_range = pd.date_range(start=env.sim_starting_date,
                                 end=env.sim_starting_date +
                                 (env.simulation_length - 1) *
                                 datetime.timedelta(
                                      minutes=env.timescale),
                                 freq=f'{env.timescale}min')
    date_range_print = pd.date_range(start=env.sim_starting_date,
                                     end=env.sim_date,
                                     periods=7)
    
    plt.close('all')
    fig, ax = plt.subplots()
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    plt.grid(True, which='major', axis='both')    
    plt.rc('font', family='serif')
    
    charge_prices = env.charge_prices[0, :]
    discharge_prices = env.discharge_prices[0, :]
    
    plt.step(date_range, -charge_prices, alpha=0.9, color='#00429d',label='Charge Prices')
    plt.step(date_range, discharge_prices, alpha=1, color='#93003a', label='Discharge Prices')
    
    plt.xlim([env.sim_starting_date, env.sim_date])
    # plt.ylim()
    # plt.axhline(0, color='black', lw=2)    
    y_ticks = np.arange(0.150, 0.351, 0.05)
    plt.yticks(y_ticks,fontsize=28)

    plt.ylim([0.12, 0.35])
    plt.xticks(ticks=date_range_print,
               labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print],
            #    rotation=45,
               fontsize=28)
    plt.ylabel('Price (€/kWh)', fontsize=28)
    
    plt.legend(fontsize=28)
    
    #show grid lines
    
    
    # plt.tight_layout()
    fig_name = f'{save_path}/Prices.png'
    plt.savefig(fig_name, format='png',
                dpi=60, bbox_inches='tight')
    #plt.show()


def plot_energy_tracking_error(csv_file, save_path):
    # Read the csv file
    results = pd.read_csv(csv_file)

    # Get the unique algorithm names
    algorithm_names = results['Algorithm'].unique()

    special_names = {
        "DDPG": "DDPG",
        "ChargeAsLateAsPossible": "ChargeAsLateAsPossible",
        "PowerTrackingErrorrMin": "Optimal",
        "ChargeAsFastAsPossible": "CAFAP",
        "RoundRobin": "RoundRobin" }

    # Create the plot
    plt.figure(figsize=(12, 4))

    
    # Plot 'energy_tracking_error' for each algorithm
    for algorithm_name in algorithm_names:
        algorithm_results = results[results['Algorithm'] == algorithm_name]
        plt.plot(algorithm_results['run'], algorithm_results['energy_tracking_error'], marker='.' , label=special_names[algorithm_name])


    plt.xlabel('Replays', fontsize=14)
    plt.ylabel('Energy Tracking Error (kWh)', fontsize=14)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
               fancybox=True, shadow=True, ncol=3, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    # Save the plot as an SVG file
    plt.savefig(save_path + 'energy_tracking_error_plot.svg', format='svg', bbox_inches='tight')


def plot_energy_tracking_error_hist(csv_file, save_path):
    # Read the csv file
    results = pd.read_csv(csv_file)

    # Get the unique algorithm names
    algorithm_names = results['Algorithm'].unique()

    special_names = {
        "DDPG": "DDPG",
        "ChargeAsLateAsPossible": "ChargeAsLateAsPossible",
        "PowerTrackingErrorrMin": "Optimal",
        "ChargeAsFastAsPossible": "CAFAP",
        "RoundRobin": "RoundRobin" }

    # Create the plot
    plt.figure(figsize=(12, 4))

    # Plot 'energy_tracking_error' for each algorithm
    for algorithm_name in algorithm_names:
        algorithm_results = results[results['Algorithm'] == algorithm_name]
        plt.hist(algorithm_results['energy_tracking_error'], label=special_names[algorithm_name], alpha=0.5)

    plt.xlabel('Energy Tracking Error (kWh)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend(loc='upper right',
               fancybox=True, shadow=True, ncol=1, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    # Save the plot as an SVG file
    plt.savefig(save_path + 'energy_tracking_error_histogram.svg', format='svg', bbox_inches='tight')


def plot_energy_tracking_error_boxplot(csv_file, save_path):
    # Read the csv file
    results = pd.read_csv(csv_file)

    # Get the unique algorithm names
    algorithm_names = results['Algorithm'].unique()

    special_names = {
        "DDPG": "DDPG",
        "ChargeAsLateAsPossible": "ChargeAsLateAsPossible",
        "PowerTrackingErrorrMin": "Optimal",
        "ChargeAsFastAsPossible": "CAFAP",
        "RoundRobin": "RoundRobin" }

    # Define colors for each algorithm
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']    # Create the plot

    # Create the plot
    plt.figure(figsize=(12, 4))

    # Plot 'energy_tracking_error' for each algorithm
    data = []
    for algorithm_name in algorithm_names:
        algorithm_results = results[results['Algorithm'] == algorithm_name]
        data.append(algorithm_results['energy_tracking_error'])

    box_plot = plt.boxplot(data, labels=[special_names[name] for name in algorithm_names], patch_artist=True)

    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('Energy Tracking Error (kWh)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    # Save the plot as an SVG file
    plt.savefig(save_path + 'energy_tracking_error_boxplot.svg', format='svg', bbox_inches='tight')


def plot_energy_tracking_error_bar(csv_file, save_path):
    # Read the csv file
    results = pd.read_csv(csv_file)

    # Get the unique algorithm names
    algorithm_names = results['Algorithm'].unique()

    special_names = {
        "DDPG": "DDPG",
        "ChargeAsLateAsPossible": "ChargeAsLateAsPossible",
        "PowerTrackingErrorrMin": "Optimal",
        "ChargeAsFastAsPossible": "CAFAP",
        "RoundRobin": "RoundRobin" }
    
    # Define colors for each algorithm
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']    # Create the plot
    plt.figure(figsize=(12, 4))

    # Plot 'energy_tracking_error' for each algorithm
    bar_width = 0.3
    for i, algorithm_name in enumerate(algorithm_names):
        algorithm_results = results[results['Algorithm'] == algorithm_name]
        plt.bar(algorithm_results['run'] + i * bar_width, algorithm_results['energy_tracking_error'], width=bar_width, label=special_names[algorithm_name], color=colors[i])

    plt.xlabel('Replays', fontsize=14)
    plt.ylabel('Energy Tracking Error (kWh)', fontsize=14)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
               fancybox=True, shadow=True, ncol=3, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    # Save the plot as an SVG file
    plt.savefig(save_path + 'energy_tracking_error_bar_plot.svg', format='svg', bbox_inches='tight')

def plot_squared_power_tracking_error(csv_file, save_path):
    # Read the csv file
    results = pd.read_csv(csv_file)

    # Get the unique algorithm names
    algorithm_names = results['Algorithm'].unique()

    special_names = {
        "DDPG": "DDPG",
        "ChargeAsLateAsPossible": "ChargeAsLateAsPossible",
        "PowerTrackingErrorrMin": "Optimal",
        "ChargeAsFastAsPossible": "CAFAP",
        "RoundRobin": "RoundRobin" }

    #colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']    # Create the plot

    # Create the plot
    plt.figure(figsize=(12, 4))

    
    # Plot 'energy_tracking_error' for each algorithm
    for algorithm_name in algorithm_names:
        algorithm_results = results[results['Algorithm'] == algorithm_name]
        plt.plot(algorithm_results['run'], algorithm_results['tracking_error'], marker='.', label=special_names[algorithm_name])


    plt.xlabel('Replays', fontsize=14)
    plt.ylabel('Squared Tracking Error (kW$^2$)', fontsize=14)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
               fancybox=True, shadow=True, ncol=3, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)

    plt.tight_layout()

    # Save the plot as an SVG file
    plt.savefig(save_path + 'squared_tracking_error_plot.svg', format='svg', bbox_inches='tight')

def plot_user_satisfaction(csv_file, save_path):
    # Read the csv file
    results = pd.read_csv(csv_file)

    # Get the unique algorithm names
    algorithm_names = results['Algorithm'].unique()

    special_names = {
        "DDPG": "DDPG",
        "ChargeAsLateAsPossible": "ChargeAsLateAsPossible",
        "PowerTrackingErrorrMin": "Optimal",
        "ChargeAsFastAsPossible": "CAFAP",
        "RoundRobin": "RoundRobin" }

    # Create the plot
    plt.figure(figsize=(12, 4))

    min_y = results['average_user_satisfaction'].min() * 3 / 4
    
    # Plot 'energy_tracking_error' for each algorithm
    for algorithm_name in algorithm_names:
        algorithm_results = results[results['Algorithm'] == algorithm_name]
        plt.plot(algorithm_results['run'], algorithm_results['average_user_satisfaction'], marker='.' , label=special_names[algorithm_name])


    plt.xlabel('Replays', fontsize=14)
    plt.ylabel('User Satisfaction (%)', fontsize=14)
    plt.ylim(min_y, 1.1)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
               fancybox=True, shadow=True, ncol=3, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    # Save the plot as an SVG file
    plt.savefig(save_path + 'user_satisfaction.svg', format='svg', bbox_inches='tight')


def plot_tracker_surplus(csv_file, save_path):
    # Read the csv file
    results = pd.read_csv(csv_file)

    # Get the unique algorithm names
    algorithm_names = results['Algorithm'].unique()

    special_names = {
        "DDPG": "DDPG",
        "ChargeAsLateAsPossible": "ChargeAsLateAsPossible",
        "PowerTrackingErrorrMin": "Optimal",
        "ChargeAsFastAsPossible": "CAFAP",
        "RoundRobin": "RoundRobin" }

    # Create the plot
    plt.figure(figsize=(12, 4))

    
    # Plot 'energy_tracking_error' for each algorithm
    for algorithm_name in algorithm_names:
        algorithm_results = results[results['Algorithm'] == algorithm_name]
        plt.plot(algorithm_results['run'], algorithm_results['power_tracker_violation'], marker='.', label=special_names[algorithm_name])


    plt.xlabel('Replays', fontsize=14)
    plt.ylabel('Power Tracker Surplus (kW)', fontsize=14)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
               fancybox=True, shadow=True, ncol=3, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)

    plt.tight_layout()

    # Save the plot as an SVG file
    plt.savefig(save_path + 'tracker_surplus_plot.svg', format='svg', bbox_inches='tight')

def plot_transformer_overload(csv_file, save_path):
    # Read the csv file
    results = pd.read_csv(csv_file)

    # Get the unique algorithm names
    algorithm_names = results['Algorithm'].unique()

    special_names = {
        "DDPG": "DDPG",
        "ChargeAsLateAsPossible": "ChargeAsLateAsPossible",
        "PowerTrackingErrorrMin": "Optimal",
        "ChargeAsFastAsPossible": "CAFAP",
        "RoundRobin": "RoundRobin" }

    # Create the plot
    plt.figure(figsize=(12, 4))

    # Plot 'energy_tracking_error' for each algorithm
    bar_width = 1
    for i, algorithm_name in enumerate(algorithm_names):
        algorithm_results = results[results['Algorithm'] == algorithm_name]
        plt.bar(algorithm_results['run'] + i * bar_width, algorithm_results['total_transformer_overload'], width=bar_width, label=special_names[algorithm_name])

    plt.xlabel('Replays', fontsize=14)
    plt.ylabel('Transformer Overload (kW)', fontsize=14)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
               fancybox=True, shadow=True, ncol=3, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)

    plt.tight_layout()

    # Save the plot as an SVG file
    plt.savefig(save_path + 'transformer_overload.svg', format='svg', bbox_inches='tight')


def plot_squared_power_tracking_error_boxplot(csv_file, save_path):
    # Read the csv file
    results = pd.read_csv(csv_file)

    # Get the unique algorithm names
    algorithm_names = results['Algorithm'].unique()

    special_names = {
        "DDPG": "DDPG",
        "ChargeAsLateAsPossible": "ChargeAsLateAsPossible",
        "PowerTrackingErrorrMin": "Optimal",
        "ChargeAsFastAsPossible": "CAFAP",
        "RoundRobin": "RoundRobin" }

    # Define colors for each algorithm
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']    # Create the plot

    # Create the plot
    plt.figure(figsize=(12, 4))

    # Plot 'tracking_error' for each algorithm
    data = []
    for algorithm_name in algorithm_names:
        algorithm_results = results[results['Algorithm'] == algorithm_name]
        data.append(algorithm_results['tracking_error'])

    box_plot = plt.boxplot(data, labels=[special_names[name] for name in algorithm_names], patch_artist=True)

    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('Squared Tracking Error (kW$^2$)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    # Save the plot as an SVG file
    plt.savefig(save_path + 'squared_tracking_error_boxplot.svg', format='svg', bbox_inches='tight')

def plot_user_satisfaction_boxplot(csv_file, save_path):
    # Read the csv file
    results = pd.read_csv(csv_file)

    # Get the unique algorithm names
    algorithm_names = results['Algorithm'].unique()

    special_names = {
        "DDPG": "DDPG",
        "ChargeAsLateAsPossible": "ChargeAsLateAsPossible",
        "PowerTrackingErrorrMin": "Optimal",
        "ChargeAsFastAsPossible": "CAFAP",
        "RoundRobin": "RoundRobin" }

    # Define colors for each algorithm
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']    # Create the plot

    # Create the plot
    plt.figure(figsize=(12, 4))

    # Plot 'average_user_satisfaction' for each algorithm
    data = []
    for algorithm_name in algorithm_names:
        algorithm_results = results[results['Algorithm'] == algorithm_name]
        data.append(algorithm_results['average_user_satisfaction'])

    box_plot = plt.boxplot(data, labels=[special_names[name] for name in algorithm_names], patch_artist=True)

    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('User Satisfaction (%)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    # Save the plot as an SVG file
    plt.savefig(save_path + 'user_satisfaction_boxplot.svg', format='svg', bbox_inches='tight')

def plot_tracker_surplus_boxplot(csv_file, save_path):
    # Read the csv file
    results = pd.read_csv(csv_file)

    # Get the unique algorithm names
    algorithm_names = results['Algorithm'].unique()

    special_names = {
        "DDPG": "DDPG",
        "ChargeAsLateAsPossible": "ChargeAsLateAsPossible",
        "PowerTrackingErrorrMin": "Optimal",
        "ChargeAsFastAsPossible": "CAFAP",
        "RoundRobin": "RoundRobin" }

    # Define colors for each algorithm
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']    # Create the plot

    # Create the plot
    plt.figure(figsize=(12, 4))

    # Plot 'power_tracker_violation' for each algorithm
    data = []
    for algorithm_name in algorithm_names:
        algorithm_results = results[results['Algorithm'] == algorithm_name]
        data.append(algorithm_results['power_tracker_violation'])

    box_plot = plt.boxplot(data, labels=[special_names[name] for name in algorithm_names], patch_artist=True)

    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('Power Tracker Surplus (kW)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    # Save the plot as an SVG file
    plt.savefig(save_path + 'tracker_surplus_boxplot.svg', format='svg', bbox_inches='tight')

def plot_squared_power_tracking_error_bar(csv_file, save_path):
    # Read the csv file
    results = pd.read_csv(csv_file)

    # Get the unique algorithm names
    algorithm_names = results['Algorithm'].unique()

    special_names = {
        "DDPG": "DDPG",
        "ChargeAsLateAsPossible": "ChargeAsLateAsPossible",
        "PowerTrackingErrorrMin": "Optimal",
        "ChargeAsFastAsPossible": "CAFAP",
        "RoundRobin": "RoundRobin" }

    # Define colors for each algorithm
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Create the plot
    plt.figure(figsize=(12, 4))

    # Plot 'tracking_error' for each algorithm
    bar_width = 0.3
    for i, algorithm_name in enumerate(algorithm_names):
        algorithm_results = results[results['Algorithm'] == algorithm_name]
        plt.bar(algorithm_results['run'] + i * bar_width, algorithm_results['tracking_error'], width=bar_width, label=special_names[algorithm_name], color=colors[i])

    plt.xlabel('Replays', fontsize=14)
    plt.ylabel('Squared Tracking Error (kW$^2$)', fontsize=14)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
               fancybox=True, shadow=True, ncol=3, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)

    plt.tight_layout()

    # Save the plot as an SVG file
    plt.savefig(save_path + 'squared_tracking_error_plot_bar.svg', format='svg', bbox_inches='tight')

def plot_user_satisfaction_bar(csv_file, save_path):
    # Read the csv file
    results = pd.read_csv(csv_file)

    # Get the unique algorithm names
    algorithm_names = results['Algorithm'].unique()

    special_names = {
        "DDPG": "DDPG",
        "ChargeAsLateAsPossible": "ChargeAsLateAsPossible",
        "PowerTrackingErrorrMin": "Optimal",
        "ChargeAsFastAsPossible": "CAFAP",
        "RoundRobin": "RoundRobin" }

    # Define colors for each algorithm
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Create the plot
    plt.figure(figsize=(12, 4))

    min_y = results['average_user_satisfaction'].min() * 3 / 4

    # Plot 'average_user_satisfaction' for each algorithm
    bar_width = 0.3
    for i, algorithm_name in enumerate(algorithm_names):
        algorithm_results = results[results['Algorithm'] == algorithm_name]
        plt.bar(algorithm_results['run'] + i * bar_width, algorithm_results['average_user_satisfaction'], width=bar_width, label=special_names[algorithm_name], color=colors[i])

    plt.xlabel('Replays', fontsize=14)
    plt.ylabel('User Satisfaction (%)', fontsize=14)
    plt.ylim(min_y, 1.1)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
               fancybox=True, shadow=True, ncol=3, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    # Save the plot as an SVG file
    plt.savefig(save_path + 'user_satisfaction_bar.svg', format='svg', bbox_inches='tight')


def plot_tracker_surplus_bar(csv_file, save_path):
    # Read the csv file
    results = pd.read_csv(csv_file)

    # Get the unique algorithm names
    algorithm_names = results['Algorithm'].unique()

    special_names = {
        "DDPG": "DDPG",
        "ChargeAsLateAsPossible": "ChargeAsLateAsPossible",
        "PowerTrackingErrorrMin": "Optimal",
        "ChargeAsFastAsPossible": "CAFAP",
        "RoundRobin": "RoundRobin" }

    # Define colors for each algorithm
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Create the plot
    plt.figure(figsize=(12, 4))

    # Plot 'power_tracker_violation' for each algorithm
    bar_width = 0.3
    for i, algorithm_name in enumerate(algorithm_names):
        algorithm_results = results[results['Algorithm'] == algorithm_name]
        plt.bar(algorithm_results['run'] + i * bar_width, algorithm_results['power_tracker_violation'], width=bar_width, label=special_names[algorithm_name], color=colors[i])

    plt.xlabel('Replays', fontsize=14)
    plt.ylabel('Power Tracker Surplus (kW)', fontsize=14)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
               fancybox=True, shadow=True, ncol=3, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)

    plt.tight_layout()

    # Save the plot as an SVG file
    plt.savefig(save_path + 'tracker_surplus_plot_bar.svg', format='svg', bbox_inches='tight')


if __name__ == "__main__":

    plot_total_power_V2G(results_path='E:/GitHub\ev2gym/results/eval_5cs_1tr_V2G_MPC_5_algos_1_exp_2024_03_03_727260/plot_results_dict.pkl',
                     save_path='E:\GitHub\ev2gym\results\eval_5cs_1tr_V2G_MPC_5_algos_1_exp_2024_03_03_727260',
                     algorithm_names=['Charge As Fast As Possible', 'OCCF V2G', 'OCCF G2V', 'eMPC V2G', 'eMPC G2V'])
    # plot_comparable_EV_SoC(results_path='plot_results_dict.pkl')
    pass
