'''
This file is used to plot the comparatigve results of the different algorithms.
'''

import pickle
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import os
from ev2gym.models.ev2gym_env import EV2Gym


marker_list = ['.', 'x', 'o', 'v', 's', 'p',
               'P', '*', 'h', 'H', '+', 'X', 'D', 'd', '|', '_']

# color_list = ['#00429d', '#5681b9', '#93c4d2', '#ffa59e', '#dd4c65', '#93003a']

color_list = ['#00429d', '#5681b9', '#93c4d2', '#ffa59e',
              '#dd4c65', '#93003a', 'b', 'g', 'r', 'c', 'm', 'y', 'k']

algorithm_names = [
    'Charge As Fast As Possible',
    # 'Charge As Late As Possible',
    # 'Round Robin',
    'OCCF V2G',
    'OCCF G2V',
    'eMPC V2G',
    'eMPC G2V',
]


def plot_total_power(results_path, save_path=None, algorithm_names=None):

    # Load the env pickle files
    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.figure(figsize=(10, 7))
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

    plt.title(f'Transformer {tr.id+1}', fontsize=28)
    plt.xlabel(f'Time', fontsize=28)
    plt.ylabel(f'Power (kW)', fontsize=28)
    plt.xlim([env.sim_starting_date, env.sim_date])
    plt.xticks(ticks=date_range_print,
               labels=[
                   f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print],
               rotation=45,
               fontsize=28)
    plt.yticks(fontsize=28)
    # put legend under the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               fancybox=True, shadow=True, ncol=3, fontsize=24)

    plt.grid(True, which='minor', axis='both')
    plt.tight_layout()

    fig_name = f'{save_path}/Transformer_Aggregated_Power.png'
    plt.savefig(fig_name, format='png',
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

    plt.show()

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

    plt.show() 
    

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

    plt.show()   


def plot_actual_power_vs_setpoint(results_path, save_path=None, algorithm_names=None):
    
    '''
    This function is used to plot the actual power vs the setpoint power.
    It plots the behavior of each algorithm in subplots vertically.
    '''
    
    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.close('all')
    plt.figure(figsize=(7, 11))
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
        
        plt.axhline(0, color='black', lw=2)
        plt.title(f'{algorithm_names[index]}', fontsize=22)
        
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
            
        plt.xlim([env.sim_starting_date, env.sim_date])
        plt.ylim([0, 1.1*env.current_power_usage.max()])
        
    # Put the legend under the plot in a separate axis           
    plt.legend(['Actual Power', 'Setpoint'], loc='upper center',
               bbox_to_anchor=(0.5, -0.5),
               fancybox=True, shadow=True, ncol=2, fontsize=22)
        
    plt.tight_layout()
    fig_name = f'{save_path}/Actual_vs_Setpoint_Power.png'
    plt.savefig(fig_name, format='png',
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
    plt.show()

if __name__ == "__main__":

    plot_total_power_V2G(results_path='E:/GitHub\ev2gym/results/eval_5cs_1tr_V2G_MPC_5_algos_1_exp_2024_03_03_727260/plot_results_dict.pkl',
                     save_path='E:\GitHub\ev2gym\results\eval_5cs_1tr_V2G_MPC_5_algos_1_exp_2024_03_03_727260',
                     algorithm_names=['Charge As Fast As Possible', 'OCCF V2G', 'OCCF G2V', 'eMPC V2G', 'eMPC G2V'])
    # plot_comparable_EV_SoC(results_path='plot_results_dict.pkl')
    pass
