'''
This file is used to plot the comparatigve results of the different algorithms.
'''

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import os
from EVsSimulator.ev_city import EVsSimulator


marker_list = ['.', 'x', 'o', 'v', '<', 's', 'p', 'P', '*', 'h', 'H', '+', 'X', 'D', 'd', '|', '_']

# color_list = ['#00429d', '#5681b9', '#93c4d2', '#ffa59e', '#dd4c65', '#93003a']

color_list = ['#00429d', '#5681b9', '#93c4d2', '#ffa59e', '#dd4c65', '#93003a','b','g','r','c','m','y','k']
algorithm_names =[
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
        print(f'{index}: {key}')
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
                #plot the inflexible loads as a fill between
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
                    
                #plot the solar power as a fill between the inflexible loads and the solar power
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
                plt.plot([env.sim_starting_date, env.sim_date], [0, 0], 'black')
            

            df['total'] = df.sum(axis=1)            
            
            #plot total and use different color and linestyle for each algorithm
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
    
    # plt.show()    
    
def plot_comparable_EV_SoC(results_path, save_path=None, algorithm_names=None):
    '''
    This function is used to plot the SoC of the EVs in the same plot
    '''

    with open(results_path, 'rb') as f:
        replay = pickle.load(f)

    plt.figure(figsize=(10, 7))
    plt.rc('font', family='serif')                
    
    for index, key in enumerate(replay.keys()):
        print(f'{index}: {key}')
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
    
    # plt.show()


def plot_actual_power_vs_setpoint(save_path):
    '''
    This function is used to plot the actual power vs the setpoint power    
    '''
    
    
if __name__ == "__main__":

    plot_total_power(results_path='./results/eval_20cs_1tr_PublicPST_5_algos_3_cycles_2024_02_27_109389/plot_results_dict.pkl',
                     save_path=None,
                     algorithm_names=algorithm_names)
    # plot_comparable_EV_SoC(results_path='plot_results_dict.pkl')
    pass
