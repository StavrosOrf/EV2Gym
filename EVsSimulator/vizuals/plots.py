# This file contains functions for plotting

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime


def visualize_step(env):
    '''Renders the current state of the environment in the terminal'''

    print(f"\n Step: {env.current_step}" +
          f" | {str(env.sim_date.weekday())} {env.sim_date.hour:2d}:{env.sim_date.minute:2d}:{env.sim_date.second:2d} |" +
          f" \tEVs +{env.current_ev_arrived} / -{env.current_ev_departed}" +
          f" | Total: {env.current_evs_parked} / {env.number_of_ports}")

    if env.verbose:
        for cs in env.charging_stations:
            print(f'  - Charging station {cs.id}:')
            print(f'\t Power: {cs.current_power_output:4.1f} kW |' +
                  f' \u2197 {env.charge_prices[cs.id, env.current_step -1 ]:4.2f} €/kWh ' +
                  f' \u2198 {env.discharge_prices[cs.id, env.current_step - 1]:4.2f} €/kWh |' +
                  f' EVs served: {cs.total_evs_served:3d} ' +
                  f' {cs.total_profits:4.2f} €')

            for port in range(cs.n_ports):
                ev = cs.evs_connected[port]
                if ev is not None:
                    print(f'\t\tPort {port}: {ev}')
                else:
                    print(f'\t\tPort {port}:')
        print("")
        for tr in env.transformers:
            print(tr)

        # print current current power setpoint
        print(f' Power setpoints (prev-step): {env.current_power_usage[env.current_step - 1]:.1f} Actual/' +
              f' {env.power_setpoints[env.current_step - 1]:.1f} Setpoint/'
              f' {env.charge_power_potential[env.current_step - 1]:.1f} Potential in kW')


def ev_city_plot(env):
    '''Plots the simulation data

    Plots:
        - The total power and current of each transformer
        - The current of each charging station
        - The energy level of each EV in charging stations
        - The total power of the CPO
    '''
    print("Plotting simulation data at ./results/" + env.sim_name + "/")

    date_range = pd.date_range(start=env.sim_starting_date,
                               end=env.sim_starting_date +
                               (env.simulation_length - 1) *
                               datetime.timedelta(
                                   minutes=env.timescale),
                               freq=f'{env.timescale}min')
    date_range_print = pd.date_range(start=env.sim_starting_date,
                                     end=env.sim_date,
                                     periods=10)
    plt.close('all')
    # close plt ion
    plt.ioff()
    plt.close('all')
    
    # light weight plots when there are too many charging stations
    if not env.lightweight_plots:
        # Plot the energy level of each EV for each charging station
        plt.figure(figsize=(20, 17))        
        # plt.style.use('seaborn-darkgrid')        
        plt.rcParams.update({'font.size': 16})
        plt.rcParams['font.family'] = ['serif']
        
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

                    plt.step(df.index, y, where='post')
                    plt.fill_between(df.index,
                                     y,
                                     step='post',
                                     alpha=0.7,
                                     label=f'EV {i}, Port {port}')

            plt.title(f'Charging Station {cs.id}', fontsize=24)
            plt.xlabel(f'Time', fontsize=24)
            plt.ylabel('State of Charge (0-1)', fontsize=24)
            plt.xlim([env.sim_starting_date, env.sim_date])
            plt.xticks(ticks=date_range_print,
                       labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print], rotation=45,
                       fontsize=22)
            # if len(env.port_arrival[f'{cs.id}.{port}']) < 6:
            if dim_x < 3:
                plt.legend()
            plt.grid(True, which='minor', axis='both')
            counter += 1

        plt.tight_layout()
        # Save plt to html
        fig_name = f'results/{env.sim_name}/EV_Energy_Level.png'  # .html
        # plt.show()
        # save in pdf format
        plt.savefig(fig_name, format='png',  # svg
                    dpi=60, bbox_inches='tight')
        
        plt.close('all')
        # Plot the charging and discharging prices
        plt.figure(figsize=(20, 17))

        df = pd.DataFrame([], index=date_range)
        df['charge'] = - env.charge_prices[0, :]
        df['discharge'] = env.discharge_prices[0, :]
        plt.plot(df['charge'], label='Charge prices (€/kW))')
        plt.plot(df['discharge'], label='Discharge prices (€/kW))')
        # plot y = 0 line
        plt.plot([env.sim_starting_date, env.sim_date], [0, 0], 'black')
        plt.legend(fontsize=24)
        plt.grid(True, which='major', axis='both')
        plt.ylabel('Price (€/kW)', fontsize=24)
        plt.xlabel('Time', fontsize=24)
        plt.xlim([env.sim_starting_date, env.sim_date])
        plt.xticks(ticks=date_range_print,
                   labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print], rotation=45,
                   fontsize=22)
        plt.tight_layout()
        fig_name = f'results/{env.sim_name}/Prices.png'
        plt.savefig(fig_name, format='png',
                    dpi=60, bbox_inches='tight')

        # Plot the total power of each transformer
        plt.figure(figsize=(20, 17))
        counter = 1
        dim_x = int(np.ceil(np.sqrt(env.number_of_transformers)))
        dim_y = int(np.ceil(env.number_of_transformers/dim_x))
        for tr in env.transformers:

            plt.subplot(dim_x, dim_y, counter)
            df = pd.DataFrame([],
                              index=date_range)
            
            colors = plt.cm.gist_earth(np.linspace(0.1, 0.8, len(tr.cs_ids)+1))

            if env.config['inflexible_loads']['include']:
                df['inflexible'] = env.tr_inflexible_loads[tr.id, :] * 1000 / 400
                blue = np.array([0.529, 0.808, 0.922, 1])                
                colors = np.insert(colors, 0, blue, axis=0)
                
            if env.config['solar_power']['include']:
                df['solar'] = env.tr_solar_power[tr.id, :] * 1000 / 400
                gold = np.array([1, 0.843, 0, 1])
                colors = np.insert(colors, 0, gold, axis=0)
                

            for cs in tr.cs_ids:
                df[cs] = env.cs_current[cs, :]

            # create 2 dfs, one for positive power and one for negative
            df_pos = df.copy()
            df_pos[df_pos <= 0] = 0
            df_neg = df.copy()
            df_neg[df_neg > 0] = 0
            

            # Add another row with one datetime step to make the plot look better
            df_pos.loc[df_pos.index[-1] +
                       datetime.timedelta(minutes=env.timescale)] = df_pos.iloc[-1]
            df_neg.loc[df_neg.index[-1] +
                       datetime.timedelta(minutes=env.timescale)] = df_neg.iloc[-1]

            # plot the positive power
            plt.stackplot(df_pos.index, df_pos.values.T,
                          interpolate=True,
                          step='post',
                          alpha=0.7,
                          colors=colors,
                          linestyle='--')

            df['total'] = df.sum(axis=1)
            # print(df)
            max_current = tr.max_current  # * env.timescale / 60
            min_current = tr.min_current  # * env.timescale / 60
            plt.step(df.index,
                     max_current, where='post', color='r', linestyle='--')
            plt.step(df.index, df['total'], 'darkgreen',
                     where='post', linestyle='--')
            plt.plot(df.index,
                     min_current, 'r--')
            plt.stackplot(df_neg.index, df_neg.values.T,
                          interpolate=True,
                          step='post',
                          colors=colors,
                          alpha=0.7,
                          linestyle='--')
            plt.plot([env.sim_starting_date,
                     env.sim_date], [0, 0], 'black')

            # for cs in tr.cs_ids:
            #     plt.step(df.index, df[cs], 'white', where='post', linestyle='--')
            plt.title(f'Transformer {tr.id}')
            plt.xlabel(f'Time')
            plt.ylabel(f'Current (A)')
            plt.xlim([env.sim_starting_date, env.sim_date])
            plt.xticks(ticks=date_range_print,
                       labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print], rotation=45)
            if len(tr.cs_ids) < 4:
                if env.config['inflexible_loads']['include']:
                    plt.legend(['Inflexible Loads'] +
                               [f'CS {i}' for i in tr.cs_ids] +
                               ['Circuit Breaker Limit (A)', 'Total Current (A)'])
                else:
                    plt.legend([f'CS {i}' for i in tr.cs_ids] +
                               ['Circuit Breaker Limit (A)', 'Total Current (A)'])
            plt.grid(True, which='minor', axis='both')
            counter += 1

        plt.tight_layout()
        # plt.show()
        fig_name = f'results/{env.sim_name}/Transformer_Current.png'
        plt.savefig(fig_name, format='png',
                    dpi=60, bbox_inches='tight')

        # Plot the power of each charging station
        counter = 1
        plt.figure(figsize=(20, 17))
        dim_x = int(np.ceil(np.sqrt(env.cs)))
        dim_y = int(np.ceil(env.cs/dim_x))
        for cs in env.charging_stations:

            plt.subplot(dim_x, dim_y, counter)
            df = pd.DataFrame([], index=date_range)
            df_signal = pd.DataFrame([], index=date_range)

            for port in range(cs.n_ports):
                df[port] = env.port_current[port, cs.id, :]
                df_signal[port] = env.port_current_signal[port, cs.id, :]
                # create 2 dfs, one for positive power and one for negative
            df_pos = df.copy()
            df_pos[df_pos < 0] = 0
            df_neg = df.copy()
            df_neg[df_neg > 0] = 0

            colors = plt.cm.gist_earth(np.linspace(0.1, 0.8, cs.n_ports))

            # Add another row with one datetime step to make the plot look better
            df_pos.loc[df_pos.index[-1] +
                       datetime.timedelta(minutes=env.timescale)] = df_pos.iloc[-1]
            df_neg.loc[df_neg.index[-1] +
                       datetime.timedelta(minutes=env.timescale)] = df_neg.iloc[-1]

            plt.stackplot(df_pos.index, df_pos.values.T,
                          interpolate=True,
                          step='post',
                          alpha=0.7,
                          colors=colors)

            df['total'] = df.sum(axis=1)
            df_signal['total'] = df_signal.sum(axis=1)

            # plot the power limit
            max_charge_current = cs.max_charge_current  # * env.timescale / 60
            max_discharge_current = cs.max_discharge_current  # * env.timescale / 60
            min_charge_current = cs.min_charge_current  # * env.timescale / 60
            min_discharge_current = cs.min_discharge_current  # * env.timescale / 60
            plt.plot([env.sim_starting_date, env.sim_date],
                     [max_charge_current, max_charge_current], 'r--')
            plt.step(df.index, df['total'], 'darkgreen',
                     where='post', linestyle='--')
            plt.step(df_signal.index, df_signal['total'], 'cyan', where='post', alpha=1,
                     linestyle='--')

            plt.plot([env.sim_starting_date, env.sim_date],
                     [min_charge_current, min_charge_current], 'b--')
            plt.plot([env.sim_starting_date, env.sim_date],
                     [max_discharge_current, max_discharge_current], 'r--')
            plt.plot([env.sim_starting_date, env.sim_date],
                     [min_discharge_current, min_discharge_current], 'b--')

            plt.stackplot(df_neg.index, df_neg.values.T,
                          interpolate=True,
                          step='post',
                          colors=colors,
                          alpha=0.7)

            plt.plot([env.sim_starting_date,
                     env.sim_date], [0, 0], 'black')

            # for i in range(cs.n_ports):
            #     plt.step(df.index, df[i], 'grey', where='post', linestyle='--')

            plt.title(f'Charging Station {cs.id}', fontsize=24)
            plt.xlabel(f'Time', fontsize=24)
            plt.ylabel(f'Current (A)', fontsize=24)
            plt.ylim([max_discharge_current*1.1, max_charge_current*1.1])
            plt.xlim([env.sim_starting_date, env.sim_date])
            plt.xticks(ticks=date_range_print,
                       labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print], rotation=45,
                       fontsize=24)
            # place the legend under each plot

            if dim_x < 3:
                plt.legend([f'Port {i}' for i in range(
                    cs.n_ports)] + ['Max. Current',
                                    'Actual Current',
                                    'Current Signal',
                                    'Min. Current'],
                    fontsize=22,)
            plt.grid(True, which='minor', axis='both')
            counter += 1

        plt.tight_layout()
        # Save plt to html
        fig_name = f'results/{env.sim_name}/CS_Current_signals.png'
        plt.savefig(fig_name, format='png', dpi=60, bbox_inches='tight')

    plt.close('all')
    # Plot the total power for each CS group
    df_total_power = pd.DataFrame([], index=date_range)
    plt.figure(figsize=(10, 8))

    counter = 1
    dim_x = int(np.ceil(np.sqrt(env.number_of_transformers)))
    dim_y = int(np.ceil(env.number_of_transformers/dim_x))
    for tr in env.transformers:

        plt.subplot(dim_x, dim_y, counter)
        df = pd.DataFrame([],
                          index=date_range)

        colors = plt.cm.gist_earth(np.linspace(0.1, 0.8, len(tr.cs_ids)))        
        extra_col = 0

        if env.config['inflexible_loads']['include']:
            df['inflexible'] = env.tr_inflexible_loads[tr.id, :]
            extra_col += 1

        if env.config['solar_power']['include']:
            df['solar'] = env.tr_solar_power[tr.id, :]
            gold = np.array([1, 0.843, 0, 1])
            colors = np.insert(colors, 0, gold, axis=0)
            extra_col += 1

        if env.config['inflexible_loads']['include']:
            # grey = np.array([0.5, 0.5, 0.5, 1])
            light_blue = np.array([0.529, 0.808, 0.922, 1])
            colors = np.insert(colors, 0, light_blue, axis=0)

        for cs in tr.cs_ids:
            df[cs] = env.cs_power[cs, :]

        # create 2 dfs, one for positive power and one for negative
        df_pos = df.copy()
        df_pos[df_pos < 0] = 0
        df_neg = df.copy()
        df_neg[df_neg > 0] = 0

        # Add another row with one datetime step to make the plot look better
        df_pos.loc[df_pos.index[-1] +
                   datetime.timedelta(minutes=env.timescale)] = df_pos.iloc[-1]
        df_neg.loc[df_neg.index[-1] +
                   datetime.timedelta(minutes=env.timescale)] = df_neg.iloc[-1]

        # plot the positive power        
        stacks = plt.stackplot(df_pos.index, df_pos.values.T,
                      interpolate=True,
                      step='post',
                      alpha=0.7,
                      colors=colors,
                    #   hatch=['//'],#, '\\'],#  +[' ']*len(tr.cs_ids),
                    # hatch=tuple(['\\'] +[' ']*len(tr.cs_ids)),
                      linestyle='--')
        
        # for index, stack in enumerate(stacks):
        #     if env.config['inflexible_loads']['include']:
                # stack.set_hatch(['//')

        df['total'] = df.sum(axis=1)
        df_total_power[tr.id] = df['total']
        
        if env.config['demand_response']['include']:
            plt.fill_between(df.index,
                             np.array([tr.max_power.max()] * len(df.index)),
                             tr.max_power,
                             step='post',
                             alpha=0.7,
                             color='r',
                             hatch='xx',
                             linestyle='--',
                             linewidth=2)


        plt.step(df.index, df['total'],
                 'darkgreen',
                 where='post',
                #  linestyle='-',
                 linewidth=3)

        plt.step(df.index,
                 #  tr.max_power
                 [tr.max_power.max()] * len(df.index),
                 where='post',
                 color='r',
                 linestyle='--',
                 linewidth=2)

        plt.stackplot(df_neg.index, df_neg.values.T,
                      interpolate=True,
                      step='post',
                      colors=colors,
                      alpha=0.7,
                    #   hatch='-\ ',
                      linestyle='--')
        plt.plot([env.sim_starting_date, env.sim_date], [0, 0], 'black')

        # for cs in tr.cs_ids:
        #     plt.step(df.index, df[cs], 'white', where='post', linestyle='--')

        plt.title(f'Transformer {tr.id}')
        plt.xlabel(f'Time', fontsize=28)
        plt.ylabel(f'Power (kW)', fontsize=28)
        plt.xlim([env.sim_starting_date, env.sim_date])
        plt.xticks(ticks=date_range_print,
                   labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print],
                   rotation=45,
                   fontsize=28)
        plt.yticks(fontsize=28)

        legend_list = [f'CS {i+1}' for i in tr.cs_ids] + \
            ['Total Power (kW)'] + \
            ['Power limit (kW)']
            
        if env.config['demand_response']['include']:
            legend_list = legend_list[:-2] + ['Demand Response (kW)'] + legend_list[-2:]

        if len(tr.cs_ids) < 4:

            if env.config['solar_power']['include']:
                legend_list = ['Solar Power'] + legend_list

            if env.config['inflexible_loads']['include']:
                legend_list = ['Inflexible Loads'] + legend_list

            plt.legend(legend_list)
        else:
            if env.config['solar_power']['include']:
                legend_list = ['Solar Power'] + legend_list

            if env.config['inflexible_loads']['include']:
                legend_list = ['Inflexible Loads'] + legend_list

            plt.legend(loc='lower right', fontsize=24, labels=legend_list)

        plt.grid(True, which='minor', axis='both')
        counter += 1

    if len(env.transformers) < 10:
        plt.tight_layout()
        fig_name = f'results/{env.sim_name}/Transformer_Aggregated_Power.png'
        plt.savefig(fig_name, format='png',
                    dpi=60, bbox_inches='tight')
    else:
        # clear plt canvas
        plt.close('all')
        
    # plt.show()
    # Plot the total power of the CPO
    plt.figure(figsize=(20, 17))

    # plt.style.use('seaborn-darkgrid')
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['font.family'] = ['serif']

    # create 2 dfs, one for positive power and one for negative
    df_pos = df_total_power.copy()
    df_pos[df_pos < 0] = 0
    df_neg = df_total_power.copy()
    df_neg[df_neg > 0] = 0

    colors = plt.cm.gist_earth(np.linspace(0.1, 0.8, len(env.transformers)))

    # Add another row with one datetime step to make the plot look better
    df_pos.loc[df_pos.index[-1] +
               datetime.timedelta(minutes=env.timescale)] = df_pos.iloc[-1]
    df_neg.loc[df_neg.index[-1] +
               datetime.timedelta(minutes=env.timescale)] = df_neg.iloc[-1]

    df_total_power['total'] = df_total_power.sum(axis=1)

    plt.step(df_total_power.index, df_total_power['total'], 'darkgreen',
             where='post', linestyle='--')

    plt.step(df_total_power.index, env.power_setpoints, 'r--', where='post',)

    if env.load_from_replay_path is not None:
        plt.step(df_total_power.index, env.replay.ev_load_potential,
                 'b--', where='post', alpha=0.4,)
    # else:
    #     plt.step(df_total_power.index, env.current_power_usage,
    #              'b--', where='post', alpha=0.4,)

    # plot the positive power
    plt.stackplot(df_pos.index, df_pos.values.T,
                  interpolate=True,
                  step='post',
                  alpha=0.7,
                  colors=colors,
                  linestyle='--')

    plt.stackplot(df_neg.index, df_neg.values.T,
                  interpolate=True,
                  step='post',
                  colors=colors,
                  alpha=0.7,
                  linestyle='--')

    plt.plot([env.sim_starting_date, env.sim_date], [0, 0], 'black')

    # for cs in tr.cs_ids:
    #     plt.step(df.index, df[cs], 'white', where='post', linestyle='--')
    # plt.title(f'Aggreagated Power vs Power Setpoint', fontsize=44)
    plt.xlabel(f'Time', fontsize=38)
    plt.ylabel(f'Power (kW)', fontsize=38)
    plt.xlim([env.sim_starting_date, env.sim_date])

    date_range_print = pd.date_range(start=env.sim_starting_date,
                                     end=env.sim_date,
                                     periods=7)
    plt.xticks(ticks=date_range_print,
               labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print], rotation=45, fontsize=28)
    # plt.xticks(ticks=date_range_print,
    #            labels=[f'{d.strftime("%A")}' for d in date_range_print], rotation=45, fontsize=28)

    # set ytick font size
    plt.yticks(fontsize=28)
    if len(env.transformers) <= 10:
        if env.load_from_replay_path is not None:
            plt.legend(['Total Power'] +
                       [f'Power Setpoint'] +
                       ['EV Unsteered Load Potential (kW)']
                       + [f'Tr {i}' for i in range(len(env.transformers))])
        else:
            plt.legend(['Total Power'] +
                       [f'Power Setpoint']
                       + [f'Tr {i}' for i in range(len(env.transformers))])
    else:
        # plt.legend(['Total Power (kW)']+[f'Power Setpoint (kW)']+['EV Unsteered Load Potential (kW)'])
        plt.legend(['Total Power (kW)'], fontsize=28)
    plt.grid(True, which='minor', axis='both')

    plt.tight_layout()
    # plt.show()
    fig_name = f'results/{env.sim_name}/Total_Aggregated_Power.png'
    plt.savefig(fig_name, format='png',
                dpi=60, bbox_inches='tight')

    # plot prices
    # plt.figure(figsize=(20, 17))
    # plt.plot(env.charge_prices[0,:], label='Charge prices (€/kW))')
    # plt.plot(env.discharge_prices[0,:], label='Discharge prices (€/kW))')
    # plt.legend()
    # plt.grid(True, which='minor', axis='both')
    # plt.tight_layout()
    # fig_name = f'results/{env.sim_name}/Prices.png'
    # plt.savefig(fig_name, format='png',
    #             dpi=60, bbox_inches='tight')

    plt.close('all')
