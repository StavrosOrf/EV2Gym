# This file contains support functions for the EV City environment.
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ev_city_plot(ev_env):
    '''Plots the simulation data

    Plots:
        - The total power of each transformer
        - The power of each charging station
        - The energy level of each EV in charging stations
    '''
    print("Plotting simulation data at ./plots/" + ev_env.sim_name + "/")
    # date_range = pd.date_range(start=ev_env.sim_starting_date,
    #                            end=ev_env.sim_date -
    #                            datetime.timedelta(
    #                                minutes=ev_env.timescale),
    #                            freq=f'{ev_env.timescale}min')
    date_range = pd.date_range(start=ev_env.sim_starting_date,
                                end=ev_env.sim_starting_date +
                                (ev_env.simulation_length - 1) *
                                datetime.timedelta(
                                    minutes=ev_env.timescale),
                                freq=f'{ev_env.timescale}min')
    date_range_print = pd.date_range(start=ev_env.sim_starting_date,
                                        end=ev_env.sim_date,
                                        periods=10)

    # Plot the energy level of each EV for each charging station
    plt.figure(figsize=(20, 17))
    plt.style.use('seaborn-darkgrid')
    plt.rcParams.update({'font.size': 16})
    counter = 1
    dim_x = int(np.ceil(np.sqrt(ev_env.cs)))
    dim_y = int(np.ceil(ev_env.cs/dim_x))
    for cs in ev_env.charging_stations:

        plt.subplot(dim_x, dim_y, counter)
        df = pd.DataFrame([], index=date_range)

        for port in range(cs.n_ports):
            df[port] = ev_env.port_energy_level[port, cs.id, :]

        # Add another row with one datetime step to make the plot look better
        df.loc[df.index[-1] +
                datetime.timedelta(minutes=ev_env.timescale)] = df.iloc[-1]

        for port in range(cs.n_ports):
            for i, (t_arr, t_dep) in enumerate(ev_env.port_arrival[f'{cs.id}.{port}']):

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

        plt.title(f'Charging Station {cs.id}')
        plt.xlabel(f'Time')
        plt.ylabel('Energy Level (kW)')
        plt.xlim([ev_env.sim_starting_date, ev_env.sim_date])
        plt.xticks(ticks=date_range_print,
                    labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print], rotation=45)
        if len(ev_env.port_arrival[f'{cs.id}.{port}']) < 6:
            plt.legend()
        plt.grid(True, which='minor', axis='both')
        counter += 1

    plt.tight_layout()
    # Save plt to html
    fig_name = f'plots/{ev_env.sim_name}/EV_Energy_Level.png' #.html
    plt.savefig(fig_name, format='png', #svg
                dpi=60, bbox_inches='tight')

    # Plot the total power of each transformer
    plt.figure(figsize=(20, 17))
    counter = 1
    dim_x = int(np.ceil(np.sqrt(ev_env.number_of_transformers)))
    dim_y = int(np.ceil(ev_env.number_of_transformers/dim_x))
    for tr in ev_env.transformers:

        plt.subplot(dim_x, dim_y, counter)
        df = pd.DataFrame([],
                            index=date_range)

        for cs in tr.cs_ids:
            df[cs] = ev_env.cs_power[cs, :]

        # create 2 dfs, one for positive power and one for negative
        df_pos = df.copy()
        df_pos[df_pos < 0] = 0
        df_neg = df.copy()
        df_neg[df_neg > 0] = 0
        colors = plt.cm.gist_earth(np.linspace(0.1, 0.8, len(tr.cs_ids)))

        # Add another row with one datetime step to make the plot look better
        df_pos.loc[df_pos.index[-1] +
                    datetime.timedelta(minutes=ev_env.timescale)] = df_pos.iloc[-1]
        df_neg.loc[df_neg.index[-1] +
                    datetime.timedelta(minutes=ev_env.timescale)] = df_neg.iloc[-1]

        # plot the positive power
        plt.stackplot(df_pos.index, df_pos.values.T,
                        interpolate=True,
                        step='post',
                        alpha=0.7,
                        colors=colors,
                        linestyle='--')

        df['total'] = df.sum(axis=1)
        # print(df)
        max_power = tr.max_power * ev_env.timescale / 60
        min_power = tr.min_power * ev_env.timescale / 60
        plt.plot([ev_env.sim_starting_date, ev_env.sim_date],
                    [max_power, max_power], 'r--')
        plt.step(df.index, df['total'], 'darkgreen',
                    where='post', linestyle='--')
        plt.plot([ev_env.sim_starting_date, ev_env.sim_date],
                    [min_power, min_power], 'r--')
        plt.stackplot(df_neg.index, df_neg.values.T,
                        interpolate=True,
                        step='post',
                        colors=colors,
                        alpha=0.7,
                        linestyle='--')
        plt.plot([ev_env.sim_starting_date, ev_env.sim_date], [0, 0], 'black')

        # for cs in tr.cs_ids:
        #     plt.step(df.index, df[cs], 'white', where='post', linestyle='--')
        plt.title(f'Transformer {tr.id}')
        plt.xlabel(f'Time')
        plt.ylabel(f'Power (kWh per {ev_env.timescale} min)')
        plt.xlim([ev_env.sim_starting_date, ev_env.sim_date])
        plt.xticks(ticks=date_range_print,
                    labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print], rotation=45)
        plt.legend([f'CS {i}' for i in tr.cs_ids] +
                    ['Total Power Limit', 'Total Power'])
        plt.grid(True, which='minor', axis='both')
        counter += 1

    plt.tight_layout()
    # plt.show()
    fig_name = f'plots/{ev_env.sim_name}/Transformer_Power.png'
    plt.savefig(fig_name, format='png',
                dpi=60, bbox_inches='tight')

    # Plot the power of each charging station
    counter = 1
    plt.figure(figsize=(20, 17))
    dim_x = int(np.ceil(np.sqrt(ev_env.cs)))
    dim_y = int(np.ceil(ev_env.cs/dim_x))
    for cs in ev_env.charging_stations:

        plt.subplot(dim_x, dim_y, counter)
        df = pd.DataFrame([], index=date_range)

        for port in range(cs.n_ports):
            df[port] = ev_env.port_power[port, cs.id, :]
            # create 2 dfs, one for positive power and one for negative

        df_pos = df.copy()
        df_pos[df_pos < 0] = 0
        df_neg = df.copy()
        df_neg[df_neg > 0] = 0
        colors = plt.cm.gist_earth(np.linspace(0.1, 0.8, cs.n_ports))

        # Add another row with one datetime step to make the plot look better
        df_pos.loc[df_pos.index[-1] +
                    datetime.timedelta(minutes=ev_env.timescale)] = df_pos.iloc[-1]
        df_neg.loc[df_neg.index[-1] +
                    datetime.timedelta(minutes=ev_env.timescale)] = df_neg.iloc[-1]

        plt.stackplot(df_pos.index, df_pos.values.T,
                        interpolate=True,
                        step='post',
                        alpha=0.7,
                        colors=colors)
        df['total'] = df.sum(axis=1)

        # plot the power limit
        max_power = cs.max_charge_power * ev_env.timescale / 60
        min_power = -cs.max_discharge_power * ev_env.timescale / 60
        plt.plot([ev_env.sim_starting_date, ev_env.sim_date],
                    [max_power, max_power], 'r--')
        plt.step(df.index, df['total'], 'darkgreen',
                    where='post', linestyle='--')
        plt.plot([ev_env.sim_starting_date, ev_env.sim_date],
                    [min_power, min_power], 'r--')
        plt.stackplot(df_neg.index, df_neg.values.T,
                        interpolate=True,
                        step='post',
                        colors=colors,
                        alpha=0.7)
        plt.plot([ev_env.sim_starting_date, ev_env.sim_date], [0, 0], 'black')

        # for i in range(cs.n_ports):
        #     plt.step(df.index, df[i], 'grey', where='post', linestyle='--')

        plt.title(f'Charging Station {cs.id}')
        plt.xlabel(f'Time')
        plt.ylabel(f'Power (kWh per {ev_env.timescale} min)')
        plt.ylim([min_power*1.1, max_power*1.1])
        plt.xlim([ev_env.sim_starting_date, ev_env.sim_date])
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
    fig_name = f'plots/{ev_env.sim_name}/CS_Power.png'
    plt.savefig(fig_name, format='png', dpi=60, bbox_inches='tight')
    plt.close('all')


def get_statistics(ev_env):
    total_ev_served = np.array(
        [cs.total_evs_served for cs in ev_env.charging_stations]).sum()
    total_profits = np.array(
        [cs.total_profits for cs in ev_env.charging_stations]).sum()
    toal_energy_charged = np.array(
        [cs.total_energy_charged for cs in ev_env.charging_stations]).sum()
    total_energy_discharged = np.array(
        [cs.total_energy_discharged for cs in ev_env.charging_stations]).sum()
    average_user_satisfaction = np.average(np.array(
        [cs.get_avg_user_satisfaction() for cs in ev_env.charging_stations]))
    
    
    stats = {'total_ev_served': total_ev_served,
            'total_profits': total_profits,
            'toal_energy_charged': toal_energy_charged,
            'total_energy_discharged': total_energy_discharged,
            'average_user_satisfaction': average_user_satisfaction,
            'ev_spawn_rate': ev_env.spawn_rate,                
            }
    
    if ev_env.replay is not None:
        stats['opt_profits'] = ev_env.replay.stats["total_profits"]

    
    return stats

def print_statistics(ev_env):
    '''Prints the statistics of the simulation'''
    total_ev_served = np.array(
        [cs.total_evs_served for cs in ev_env.charging_stations]).sum()
    total_profits = np.array(
        [cs.total_profits for cs in ev_env.charging_stations]).sum()
    toal_energy_charged = np.array(
        [cs.total_energy_charged for cs in ev_env.charging_stations]).sum()
    total_energy_discharged = np.array(
        [cs.total_energy_discharged for cs in ev_env.charging_stations]).sum()
    average_user_satisfaction = np.average(np.array(
        [cs.get_avg_user_satisfaction() for cs in ev_env.charging_stations]))

    print("\n\n==============================================================")
    print("Simulation statistics:")
    print(f'  - Total EVs spawned: {ev_env.total_evs_spawned}')
    print(f'  - Total EVs served: {total_ev_served}')
    print(f'  - Total profits: {total_profits*100:.2f} €')
    print(
        f'  - Average user satisfaction: {average_user_satisfaction:.2f} %')

    print(f'  - Total energy charged: {toal_energy_charged:.1f} kW')
    print(
        f'  - Total energy discharged: {total_energy_discharged:.1f} kW\n')

    for cs in ev_env.charging_stations:
        print(cs)
    print("==============================================================\n\n")

def visualize(ev_env):
    '''Renders the current state of the environment in the terminal'''

    print(f"\n Step: {ev_env.current_step}" +
            f" | {ev_env.sim_date.hour}:{ev_env.sim_date.minute}:{ev_env.sim_date.second} |" +
            f" \tEVs +{ev_env.current_ev_arrived} / -{ev_env.current_ev_departed}" +
            f" | Total: {ev_env.current_evs_parked} / {ev_env.number_of_ports}")

    if ev_env.verbose:
        for cs in ev_env.charging_stations:
            print(f'  - Charging station {cs.id}:')
            print(f'\t Power: {cs.current_power_output:4.1f} kWh |' +
                    f' \u2197 {ev_env.charge_prices[cs.id, ev_env.current_step -1 ]:4.2f} €/kWh ' +
                    f' \u2198 {ev_env.discharge_prices[cs.id, ev_env.current_step - 1]:4.2f} €/kWh |' +
                    f' EVs served: {cs.total_evs_served:3d} ' +
                    f' {cs.total_profits:4.2f} €')

            for port in range(cs.n_ports):
                ev = cs.evs_connected[port]
                if ev is not None:
                    print(f'\t\tPort {port}: {ev}')
                else:
                    print(f'\t\tPort {port}:')
        print("")
        for tr in ev_env.transformers:
            print(tr)

def spawn_EVs(number_of_ports):
    '''Spawns EVs based on the spawn rate'''
    evs_to_spawn = np.random.poisson(number_of_ports * ev_env.spawn_rate)
    return evs_to_spawn

