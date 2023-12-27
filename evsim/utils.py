# This file contains support functions for the EV City environment.
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .ev import EV


def ev_city_plot(ev_env):
    '''Plots the simulation data

    Plots:
        - The total power and current of each transformer
        - The current of each charging station
        - The energy level of each EV in charging stations
        - The total power of the CPO
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
    if not ev_env.lightweight_plots:
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
            plt.ylabel('Energy Level (kWh)')
            plt.xlim([ev_env.sim_starting_date, ev_env.sim_date])
            plt.xticks(ticks=date_range_print,
                       labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print], rotation=45)
            # if len(ev_env.port_arrival[f'{cs.id}.{port}']) < 6:
            if dim_x < 5:
                plt.legend()
            plt.grid(True, which='minor', axis='both')
            counter += 1

        plt.tight_layout()
        # Save plt to html
        fig_name = f'plots/{ev_env.sim_name}/EV_Energy_Level.png'  # .html
        plt.savefig(fig_name, format='png',  # svg
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
                df[cs] = ev_env.cs_current[cs, :]

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
            max_current = tr.max_current  # * ev_env.timescale / 60
            min_current = tr.min_current  # * ev_env.timescale / 60
            plt.plot([ev_env.sim_starting_date, ev_env.sim_date],
                     [max_current, max_current], 'r--')
            plt.step(df.index, df['total'], 'darkgreen',
                     where='post', linestyle='--')
            plt.plot([ev_env.sim_starting_date, ev_env.sim_date],
                     [min_current, min_current], 'r--')
            plt.stackplot(df_neg.index, df_neg.values.T,
                          interpolate=True,
                          step='post',
                          colors=colors,
                          alpha=0.7,
                          linestyle='--')
            plt.plot([ev_env.sim_starting_date,
                     ev_env.sim_date], [0, 0], 'black')

            # for cs in tr.cs_ids:
            #     plt.step(df.index, df[cs], 'white', where='post', linestyle='--')
            plt.title(f'Transformer {tr.id}')
            plt.xlabel(f'Time')
            plt.ylabel(f'Current (A)')
            plt.xlim([ev_env.sim_starting_date, ev_env.sim_date])
            plt.xticks(ticks=date_range_print,
                       labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print], rotation=45)
            if len(tr.cs_ids) < 3:
                plt.legend([f'CS {i}' for i in tr.cs_ids] +
                           ['Circuit Breaker Limit (A)', 'Total Current (A)'])
            plt.grid(True, which='minor', axis='both')
            counter += 1

        plt.tight_layout()
        # plt.show()
        fig_name = f'plots/{ev_env.sim_name}/Transformer_Current.png'
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
                df[port] = ev_env.port_current[port, cs.id, :]
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
            max_charge_current = cs.max_charge_current  # * ev_env.timescale / 60
            max_discharge_current = cs.max_discharge_current  # * ev_env.timescale / 60
            min_charge_current = cs.min_charge_current  # * ev_env.timescale / 60
            min_discharge_current = cs.min_discharge_current  # * ev_env.timescale / 60
            plt.plot([ev_env.sim_starting_date, ev_env.sim_date],
                     [max_charge_current, max_charge_current], 'r--')
            plt.step(df.index, df['total'], 'darkgreen',
                     where='post', linestyle='--')
            plt.plot([ev_env.sim_starting_date, ev_env.sim_date],
                     [min_charge_current, min_charge_current], 'b--')
            plt.plot([ev_env.sim_starting_date, ev_env.sim_date],
                     [max_discharge_current, max_discharge_current], 'r--')
            plt.plot([ev_env.sim_starting_date, ev_env.sim_date],
                     [min_discharge_current, min_discharge_current], 'b--')

            plt.stackplot(df_neg.index, df_neg.values.T,
                          interpolate=True,
                          step='post',
                          colors=colors,
                          alpha=0.7)
            plt.plot([ev_env.sim_starting_date,
                     ev_env.sim_date], [0, 0], 'black')

            # for i in range(cs.n_ports):
            #     plt.step(df.index, df[i], 'grey', where='post', linestyle='--')

            plt.title(f'Charging Station {cs.id}')
            plt.xlabel(f'Time')
            plt.ylabel(f'Current (A)')
            plt.ylim([max_discharge_current*1.1, max_charge_current*1.1])
            plt.xlim([ev_env.sim_starting_date, ev_env.sim_date])
            plt.xticks(ticks=date_range_print,
                       labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print], rotation=45)
            # place the legend under each plot

            if dim_x < 5:
                plt.legend([f'Port {i}' for i in range(
                    cs.n_ports)] + ['Total Current Limit (A)',
                                    'Total Current (A)',
                                    'Minimum EVSE Current Limit (A)'])
            plt.grid(True, which='minor', axis='both')
            counter += 1

        plt.tight_layout()
        # Save plt to html
        fig_name = f'plots/{ev_env.sim_name}/CS_Current_signals.png'
        plt.savefig(fig_name, format='png', dpi=60, bbox_inches='tight')

    # Plot the total power for each CS group
    df_total_power = pd.DataFrame([], index=date_range)
    plt.figure(figsize=(20, 17))

    counter = 1
    dim_x = int(np.ceil(np.sqrt(ev_env.number_of_transformers)))
    dim_y = int(np.ceil(ev_env.number_of_transformers/dim_x))
    for tr in ev_env.transformers:

        plt.subplot(dim_x, dim_y, counter)
        df = pd.DataFrame([],
                          index=date_range)

        for cs in tr.cs_ids:
            df[cs] = ev_env.cs_power[cs, :]*60/ev_env.timescale

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
        df_total_power[tr.id] = df['total']

        plt.step(df.index, df['total'], 'darkgreen',
                 where='post', linestyle='--')
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
        plt.ylabel(f'Power (kW)')
        plt.xlim([ev_env.sim_starting_date, ev_env.sim_date])
        plt.xticks(ticks=date_range_print,
                   labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print], rotation=45)

        if len(tr.cs_ids) < 3:
            plt.legend([f'CS {i}' for i in tr.cs_ids] +
                       ['Total Power (kW)'])
        plt.grid(True, which='minor', axis='both')
        counter += 1

    if len(ev_env.transformers) < 10:
        plt.tight_layout()
        fig_name = f'plots/{ev_env.sim_name}/Transformer_Aggregated_Power.png'
        plt.savefig(fig_name, format='png',
                    dpi=60, bbox_inches='tight')
    else:
        # clear plt canvas
        plt.close('all')

    # Plot the total power of the CPO
    plt.figure(figsize=(20, 17))

    # create 2 dfs, one for positive power and one for negative
    df_pos = df_total_power.copy()
    df_pos[df_pos < 0] = 0
    df_neg = df_total_power.copy()
    df_neg[df_neg > 0] = 0
    colors = plt.cm.gist_earth(np.linspace(0.1, 0.8, len(ev_env.transformers)))

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

    df_total_power['total'] = df_total_power.sum(axis=1)
    # print(df_total_power)

    plt.step(df_total_power.index, df_total_power['total'], 'darkgreen',
             where='post', linestyle='--')

    plt.step(df_total_power.index, ev_env.power_setpoints, 'r--', where='post',)

    if ev_env.load_from_replay_path is not None:
        plt.step(df_total_power.index, ev_env.replay.ev_load_potential,
                 'b--', where='post', alpha=0.4,)
    else:
        plt.step(df_total_power.index, ev_env.current_power_setpoints,
                 'b--', where='post', alpha=0.4,)

    plt.stackplot(df_neg.index, df_neg.values.T,
                  interpolate=True,
                  step='post',
                  colors=colors,
                  alpha=0.7,
                  linestyle='--')

    plt.plot([ev_env.sim_starting_date, ev_env.sim_date], [0, 0], 'black')

    # for cs in tr.cs_ids:
    #     plt.step(df.index, df[cs], 'white', where='post', linestyle='--')
    plt.title(f'Setpoint Tracker')
    plt.xlabel(f'Time')
    plt.ylabel(f'Power (kW)')
    plt.xlim([ev_env.sim_starting_date, ev_env.sim_date])
    plt.xticks(ticks=date_range_print,
               labels=[f'{d.hour:2d}:{d.minute:02d}' for d in date_range_print], rotation=45)
    if len(ev_env.transformers) < 10:
        plt.legend([f'Tr {i}' for i in range(len(ev_env.transformers))] +
                   ['Total Power (kW)']+[f'Power Setpoint (kW)']+['EV Unsteered Load Potential (kW)'])
    plt.grid(True, which='minor', axis='both')

    plt.tight_layout()
    # plt.show()
    fig_name = f'plots/{ev_env.sim_name}/Total_Aggregated_Power.png'
    plt.savefig(fig_name, format='png',
                dpi=60, bbox_inches='tight')

    # plot prices
    # plt.figure(figsize=(20, 17))
    # plt.plot(ev_env.charge_prices[0,:], label='Charge prices (€/kW))')
    # plt.plot(ev_env.discharge_prices[0,:], label='Discharge prices (€/kW))')
    # plt.legend()
    # plt.grid(True, which='minor', axis='both')
    # plt.tight_layout()
    # fig_name = f'plots/{ev_env.sim_name}/Prices.png'
    # plt.savefig(fig_name, format='png',
    #             dpi=60, bbox_inches='tight')

    plt.close('all')


def get_statistics(ev_env):
    total_ev_served = np.array(
        [cs.total_evs_served for cs in ev_env.charging_stations]).sum()
    total_profits = np.array(
        [cs.total_profits for cs in ev_env.charging_stations]).sum()
    total_energy_charged = np.array(
        [cs.total_energy_charged for cs in ev_env.charging_stations]).sum()
    total_energy_discharged = np.array(
        [cs.total_energy_discharged for cs in ev_env.charging_stations]).sum()
    average_user_satisfaction = np.average(np.array(
        [cs.get_avg_user_satisfaction() for cs in ev_env.charging_stations]))
    
    tracking_error = 0
    power_tracker_violation = 0
    for t in range(ev_env.simulation_length):
        tracking_error += (min(ev_env.power_setpoints[t], ev_env.charge_power_potential[t]) -
                      ev_env.current_power_setpoints[t])**2
        if ev_env.current_power_setpoints[t] > ev_env.power_setpoints[t]:
            power_tracker_violation += ev_env.current_power_setpoints[t] - \
                ev_env.power_setpoints[t]

    # find the final battery capacity of evs    
    if ev_env.load_from_replay_path is not None and len(ev_env.EVs) > 0:        
        energy_user_satisfaction = 0
        for i, ev in enumerate(ev_env.EVs):
            e_actual = ev.current_capacity
            e_max = ev_env.replay.EVs[i].current_capacity            
            # print(f'EV {i} actual: {e_actual:.2f} kWh, max: {e_max:.2f} kWh')
            energy_user_satisfaction += e_actual / e_max * 100

        energy_user_satisfaction /= len(ev_env.EVs)
    else:
        energy_user_satisfaction = 100

    stats = {'total_ev_served': total_ev_served,
             'total_profits': total_profits,
             'total_energy_charged': total_energy_charged,
             'total_energy_discharged': total_energy_discharged,
             'average_user_satisfaction': average_user_satisfaction,
             'power_tracker_violation': power_tracker_violation,
             'tracking_error': tracking_error,
             'energy_user_satisfaction': energy_user_satisfaction,
             }

    if ev_env.replay is not None:
        stats['opt_profits'] = ev_env.replay.stats["total_profits"]
        stats['opt_tracking_error'] = ev_env.replay.stats["tracking_error"]
        stats['opt_power_tracker_violation'] = ev_env.replay.stats["power_tracker_violation"]
        stats['opt_energy_user_satisfaction'] = ev_env.replay.stats["energy_user_satisfaction"]

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
    # tracking_error = ((ev_env.current_power_setpoints -
    #                   ev_env.power_setpoints)**2).sum()
    tracking_error = 0
    power_tracker_violation = 0
    for t in range(ev_env.simulation_length):
        tracking_error += (min(ev_env.power_setpoints[t], ev_env.charge_power_potential[t]) -
                      ev_env.current_power_setpoints[t])**2
        if ev_env.current_power_setpoints[t] > ev_env.power_setpoints[t]:
            power_tracker_violation += ev_env.current_power_setpoints[t] - \
                ev_env.power_setpoints[t]

    # find the final battery capacity of evs    
    if ev_env.load_from_replay_path is not None and len(ev_env.EVs) > 0:
        energy_user_satisfaction = 0
        for i, ev in enumerate(ev_env.EVs):
            e_actual = ev.current_capacity
            e_max = ev_env.replay.EVs[i].current_capacity
            # print(f'EV {i} actual: {e_actual:.2f} kWh, max: {e_max:.2f} kWh')
            energy_user_satisfaction += e_actual / e_max * 100

        energy_user_satisfaction /= len(ev_env.EVs)
    else:
        energy_user_satisfaction = 100

    print("\n\n==============================================================")
    print("Simulation statistics:")
    for cs in ev_env.charging_stations:
        print(cs)
    print(
        f'  - Total EVs spawned: {ev_env.total_evs_spawned} |  served: {total_ev_served}')
    print(f'  - Total profits: {total_profits*100:.2f} €')
    print(
        f'  - Average user satisfaction: {average_user_satisfaction*100:.2f} %')

    print(
        f'  - Total energy charged: {toal_energy_charged:.1f} | discharged: {total_energy_discharged:.1f} kWh')    
    print(
        f'  - Power Tracking squared error: {tracking_error:.2f}, Power Violation: {power_tracker_violation:.2f} kW')
    print(f'  - Energy user satisfaction: {energy_user_satisfaction:.2f} %\n')

    print("==============================================================\n\n")


def visualize_step(ev_env):
    '''Renders the current state of the environment in the terminal'''

    print(f"\n Step: {ev_env.current_step}" +
          f" | {str(ev_env.sim_date.weekday())} {ev_env.sim_date.hour:2d}:{ev_env.sim_date.minute:2d}:{ev_env.sim_date.second:2d} |" +
          f" \tEVs +{ev_env.current_ev_arrived} / -{ev_env.current_ev_departed}" +
          f" | Total: {ev_env.current_evs_parked} / {ev_env.number_of_ports}")

    if ev_env.verbose:
        for cs in ev_env.charging_stations:
            print(f'  - Charging station {cs.id}:')
            print(f'\t Power: {cs.current_power_output:4.1f} kW |' +
                  f' \u2197 {ev_env.charge_prices[cs.id, ev_env.current_step -1 ]:4.2f} €/kW ' +
                  f' \u2198 {ev_env.discharge_prices[cs.id, ev_env.current_step - 1]:4.2f} €/kW |' +
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

        # print current current power setpoint
        print(f'  - Power setpoint: {ev_env.current_power_setpoints[ev_env.current_step - 1]:.1f} Actual/' +
              f' {ev_env.power_setpoints[ev_env.current_step - 1]:.1f} Setpoint/'
              f' {ev_env.charge_power_potential[ev_env.current_step - 1]:.1f} Potential in kWh')


def spawn_EV(ev_env, cs_id):
    '''Spawns EVs based on the spawn rate'''
    time = ev_env.sim_date
    day = time.weekday()
    hour = time.hour
    minute = time.minute
    # Divide by 15 because the spawn rate is in 15 minute intervals
    i = hour*4 + minute//15

    scenario = ev_env.scenario.split("_")[0]

    if day < 5:
        tau = ev_env.df_arrival_week[scenario].iloc[i]
        multiplier = 5
    else:
        if scenario == "workplace":
            return None
        tau = ev_env.df_arrival_weekend[scenario].iloc[i]
        if day == 5:
            multiplier = 4
        else:
            multiplier = 3

    if np.random.rand(1)*100 < tau * multiplier * (ev_env.timescale/60) * 2:

        required_energy = ev_env.df_energy_demand[scenario].iloc[np.random.randint(
            0, 100, size=1)].values[0]  # kWh

        if required_energy < 5:
            required_energy = 5

        if ev_env.heterogeneous_specs:
            battery_capacity = np.random.randint(40, 80)  # kWh
        else:
            battery_capacity = 50

        if battery_capacity < required_energy:
            initial_battery_capacity = 0.05*battery_capacity
        else:
            initial_battery_capacity = battery_capacity - required_energy

        time_of_stay = np.random.choice(
            np.arange(0, 48, 1), 1, p=ev_env.time_of_connection_vs_hour[hour, :])/2
        time_of_stay = time_of_stay[0] * 60 / ev_env.timescale + 1

        # Alternative method for time of stay
        # time_of_stay = ev_env.df_connection_time[scenario].iloc[np.random.randint(
        #     0, 100, size=1)].values[0] * 60 / ev_env.timescale

        # if time_of_stay + ev_env.current_step + 1 > ev_env.simulation_length:
        # TODO think about "empty_ports_at_end_of_simulation"

        if ev_env.heterogeneous_specs:
            return EV(id=None,
                      location=cs_id,
                      battery_capacity_at_arrival=initial_battery_capacity,
                      max_ac_charge_power=np.random.rand.choice(
                          [11, 22], [0.2, 0.8]),
                      max_dc_charge_power=np.random.randint(50, 150),
                      max_discharge_power=-np.random.randint(3, 15),
                      discharge_efficiency=np.round(1 -
                                                    (np.random.rand()+0.00001)/20, 3),  # [0.95-1]
                      transition_soc=np.round(0.9 - \
                                              (np.random.rand()+0.00001)/5, 3),  # [0.7-0.9]
                      battery_capacity=battery_capacity,
                      time_of_arrival=ev_env.current_step+1,
                      earlier_time_of_departure=int(
                          time_of_stay + ev_env.current_step + 3),
                      ev_phases=np.random.choice([1, 2, 3], p=[0.2, 0.1, 0.7]),
                      timescale=ev_env.timescale,
                      simulation_length=ev_env.simulation_length,)
        else:
            return EV(id=None,
                      location=cs_id,
                      battery_capacity_at_arrival=initial_battery_capacity,
                      battery_capacity=battery_capacity,
                      time_of_arrival=ev_env.current_step+1,
                      earlier_time_of_departure=int(
                          time_of_stay + ev_env.current_step + 3),
                      ev_phases=3,
                      transition_soc=0.999,
                      timescale=ev_env.timescale,
                      simulation_length=ev_env.simulation_length,)
