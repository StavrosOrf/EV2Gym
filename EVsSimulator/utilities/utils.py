# This file contains support functions for the EV City environment.

import numpy as np
import matplotlib.pyplot as plt
import math
from EVsSimulator.models.ev import EV


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
    # get transformer overload from ev_env.tr_overload
    total_transformer_overload = np.array(ev_env.tr_overload).sum()

    tracking_error = 0
    power_tracker_violation = 0
    for t in range(ev_env.simulation_length):
        tracking_error += (min(ev_env.power_setpoints[t], ev_env.charge_power_potential[t]) -
                           ev_env.current_power_setpoints[t])**2
        if ev_env.current_power_setpoints[t] > ev_env.power_setpoints[t]:
            power_tracker_violation += ev_env.current_power_setpoints[t] - \
                ev_env.power_setpoints[t]

    # ev_percentage_charged = []
    # for i, ev in enumerate(ev_env.EVs):
    #     ev_percentage_charged.append((ev.battery_capacity-ev.current_capacity)/(ev.battery_capacity-ev.battery_capacity_at_arrival))

    # find the final battery capacity of evs
    if ev_env.eval_mode != "unstirred" and len(ev_env.EVs) > 0 \
            and ev_env.replay is not None:

        if ev_env.replay.unstirred_EVs is None:
            energy_user_satisfaction = -10000000
        else:
            energy_user_satisfaction = 0
            for i, ev in enumerate(ev_env.EVs):
                e_actual = ev.current_capacity
                e_max = ev_env.replay.unstirred_EVs[i].current_capacity
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
             'total_transformer_overload': total_transformer_overload,
             #  'ev_percentage_charged': ev_percentage_charged,
             }
    if ev_env.eval_mode != "optimal" and ev_env.replay is not None:
        if ev_env.replay.optimal_stats is not None:
            stats['opt_profits'] = ev_env.replay.optimal_stats["total_profits"]
            stats['opt_tracking_error'] = ev_env.replay.optimal_stats["tracking_error"]
            stats['opt_power_tracker_violation'] = ev_env.replay.optimal_stats["power_tracker_violation"]
            stats['opt_energy_user_satisfaction'] = ev_env.replay.optimal_stats["energy_user_satisfaction"]
            stats['opt_total_energy_charged'] = ev_env.replay.optimal_stats["total_energy_charged"]

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
    total_transformer_overload = np.array(ev_env.tr_overload).sum()

    tracking_error = 0
    power_tracker_violation = 0
    for t in range(ev_env.simulation_length):
        tracking_error += (min(ev_env.power_setpoints[t], ev_env.charge_power_potential[t]) -
                           ev_env.current_power_setpoints[t])**2
        if ev_env.current_power_setpoints[t] > ev_env.power_setpoints[t]:
            power_tracker_violation += ev_env.current_power_setpoints[t] - \
                ev_env.power_setpoints[t]

    # find the final battery capacity of evs
    if ev_env.eval_mode != "unstirred" and len(ev_env.EVs) > 0 \
            and ev_env.replay is not None:

        if ev_env.replay.unstirred_EVs is None:
            energy_user_satisfaction = -10000000
        else:
            energy_user_satisfaction = 0
            for i, ev in enumerate(ev_env.EVs):
                e_actual = ev.current_capacity
                e_max = ev_env.replay.unstirred_EVs[i].current_capacity
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
    print(f'  - Energy user satisfaction: {energy_user_satisfaction:.2f} %')
    print(
        f'  - Total transformer overload: {total_transformer_overload:.2f} Amperes / DT \n')
    print("==============================================================\n\n")


def spawn_EV(ev_env, cs_id):
    '''Spawns EVs based on the spawn rate'''
    time = ev_env.sim_date
    day = time.weekday()
    hour = time.hour
    minute = time.minute
    # Divide by 15 because the spawn rate is in 15 minute intervals (in the csv file)
    i = hour*4 + minute//15

    scenario = ev_env.scenario

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
            initial_battery_capacity = 0.05 * battery_capacity
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
                      transition_soc=0.9999,
                      timescale=ev_env.timescale,
                      simulation_length=ev_env.simulation_length,)


def CalculateChargePowerPotential(env):
    '''
    This function calculates the total charge power potential of all currently parked EVs for the current time step     
    '''

    power_potential = 0
    for cs in env.charging_stations:
        cs_power_potential = 0
        for port in range(cs.n_ports):
            ev = cs.evs_connected[port]
            if ev is not None:
                if ev.time_of_arrival-1 != env.current_step:
                    phases = min(cs.phases, ev.ev_phases)
                    ev_current = ev.max_ac_charge_power * \
                        1000/(math.sqrt(phases)*cs.voltage)
                    current = min(cs.max_charge_current, ev_current)
                    cs_power_potential += math.sqrt(phases) * \
                        cs.voltage*current/1000

        max_cs_power = math.sqrt(cs.phases) * \
            cs.voltage*cs.max_charge_current/1000
        min_cs_power = math.sqrt(cs.phases) * \
            cs.voltage*cs.min_charge_current/1000

        if cs_power_potential > max_cs_power:
            power_potential += max_cs_power
        elif cs_power_potential < min_cs_power:
            power_potential += 0
        else:
            power_potential += cs_power_potential

    return power_potential
