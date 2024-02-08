# This file contains support functions for the EV City environment.

import numpy as np
import matplotlib.pyplot as plt
import math
import datetime

from EVsSimulator.models.ev import EV


def get_statistics(env):
    total_ev_served = np.array(
        [cs.total_evs_served for cs in env.charging_stations]).sum()
    total_profits = np.array(
        [cs.total_profits for cs in env.charging_stations]).sum()
    total_energy_charged = np.array(
        [cs.total_energy_charged for cs in env.charging_stations]).sum()
    total_energy_discharged = np.array(
        [cs.total_energy_discharged for cs in env.charging_stations]).sum()
    average_user_satisfaction = np.average(np.array(
        [cs.get_avg_user_satisfaction() for cs in env.charging_stations]))
    # get transformer overload from env.tr_overload
    total_transformer_overload = np.array(env.tr_overload).sum()

    tracking_error = 0
    power_tracker_violation = 0
    for t in range(env.simulation_length):
        tracking_error += (min(env.power_setpoints[t], env.charge_power_potential[t]) -
                           env.current_power_usage[t])**2
        if env.current_power_usage[t] > env.power_setpoints[t]:
            power_tracker_violation += env.current_power_usage[t] - \
                env.power_setpoints[t]

    # ev_percentage_charged = []
    # for i, ev in enumerate(env.EVs):
    #     ev_percentage_charged.append((ev.battery_capacity-ev.current_capacity)/(ev.battery_capacity-ev.battery_capacity_at_arrival))

    # find the final battery capacity of evs
    if env.eval_mode != "unstirred" and len(env.EVs) > 0 \
            and env.replay is not None:

        if env.replay.unstirred_EVs is None:
            energy_user_satisfaction = -10000000
        else:
            energy_user_satisfaction = 0
            for i, ev in enumerate(env.EVs):
                e_actual = ev.current_capacity
                e_max = env.replay.unstirred_EVs[i].current_capacity
                # print(f'EV {i} actual: {e_actual:.2f} kWh, max: {e_max:.2f} kWh')
                energy_user_satisfaction += e_actual / e_max * 100

            energy_user_satisfaction /= len(env.EVs)
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
    if env.eval_mode != "optimal" and env.replay is not None:
        if env.replay.optimal_stats is not None:
            stats['opt_profits'] = env.replay.optimal_stats["total_profits"]
            stats['opt_tracking_error'] = env.replay.optimal_stats["tracking_error"]
            stats['opt_power_tracker_violation'] = env.replay.optimal_stats["power_tracker_violation"]
            stats['opt_energy_user_satisfaction'] = env.replay.optimal_stats["energy_user_satisfaction"]
            stats['opt_total_energy_charged'] = env.replay.optimal_stats["total_energy_charged"]

    return stats


def print_statistics(env):
    '''Prints the statistics of the simulation'''
    total_ev_served = np.array(
        [cs.total_evs_served for cs in env.charging_stations]).sum()
    total_profits = np.array(
        [cs.total_profits for cs in env.charging_stations]).sum()
    toal_energy_charged = np.array(
        [cs.total_energy_charged for cs in env.charging_stations]).sum()
    total_energy_discharged = np.array(
        [cs.total_energy_discharged for cs in env.charging_stations]).sum()
    average_user_satisfaction = np.average(np.array(
        [cs.get_avg_user_satisfaction() for cs in env.charging_stations]))
    total_transformer_overload = np.array(env.tr_overload).sum()

    tracking_error = 0
    power_tracker_violation = 0
    for t in range(env.simulation_length):
        tracking_error += (min(env.power_setpoints[t], env.charge_power_potential[t]) -
                           env.current_power_usage[t])**2
        if env.current_power_usage[t] > env.power_setpoints[t]:
            power_tracker_violation += env.current_power_usage[t] - \
                env.power_setpoints[t]

    # find the final battery capacity of evs
    if env.eval_mode != "unstirred" and len(env.EVs) > 0 \
            and env.replay is not None:

        if env.replay.unstirred_EVs is None:
            energy_user_satisfaction = -10000000
        else:
            energy_user_satisfaction = 0
            for i, ev in enumerate(env.EVs):
                e_actual = ev.current_capacity
                e_max = env.replay.unstirred_EVs[i].current_capacity
                # print(f'EV {i} actual: {e_actual:.2f} kWh, max: {e_max:.2f} kWh')
                energy_user_satisfaction += e_actual / e_max * 100

            energy_user_satisfaction /= len(env.EVs)
    else:
        energy_user_satisfaction = 100

    print("\n\n==============================================================")
    print("Simulation statistics:")
    for cs in env.charging_stations:
        print(cs)
    print(
        f'  - Total EVs spawned: {env.total_evs_spawned} |  served: {total_ev_served}')
    print(f'  - Total profits: {total_profits:.2f} €')
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


def spawn_single_EV(env, scenario, cs_id, port, hour, step, min_time_of_stay_steps) -> EV:
    '''
    This function spawns a single EV and returns it
    '''

    required_energy = env.df_energy_demand[scenario].iloc[np.random.randint(
        0, 100, size=1)].values[0]  # kWh

    if required_energy < 10:
        required_energy = np.random.randint(10, 15)

    if env.heterogeneous_specs:
        battery_capacity = np.random.randint(40, 80)  # kWh
    else:
        battery_capacity = 50

    if battery_capacity < required_energy:
        initial_battery_capacity = 0.05 * battery_capacity
    else:
        initial_battery_capacity = battery_capacity - required_energy

    # time_of_stay = np.random.choice(
    #     np.arange(0, 48, 1), 1, p=env.time_of_connection_vs_hour[hour, :])/2
    # time_of_stay = time_of_stay[0] * 60 / env.timescale + 1

    # Alternative method for time of stay
    time_of_stay = env.df_connection_time[scenario].iloc[np.random.randint(
        0, 100, size=1)].values[0] * 60 / env.timescale + 1
    
    if time_of_stay < min_time_of_stay_steps:
        time_of_stay = min_time_of_stay_steps

    if env.empty_ports_at_end_of_simulation:
        if time_of_stay + step + 3 > env.simulation_length:
            time_of_stay = env.simulation_length - step - 4
            
        assert time_of_stay >= min_time_of_stay_steps, "Time of stay is less than min_time_of_stay_steps"

    # "empty_ports_at_end_of_simulation"

    if env.heterogeneous_specs:
        return EV(id=port,
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
                  time_of_arrival=step+1,
                  time_of_departure=int(
                      time_of_stay + step + 3),
                  ev_phases=np.random.choice([1, 2, 3], p=[0.2, 0.1, 0.7]),
                  timescale=env.timescale,
                  )
    else:
        return EV(id=port,
                  location=cs_id,
                  battery_capacity_at_arrival=initial_battery_capacity,
                  battery_capacity=battery_capacity,
                  time_of_arrival=step+1,
                  time_of_departure=int(
                      time_of_stay + step + 3),
                  ev_phases=3,
                  transition_soc=0.9999,
                  timescale=env.timescale,
                  )


def EV_spawner(env) -> list[EV]:
    '''
    This function spawns all the EVs of the current simulation and returns the list of EVs

    Returns:
        EVs: list of EVs
    '''

    ev_list = []

    occupancy_list = np.zeros((env.number_of_ports, env.simulation_length))

    arrival_probabilities = np.random.rand(
        env.number_of_ports, env.simulation_length)

    scenario = env.scenario
    user_spawn_multiplier = env.config["spawn_multiplier"]
    time = env.sim_date
    
    #Define minimum time of stay duration so that an EV can fully charge
    min_time_of_stay = 120 #minutes
    min_time_of_stay_steps = min_time_of_stay // env.timescale

    for t in range(env.simulation_length-min_time_of_stay_steps-1):
        day = time.weekday()
        hour = time.hour
        minute = time.minute
        # Divide by 15 because the spawn rate is in 15 minute intervals (in the csv file)
        i = hour*4 + minute//15

        if day < 5:
            tau = env.df_arrival_week[scenario].iloc[i]
            multiplier = 10
        else:
            if scenario == "workplace":
                multiplier = 0
                tau = 1
            else:
                tau = env.df_arrival_weekend[scenario].iloc[i]
            if day == 5:
                multiplier = 8
            else:
                multiplier = 6

        counter = 0
        for cs in env.charging_stations:
            for port in range(cs.n_ports):
                # if port is empty 
                if occupancy_list[counter, t] == 0:
                    # and there is an EV arriving
                    if arrival_probabilities[counter, t]*100 < tau * multiplier * (env.timescale/60) * user_spawn_multiplier:
                        ev = spawn_single_EV(
                            env,
                            scenario,
                            cs.id,
                            port,
                            hour,
                            t,
                            min_time_of_stay_steps=min_time_of_stay_steps)

                        ev_list.append(ev)
                        occupancy_list[counter, t +
                                       1:ev.time_of_departure] = 1

                counter += 1
        # step the time
        time = time + datetime.timedelta(minutes=env.timescale)

    return ev_list


def create_power_setpoint_one_step(env) -> float:
    '''
    This function creates the power setpoint for the current time step
    '''

    # get prices
    prices = env.discharge_prices[0]

    power_potential = 0
    for cs in env.charging_stations:
        cs_power_potential = 0
        for port in range(cs.n_ports):
            ev = cs.evs_connected[port]
            if ev is not None:
                if ev.get_soc() < 1 and ev.time_of_departure > env.current_step:
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

    # normalize prices
    prices = (prices / np.max(prices))/2

    return power_potential * (1 - prices[env.current_step])


def calculate_charge_power_potential(env) -> float:
    '''
    This function calculates the total charge power potential of all currently parked EVs for the current time step     
    '''

    power_potential = 0
    for cs in env.charging_stations:
        cs_power_potential = 0
        for port in range(cs.n_ports):
            ev = cs.evs_connected[port]
            if ev is not None:
                if ev.get_soc() < 1 and ev.time_of_departure > env.current_step:
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
