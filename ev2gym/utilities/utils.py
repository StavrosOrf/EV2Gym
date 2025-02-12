# This file contains support functions for the EV City environment.

import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
from typing import List, Dict

from ev2gym.models.ev import EV


def get_statistics(env) -> Dict:
    total_ev_served = np.array(
        [cs.total_evs_served for cs in env.charging_stations]).sum()
    total_profits = np.array(
        [cs.total_profits for cs in env.charging_stations]).sum()
    total_energy_charged = np.array(
        [cs.total_energy_charged for cs in env.charging_stations]).sum()
    total_energy_discharged = np.array(
        [cs.total_energy_discharged for cs in env.charging_stations]).sum()
    average_user_satisfaction = np.array(
        [cs.get_avg_user_satisfaction() for cs in env.charging_stations
         if cs.total_evs_served > 0]).mean()

    # get transformer overload from env.tr_overload
    total_transformer_overload = np.array(env.tr_overload).sum()

    tracking_error = 0
    energy_tracking_error = 0
    power_tracker_violation = 0
    for t in range(env.simulation_length):
        # tracking_error += (min(env.power_setpoints[t], env.charge_power_potential[t]) -
        #                    env.current_power_usage[t])**2
        # energy_tracking_error += abs(min(env.power_setpoints[t], env.charge_power_potential[t]) -
        #                              env.current_power_usage[t])

        tracking_error += (env.power_setpoints[t] -
                           env.current_power_usage[t])**2
        energy_tracking_error += abs(env.power_setpoints[t] -
                                     env.current_power_usage[t])

        if env.current_power_usage[t] > env.power_setpoints[t]:
            power_tracker_violation += env.current_power_usage[t] - \
                env.power_setpoints[t]

    energy_tracking_error *= env.timescale / 60

    # calculate total batery degradation
    battery_degradation = np.array(
        [np.array(ev.get_battery_degradation()).reshape(-1) for ev in env.EVs])
    if len(battery_degradation) == 0:
        battery_degradation = np.zeros((1, 2))
    battery_degradation_calendar = battery_degradation[:, 0].sum()
    battery_degradation_cycling = battery_degradation[:, 1].sum()
    battery_degradation = battery_degradation.sum()

    total_steps_min_emergency_battery_capacity_violation = 0
    energy_user_satisfaction = np.zeros((len(env.EVs)))
    for i, ev in enumerate(env.EVs):
        e_actual = ev.current_capacity
        e_max = ev.max_energy_AFAP
        energy_user_satisfaction[i] = (e_actual / e_max) * 100
        total_steps_min_emergency_battery_capacity_violation += ev.min_emergency_battery_capacity_metric

    stats = {'total_ev_served': total_ev_served,
             'total_profits': total_profits,
             'total_energy_charged': total_energy_charged,
             'total_energy_discharged': total_energy_discharged,
             'average_user_satisfaction': average_user_satisfaction,
             'power_tracker_violation': power_tracker_violation,
             'tracking_error': tracking_error,
             'energy_tracking_error': energy_tracking_error,
             'energy_user_satisfaction': np.mean(energy_user_satisfaction),
             'std_energy_user_satisfaction': np.std(energy_user_satisfaction),
             'min_energy_user_satisfaction': np.min(energy_user_satisfaction),
             'total_steps_min_emergency_battery_capacity_violation': total_steps_min_emergency_battery_capacity_violation,
             'total_transformer_overload': total_transformer_overload,
             'battery_degradation': battery_degradation,
             'battery_degradation_calendar': battery_degradation_calendar,
             'battery_degradation_cycling': battery_degradation_cycling,
             'total_reward': env.total_reward,
             }

    if env.eval_mode != "optimal" and env.replay is not None:
        if env.replay.optimal_stats is not None:
            stats['opt_profits'] = env.replay.optimal_stats["total_profits"]
            stats['opt_tracking_error'] = env.replay.optimal_stats["tracking_error"]
            stats['opt_actual_tracking_error'] = env.replay.optimal_stats["energy_tracking_error"]
            stats['opt_power_tracker_violation'] = env.replay.optimal_stats["power_tracker_violation"]
            stats['opt_energy_user_satisfaction'] = env.replay.optimal_stats["energy_user_satisfaction"]
            stats['opt_total_energy_charged'] = env.replay.optimal_stats["total_energy_charged"]

    return stats


def print_statistics(env) -> None:

    assert env.stats is not None, "No statistics available. Run the simulation first!"

    stats = env.stats

    total_ev_served = stats['total_ev_served']
    total_profits = stats['total_profits']
    total_energy_charged = stats['total_energy_charged']
    total_energy_discharged = stats['total_energy_discharged']
    average_user_satisfaction = stats['average_user_satisfaction']
    total_transformer_overload = stats['total_transformer_overload']
    tracking_error = stats['tracking_error']
    energy_tracking_error = stats['energy_tracking_error']
    power_tracker_violation = stats['power_tracker_violation']
    energy_user_satisfaction = stats['energy_user_satisfaction']
    std_energy_user_satisfaction = stats['std_energy_user_satisfaction']
    min_energy_user_satisfaction = stats['min_energy_user_satisfaction']

    total_transformer_overload = stats['total_transformer_overload']
    battery_degradation = stats['battery_degradation']
    battery_degradation_calendar = stats['battery_degradation_calendar']
    battery_degradation_cycling = stats['battery_degradation_cycling']

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
        f'  - Total energy charged: {total_energy_charged:.1f} | discharged: {total_energy_discharged:.1f} kWh')
    print(
        f'  - Power Tracking squared error: {tracking_error:.2f}, Power Violation: {power_tracker_violation:.2f} kW')
    print(f' - Actual Energy Tracking error: {energy_tracking_error:.2f} kW')
    print(
        f'  - Mean energy user satisfaction: {energy_user_satisfaction:.2f} % | Min: {min_energy_user_satisfaction:.2f} %')
    print(
        f'  - Std Energy user satisfaction: {std_energy_user_satisfaction:.2f} %')
    print(
        f'  - Total Battery degradation: {battery_degradation:.5f}% | Calendar: {battery_degradation_calendar:.5f}%, Cycling: {battery_degradation_cycling:.5f}%')
    print(
        f'  - Total transformer overload: {total_transformer_overload:.2f} kWh \n')

    print("==============================================================\n\n")


def spawn_single_EV(env,
                    scenario,
                    cs_id,
                    port,
                    hour,
                    minute,
                    step,
                    min_time_of_stay_steps
                    ) -> EV:
    '''
    This function spawns a single EV and returns it
    '''

    # required energy independent of time of arrival
    # required_energy = env.df_energy_demand[scenario].iloc[np.random.randint(
    #     0, 100, size=1)].values[0]  # kWh

    # roound minute to 30 or 0
    if minute < 30:
        minute = 0
    else:
        minute = 30

    # required energy dependent on time of arrival
    arrival_time = f'{hour:02d}:{minute:02d}'

    required_energy_mean = env.df_req_energy[
        (env.df_req_energy['Arrival Time'] == arrival_time)
    ][scenario].values[0]

    required_energy = np.random.normal(
        required_energy_mean, 0.5*required_energy_mean)  # kWh

    if required_energy < 5:
        required_energy = np.random.randint(5, 10)

    if env.heterogeneous_specs:
        sampled_ev = np.random.choice(
            list(env.ev_specs.keys()), p=env.normalized_ev_registrations)
        battery_capacity = env.ev_specs[sampled_ev]["battery_capacity"]
    else:
        battery_capacity = env.config["ev"]["battery_capacity"]

    if battery_capacity < required_energy:
        initial_battery_capacity = np.random.randint(1, battery_capacity)
    else:
        initial_battery_capacity = battery_capacity - required_energy

    if initial_battery_capacity > env.config["ev"]['desired_capacity']:
        initial_battery_capacity = np.random.randint(1, battery_capacity)

    if initial_battery_capacity < env.config["ev"]['min_battery_capacity'] and battery_capacity > 2*env.config["ev"]['min_battery_capacity']:
        initial_battery_capacity = env.config["ev"]['min_battery_capacity']

    # time of stay dependent on time of arrival
    time_of_stay_mean = env.df_time_of_stay_vs_arrival[(
        env.df_time_of_stay_vs_arrival['Arrival Time'] == arrival_time)
    ][scenario].values[0]

    time_of_stay = np.random.normal(
        time_of_stay_mean, 0.2*time_of_stay_mean)  # hours

    # turn from hours to steps
    time_of_stay = time_of_stay * 60 / env.timescale + 1

    # Alternative method for time of stay based on 10.000 charging sessions
    # time_of_stay = np.random.choice(
    #     np.arange(0, 48, 1), 1, p=env.time_of_connection_vs_hour[hour, :])/2
    # time_of_stay = time_of_stay[0] * 60 / env.timescale + 1

    # Alternative method for time of stay without taking into account the hour
    # time_of_stay = env.df_connection_time[scenario].iloc[np.random.randint(
    #     0, 100, size=1)].values[0] * 60 / env.timescale + 1

    if time_of_stay < min_time_of_stay_steps:
        time_of_stay = min_time_of_stay_steps

    if env.empty_ports_at_end_of_simulation:
        if time_of_stay + step + 4 >= env.simulation_length:
            return None
            time_of_stay = env.simulation_length - step - 4 - 2

    if "transition_soc_multiplier" in env.config["ev"]:
        transition_soc_multiplier = env.config["ev"]["transition_soc_multiplier"]
    else:
        transition_soc_multiplier = 1

    min_emergency_battery_capacity = env.config["ev"]["min_emergency_battery_capacity"]

    if min_emergency_battery_capacity > battery_capacity:
        min_emergency_battery_capacity = 0.7*battery_capacity

    if env.heterogeneous_specs:

        # get charge efficiency from env.ev_specs dict
        # if there is key charge_efficiency_v
        if "3ph_ch_efficiency" in env.ev_specs[sampled_ev]:
            charge_efficiency_v = env.ev_specs[sampled_ev]["3ph_ch_efficiency"]
            current_levels = env.ev_specs[sampled_ev]["ch_current"]
            assert len(charge_efficiency_v) == len(current_levels)
            assert all([0 <= x <= 100 for x in charge_efficiency_v])

            # make a dict with charge leves kay and charge efficiency value
            charge_efficiency = dict(zip(current_levels, charge_efficiency_v))

            for i in range(0, 101):
                if i not in charge_efficiency or charge_efficiency[i] == 0:
                    nonzero_keys = [
                        k for k, v in charge_efficiency.items() if v != 0]
                    if nonzero_keys:
                        closest = min(nonzero_keys, key=lambda x: abs(x - i))
                        charge_efficiency[i] = charge_efficiency[closest]

            discharge_efficiency = charge_efficiency.copy()

        else:
            charge_efficiency = np.round(1 -
                                         (np.random.rand()+0.00001)/20, 3)  # [0.95-1]
            discharge_efficiency = np.round(1 -
                                            (np.random.rand()+0.00001)/20, 3)  # [0.95-1]

        return EV(id=port,
                  location=cs_id,
                  battery_capacity_at_arrival=initial_battery_capacity,
                  max_ac_charge_power=env.ev_specs[sampled_ev]["max_ac_charge_power"],
                  max_dc_charge_power=env.ev_specs[sampled_ev]["max_dc_charge_power"],
                  max_discharge_power=-
                  env.ev_specs[sampled_ev]["max_dc_discharge_power"],
                  min_emergency_battery_capacity=min_emergency_battery_capacity,
                  charge_efficiency=charge_efficiency,
                  discharge_efficiency=discharge_efficiency,

                  transition_soc=np.round(0.9 -
                                          (np.random.rand()+0.00001)/5, 3),  # [0.7-0.9]
                  transition_soc_multiplier=transition_soc_multiplier,
                  battery_capacity=battery_capacity,
                  desired_capacity=env.config["ev"]['desired_capacity'] *
                  battery_capacity,
                  time_of_arrival=step+1,
                  time_of_departure=int(
                      time_of_stay + step + 3),
                  ev_phases=3,
                  timescale=env.timescale,
                  )
    else:
        return EV(id=port,
                  location=cs_id,
                  battery_capacity_at_arrival=initial_battery_capacity,
                  battery_capacity=battery_capacity,
                  desired_capacity=env.config["ev"]['desired_capacity'] *
                  battery_capacity,
                  min_emergency_battery_capacity=min_emergency_battery_capacity,
                  max_ac_charge_power=env.config["ev"]['max_ac_charge_power'],
                  min_ac_charge_power=env.config["ev"]['min_ac_charge_power'],
                  max_dc_charge_power=env.config["ev"]['max_dc_charge_power'],
                  max_discharge_power=env.config["ev"]['max_discharge_power'],
                  min_discharge_power=env.config["ev"]['min_discharge_power'],
                  time_of_arrival=step+1,
                  time_of_departure=int(
                      time_of_stay + step + 3),
                  ev_phases=env.config["ev"]['ev_phases'],
                  transition_soc=env.config["ev"]['transition_soc'],
                  transition_soc_multiplier=transition_soc_multiplier,
                  charge_efficiency=env.config["ev"]['charge_efficiency'],
                  discharge_efficiency=env.config["ev"]['discharge_efficiency'],
                  timescale=env.timescale,
                  )


def spawn_single_EV_GF(env,
                       day,
                       cs_id,
                       port,
                       hour,
                       minute,
                       step,
                       min_time_of_stay_steps
                       ) -> EV:
    '''
    This function spawns a single EV and returns it
    '''

    # round minute to 30 or 0
    if minute < 30:
        minute = 0
    else:
        minute = 30

    if day < 5:
        size = env.df_req_energy_weekday.shape[1]
        required_energy = 2 * \
            np.random.choice(size, 1, p=env.df_req_energy_weekday[hour])
        size = env.time_of_connection_vs_hour_weekday.shape[1]
        time_of_stay = 10 * \
            np.random.choice(
                size, 1, p=env.time_of_connection_vs_hour_weekday[hour])
    else:
        size = env.df_req_energy_weekend.shape[1]
        required_energy = 2 * \
            np.random.choice(size, 1, p=env.df_req_energy_weekend[hour])
        size = env.time_of_connection_vs_hour_weekend.shape[1]
        time_of_stay = 10 * \
            np.random.choice(
                size, 1, p=env.time_of_connection_vs_hour_weekend[hour])

    required_energy = float(required_energy)
    time_of_stay = float(time_of_stay)

    if required_energy < 2:
        required_energy = np.random.randint(5, 10)

    if env.heterogeneous_specs:
        sampled_ev = np.random.choice(
            list(env.ev_specs.keys()), p=env.normalized_ev_registrations)
        battery_capacity = env.ev_specs[sampled_ev]["battery_capacity"]
    else:
        battery_capacity = env.config["ev"]["battery_capacity"]

    if battery_capacity < required_energy:
        initial_battery_capacity = np.random.randint(1, battery_capacity)
    else:
        initial_battery_capacity = battery_capacity - required_energy

    if initial_battery_capacity > env.config["ev"]['desired_capacity']:
        initial_battery_capacity = np.random.randint(1, battery_capacity)

    if initial_battery_capacity < env.config["ev"]['min_battery_capacity'] and battery_capacity > 2*env.config["ev"]['min_battery_capacity']:
        initial_battery_capacity = env.config["ev"]['min_battery_capacity']

    # turn from minutes to steps
    time_of_stay = time_of_stay // env.timescale + 1

    if time_of_stay < min_time_of_stay_steps:
        time_of_stay = min_time_of_stay_steps

    if env.empty_ports_at_end_of_simulation:
        if time_of_stay + step + 4 >= env.simulation_length:
            return None
            time_of_stay = env.simulation_length - step - 4 - 2
    if initial_battery_capacity > battery_capacity:
        print(f"Initial battery capacity: {initial_battery_capacity}")
        print(f"Battery capacity: {battery_capacity}")
        raise ValueError(
            "Initial battery capacity cannot be higher than battery capacity!")

    if "transition_soc_multiplier" in env.config["ev"]:
        transition_soc_multiplier = env.config["ev"]["transition_soc_multiplier"]
    else:
        transition_soc_multiplier = 1

    if env.heterogeneous_specs:
        return EV(id=port,
                  location=cs_id,
                  battery_capacity_at_arrival=initial_battery_capacity,
                  max_ac_charge_power=env.ev_specs[sampled_ev]["max_ac_charge_power"],
                  max_dc_charge_power=env.ev_specs[sampled_ev]["max_dc_charge_power"],
                  max_discharge_power=-
                  env.ev_specs[sampled_ev]["max_dc_discharge_power"],
                  discharge_efficiency=np.round(1 -
                                                (np.random.rand()+0.00001)/20, 3),  # [0.95-1]
                  transition_soc=np.round(0.9 -
                                          (np.random.rand()+0.00001)/5, 3),  # [0.7-0.9]
                  transition_soc_multiplier=transition_soc_multiplier,
                  battery_capacity=battery_capacity,
                  desired_capacity=env.config["ev"]['desired_capacity'] *
                  battery_capacity,
                  time_of_arrival=step+1,
                  time_of_departure=int(
                      time_of_stay + step + 3),
                  ev_phases=3,
                  timescale=env.timescale,
                  )
    else:
        return EV(id=port,
                  location=cs_id,
                  battery_capacity_at_arrival=initial_battery_capacity,
                  battery_capacity=battery_capacity,
                  desired_capacity=env.config["ev"]['desired_capacity'] *
                  battery_capacity,
                  max_ac_charge_power=env.config["ev"]['max_ac_charge_power'],
                  min_ac_charge_power=env.config["ev"]['min_ac_charge_power'],
                  max_dc_charge_power=env.config["ev"]['max_dc_charge_power'],
                  max_discharge_power=env.config["ev"]['max_discharge_power'],
                  min_discharge_power=env.config["ev"]['min_discharge_power'],
                  time_of_arrival=step+1,
                  time_of_departure=int(
                      time_of_stay + step + 3),
                  ev_phases=env.config["ev"]['ev_phases'],
                  transition_soc=env.config["ev"]['transition_soc'],
                  transition_soc_multiplier=transition_soc_multiplier,
                  charge_efficiency=env.config["ev"]['charge_efficiency'],
                  discharge_efficiency=env.config["ev"]['discharge_efficiency'],
                  timescale=env.timescale,
                  )


def EV_spawner(env) -> List[EV]:
    '''
    This function spawns all the EVs of the current simulation and returns the list of EVs

    Returns:
        EVs: list of EVs
    '''

    ev_list = []

    occupancy_list = np.zeros((env.number_of_ports, env.simulation_length))

    arrival_probabilities = np.random.rand(env.number_of_ports,
                                           env.simulation_length)

    scenario = env.scenario
    user_spawn_multiplier = env.config["spawn_multiplier"]
    time = env.sim_date

    # Define minimum time of stay duration so that an EV can fully charge
    min_time_of_stay = env.config['ev']["min_time_of_stay"]
    min_time_of_stay_steps = min_time_of_stay // env.timescale

    if env.simulation_length-min_time_of_stay_steps-1 < 0:
        raise ValueError(
            "Simulation length is too short for the minimum time of stay! Increase the simulation length or decrease the minimum time of stay.")

    for t in range(2, env.simulation_length-min_time_of_stay_steps-1):
        day = time.weekday()
        hour = time.hour
        minute = time.minute
        # Divide by 15 because the spawn rate is in 15 minute intervals (in the csv file)
        i = hour*4 + minute//15

        if day < 5:
            if scenario == "workplace" and (hour < 6 or hour > 18):
                time = time + datetime.timedelta(minutes=env.timescale)
                continue
            else:
                tau = env.df_arrival_week[scenario].iloc[i]
                multiplier = 1  # 10
        else:
            if scenario == "workplace":
                time = time + datetime.timedelta(minutes=env.timescale)
                continue
            else:
                tau = env.df_arrival_weekend[scenario].iloc[i]

            if day == 5:
                multiplier = 1  # 8
            else:
                multiplier = 1  # 6

        counter = 0
        for cs in env.charging_stations:
            for port in range(cs.n_ports):
                # if port is empty
                if occupancy_list[counter, t] == 0 and \
                    occupancy_list[counter, t-1] == 0 and \
                        occupancy_list[counter, t-2] == 0:
                    # and there is an EV arriving
                    if arrival_probabilities[counter, t]*100 < tau * multiplier * (env.timescale/60) * user_spawn_multiplier:
                        ev = spawn_single_EV(env=env,
                                             scenario=scenario,
                                             cs_id=cs.id,
                                             port=port,
                                             hour=hour,
                                             minute=minute,
                                             step=t,
                                             min_time_of_stay_steps=min_time_of_stay_steps)

                        if ev is not None:
                            ev_list.append(ev)

                            occupancy_list[counter, t +
                                           1:ev.time_of_departure] = 1
                counter += 1
        # step the time
        time = time + datetime.timedelta(minutes=env.timescale)

    return ev_list


def EV_spawner_GF(env) -> List[EV]:
    '''
    This function spawns all the EVs of the current simulation and returns the list of EVs

    Returns:
        EVs: list of EVs
    '''

    ev_list = []

    occupancy_list = np.zeros((env.number_of_ports, env.simulation_length))

    arrival_probabilities = np.random.rand(env.number_of_ports,
                                           env.simulation_length)

    user_spawn_multiplier = env.config["spawn_multiplier"]
    time = env.sim_date

    # Define minimum time of stay duration so that an EV can fully charge
    min_time_of_stay = env.config['ev']["min_time_of_stay"]
    min_time_of_stay_steps = min_time_of_stay // env.timescale

    if env.simulation_length-min_time_of_stay_steps-1 < 0:
        raise ValueError(
            "Simulation length is too short for the minimum time of stay! Increase the simulation length or decrease the minimum time of stay.")

    for t in range(2, env.simulation_length-min_time_of_stay_steps-1):
        day = time.weekday()
        hour = time.hour
        minute = time.minute
        # Divide by 10 because the spawn rate is in 10 minute intervals
        i = hour*6 + minute//10

        tau = env.df_arrival[day, i]
        multiplier = 0.5

        counter = 0
        for cs in env.charging_stations:
            for port in range(cs.n_ports):
                # if port is empty
                if occupancy_list[counter, t] == 0 and \
                    occupancy_list[counter, t-1] == 0 and \
                        occupancy_list[counter, t-2] == 0:
                    # and there is an EV arriving
                    # if arrival_probabilities[counter, t] < tau * multiplier * (env.timescale/60) * user_spawn_multiplier:
                    if arrival_probabilities[counter, t] < tau * 100 * (env.timescale/60) * user_spawn_multiplier * multiplier:
                        ev = spawn_single_EV_GF(env=env,
                                                day=day,
                                                cs_id=cs.id,
                                                port=port,
                                                hour=hour,
                                                minute=minute,
                                                step=t,
                                                min_time_of_stay_steps=min_time_of_stay_steps)

                        if ev is not None:
                            ev_list.append(ev)

                            occupancy_list[counter, t +
                                           1:ev.time_of_departure] = 1
                counter += 1
        # step the time
        time = time + datetime.timedelta(minutes=env.timescale)

    return ev_list


def smooth_vector(v) -> np.ndarray:
    n = len(v)
    smoothed_v = [0] * n

    # Calculate the sum of the original vector
    total_sum = sum(v)

    for i in range(n):
        # Calculate the range for averaging
        start = max(0, i - 1)
        end = min(n, i + 2)

        # Calculate the average of neighboring elements
        smoothed_v[i] = sum(v[start:end]) / (end - start)

    # Adjust the smoothed vector to maintain the original sum
    smoothed_sum = sum(smoothed_v)
    sum_ratio = total_sum / smoothed_sum

    # Apply the ratio to each element of the smoothed vector
    smoothed_v = [value * sum_ratio for value in smoothed_v]

    return smoothed_v


def median_smoothing(v, window_size) -> np.ndarray:
    smoothed_v = np.zeros_like(v)
    half_window = window_size // 2

    for i in range(len(v)):
        start = max(0, i - half_window)
        end = min(len(v), i + half_window + 1)
        smoothed_v[i] = np.median(v[start:end])

    return smoothed_v


def generate_power_setpoints(env) -> np.ndarray:
    '''
    This function generates the power setpoints for the entire simulation using
    the list of EVs and the charging stations from the environment.

    It considers the ev SoC and teh steps required to fully charge the EVs.

    Returns:
        power_setpoints: np.ndarray

    '''

    power_setpoints = np.zeros(env.simulation_length)
    # get normalized prices
    prices = abs(env.charge_prices[0])
    prices = prices / np.max(prices)

    required_energy_multiplier = 100 + \
        env.config["power_setpoint_flexiblity"]

    min_cs_power = env.charging_stations[0].get_min_charge_power()
    max_cs_power = env.charging_stations[0].get_max_power()

    total_evs_spawned = 0
    for t in range(env.simulation_length):
        counter = total_evs_spawned
        for _, ev in enumerate(env.EVs_profiles[counter:]):
            if ev.time_of_arrival == t + 1:
                total_evs_spawned += 1

                required_energy = ev.battery_capacity - ev.battery_capacity_at_arrival
                required_energy = required_energy * required_energy_multiplier / 100
                min_power_limit = max(ev.min_ac_charge_power, min_cs_power)
                max_power_limit = min(ev.max_ac_charge_power, max_cs_power)

                # Spread randomly the required energy over the time of stay using the prices as weights
                shifted_load = np.random.normal(loc=1 - prices[t+2:ev.time_of_departure],
                                                scale=min(
                                                    prices[t+2:ev.time_of_departure]),
                                                size=ev.time_of_departure - t - 2)
                # make shifted load positive
                shifted_load = np.abs(shifted_load)
                shifted_load = shifted_load / np.sum(shifted_load)
                shifted_load = shifted_load * required_energy * 60 / env.timescale

                # find power lower than min_power_limit and higher than max_power_limit
                step = 0
                while np.min(shifted_load[shifted_load != 0]) < min_power_limit or \
                        np.max(shifted_load) > max_power_limit:

                    if step > 10:
                        break

                    # print(f"Shifted load: {shifted_load}")
                    for i in range(len(shifted_load)):
                        if shifted_load[i] < min_power_limit and shifted_load[i] > 0:
                            load_to_shift = shifted_load[i]
                            shifted_load[i] = 0

                            if i == len(shifted_load) - 1:
                                shifted_load[0] += load_to_shift
                            else:
                                shifted_load[i+1] += load_to_shift

                        elif shifted_load[i] > max_power_limit:
                            load_to_shift = shifted_load[i] - max_power_limit
                            shifted_load[i] = max_power_limit

                            if i == len(shifted_load) - 1:
                                shifted_load[0] += load_to_shift
                            else:
                                shifted_load[i+1] += load_to_shift
                    step += 1

                power_setpoints[t+2:ev.time_of_departure] += shifted_load

            elif ev.time_of_arrival > t + 1:
                break

    # return smooth_vector(power_setpoints)

    # if env.timescale < 15:
    #     power_setpoints = median_smoothing(power_setpoints, 5)
    #     # make the setpoint have the same value for 15 minutes
    #     new_setpoints = np.zeros(env.simulation_length)
    #     for t in range(env.simulation_length):
    #         # average of the setpoints for the next 15 minutes
    #         new_setpoints[t] = np.mean(power_setpoints[t:t+15])

    #     return new_setpoints
    multiplier = int(15 / env.timescale)
    if multiplier < 1:
        multiplier = 1
    return median_smoothing(power_setpoints, 5 * multiplier)


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
