# This file contains the state function for the RL agent
import math
import numpy as np


def PublicPST(env, *args):
    # This state function is the public power setpoints
    # The state is the public power setpoints
    # The state is a vector

    state = [
        (env.current_step) / env.simulation_length,
        env.sim_date.weekday() / 7,
        # turn hour and minutes in sin and cos
        # math.sin(env.sim_date.hour/24*2*math.pi),
        # math.cos(env.sim_date.hour/24*2*math.pi),
    ]

    # the final state of each simulation
    if env.current_step < env.simulation_length:
        state.append(env.power_setpoints[env.current_step]/100)
        state.append(env.charge_power_potential[env.current_step]/100)
    else:
        state.append(env.power_setpoints[env.current_step-1]/100)
        state.append(env.charge_power_potential[env.current_step-1]/100)

    state.append(env.current_power_usage[env.current_step-1]/100)

    # For every transformer
    for tr in env.transformers:

        # For every charging station connected to the transformer
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:

                # For every EV connected to the charging station
                for EV in cs.evs_connected:
                    # If there is an EV connected
                    if EV is not None:
                        state.append([
                            1 if EV.get_soc() == 1 else 0,  # we know if the EV is full
                            EV.total_energy_exchanged/100,
                            # EV.max_ac_charge_power*1000 /
                            # (cs.voltage*math.sqrt(cs.phases))/100,
                            # EV.min_ac_charge_power*1000 /
                            # (cs.voltage*math.sqrt(cs.phases))/100,
                            (env.current_step-EV.time_of_arrival) / env.simulation_length])

                    # else if there is no EV connected put zeros
                    else:
                        state.append(np.zeros(3))

    state = np.array(np.hstack(state))

    np.set_printoptions(suppress=True)

    return state


def BusinessPSTwithMoreKnowledge(env, *args):
    '''
    This state function is used for the business case scenario that requires more knowledge such as SoC and time of departure for each EV present.
    '''

    state = [
        (env.current_step) / env.simulation_length,
        env.sim_date.weekday() / 7,
        # turn hour and minutes in sin and cos
        math.sin(env.sim_date.hour/24*2*math.pi),
        math.cos(env.sim_date.hour/24*2*math.pi),
    ]

    # the final state of each simulation
    if env.current_step < env.simulation_length:
        state.append(env.power_setpoints[env.current_step]/100)
        state.append(env.charge_power_potential[env.current_step]/100)
    else:
        state.append(env.power_setpoints[env.current_step-1]/100)
        state.append(env.charge_power_potential[env.current_step-1]/100)

    for tr in env.transformers:
        state.append(tr.max_current/100)
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:
                for EV in cs.evs_connected:
                    if EV is not None:
                        state.append([EV.total_energy_exchanged,
                                      EV.max_ac_charge_power*1000 /
                                      (cs.voltage*math.sqrt(cs.phases)),
                                      EV.min_ac_charge_power*1000 /
                                      (cs.voltage*math.sqrt(cs.phases)),
                                      (env.current_step-EV.time_of_arrival) /
                                      env.simulation_length,  # time stayed
                                      # total time stayed
                                      (EV.time_of_departure - \
                                       EV.time_of_arrival) / env.simulation_length,
                                      (((EV.battery_capacity - EV.battery_capacity_at_arrival) /
                                        (EV.time_of_departure - EV.time_of_arrival)) / EV.max_ac_charge_power),  # average charging speed
                                      EV.time_of_departure / env.simulation_length,  # time of departure
                                      EV.get_soc(),  # soc
                                      EV.required_power / EV.battery_capacity,  # required energy
                                      EV.time_of_arrival / env.simulation_length,  # time of arrival
                                      ])
                    else:
                        state.append(np.zeros(10))

    state = np.array(np.hstack(state))

    np.set_printoptions(suppress=True)

    return state


def RewardMaximizationState(env, *args):
    '''
    This state function is used for the business case scenario that requires more knowledge such as SoC and time of departure for each EV present.
    '''

    state = [
        (env.current_step) / env.simulation_length,
        env.sim_date.weekday() / 7,
        # turn hour and minutes in sin and cos
        math.sin(env.sim_date.hour/24*2*math.pi),
        math.cos(env.sim_date.hour/24*2*math.pi),
    ]

    state.append(env.charge_prices[env.current_step])
    state.append(env.discharge_prices[env.current_step])

    for tr in env.transformers:
        state.append(tr.max_current/100)
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:
                for EV in cs.evs_connected:
                    if EV is not None:
                        state.append([EV.total_energy_exchanged,
                                      EV.max_ac_charge_power*1000 /
                                      (env.current_step-EV.time_of_arrival) /
                                      env.simulation_length,  # time stayed
                                      # total time stayed
                                      (EV.time_of_departure - \
                                       EV.time_of_arrival) / env.simulation_length,
                                      (((EV.battery_capacity - EV.battery_capacity_at_arrival) /
                                        (EV.time_of_departure - EV.time_of_arrival)) / EV.max_ac_charge_power),  # average charging speed
                                      EV.time_of_departure / env.simulation_length,  # time of departure
                                      EV.get_soc(),  # soc
                                      EV.required_power / EV.battery_capacity,  # required energy
                                      EV.time_of_arrival / env.simulation_length,  # time of arrival
                                      ])
                    else:
                        state.append(np.zeros(8))

    state = np.array(np.hstack(state))

    np.set_printoptions(suppress=True)

    return state
