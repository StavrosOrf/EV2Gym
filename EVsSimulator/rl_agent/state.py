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
        math.sin(env.sim_date.hour/24*2*math.pi),
        math.cos(env.sim_date.hour/24*2*math.pi),
        env.power_setpoints[env.current_step]/100,
        env.power_potential[env.current_step]/100,
    ]

    #For every transformer
    for tr in env.transformers:
        
        #For every charging station connected to the transformer
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:
                
                #For every EV connected to the charging station
                for EV in cs.evs_connected:                    
                    #If there is an EV connected
                    if EV is not None:
                        state.append([
                            EV.total_energy_exchanged/100, \
                            EV.max_ac_charge_power*1000/(cs.voltage*math.sqrt(cs.phases))/100, 
                            EV.min_ac_charge_power*1000/(cs.voltage*math.sqrt(cs.phases))/100,                            
                            (env.current_step-EV.time_of_arrival) / env.simulation_length])                                           
                        
                    # else if there is no EV connected put zeros
                    else:
                        state.append(np.zeros(4))                

    state = np.array(np.hstack(state))

    np.set_printoptions(suppress=True)

    return state


def BusinessPSTwithMoreKnowledge(env, *args):
    # This state function is the business power setpoints with complete knowledge
    # The state is the business power setpoints
    # The state is a vector

    state = [
        (env.current_step-1) / env.simulation_length,
        env.sim_date.weekday() / 7,
        # turn hour and minutes in sin and cos
        math.sin(env.sim_date.hour/24*2*math.pi),
        math.cos(env.sim_date.hour/24*2*math.pi),
        env.power_setpoints[env.current_step-1]/100,
        env.power_potential[env.current_step-1]/100,
    ]

    for tr in env.transformers:
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:
                for EV in cs.evs_connected:
                    if EV is not None:
                        state.append(EV.get_state(self.current_step,
                                                  scenario=scenario,
                                                  voltage=self.voltage,
                                                  phases=self.phases,))
                    else:
                        state.append(np.zeros(4))
                state.append()

    state = np.array(np.hstack(state))

    np.set_printoptions(suppress=True)

    return state
