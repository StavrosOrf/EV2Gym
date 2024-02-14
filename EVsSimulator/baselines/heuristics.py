# this class contains heurisyic algorithms for the power setpoint tracking problem
import math
import numpy as np


class RoundRobin():
    def __init__(self, env, verbose=False):

        self.verbose = verbose

        # find average charging power of the simulation
        self.average_power = 0
        for cs in env.charging_stations:
            self.average_power += cs.max_charge_current * \
                cs.voltage * math.sqrt(cs.phases) /cs.n_ports
        self.average_power /= len(env.charging_stations)

        # list with the ids of EVs that were already served in this round
        self.ev_buffer = []

    def update_ev_buffer(self, env) -> None:
        '''
        This function updates the EV buffer list with the EVs that are currently parked by adding or removing them.
        '''
        counter = 0
        # iterate over all ports
        for cs in env.charging_stations:
            for port in range(cs.n_ports):
                if cs.evs_connected[port] is not None:
                    if cs.evs_connected[port].get_soc() < 1:                        
                        
                        if counter not in self.ev_buffer:
                            self.ev_buffer.insert(0,counter)
                    else:
                        if counter in self.ev_buffer:
                            self.ev_buffer.remove(counter)
                else:
                    if counter in self.ev_buffer:
                        self.ev_buffer.remove(counter)            
                counter += 1

    def get_action(self, env):

        # this function returns the action list based on the round robin algorithm

        total_power = env.power_setpoints[env.current_step] * 1000  # in W

        number_of_EVs_to_charge = total_power / self.average_power
        
        if self.verbose:
            print("-------------------Round Robin-------------------")
            print(  
            f'Number of EVs to charge: {number_of_EVs_to_charge:.2f},\
                total power: {total_power:.2f}, average power: {self.average_power:.2f}')
            
        # get currently parked EVs
        self.update_ev_buffer(env)
        max_number_of_EVs_to_charge = len(self.ev_buffer)
        
        if self.verbose:
            print(f'EV buffer: {self.ev_buffer}')
        
        # get the EVs to charge in this round
        evs_to_charge = self.ev_buffer[:min(int(np.ceil(number_of_EVs_to_charge)), max_number_of_EVs_to_charge)]                
        self.ev_buffer = self.ev_buffer[min(int(np.ceil(number_of_EVs_to_charge)), max_number_of_EVs_to_charge):]        

        self.ev_buffer.extend(evs_to_charge)
        
        # create action list
        action_list = np.zeros(env.number_of_ports)

        # set the action for the EVs to charge
        for i, ev in enumerate(evs_to_charge):
            action_list[ev] = 1
            if i == len(evs_to_charge) - 1:
                action_list[ev] = (number_of_EVs_to_charge - i)                

        print(f'Evs to charge: {evs_to_charge}')
        return action_list
