# this class contains heurisyic algorithms for the power setpoint tracking problem
import math
import numpy as np

class RoundRobin():
    def __init__(self, env):

        #find average charging power of the simulation
        self.average_power = 0
        for cs in env.charging_stations:
            self.average_power += cs.max_charge_current * cs.voltage * math.sqrt(cs.phases)/2
        self.average_power /= len(env.charging_stations)

        #list with the ids of EVs that were already served in this round
        self.served_ev_list = []
    
    def get_currently_parked_ev(self, env):
        # iterate over all ports and check if there is an uncharged EV parked
        # if so, add the EV id to the ev_list

        current_parked_ev_list = []

        counter = 0
        # iterate over all ports
        for cs in env.charging_stations:
            for port in range(cs.n_ports):
                if cs.evs_connected[port] is not None:        
                    if cs.evs_connected[port].get_soc() < 1:
                        current_parked_ev_list.append(counter)
                counter += 1

        return current_parked_ev_list

    def get_action(self, env):
        #this function returns the actino list based on the round robin algorithm

        total_power = env.power_setpoints[env.current_step]*1000

        number_of_EVs_to_charge = math.ceil(total_power // self.average_power)
        print(f'Number of EVs to charge: {number_of_EVs_to_charge}, total power: {total_power:.2f}, average power: {self.average_power:.2f}')
        #get currently parked EVs
        currently_parked_ev_list = self.get_currently_parked_ev(env)
        print(f'Currently parked EVs: {currently_parked_ev_list}')
        print(f'Served EVs: {self.served_ev_list}')

        #remove indexex of already served EVs in this round
        for ev in self.served_ev_list:
            currently_parked_ev_list.remove(ev)

        #if there are more EVs to charge than currently parked EVs, pop the first EVs from the list required to charge
        if number_of_EVs_to_charge > len(currently_parked_ev_list):
            number_of_EVs_to_charge = len(currently_parked_ev_list)
            number_of_EVs_to_charge = number_of_EVs_to_charge - len(currently_parked_ev_list)

        #get the EVs to charge in this round
        evs_to_charge = currently_parked_ev_list[:number_of_EVs_to_charge]

        #add the EVs to the served list
        self.served_ev_list.extend(evs_to_charge)

        #create action list
        action_list = np.zeros(env.number_of_ports)

        #set the action for the EVs to charge
        for ev in evs_to_charge:
            action_list[ev] = 1
        
        print(f'Evs to charge: {evs_to_charge}')        
        return action_list