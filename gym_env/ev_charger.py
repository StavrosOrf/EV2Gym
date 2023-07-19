'''=================================================================================================
Author: Stavros Orfanoudakis 2023

This file contains the EV_Charger class, which is used to represent the EV chargers in the environment.

The EV_Charger class contains the following attributes:
    - id: unique identifier of the EV charger
    - connected_bus: the bus to which the EV charger is connected
    - connected_transformer: the transformer(s) to which the EV charger is connected
    - geo_location: the geographical location of the EV charger
    - max_charge_power: the maximum total power that the EV charger can provide to all EVs connected to it per hour
    - max_discharge_power: the maximum total power that the EV charger can receive from all EVs connected to it per hour
    - n_ports: the number of ports of the EV charger
    - charger_type: the type of the EV charger (typ1, type2, or DC)
    - bi_directional: whether the EV charger can provide power to the grid or not
    - timescale: the timescale of the simulation (useful for determining the charging speed)

The EV_Charger class contains the following status variables:
    - current_power_output: the current total power output of the EV charger (positive for draining energy from the grid, negative for providing energy to the grid)
    - evs_connected: the list of EVs connected to the EV charger
    - current_step: the current simulation timestep
    
The EV_Charger class contains the following methods:
    - step: updates the EV charger status according to the actions taken by the EVs
    - reset: resets the EV charger status to the initial state
===================================================================================================='''

class EV_Charger:
    def __init__(self,
                 id,
                 connected_bus,
                 connected_transformer,
                 geo_location = None,
                 max_charge_power=22, #kW
                 max_discharge_power=22, #kW                 
                 n_ports=2,
                 charger_type="Type2",
                 bi_directional=True,
                 timescale=5):
        
        self.id = id

        #EV Charger location and grid characteristics
        self.connected_bus = connected_bus
        self.connected_transformer = connected_transformer
        self.geo_location = geo_location
        
        #EV Charger technical characteristics
        self.max_charge_power = max_charge_power
        self.max_discharge_power = max_discharge_power
        self.n_ports = n_ports
        self.charger_type = charger_type
        self.bi_directional = bi_directional

        #EV Charger status
        self.current_power_output = 0        
        self.evs_connected = []        
        self.current_step = 0


    def step(self,actions):
        #actions are in the format of (power,n_ports) positive for charging negative for discharging
        #default to 0 if no ev is connected

        costs = 0
        user_satisfaction = []

        #TODO: check if the power requested is within the limits of the charger, AND NORMALIZE ELSEWISE
        #TODO: update the information of the connected EVs according to actions

        #TODO: remove departed EVs and add new EVs that just arrived

        self.current_step += 1

        return costs, user_satisfaction      
    
    def spawn_ev(self,ev):
        self.evs_connected.append(ev)

    def reset(self):
        self.current_power_output = 0        
        self.evs_connected = []
        self.current_step = 0