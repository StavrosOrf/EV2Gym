'''
Author: Stavros Orfanoudakis 2023

This file contains the EV class, which is used to represent the EVs in the environment.

The EV class contains the following attributes:
    - id: unique identifier of the EV (uniquep per charging station)
    - location: the location of the EV (charging station id)
    - battery_capacity_at_arrival: the battery capacity of the EV at arrival time in kWh
    - time_of_arrival: the time of arrival of the EV in the charging station in simulation timesteps
    - earlier_time_of_departure: the earliest time of departure of the EV in the charging station in simulation timesteps (if use_probabilistic_time_of_departure is False, then earlier_time_of_departure is equal to time_of_departure)
    - use_probabilistic_time_of_departure: whether the EV will use a probabilistic time of departure or not
    - battery_capacity: the battery capacity of the EV in kWh
    - min_desired_capacity: the minimum desired capacity of the EV in kWh to maximize battery life
    - max_desired_capacity: the maximum desired capacity of the EV in kWh to maximize battery life
    - charge_efficiency: the efficiency of the EV when charging
    - discharge_efficiency: the efficiency of the EV when discharging
    - v2g_enabled: whether the EV can provide power to the grid or not
    - timescale: the timescale of the simulation (useful for determining the charging speed)

The EV class contains the following status variables:
    - current_capacity: the current battery capacity of the EV in kWh
    - charging_cycles: the number of charging/discharging cycles of the EV (useful for determining battery life parameters)

The EV class contains the following methods:
    
    '''


class EV():
    def __init__(self,
                 id,
                 location,
                 battery_capacity_at_arrival,
                 time_of_arrival,
                 earlier_time_of_departure,
                 use_probabilistic_time_of_departure=False
                 battery_capacity=50,  # kWh
                 min_desired_capacity=8,  # kWh
                 max_desired_capacity=45,  # kWh
                 charge_efficiency=1,
                 discharge_efficiency=1,
                 v2g_enabled=True,
                 timescale=5,):

        self.id = id
        self.location = location
        self.timescale = timescale

        # EV simulation characteristics
        self.current_capacity = battery_capacity_at_arrival  # kWh
        self.time_of_arrival = time_of_arrival
        self.earlier_time_of_departure = earlier_time_of_departure
        self.use_probabilistic_time_of_departure = use_probabilistic_time_of_departure

        # EV technical characteristics
        self.battery_capacity = battery_capacity  # kWh
        self.min_desired_capacity = min_desired_capacity  # kWh
        self.max_desired_capacity = max_desired_capacity  # kWh
        self.efficiency = efficiency
        self.v2g_enabled = v2g_enabled

        # EV status
        self.current_capacity = battery_capacity_at_arrival  # kWh
        self.charging_cycles = 0

    def step(self, action):
        '''
        TODO: This method is used to update the EV's status after a timestep.
        '''
        pass
