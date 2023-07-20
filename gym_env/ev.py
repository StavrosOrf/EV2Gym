''' =================================================================================================
Author: Stavros Orfanoudakis 2023

This file contains the EV class, which is used to represent the EVs in the environment.

The EV class contains the following attributes:
    - id: unique identifier of the EV (uniquep per charging station)
    - location: the location of the EV (charging station id)
    - battery_capacity_at_arrival: the battery capacity of the EV at arrival time in kWh
    - time_of_arrival: the time of arrival of the EV in the charging station in simulation timesteps
    - earlier_time_of_departure: the earliest time of departure of the EV in the charging station in simulation timesteps (if use_probabilistic_time_of_departure is False, then earlier_time_of_departure is equal to time_of_departure)
    - desired_capacity: the desired capacity of the EV at departure time in kWh
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
    - current_power: the current power input of the EV in kW (positive for charging, negative for discharging)
    - charging_cycles: the number of charging/discharging cycles of the EV (useful for determining battery life parameters)
    - previous_power: the power input of the EV in the previous timestep in kW (positive for charging, negative for discharging)

The EV class contains the following methods:
    - step: updates the EV status according to the actions taken by the EV charger
    - _charge: charges the EV
    - _discharge: discharges the EV

==================================================================================================='''
import numpy as np


class EV():
    def __init__(self,
                 id,
                 location,
                 battery_capacity_at_arrival,
                 time_of_arrival,
                 earlier_time_of_departure,
                 use_probabilistic_time_of_departure=False,
                 desired_capacity=50,  # kWh
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
        self.desired_capacity = desired_capacity  # kWh

        # EV technical characteristics
        self.battery_capacity = battery_capacity  # kWh
        self.min_desired_capacity = min_desired_capacity  # kWh
        self.max_desired_capacity = max_desired_capacity  # kWh
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.v2g_enabled = v2g_enabled

        # EV status
        self.current_capacity = battery_capacity_at_arrival  # kWh
        self.current_power = 0  # kWh
        self.charging_cycles = 0
        self.previous_power = 0

    def __str__(self):
        return f' {self.current_power:5.1f} kWh | {(self.current_capacity/self.battery_capacity)[0]*100:5.1f} %'

    def step(self, action):
        '''
        The step method is used to update the EV's status according to the actions taken by the EV charger.
        Inputs:
            - action: the power input in kW

        Outputs:
            - self.current_power: the current power input of the EV in kW (positive for charging, negative for discharging)
        '''
        if action == 0:
            return 0
        action = action[0]

        # If the action is different than the previous action, then increase the charging cycles
        if self.previous_power == 0 or (self.previous_power/action) < 0:
            self.charging_cycles += 1

        if action > 0:
            self._charge(action)
        else:
            self._discharge(action)

        self.previous_power = self.current_power

        return self.current_power

    def is_departing(self, timestep):
        '''
        The is_departing method is used to determine whether the EV is departing or not.
        Inputs:
            - timestep: the current timestep of the simulation
        Outputs:
            - Returns the user satisfaction of the EV in departing else None
        '''
        if timestep < self.earlier_time_of_departure:
            return None

        if self.use_probabilistic_time_of_departure:
            if np.random.poisson(lam=2.0) < timestep - self.earlier_time_of_departure:
                return self.get_user_satisfaction()
        else:
            if timestep == self.time_of_departure:
                return self.get_user_satisfaction()

    def get_user_satisfaction(self):
        '''
        A function that returns the user satisfaction of the EV when departing.
        Outputs: 
            - Score: a value between 0 and 1
        '''

        if self.current_capacity < self.desired_capacity:
            return 0
        elif self.current_capacity >= self.desired_capacity:
            return 1

    def _charge(self, power):
        '''
        The _charge method is used to charge the EV's battery.
        Inputs:
            - power: the power input in kW
        '''

        assert (power > 0)
        given_power = power * self.charge_efficiency * self.timescale / 60

        if self.current_capacity + given_power > self.battery_capacity:
            self.current_power = self.battery_capacity - self.current_capacity
            self.current_capacity = self.battery_capacity
        else:
            self.current_power = given_power
            self.current_capacity += given_power

    def _discharge(self, power):
        '''
        The _discharge method is used to discharge the EV's battery.
        Inputs:
            - power: the power input in kW (it is negative because of the discharge)
        '''
        assert (power < 0)
        giving_power = power * self.discharge_efficiency * self.timescale / 60

        if self.current_capacity + giving_power < 0:
            self.current_power = -self.current_capacity
            self.current_capacity = 0
        else:
            self.current_power = giving_power
            self.current_capacity -= giving_power
