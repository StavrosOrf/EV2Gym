'''
===================================
Author: Stavros Orfanoudakis 2023
===================================
'''

import numpy as np
import warnings
from typing import Optional, Dict, Any, Tuple


class EV():
    '''
    This file contains the EV class, which is used to represent the EVs in the environment.
    The battery model was adapted from https://github.com/zach401/acnportal/blob/master/acnportal/acnsim/models/battery.py#L186


    Attributes:
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

    Status variables:
        - current_capacity: the current battery capacity of the EV in kWh
        - current_power: the current power input of the EV in kW (positive for charging, negative for discharging)
        - charging_cycles: the number of charging/discharging cycles of the EV (useful for determining battery life parameters)
        - previous_power: the power input of the EV in the previous timestep in kW (positive for charging, negative for discharging)

    Methods:
        - step: updates the EV status according to the actions taken by the EV charger
        - _charge: charges the EV
        - _discharge: discharges the EV

    '''

    def __init__(self,
                 id,
                 location,
                 battery_capacity_at_arrival,
                 time_of_arrival,
                 earlier_time_of_departure,
                 use_probabilistic_time_of_departure=False,
                 desired_capacity=None,  # kWh
                 battery_capacity=50,  # kWh
                 max_ac_charge_power=11,  # kW
                 max_dc_charge_power=50,  # kW
                 noise_level=0,
                 transition_soc=0.8,
                 max_discharge_power=-5,  # kWh
                 charge_efficiency=1,
                 discharge_efficiency=1,
                 v2g_enabled=True,
                 timescale=5,
                 simulation_length=150,):

        self.id = id
        self.location = location
        self.timescale = timescale
        self.simulation_length = simulation_length

        # EV simulation characteristics
        self.time_of_arrival = time_of_arrival
        self.earlier_time_of_departure = earlier_time_of_departure
        self.use_probabilistic_time_of_departure = use_probabilistic_time_of_departure
        self.desired_capacity = battery_capacity if desired_capacity is None else desired_capacity
        self.battery_capacity_at_arrival = battery_capacity_at_arrival  # kWh

        # EV technical characteristics
        self.battery_capacity = battery_capacity  # kWh
        self.max_ac_charge_power = max_ac_charge_power  # kW
        self.max_discharge_power = max_discharge_power  # kW
        self.transition_soc = transition_soc
        self.noise_level = noise_level

        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.v2g_enabled = v2g_enabled

        # EV status
        self.current_capacity = battery_capacity_at_arrival  # kWh
        self.current_power = 0  # kWh
        self.actual_current = 0  # A
        self.charging_cycles = 0
        self.previous_power = 0
        self.required_power = self.battery_capacity - self.battery_capacity_at_arrival

    def reset(self):
        '''
        The reset method is used to reset the EV's status to the initial state.
        '''
        self.current_capacity = self.battery_capacity_at_arrival
        self.current_power = 0
        self.charging_cycles = 0
        self.previous_power = 0
        self.required_power = self.battery_capacity - self.battery_capacity_at_arrival

    def step(self, amps, voltage, type='AC'):
        '''
        The step method is used to update the EV's status according to the actions taken by the EV charger.
        Inputs:
            - action: the power input in kW (positive for charging, negative for discharging)
        Outputs:
            - self.current_power: the current power input of the EV in kW (positive for charging, negative for discharging)
            - self.actual_curent: the actual current input of the EV in A (positive for charging, negative for discharging)
        '''

        if type == 'DC':
            raise NotImplementedError

        if amps == 0:
            self.current_power = 0
            self.actual_current = 0
            return 0, 0

        # If the action is different than the previous action, then increase the charging cycles
        if self.previous_power == 0 or (self.previous_power/amps) < 0:
            self.charging_cycles += 1

        print(f'action: {amps} A, {voltage} V')

        if amps > 0:
            self.actual_current = self._charge(amps, voltage)
        elif amps < 0:
            self.actual_current = self._discharge(amps, voltage)

        self.previous_power = self.current_power

        return self.current_power, self.actual_current

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
            raise NotImplementedError
            if np.random.poisson(lam=2.0) < timestep - self.earlier_time_of_departure:
                return self.get_user_satisfaction()
        else:
            # if timestep >= self.earlier_time_of_departure:
            return self.get_user_satisfaction()

    def get_user_satisfaction(self):
        '''
        A function that returns the user satisfaction of the EV when departing.
        Outputs: 
            - Score: a value between 0 and 1
        '''

        if self.current_capacity < self.desired_capacity - 0.001:
            # print (f'EV {self.id} is departing with {self.current_capacity} kWh out of {self.desired_capacity} kWh')
            # return 0
            return self.current_capacity / self.desired_capacity
        else:
            return 1

    def get_soc(self):
        '''
        A function that returns the state of charge of the EV.
        Outputs: 
            - SoC: the state of charge of the EV in [0,1]
        '''
        return (self.current_capacity/self.battery_capacity)

    def get_state(self, current_step):
        '''
        A function that returns the state of the EV.
        Outputs: 
            - State: the state of the EV
        '''
        timestep_left = self.earlier_time_of_departure - current_step

        # , self.charging_cycles
        # return self.get_soc(), timestep_left / self.simulation_length
        return self.required_power/self.battery_capacity, \
            (current_step-self.time_of_arrival) / self.simulation_length
        # return self.required_power, self.current_step-self.time_of_arrival

    def __str__(self):
        return f' {self.current_power*60/self.timescale :5.1f} kWh |' + \
            f' {(self.current_capacity/self.battery_capacity)*100:5.1f} % |' + \
            f' {self.charging_cycles:2d}  |' + \
            f' t_dep: {self.earlier_time_of_departure}'

    def _charge(self, amps, voltage):

        assert (amps > 0)
        # given_power = (amps * voltage / 1000) * \
        #     self.charge_efficiency * self.timescale / 60  # KW

        # if self.current_capacity + given_power > self.battery_capacity:
        #     self.current_power = self.battery_capacity - self.current_capacity
        #     self.current_capacity = self.battery_capacity
        # else:
        #     self.current_power = given_power
        #     self.current_capacity += given_power

        """ Method to "charge" the battery based on a two-stage linear
        battery model.
        
        Battery model based on a piecewise linear approximation of battery behavior. The battery will charge at the
        minimum of max_rate and the pilot until it reaches _transition_soc. After this, the maximum charging rate of the
        battery will decrease linearly to 0 at 100% state of charge.

        For more info on model: https://www.sciencedirect.com/science/article/pii/S0378775316317396

        All calculations are done in terms fo battery state of charge
        (SoC). Results are converted back to power units at the end.
        Code adapted from: https://github.com/zach401/acnportal/blob/master/acnportal/acnsim/models/battery.py#L186

        Args:
            pilot (float): Pilot signal passed to the battery. [A]
            voltage (float): AC voltage provided to the battery
                charger. [V]
            period (float): Length of the charging period. [minutes]

        Returns:
            float: average charging rate of the battery over this single
                period.

        """

        pilot = amps
        voltage = voltage
        period = self.timescale
        # All calculations are done in terms of battery SoC, so we
        # convert pilot signal and max power into pilot and max rate of
        # change of SoC.
        pilot_dsoc = pilot * voltage / 1000 / \
            self.battery_capacity / (60 / period)
        max_dsoc = self.max_ac_charge_power / \
            self.battery_capacity / (60 / period)

        if pilot_dsoc > max_dsoc:
            pilot_dsoc = max_dsoc

        # The pilot SoC rate of change has a new transition SoC at
        # which decreasing of max charging rate occurs.
        pilot_transition_soc = self.transition_soc + (
            pilot_dsoc - max_dsoc
        ) / max_dsoc * (self.transition_soc - 1)

        if pilot < 0:
            warnings.warn(
                f"Negative pilot signal input. Battery models"
                f"may not be accurate for pilot {pilot} A."
            )

        # The charging equation depends on whether the current SoC of
        # the battery is above or below the new transition SoC.
        if self.get_soc() < pilot_transition_soc:
            # In the pre-rampdown region, the charging equation changes
            # depending on whether charging the battery over this
            # time period causes the battery to transition between
            # charging regions.
            if 1 <= (pilot_transition_soc - self.get_soc()) / pilot_dsoc:
                curr_soc = pilot_dsoc + self.get_soc()
            else:
                curr_soc = 1 + np.exp(
                    (pilot_dsoc + self.get_soc() - pilot_transition_soc)
                    / (pilot_transition_soc - 1)
                ) * (pilot_transition_soc - 1)
        else:
            curr_soc = 1 + np.exp(pilot_dsoc / (pilot_transition_soc - 1)) * (
                self.get_soc() - 1
            )

        # Add subtractive noise to the final SoC, scaling the noise
        # such that _noise_level is the standard deviation of the noise
        # in the battery charging power.
        if self.noise_level > 0:
            raw_noise = np.random.normal(0, self.noise_level)
            scaled_noise = raw_noise * (period / 60) / self.battery_capacity
            curr_soc -= abs(scaled_noise)

        dsoc = curr_soc - self.get_soc()
        self.current_capacity = curr_soc * self.battery_capacity

        # For charging power and charging rate (current), we use the
        # the average over this time period.
        self.current_power = dsoc * self.battery_capacity  # / (period / 60)
        self.required_power = self.required_power - self.current_power  # * period / 60
        return self.current_power / (period / 60) * 1000 / voltage

    def _discharge(self, amps, voltage):
        '''
        The _discharge method is used to discharge the EV's battery.
        Inputs:
            - power: the power input in kW (it is negative because of the discharge)
        '''
        assert (amps < 0)

        given_power = (amps * voltage / 1000)
        print(f'given_power: {given_power} kWh, {self.current_capacity} kWh', )

        if abs(given_power/1000) > abs(self.max_discharge_power):
            given_power = self.max_discharge_power

        given_power = given_power * self.discharge_efficiency * self.timescale / 60
        print(f'given_power: {given_power} kWh')

        if self.current_capacity + given_power < 0:
            self.current_power = -self.current_capacity  # * 60 / self.timescale
            self.current_capacity = 0
        else:
            self.current_power = given_power  # * 60 / self.timescale
            self.current_capacity += given_power

        self.required_power = self.required_power + self.current_power

        return given_power*60/self.timescale * 1000 / voltage
