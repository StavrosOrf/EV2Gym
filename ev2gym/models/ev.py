'''
This file contains the EV class.
'''

import numpy as np
import warnings
import math
from typing import Tuple, Union


class EV():
    '''
     which is used to represent the EVs in the environment.
    The two-stage battery model was adapted from https://github.com/zach401/acnportal/blob/master/acnportal/acnsim/models/battery.py#L186


    Attributes:
        - id: unique identifier of the EV (uniquep per charging station)
        - location: the location of the EV (charging station id)
        - battery_capacity_at_arrival: the battery capacity of the EV at arrival time in kWh
        - time_of_arrival: the time of arrival of the EV in the charging station in simulation timesteps
        - time_of_departure: the earliest time of departure of the EV in the charging station in simulation timesteps (if use_probabilistic_time_of_departure is False, then time_of_departure is equal to time_of_departure)
        - desired_capacity: the desired capacity of the EV at departure time in kWh
        - use_probabilistic_time_of_departure: whether the EV will use a probabilistic time of departure or not
        - battery_capacity: the battery capacity of the EV in kWh
        - min_desired_capacity: the minimum desired capacity of the EV in kWh to maximize battery life
        - max_desired_capacity: the maximum desired capacity of the EV in kWh to maximize battery life
        - charge_efficiency: the efficiency of the EV when charging
        - discharge_efficiency: the efficiency of the EV when discharging        
        - timescale: the timescale of the simulation (useful for determining the charging speed)

    Status variables:
        - current_capacity: the current battery capacity of the EV in kWh
        - current_energy: the current power input of the EV in kW (positive for charging, negative for discharging)
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
                 time_of_departure,                 
                 desired_capacity=None,  # kWh
                 battery_capacity=50,  # kWh
                 min_battery_capacity=10,  # kWh
                 max_ac_charge_power=22,  # kW
                 min_ac_charge_power=0,  # kW
                 max_dc_charge_power=50,  # kW
                 max_discharge_power=-22,  # kW
                 min_discharge_power=0,  # kW                 
                 ev_phases=3,                 
                 transition_soc=0.8,
                 charge_efficiency=1,
                 discharge_efficiency=1,                 
                 timescale=5,
                 ):

        self.id = id
        self.location = location
        self.timescale = timescale        

        # EV simulation characteristics
        self.time_of_arrival = time_of_arrival
        self.time_of_departure = time_of_departure        
        self.desired_capacity = battery_capacity if desired_capacity is None else desired_capacity
        self.battery_capacity_at_arrival = battery_capacity_at_arrival  # kWh

        # EV technical characteristics
        self.battery_capacity = battery_capacity  # kWh
        self.min_battery_capacity = min_battery_capacity  # kWh
        self.max_ac_charge_power = max_ac_charge_power  # kW
        self.min_ac_charge_power = min_ac_charge_power  # kW
        self.max_discharge_power = max_discharge_power  # kW
        self.min_discharge_power = min_discharge_power  # kW
        self.max_dc_charge_power = max_dc_charge_power  # kW
        self.transition_soc = transition_soc        
        self.ev_phases = ev_phases

        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency        

        # EV status
        self.current_capacity = battery_capacity_at_arrival  # kWh
        self.prev_capacity = self.current_capacity
        self.current_energy = 0  # kW
        self.actual_current = 0  # A
        self.charging_cycles = 0
        self.previous_power = 0
        self.required_energy = self.battery_capacity - self.battery_capacity_at_arrival
        self.total_energy_exchanged = 0

        # Baterry degradation        
        self.abs_total_energy_exchanged = 0
        self.historic_soc = []
        self.active_steps = []
        
        self.calendar_loss = 0
        self.cyclic_loss = 0

    def reset(self):
        '''
        The reset method is used to reset the EV's status to the initial state.
        '''
        self.current_capacity = self.battery_capacity_at_arrival
        self.prev_capacity = self.current_capacity
        self.current_energy = 0
        self.actual_current = 0
        self.charging_cycles = 0
        self.previous_power = 0
        self.required_energy = self.battery_capacity - self.battery_capacity_at_arrival
        self.total_energy_exchanged = 0
        self.c_lost = 0
        
        self.abs_total_energy_exchanged = 0
        self.historic_soc = []
        self.active_steps = []
        
        self.calendar_loss = 0
        self.cyclic_loss = 0

    def step(self, amps, voltage, phases=1, type='AC') -> Tuple[float, float]:
        '''
        The step method is used to update the EV's status according to the actions taken by the EV charger.
        Inputs:
            - action: the power input in kW (positive for charging, negative for discharging)
        Outputs:
            - self.current_energy: the current power input of the EV in kW (positive for charging, negative for discharging)
            - self.actual_curent: the actual current input of the EV in A (positive for charging, negative for discharging)
        '''

        if type == 'DC':
            raise NotImplementedError

        if amps > 0 and amps < self.min_ac_charge_power*1000/(voltage*math.sqrt(phases)):
            amps = 0
        elif amps < 0 and amps > self.min_discharge_power*1000/(voltage*math.sqrt(phases)):
            amps = 0
            
        self.historic_soc.append(self.get_soc())

        
        if amps == 0:
            self.current_energy = 0
            self.actual_current = 0
            
            self.active_steps.append(0)
            return 0, 0

        # If the action is different than the previous action, then increase the charging cycles
        if self.previous_power == 0 or (self.previous_power/amps) < 0:
            self.charging_cycles += 1

        phases = min(phases, self.ev_phases)

        if amps > 0:
            self.actual_current = self._charge(amps, voltage, phases)
        elif amps < 0:
            self.actual_current = self._discharge(amps, voltage, phases)

        self.previous_power = self.current_energy

        self.total_energy_exchanged += self.current_energy #* self.timescale / 60
        self.abs_total_energy_exchanged += abs(self.current_energy) #* self.timescale / 60
        
        #round up to the nearest 0.01 the current capacity
        self.current_capacity = self.my_ceil(self.current_capacity, 2)
        
        self.active_steps.append(1 if self.actual_current != 0 else 0)
        return self.current_energy, self.actual_current

    def my_ceil(self, a, precision=2):
        return np.true_divide(np.ceil(a * 10**precision), 10**precision)

    def is_departing(self, timestep) -> Union[float, None]:
        '''
        The is_departing method is used to determine whether the EV is departing or not.
        Inputs:
            - timestep: the current timestep of the simulation
        Outputs:
            - Returns the user satisfaction of the EV in departing else None
        '''
        if timestep < self.time_of_departure:
            return None

        return self.get_user_satisfaction()

    def get_user_satisfaction(self) -> float:
        '''
        A function that returns the user satisfaction of the EV when departing.
        Outputs: 
            - Score: a value between 0 and 1
        '''

        if self.current_capacity < self.desired_capacity - 0.001:            
            return self.current_capacity / self.desired_capacity
        else:
            return 1

    def get_soc(self) -> float:
        '''
        A function that returns the state of charge of the EV.
        Outputs: 
            - SoC: the state of charge of the EV in [0,1]
        '''
        return (self.current_capacity/self.battery_capacity)

    def __str__(self):
        return f' {self.current_energy*60/self.timescale :5.1f} kWh |' + \
            f' {(self.current_capacity/self.battery_capacity)*100:5.1f} % |' + \
            f't_stay: {self.time_of_arrival}-' + \
            f'{self.time_of_departure} |' + \
            f' {self.max_ac_charge_power}/' + \
            f'{self.max_discharge_power} kWh|' + \
            f' {self.battery_capacity} kW |'

    def _charge(self, amps, voltage, phases=1) -> float:

        assert (amps > 0)
        # given_power = (amps * voltage / 1000) * \
        #     self.charge_efficiency * self.timescale / 60  # KW

        # if self.current_capacity + given_power > self.battery_capacity:
        #     self.current_energy = self.battery_capacity - self.current_capacity
        #     self.current_capacity = self.battery_capacity
        # else:
        #     self.current_energy = given_power
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
        voltage = voltage * math.sqrt(phases)
        period = self.timescale
        # All calculations are done in terms of battery SoC, so we
        # convert pilot signal and max power into pilot and max rate of
        # change of SoC.
        pilot_dsoc = self.charge_efficiency * pilot * voltage / 1000 / \
            self.battery_capacity / (60 / period)
        max_dsoc = self.charge_efficiency * self.max_ac_charge_power / \
            self.battery_capacity / (60 / period)

        if pilot_dsoc > max_dsoc:
            pilot_dsoc = max_dsoc

        if self.transition_soc == 1:
            curr_soc = pilot_dsoc + self.get_soc()
            if curr_soc > 1:
                curr_soc = 1
        
        else:
            # The pilot SoC rate of change has a new transition SoC at
            # which decreasing of max charging rate occurs.
            pilot_transition_soc = self.transition_soc + (
                pilot_dsoc - max_dsoc
            ) / max_dsoc * (self.transition_soc - 1)

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

        dsoc = curr_soc - self.get_soc()
        self.prev_capacity = self.current_capacity
        self.current_capacity = curr_soc * self.battery_capacity

        # For charging power and charging rate (current), we use the
        # the average over this time period.
        self.current_energy = dsoc * self.battery_capacity  # / (period / 60)
        self.required_energy = self.required_energy - self.current_energy  # * period / 60
        return self.current_energy / (period / 60) * 1000 / voltage

    def _discharge(self, amps, voltage, phases) -> float:
        '''
        The _discharge method is used to discharge the EV's battery.
        Inputs:
            - power: the power input in kW (it is negative because of the discharge)
        '''
        assert (amps < 0)

        voltage = voltage * math.sqrt(phases)
        
        given_power = (amps * voltage / 1000)
        
        if abs(given_power/1000) > abs(self.max_discharge_power):
            given_power = self.max_discharge_power

        given_energy = given_power * self.discharge_efficiency * self.timescale / 60        
        if self.current_capacity + given_energy < self.min_battery_capacity:
            self.current_energy = -(self.current_capacity - self.min_battery_capacity)
            given_energy = self.current_energy
            self.prev_capacity = self.current_capacity
            self.current_capacity = self.min_battery_capacity
        else:
            self.current_energy = given_energy  # * 60 / self.timescale
            self.prev_capacity = self.current_capacity
            self.current_capacity += given_energy        
        
        self.required_energy = self.required_energy + self.current_energy

        return given_energy*60/self.timescale * 1000 / voltage

    def get_battery_degradation(self) -> Tuple[float, float]:
        '''
        A function that returns the capacity loss of the EV.

        Qacc := Accumulated battery cell throughput (Ah)
        Qsim := Battery cell throughput during simulation (Ah)        
        Tacc := Battery age (days)
        Tsim := Simulation time (days)
        theta := Battery temperature (K)

        Outputs: 
            - Capacity loss: the capacity loss
        '''

        # Degradation modelling parameters
        e0 = 7.543e6
        e1 = 23.75e6
        e2 = 6976

        z0 = 7.348e-3
        z1 = 3.667
        z2 = 7.6e-4
        z3 = 4.081e-3

        b_cap_ah = 2.05  # ah
        b_cap_kwh = 78  # kwh

        d_dist = 15000  # km
        b_age = 2*365  # days
        G = 0.186  # kwh/km

        # Age of the battery in days
        T_acc = b_age

        # Simulation time in days
        T_sim = (self.time_of_departure - self.time_of_arrival + 1)*self.timescale/ (60*24) # days

        theta = 298.15  # Kelvin
        k = 0.8263  # Volts

        v_min = 3.3324  # Volts
        self.historic_soc.append(self.get_soc()) # Add the final soc to the historic soc
        avg_soc = np.mean(self.historic_soc) 
        v_avg = v_min + k * avg_soc

        # alpha(v_avg)
        alpha = (e0 * v_avg - e1) * math.exp(-e2 / theta)
        d_cal = alpha * 0.75 * T_sim / (T_acc)**0.25

        # beta(v_avg, soc_avg)
        # print(f'avg_soc: {avg_soc}')        
        self.active_steps.append(1)
        
        # get historic soc that self.active_steps == 1
        filtered_historic_soc = [soc for i, soc in enumerate(self.historic_soc) if self.active_steps[i] == 1]
        # print(f'filtered soc {filtered_historic_soc}')
        avg_filtered_soc = np.mean(filtered_historic_soc) 
        
        delta_DoD = 2 * abs(avg_filtered_soc.repeat(len( filtered_historic_soc)) - filtered_historic_soc).mean()
        # print(f'delta_DoD: {delta_DoD}')
        v_half_soc = v_min + k * 0.5
        beta = z0 * (v_half_soc - z1)**2 + z2 + z3 * delta_DoD        

        Q_sim = (self.abs_total_energy_exchanged / b_cap_kwh) * b_cap_ah

        # accumulated throughput
        Q_acc = 2 * (b_age * (d_dist / 365) * G * b_cap_ah) / b_cap_kwh
        # print(f'Q_acc: {Q_acc}')

        d_cyc = beta * 0.5 * Q_sim / (Q_acc)**0.5

        self.calendar_loss = d_cal
        self.cyclic_loss = d_cyc
        
        return d_cal, d_cyc
