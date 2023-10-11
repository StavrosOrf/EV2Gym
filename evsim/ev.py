'''
===================================
Author: Stavros Orfanoudakis 2023
===================================
'''

import numpy as np

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
                 desired_capacity=50,  # kWh
                 battery_capacity=50,  # kWh
                 min_desired_capacity=8,  # kWh
                 max_desired_capacity=45,  # kWh
                 max_charge_current=20,  # A
                 min_charge_current=6,  # A
                 max_discharge_current=20,  # A
                 min_discharge_current=6,  # A
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
        self.desired_capacity = desired_capacity  # kWh
        self.battery_capacity_at_arrival = battery_capacity_at_arrival  # kWh

        # EV technical characteristics
        self.battery_capacity = battery_capacity  # kWh
        self.min_desired_capacity = min_desired_capacity  # kWh
        self.max_desired_capacity = max_desired_capacity  # kWh
        self.max_charge_current = max_charge_current  # A
        self.max_discharge_current = max_discharge_current  # A

        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.v2g_enabled = v2g_enabled

        # EV status
        self.current_capacity = battery_capacity_at_arrival  # kWh
        self.current_power = 0  # kWh
        self.actual_current = 0  # A
        self.charging_cycles = 0
        self.previous_power = 0
        self.required_power = 0

    def reset(self):
        '''
        The reset method is used to reset the EV's status to the initial state.
        '''
        self.current_capacity = self.battery_capacity_at_arrival
        self.current_power = 0
        self.charging_cycles = 0
        self.previous_power = 0

    def step(self, amps, voltage):
        '''
        The step method is used to update the EV's status according to the actions taken by the EV charger.
        Inputs:
            - action: the power input in kW (positive for charging, negative for discharging)
        Outputs:
            - self.current_power: the current power input of the EV in kW (positive for charging, negative for discharging)
            - self.actual_curent: the actual current input of the EV in A (positive for charging, negative for discharging)
        '''
        if amps == 0:
            self.current_power = 0
            return 0, 0

        # If the action is different than the previous action, then increase the charging cycles
        if self.previous_power == 0 or (self.previous_power/amps) < 0:
            self.charging_cycles += 1

        if amps > 0:
            self._charge(amps, voltage)
        elif amps < 0:
            self._discharge(amps, voltage)

        self.previous_power = self.current_power

        return self.current_power, self

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
            return 0
        else:
            return 1

    def get_soc(self):
        '''
        A function that returns the state of charge of the EV.
        Outputs: 
            - SoC: the state of charge of the EV in [0,100] %
        '''
        return (self.current_capacity/self.battery_capacity)*100

    def get_state(self, current_step):
        '''
        A function that returns the state of the EV.
        Outputs: 
            - State: the state of the EV
        '''
        timestep_left = self.earlier_time_of_departure - current_step

        # , self.charging_cycles
        return self.get_soc()/100, timestep_left / self.simulation_length

    def __str__(self):
        return f' {self.current_power:5.1f} kWh |' + \
            f' {(self.current_capacity/self.battery_capacity)*100:5.1f} % |' + \
            f' {self.charging_cycles:2d}  |' + \
            f' t_dep: {self.earlier_time_of_departure}'

    def _charge(self, amps, voltage):
        '''
        The _charge method is used to charge the EV's battery.
        Inputs:
            - power: the power input in kW
        '''

        assert (amps > 0)
        given_power = (amps * voltage / 1000) * \
            self.charge_efficiency * self.timescale / 60  # KW

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
            self.current_capacity += giving_power



class Battery():
    """This class models the behavior of a battery and battery management system (BMS).

    Args:
        capacity (float): Capacity of the battery [kWh]
        init_charge (float): Initial charge of the battery [kWh]
        max_power (float): Maximum charging rate of the battery [kW]
    """

    def __init__(self, capacity, init_charge, max_power):
        if init_charge > capacity:
            raise ValueError("Initial Charge cannot be greater than capacity.")
        self._capacity = capacity
        self._current_charge = init_charge
        self._init_charge = init_charge
        self._max_power = max_power
        self._current_charging_power = 0

    @property
    def _soc(self):
        """ Returns the state of charge of the battery as a percent."""
        return self._current_charge / self._capacity

    @property
    def max_charging_power(self):
        """ Returns the maximum charging power of the Battery."""
        return self._max_power

    @property
    def current_charging_power(self):
        """ Returns the current draw of the battery on the AC side."""
        return self._current_charging_power

    def charge(self, pilot, voltage, period):
        """ Method to "charge" the battery

        Args:
            pilot (float): Pilot signal passed to the battery. [A]
            voltage (float): AC voltage provided to the battery charger. [V]
            period (float): Length of the charging period. [minutes]

        Returns:
            float: actual charging rate of the battery. [A]

        Raises:
            ValueError: if voltage or period are <= 0.
        """
        if voltage <= 0:
            raise ValueError("Voltage must be greater than 0. Got {0}".format(voltage))
        if period <= 0:
            raise ValueError("period must be greater than 0. Got {0}".format(voltage))

        # Rate which would fill the battery in period minutes.
        rate_to_full = (self._capacity - self._current_charge) / (period / 60)

        charge_power = min([pilot * voltage / 1000, self._max_power, rate_to_full])
        self._current_charge += charge_power * (period / 60)
        self._current_charging_power = charge_power
        return charge_power * 1000 / voltage

    def reset(self, init_charge=None):
        """ Reset battery to initial state. If init_charge is not
        given (is None), the battery is reset to its initial charge
        on initialization.

        Args:
            init_charge (float): charge battery should be reset to. [acnsim units]

        Returns:
            None
        """
        if init_charge is None:
            self._current_charge = self._init_charge
        else:
            if init_charge > self._capacity:
                raise ValueError("Initial Charge cannot be greater than capacity.")
            self._current_charge = init_charge
        self._current_charging_power = 0

    def _to_dict(
        self, context_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """ Implements BaseSimObj._to_dict. """
        attribute_dict = {}
        nn_attr_lst = [
            "_max_power",
            "_current_charging_power",
            "_current_charge",
            "_capacity",
            "_init_charge",
        ]
        for attr in nn_attr_lst:
            attribute_dict[attr] = getattr(self, attr)
        return attribute_dict, context_dict

class Linear2StageBattery(Battery):
    """ Extends Battery with a simple piecewise linear model of battery dynamics based on SoC.

    Battery model based on a piecewise linear approximation of battery behavior. The battery will charge at the
    minimum of max_rate and the pilot until it reaches _transition_soc. After this, the maximum charging rate of the
    battery will decrease linearly to 0 at 100% state of charge.

    For more info on model: https://www.sciencedirect.com/science/article/pii/S0378775316317396

    All public attributes are the same as Battery.

    Args:
        noise_level (float): Standard deviation of the noise to add to the charging process. (kW)
        transition_soc (float): State of charging when transitioning from constant current to constraint voltage.
        charge_calculation (str): If 'stepwise', use the charging
            method from a previous version of acnportal, which
            assumes a constant maximal charging rate for the entire
            timestep during which the pilot signal is input. This
            charging method is less accurate than the _charge method,
            and should only be used for reproducing results from
            older versions of acnportal.

            If 'continuous' or not provided, use the _charge method,
            which assumes a continuously varying maximal charging rate.
    """

    charging_methods = ["continuous", "stepwise"]

    def __init__(
        self,
        capacity,
        init_charge,
        max_power,
        noise_level=0,
        transition_soc=0.8,
        charge_calculation="continuous",
    ):
        super().__init__(capacity, init_charge, max_power)
        self._noise_level = noise_level
        if transition_soc < 0:
            raise ValueError(
                f"transition_soc must be non-negative. " f"Got {transition_soc}."
            )
        elif transition_soc >= 1:
            raise ValueError(
                f"transition_soc must be less than 1. " f"Got {transition_soc}."
            )
        self._transition_soc = transition_soc
        if charge_calculation not in self.charging_methods:
            raise ValueError(
                f"Charging method {charge_calculation} specified in "
                f"charge_calculation attribute not recognized by "
                f"Linear2StageBattery class. use one of "
                f"{self.charging_methods}."
            )
        self.charge_calculation = charge_calculation

    def charge(self, pilot, voltage, period):
        """ Method to "charge" the battery based on a two-stage linear
        battery model.

        Uses one of
        {_charge, _charge_stepwise}
        to charge the battery depending on the value of the
        charge_calculation attribute of this object.
        """
        if self.charge_calculation == "stepwise":
            return self._charge_stepwise(pilot, voltage, period)
        elif self.charge_calculation == "continuous":
            return self._charge(pilot, voltage, period)
        else:
            raise ValueError(
                f"Charging method {self.charge_calculation} specified "
                f"in charge_calculation attribute not recognized by "
                f"Linear2StageBattery class. use one of "
                f"{self.charging_methods}."
            )

    def _charge(self, pilot, voltage, period):
        """ Method to "charge" the battery based on a two-stage linear
        battery model.

        All calculations are done in terms fo battery state of charge
        (SoC). Results are converted back to power units at the end.

        Args:
            pilot (float): Pilot signal passed to the battery. [A]
            voltage (float): AC voltage provided to the battery
                charger. [V]
            period (float): Length of the charging period. [minutes]

        Returns:
            float: average charging rate of the battery over this single
                period.

        """
        if voltage <= 0:
            raise ValueError(f"Voltage must be greater than 0. Got {voltage}.")
        if period <= 0:
            raise ValueError(f"Period must be greater than 0. Got {period}.")
        if pilot == 0:
            self._current_charging_power = 0
            return 0
        # All calculations are done in terms of battery SoC, so we
        # convert pilot signal and max power into pilot and max rate of
        # change of SoC.
        pilot_dsoc = pilot * voltage / 1000 / self._capacity / (60 / period)
        max_dsoc = self._max_power / self._capacity / (60 / period)

        if pilot_dsoc > max_dsoc:
            pilot_dsoc = max_dsoc

        # The pilot SoC rate of change has a new transition SoC at
        # which decreasing of max charging rate occurs.
        pilot_transition_soc = self._transition_soc + (
            pilot_dsoc - max_dsoc
        ) / max_dsoc * (self._transition_soc - 1)

        if pilot < 0:
            warnings.warn(
                f"Negative pilot signal input. Battery models"
                f"may not be accurate for pilot {pilot} A."
            )

        # The charging equation depends on whether the current SoC of
        # the battery is above or below the new transition SoC.
        if self._soc < pilot_transition_soc:
            # In the pre-rampdown region, the charging equation changes
            # depending on whether charging the battery over this
            # time period causes the battery to transition between
            # charging regions.
            if 1 <= (pilot_transition_soc - self._soc) / pilot_dsoc:
                curr_soc = pilot_dsoc + self._soc
            else:
                curr_soc = 1 + np.exp(
                    (pilot_dsoc + self._soc - pilot_transition_soc)
                    / (pilot_transition_soc - 1)
                ) * (pilot_transition_soc - 1)
        else:
            curr_soc = 1 + np.exp(pilot_dsoc / (pilot_transition_soc - 1)) * (
                self._soc - 1
            )

        # Add subtractive noise to the final SoC, scaling the noise
        # such that _noise_level is the standard deviation of the noise
        # in the battery charging power.
        if self._noise_level > 0:
            raw_noise = np.random.normal(0, self._noise_level)
            scaled_noise = raw_noise * (period / 60) / self._capacity
            curr_soc -= abs(scaled_noise)

        dsoc = curr_soc - self._soc
        self._current_charge = curr_soc * self._capacity

        # For charging power and charging rate (current), we use the
        # the average over this time period.
        self._current_charging_power = dsoc * self._capacity / (period / 60)
        return self._current_charging_power * 1000 / voltage