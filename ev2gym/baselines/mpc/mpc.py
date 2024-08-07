"""
This file contains the implementation of the MPC baseline for the profit and flexibility maximization problem.

Authors: Cesar Diaz-Londono, Stavros Orfanoudakis
"""

import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod


class MPC(ABC):

    def __init__(self,
                 env,
                 control_horizon=25,
                 verbose=False,
                 time_limit=200,
                 output_flag=0,
                 MIPGap=None,
                 **kwargs):
        """
        Initialize the MPC baseline.

        Args:
            env: The environment to be used for the MPC baseline.
            horizon: The horizon of the MPC baseline.
            verbose: Whether to print debug information.
        """

        self.env = env
        self.n_ports = env.number_of_ports  # Total number of EVSEs to be controlled
        self.T = env.timescale/60  # Time scale in hours
        # Total number of EVs that will spawn
        self.EV_number = len(env.EVs_profiles)
        assert self.EV_number > 0, "No EVs in the simulation, reset the environment and try again."

        self.simulation_length = env.simulation_length  # Simulation length in steps
        self.t_min = env.timescale  # Time scale in minutes
        self.control_horizon = control_horizon  # prediction horizon in steps
        self.total_exec_time = 0

        self.output_flag = output_flag
        self.time_limit = time_limit
        self.MIPGap = MIPGap
        self.verbose = verbose

        if self.verbose:
            np.set_printoptions(linewidth=np.inf)
            print(f'Number of EVs: {self.EV_number}')
            print(f'Number of ports: {self.n_ports}')
            print(f'Simulation length: {self.simulation_length}')
            print(f'Time scale: {self.T}')
            print(f'Prediction horizon: {self.control_horizon}')

        # Assume all Chargers have the same characteristics and have only one port!!!
        assert env.charging_stations[0].n_ports == 1, "MPC baseline only works with one port per charger."
        # Maximum power per EVSE
        Pmax = env.charging_stations[0].get_max_power()
        Pmin = env.charging_stations[0].get_min_power()

        # Assume all EVs have the same power intake characteristics, and can receive Pmax !!!
        # Initial SoC conditions [kWh] for EVs
        self.Cx0 = np.zeros(self.EV_number)

        # Assume all EVs have enough time to reach the final SoC !!!
        # Final SoC conditions [kWh] for EVs
        self.Cxf = np.zeros(self.EV_number)
        # Arrival time of each EV in steps
        self.arrival_times = np.zeros(self.EV_number, dtype=int)
        # Departure time of each EV in steps
        self.departure_times = np.zeros(self.EV_number, dtype=int)

        # Scheduling matrix [1, 0] (if EV is connected or not)
        self.u = np.zeros(
            (self.n_ports, self.simulation_length + self.control_horizon + 1))
        # Matrix for the location of initial conditions
        self.x_init = np.zeros(
            (self.n_ports, self.simulation_length + self.control_horizon + 1))
        # Matrix for the location of final conditions
        self.x_final = np.zeros(
            (self.n_ports, self.simulation_length + self.control_horizon + 1))
        self.x_max_batt = np.zeros(
            (self.n_ports, self.simulation_length + self.control_horizon + 1))
        # Matrix with maximum powers
        self.p_max_MT = np.zeros(
            (self.n_ports, self.simulation_length + self.control_horizon + 1))
        self.p_max_MT_dis = np.zeros(
            (self.n_ports, self.simulation_length + self.control_horizon + 1))
        # Matrix with minimum powers
        self.p_min_MT = np.zeros(
            (self.n_ports, self.simulation_length + self.control_horizon + 1))
        
        # the charger each EV is connected to
        self.ev_locations = np.zeros(self.EV_number, dtype=int)
        # The maximum battery capacity of each EV
        self.ev_max_batt = np.zeros(self.EV_number, dtype=int)

        self.max_ch_power = np.zeros(self.n_ports)
        self.max_disch_power = np.zeros(self.n_ports)

        for i, cs in enumerate(env.charging_stations):
            self.max_ch_power[i] = cs.get_max_power()
            self.max_disch_power[i] = cs.get_min_power()

        # EVs Scheduling and specs based on the ev2gym environment
        for index, EV in enumerate(env.EVs_profiles):

            if index == 0:
                # Assume all EVs have the same charging and discharging efficiency !!!
                self.ch_eff = EV.charge_efficiency
                self.disch_eff = EV.discharge_efficiency
                self.min_SoC = EV.min_battery_capacity/EV.battery_capacity

            # Assume all EVs have the same characteristics !!!
            self.Cx0[index] = EV.battery_capacity_at_arrival
            self.Cxf[index] = EV.desired_capacity
            self.arrival_times[index] = EV.time_of_arrival

            if EV.time_of_departure > self.simulation_length:
                self.departure_times[index] = self.simulation_length
            else:
                self.departure_times[index] = EV.time_of_departure + 1

            ev_location = EV.location

            self.ev_locations[index] = ev_location
            self.ev_max_batt[index] = EV.battery_capacity
            
            self.u[ev_location, self.arrival_times[index]:
                   self.departure_times[index]] = 1
            self.x_init[ev_location, self.arrival_times[index]:
                        self.departure_times[index]] = self.Cx0[index]

            self.x_final[ev_location, self.arrival_times[index]:
                         self.departure_times[index]] = self.Cxf[index]

            self.x_max_batt[ev_location, self.arrival_times[index]:
                            self.departure_times[index]] = EV.battery_capacity
            ev_pmax = min(Pmax, EV.max_ac_charge_power)
            self.p_max_MT[ev_location, self.arrival_times[index]:
                          self.departure_times[index]] = ev_pmax
            ev_dis_pmax = min(abs(Pmin), abs(EV.max_discharge_power))
            self.p_max_MT_dis[ev_location, self.arrival_times[index]:
                              self.departure_times[index]] = ev_dis_pmax

            ev_pmin = max(abs(Pmin), EV.min_ac_charge_power)
            ev_pmin = 0  # formulation does not support p_min different than 0
            ev_pmin = Pmin
            self.p_min_MT[ev_location, self.arrival_times[index]:
                          self.departure_times[index]] = ev_pmin

        if self.verbose:
            print(f'Initial SoC: {self.Cx0}')
            print(f'Final SoC: {self.Cxf}')
            print(f'Arrival times: {self.arrival_times}')
            print(f'Departure times: {self.departure_times}')
            print(f'Initial conditions: {self.x_init}')
            print(f'Final conditions: {self.x_final}')
            print(f'Pmax: {self.p_max_MT}')

        self.number_of_transformers = env.number_of_transformers
        self.cs_transformers = env.cs_transformers
        self.tr_loads = np.zeros(
            (self.number_of_transformers, self.control_horizon))
        self.tr_pv = np.zeros(
            (self.number_of_transformers, self.control_horizon))
        self.tr_power_limit = np.zeros(
            (self.number_of_transformers, self.control_horizon))

        self.tr_cs = np.zeros((self.number_of_transformers,
                               self.control_horizon,
                               self.control_horizon*self.n_ports))

        for tr_index in range(self.number_of_transformers):
            # print(np.array(env.cs_transformers)==tr_index)
            for i in range(self.control_horizon):
                self.tr_cs[tr_index, i,
                           i*self.n_ports:
                               (i+1)*self.n_ports] = np.array(env.cs_transformers) == tr_index

        if self.verbose:
            print(f'Transformer loads: {self.tr_loads.shape}')
            print(f'{self.tr_loads}')
            print(f'Transformer Power Limit: {self.tr_power_limit.shape}')
            print(f'{self.tr_power_limit}')
            print(f'Transformer to CS: {self.tr_cs.shape}')
            print(f'{self.tr_cs}')

        # Assume every charging station has the same energy prices
        # prices per KWh for the whole simulation
        self.ch_prices = abs(env.charge_prices[0, :])
        self.disch_prices = abs(env.discharge_prices[0, :])

        # extend prices for the control horizon
        self.ch_prices = np.concatenate(
            (self.ch_prices, np.ones(self.control_horizon)*100000))
        self.disch_prices = np.concatenate(
            (self.disch_prices, np.zeros(self.control_horizon)))

        self.opti_info = []
        self.x_next = self.x_init[:, 0].copy()  # First initial condition

        if self.verbose:
            print(f'Prices: {self.ch_prices}')
            print(f' Discharge Prices: {self.disch_prices}')

        # parameters for the MPC v2 model

        self.varch2 = 0

        self.d_cycHist_e2 = []
        self.d_calHist_e2 = []
        
        self.Xhist_e2 = np.zeros((self.n_ports, self.simulation_length))
        self.Uhist_e2 = np.zeros((self.n_ports, self.simulation_length))
        self.Uhist_e2V = np.zeros((self.n_ports, self.simulation_length))

    @abstractmethod
    def get_action(self, env):
        pass

    def update_tr_power(self, t):
        '''
        This function updates the transformer power limits, loads and PV generation for the next control horizon based on forecasts.
        '''

        for i, tr in enumerate(self.env.transformers):
            self.tr_power_limit[i, :] = tr.get_power_limits(
                step=t, horizon=self.control_horizon)

            self.tr_pv[i, :] = np.zeros(self.control_horizon)
            self.tr_pv[i, 0] = tr.solar_power[tr.current_step+1]
            l = len(tr.pv_generation_forecast[tr.current_step + 2:
                                              tr.current_step+self.control_horizon+1])

            if l >= self.control_horizon - 1:
                l = self.control_horizon - 1
            else:
                l = l + 1
            self.tr_pv[i, 1:l] = tr.pv_generation_forecast[tr.current_step + 2:
                                                           tr.current_step+self.control_horizon]
            self.tr_loads[i, :] = np.zeros(self.control_horizon)
            self.tr_loads[i, 0] = tr.inflexible_load[tr.current_step+1]
            self.tr_loads[i, 1:l] = tr.inflexible_load_forecast[tr.current_step + 2:
                                                                tr.current_step+self.control_horizon]

    def update_tr_power_oracle(self, t):
        '''
        This function updates the transformer power limits, loads and PV generation for the next control horizon based on forecasts.
        '''

        for i, tr in enumerate(self.env.transformers):
            self.tr_power_limit[i, :] = tr.max_power
            self.tr_pv[i, :] = tr.solar_power
            self.tr_loads[i, :] = tr.inflexible_load

    def reconstruct_state(self, t):
        '''
        This function reconstructs the state of the environment using the historical data.
        '''
        counter = 0
        for charger in self.env.charging_stations:
            for ev in charger.evs_connected:
                if ev is None:
                    self.x_next[counter] = 0
                else:
                    self.x_next[counter] = ev.current_capacity
                counter += 1

        self.Gxx0 = self.x_next.copy()

        if t == 0:
            for i in range(0, self.control_horizon-1):
                self.Gxx0 = np.concatenate(
                    (self.Gxx0, self.x_init[:, i].copy()))
        else:
            for i in range(t, t + self.control_horizon-1):
                Gx1 = self.x_init[:, i].copy()
                for j in range(self.n_ports):
                    if self.x_init[j, t] > 0 and self.x_init[j, t - 1] != 0:
                        Gx1[j] = self.x_next[j].copy()
                self.Gxx0 = np.concatenate((self.Gxx0, Gx1))

        # Maximum capacity of EVs
        self.XMAX = np.array([self.x_max_batt[:, t + i]
                              for i in range(self.control_horizon)]).flatten()

    def calculate_XF_G2V(self, t):
        # Building final SoC self.XF vector
        self.XF = np.zeros(self.control_horizon * self.n_ports)
        m = self.n_ports
        for j in range(t + 1, t + self.control_horizon + 1):
            for i in range(self.n_ports):
                m += 1
                if self.u[i, j] == 0 and self.u[i, j - 1] == 1:
                    self.XF[m - self.n_ports-1] = self.x_final[i, j - 1]

    def calculate_XF_V2G(self, t):
        # Building final SoC self.XF vector
        self.XF = np.zeros(self.control_horizon * self.n_ports)
        m = self.n_ports
        for j in range(t + 1, t + self.control_horizon + 1):
            for i in range(self.n_ports):
                m += 1
                if self.u[i, j] == 0 and self.u[i, j - 1] == 1:
                    self.XF[m - self.n_ports-1] = self.x_final[i, j - 1]
                else:
                    self.XF[m - self.n_ports - 1] = self.x_max_batt[i, j - 1] * \
                        self.min_SoC

                if self.u[i, j] == 1 and self.u[i, j - 1] == 1 and self.u[i, j - 2] == 0:
                    self.XF[m - self.n_ports-1] = 0

    def v2g_station_models(self, t):
        '''
        This function builds the station models for the V2G problem.
        '''

        # Station model
        self.Amono = np.dstack([np.diag(self.u[:, i])
                                for i in range(t, t + 1 + self.control_horizon)])

        self.Bmono = np.zeros((self.n_ports, self.nb, self.control_horizon+1))
        for j in range(t, t + self.control_horizon+1):
            Bmono2 = []
            bnew = self.T * np.diag(self.u[:, j]).T
            for i in range(self.n_ports):
                Bmono2.append(self.ch_eff * bnew[:, i])
                Bmono2.append(-self.disch_eff * bnew[:, i])
            Bmono2 = np.array(Bmono2).T
            self.Bmono[:, :, j - t] = Bmono2

    def g2v_station_models(self, t):

        self.Amono = np.dstack([np.diag(self.u[:, i])
                                for i in range(t, t + 1 + self.control_horizon)])

        self.Bmono = self.ch_eff * self.T * np.dstack([np.diag(self.u[:, i])
                                                       for i in range(t, t + 1 + self.control_horizon)])

    def calculate_InequalityConstraints(self, t):
        '''
        This function calculates the inequality constraints for the optimization problem.
        Au and bu are the inequality constraints.
        '''

        # Complete model calculation Gu, this is the G in the paper
        self.Gu = np.zeros((self.control_horizon * self.na,
                            self.control_horizon * self.nb))

        for i in range(self.control_horizon):
            Bbar = self.Bmono[:, :, 0]
            for j in range(i + 1):
                Abar = np.eye(self.n_ports)
                if i == j:
                    self.Gu[i * self.na: (i+1) * self.na, j * self.nb: (j+1)
                            * self.nb] = self.Bmono[:, :, j]
                else:
                    for m in range(j + 1, i + 1):
                        Abar = np.dot(Abar, self.Amono[:, :, m])
                    self.Gu[i * self.na: (i+1) * self.na, j *
                            self.nb: (j+1) * self.nb] = np.dot(Abar, Bbar)

                Bbar = self.Bmono[:, :, j]

        # Inequality constraint
        self.AU = np.vstack((self.Gu, -self.Gu))
        self.bU = np.concatenate(
            (np.abs(self.XMAX - self.Gxx0), -self.XF + self.Gxx0))

    def set_power_limits_V2G(self, t):
        '''
        This function sets the power limits for the V2G problem.
        '''
        # Boundaries of the power
        # self.LB = np.array([self.p_min_MT[j, i + t]
        #                for i in range(self.control_horizon)
        #                for j in range(self.n_ports)])

        self.LB = np.zeros((self.control_horizon * self.nb, 1), dtype=float)

        self.UB = np.array([[self.p_max_MT[j, i + t], self.p_max_MT_dis[j, i + t]]
                            for i in range(self.control_horizon)
                            for j in range(self.n_ports)], dtype=float)

        self.LB = self.LB.flatten().reshape(-1)
        self.UB = self.UB.flatten().reshape(-1)

    def set_power_limits_G2V(self, t):
        '''
        This function sets the power limits for the G2V problem.
        '''
        self.LB = np.zeros((self.control_horizon * self.nb, 1), dtype=float)

        self.UB = np.array([self.p_max_MT[j, i + t]
                            for i in range(self.control_horizon)
                            for j in range(self.n_ports)], dtype=float)

        self.LB = self.LB.flatten().reshape(-1)
        self.UB = self.UB.flatten().reshape(-1)

    def print_info(self, t):
        '''
        This function prints the information of the optimization problem.
        '''
        print(f'-------------------------------------------- \n t: {t}')
        for tr in range(self.number_of_transformers):
            print(f'Transformer {tr}:')
            print(f' - tr_pv: {self.tr_pv[tr, :]}')
            print(f' - tr_loads: {self.tr_loads[tr, :]}')
            print(f' - tr_power_limit: {self.tr_power_limit[tr, :]}')

        print(f'x_next: {self.x_next}')
        print(f'Amono: {self.Amono.shape}')
        # print(f'Amono: {self.Amono}')
        print(f'Bmono: {self.Bmono.shape}')
        # print(f'Bmono: {self.Bmono}')
        print(f'Gxx0: {self.Gxx0.shape}')
        print(f'Gxx0:{self.Gxx0}')
        print(f'Gu:{self.Gu.shape}')
        print(f'Gu:{self.Gu}')
        print(f'self.XF: {self.XF.shape}')
        print(f'XF: {self.XF}')
        print(f'self.XMAX: {self.XMAX.shape}')
        print(f'xmax: {self.XMAX}')
        print(f'AU: {self.AU.shape}, BU: {self.bU.shape}')
        # print(f'AU: {self.AU}')
        # print(f'bu: {self.bU}')
        print(f'self.LB: {self.LB.shape}')
        print(f'self.UB: {self.UB.shape} ')
        print(f'UB: {self.UB}')
        print(f'u: {self.u[:, t:t+self.control_horizon]}')
        print(f'Initial SoC: {self.Cx0}')
        print(f'Final SoC: {self.Cxf}')
        print(f'Arrival times: {self.arrival_times}')
        print(f'Departure times: {self.departure_times}')
        print(f'P_max_MT: {self.p_max_MT}')

        # print(f'x_init: {self.x_init}')
        print(f'Desired Final: {self.x_final}')
