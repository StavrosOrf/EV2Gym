"""
This file contains the implementation of the MPC baseline for the profit and flexibility maximization problem.

Authors: Cesar Diaz-Londono, Stavros Orfanoudakis
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod


class MPC(ABC):

    def __init__(self, env, control_horizon=10, verbose=False):
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
        Cx0 = np.zeros(self.EV_number)  # Initial SoC conditions [kWh] for EVs

        # Assume all EVs have enough time to reach the final SoC !!!
        Cxf = np.zeros(self.EV_number)  # Final SoC conditions [kWh] for EVs
        # Arrival time of each EV in steps
        arrival_times = np.zeros(self.EV_number, dtype=int)
        # Departure time of each EV in steps
        departure_times = np.zeros(self.EV_number, dtype=int)

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

        # EVs Scheduling and specs based on the EVsSimulator environment
        for index, EV in enumerate(env.EVs_profiles):

            if index == 0:
                # Assume all EVs have the same charging and discharging efficiency !!!
                self.ch_eff = EV.charge_efficiency
                self.disch_eff = EV.discharge_efficiency
                self.min_SoC = EV.min_battery_capacity/EV.battery_capacity

            # Assume all EVs have the same characteristics !!!
            Cx0[index] = EV.battery_capacity_at_arrival
            Cxf[index] = EV.desired_capacity
            arrival_times[index] = EV.time_of_arrival

            if EV.time_of_departure > self.simulation_length:
                departure_times[index] = self.simulation_length
            else:
                departure_times[index] = EV.time_of_departure

            ev_location = EV.location
            self.u[ev_location, arrival_times[index]: departure_times[index]] = 1
            self.x_init[ev_location, arrival_times[index]                        : departure_times[index]] = Cx0[index]
            self.x_final[ev_location, arrival_times[index]                         : departure_times[index]] = Cxf[index]
            self.x_max_batt[ev_location, arrival_times[index]                         : departure_times[index]] = EV.battery_capacity
            ev_pmax = min(Pmax, EV.max_ac_charge_power)
            self.p_max_MT[ev_location, arrival_times[index]                          : departure_times[index]] = ev_pmax
            ev_dis_pmax = min(abs(Pmin), abs(EV.max_discharge_power))
            self.p_max_MT_dis[ev_location, arrival_times[index]                          : departure_times[index]] = ev_dis_pmax
            
            ev_pmin = max(Pmin, EV.min_ac_charge_power)
            ev_pmin = 0  # formulation does not support p_min different than 0
            ev_pmin = Pmin
            self.p_min_MT[ev_location, arrival_times[index]                          : departure_times[index]] = ev_pmin

        if self.verbose:
            print(f'Initial SoC: {Cx0}')
            print(f'Final SoC: {Cxf}')
            print(f'Arrival times: {arrival_times}')
            print(f'Departure times: {departure_times}')
            print(f'Initial conditions: {self.x_init}')
            print(f'Final conditions: {self.x_final}')
            print(f'Pmax: {self.p_max_MT}')

        self.number_of_transformers = env.number_of_transformers
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
        self.x_next = self.x_init[:, 0]  # First initial condition
        self.x_hist2 = self.x_next.reshape(-1, 1)  # Save historical SoC
        self.u_hist2 = np.empty((self.n_ports, 0))     # Save historical Power
        # Save historical flexibility
        self.cap_hist = np.empty((self.n_ports, 0))

        if self.verbose:
            print(f'Prices: {self.ch_prices}')
            print(f' Discharge Prices: {self.disch_prices}')

    @abstractmethod
    def get_action(self, t):
        pass
        
    def update_tr_power(self, t):
        '''
        This function updates the transformer power limits, loads and PV generation for the next control horizon based on forecasts.
        '''
        
        for i, tr in enumerate(self.env.transformers):
            self.tr_power_limit[i, :] = tr.get_power_limits(
                step=t, horizon=self.control_horizon)

            self.tr_pv[i, :] = np.zeros(self.control_horizon)
            self.tr_pv[i, 0] = tr.solar_power[tr.current_step]
            l = len(tr.pv_generation_forecast[tr.current_step + 1:
                                              tr.current_step+self.control_horizon+1])
            self.tr_pv[i, 1:l+1] = tr.pv_generation_forecast[tr.current_step + 1:
                                                             tr.current_step+self.control_horizon+1]

            self.tr_loads[i, :] = np.zeros(self.control_horizon)
            self.tr_loads[i, 0] = tr.inflexible_load[tr.current_step]
            l = len(tr.infelxible_load_forecast[tr.current_step + 1:
                                                tr.current_step+self.control_horizon+1])
            self.tr_loads[i, 1:l+1] = tr.infelxible_load_forecast[tr.current_step + 1:
                                                                  tr.current_step+self.control_horizon+1]

            print(f'tr_pv: {self.tr_pv[i, :]}'
                  f'\ntr_power_limit: {self.tr_power_limit[i, :]}'
                  f'\ntr_loads: {self.tr_loads[i, :]}')

    def recosntruct_state(self,t):
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
                
        self.Gxx0 = self.x_next

        if t == 0:
            for i in range(0, self.control_horizon-1):
                self.Gxx0 = np.concatenate((self.Gxx0, self.x_init[:, i]))
        else:
            for i in range(t, t + self.control_horizon-1):
                Gx1 = self.x_init[:, i]
                for j in range(self.n_ports):
                    if self.x_init[j, t] > 0 and self.x_init[j, t - 1] != 0:
                        Gx1[j] = self.x_next[j]
                self.Gxx0 = np.concatenate((self.Gxx0, Gx1))
                
        # Building final SoC self.XF vector
        self.XF = np.zeros(self.control_horizon * self.n_ports)
        m = self.n_ports
        for j in range(t + 1, t + self.control_horizon + 1):
            for i in range(self.n_ports):
                m += 1
                if self.u[i, j] == 0 and self.u[i, j - 1] == 1:
                    self.XF[m - self.n_ports-1] = self.x_final[i, j - 1]
                else:
                    self.XF[m - self.n_ports-1] = self.x_max_batt[i, j - 1] * self.min_SoC
                    
                if self.u[i, j] == 1 and self.u[i, j - 1] == 1 and self.u[i, j - 2] == 0:
                    self.XF[m - self.n_ports-1] = 0

        # Maximum capacity of EVs
        self.XMAX = np.array([self.x_max_batt[:, t + i]
                        for i in range(self.control_horizon)]).flatten()


class OCCF_V2G(MPC):

    def __init__(self, env, control_horizon=10, verbose=False):
        """
        Initialize the MPC baseline.

        Args:
            env: The environment to be used for the MPC baseline.
            horizon: The horizon of the MPC baseline.
            verbose: Whether to print debug information.
        """
        super().__init__(env, control_horizon, verbose)
        
        self.na = self.n_ports
        self.nb = 2 * self.na

    def get_action(self, t):
        """
        This function computes the MPC actions for the economic problem including V2G.
        """

        # update transformer limits 
        self.update_tr_power(t)
        
        # reconstruct self.x_next using the environment
        self.recosntruct_state(t)

        if self.verbose:
            print(f'x_next: {self.x_next}')

        

        if self.verbose:
            print(f'-------------------------------------------- \n t: {t}')
            
        # Station model
        Amono = np.dstack([np.diag(self.u[:, i])
                           for i in range(t, t + 1 + self.control_horizon)])
        
        # Bmono = self.T * np.dstack([np.diag(self.u[:, i])
        #                             for i in range(t, t + 1 + self.control_horizon)])
        
        Bmono = np.zeros((self.n_ports, self.nb, self.control_horizon+1))
        for j in range(t, t + self.control_horizon):
            Bmono2 = np.zeros((self.n_ports, self.nb))
            bnew = self.T * np.diag(self.u[:, j - 1])
            for i in range(self.n_ports):
                Bmono2[i, :self.n_ports] = self.ch_eff * bnew[:, i]
                Bmono2[i, self.n_ports:] = -bnew[:, i] / self.disch_eff
            Bmono[:, :, j - t] = Bmono2
        
        if self.verbose:
            print(f'Amono: {Amono.shape}')
            print(f'Bmono: {Bmono.shape}')
            print(f'Amono: \n {Amono}')
            print(f'Bmono: \n {Bmono}')

        # Complete model calculation Gxx0, this is the big A in the paper

        if self.verbose:
            print(f'Gxx0: {self.Gxx0.shape}')
            print(f'Gxx0: \n {self.Gxx0}')

        input("Press Enter to continue...")
        
        # Complete model calculation Gu, this is the G in the paper
        self.Gu = np.zeros((self.control_horizon * self.na,
                       self.control_horizon * self.nb))

        for i in range(self.control_horizon):
            Bbar = Bmono[:, :, 0]
            for j in range(i + 1):
                Abar = np.eye(self.n_ports)
                if i == j:
                    self.Gu[i * self.na: (i+1) * self.na, j * self.nb: (j+1)
                       * self.nb] = Bmono[:, :, j]  # H
                else:
                    for m in range(j + 1, i + 1):
                        Abar = np.dot(Abar, Amono[:, :, m])
                    self.Gu[i * self.na: (i+1) * self.na, j *
                        self.nb: (j+1) * self.nb] = np.dot(Abar, Bbar)  # H

                Bbar = Bmono[:, :, j]

        if self.verbose:
            print(f'Gu:{self.Gu.shape} \n {self.Gu}')
            
        if self.verbose:
            print(f'self.XF: {self.XF.shape} \n {self.XF}')
            print(f'self.XMAX: {self.XMAX.shape} \n {self.XMAX}')

        # Inequality constraint
        self.AU = np.vstack((self.Gu, -self.Gu))
        self.bU = np.concatenate((np.abs(self.XMAX - self.Gxx0), -self.XF + self.Gxx0))

        if self.verbose:
            print(f'AU: {self.AU.shape}, BU: {self.bU.shape}')
            # print(f'AU: \n {AU}')
            print(f'bU: \n {self.bU}')

        # Generate the min cost function
        f = []
        f2 = []
        for i in range(self.control_horizon):
            for j in range(self.n_ports):
                f.append(self.T * self.ch_prices[t + i])
                f.append(-self.T * self.disch_prices[t + i])
                
                f2.append(self.T * self.ch_prices[t + i])
                f2.append(self.T * self.disch_prices[t + i]*2)
                
        f = np.array(f).reshape(-1, 1)
        f2 = np.array(f2).reshape(-1, 1)

        if self.verbose:
            print(f'f: {f.shape}, BinEV: {BinEV.shape}')
            print(f'f: \n {f}')            
            print(f'u: {self.u.shape} \n {self.u}')

        # Boundaries of the power
        # LB = np.array([self.p_min_MT[j, i + t]
        #                for i in range(self.control_horizon)
        #                for j in range(self.n_ports)])
        
        LB = np.zeros((self.control_horizon * self.nb, 1))

        UB = np.array([[self.p_max_MT[j, i + t], self.p_max_MT_dis[j, i + t]]
                       for i in range(self.control_horizon)
                       for j in range(self.n_ports)])
        
        #flatten UB
        
        LB = LB.flatten().reshape(-1)
        UB = UB.flatten().reshape(-1)
        
    
        if self.verbose:
            print(f'LB: {LB.shape}, \n {LB}')
            print(f'UB: {UB.shape}, \n {UB}')

        # Optimization with CVXPY
        u1 = cp.Variable(self.nb * self.control_horizon, name='u1')
        CapF1 = cp.Variable(
            self.nb * self.control_horizon, name='CapF1')  # not needed for profit maxmization        
        Zbin = cp.Variable(self.n_ports * self.control_horizon, boolean=True, name='Zbin')
        
        # Constraints
        constr = [self.AU @ u1 <= self.bU,
                  CapF1 <= UB,
                  LB <= CapF1
                  ]
        
        
        
        for tr_index in range(self.number_of_transformers):
            constr.append(self.tr_cs[tr_index, :, :] @ (u1[::2] - u1[1::2])
                          + self.tr_loads[tr_index,:].T
                          - self.tr_pv[tr_index,:].T
                          <= self.tr_power_limit[tr_index])

        # Cost function
        objective = cp.Minimize(f.T @ u1 - f2.T @ CapF1)
        prob = cp.Problem(objective, constr)
        prob.solve(solver=cp.GUROBI, verbose=False)

        print("--------------------------------------------")
        if prob.status != cp.OPTIMAL:
            print(f'Objective value: {prob.status}')
            print("Optimal solution not found !!!!!")
            exit()

        u = u1.value  # Optimal power

        if self.verbose:
            print(f'u: {u.shape} \n {u}')
            # if any u is negative, then we are discharging
            # if np.any(u < 0):
            #     print("Discharging")
            #     input("Press Enter to continue...")

        # Selecting the first self.n_ports power levels
        uc = u[:self.n_ports]

        # build normalized actions
        actions = np.zeros(self.n_ports)
        for i in range(self.n_ports):
            if uc[i] > 0:
                actions[i] = uc[i]/(self.p_max_MT[i, t])
            elif uc[i] < 0:
                actions[i] = uc[i]/abs(self.p_min_MT[i, t])

        if self.verbose:
            print(f'actions: {actions.shape} \n {actions}')

        X2 = Amono[:, :, 0] @ self.x_next + Bmono[:, :, 0] @ uc
        # print(X2)

        self.x_hist2 = np.concatenate(
            (self.x_hist2, X2.reshape(-1, 1)), axis=1)
        self.u_hist2 = np.concatenate(
            (self.u_hist2, uc.reshape(-1, 1)), axis=1)
        return actions

    def plot(self):
        '''
        This function plots the results of the MPC baseline in the plot folder of the run.        
        '''

        # Colors
        rojo = [0.6350, 0.0780, 0.1840]
        azul = [0, 0.4470, 0.7410]

        # Plot price
        x_v = np.arange(0, self.simulation_length+1, step=1)
        plt.figure()
        plt.stairs(self.prices[:self.simulation_length],
                   x_v, color=azul, linewidth=1)
        plt.xlabel('Steps', fontsize=12)
        plt.ylabel('Price [€/kWh]', fontsize=12)
        # plt.legend(['Prices'], loc='best', fontsize=12)
        # plt.xlim(0.25, self.T)
        # plt.ylim(bottom=0)
        plt.grid(True)
        plt.show()

        print(f'x_v: {x_v.shape}')
        print(f'x_v: {x_v}')
        print(f'prices: {self.prices.shape}')
        print(f'prices: {self.prices}')

        print(f'x_hist2: {self.x_hist2.shape}')
        print(f'u_hist2: {self.u_hist2.shape}')
        print(f'x_hist2: {self.x_hist2}')
        print(f'u_hist2: {self.u_hist2}')

        # Charger responses
        for i in range(self.n_ports):
            x_v = np.arange(0, self.simulation_length+1, step=1)
            plt.figure()

            # plt.stairs(x_v, self.Uhist1[i, :self.simulation_length], color=rojo, linewidth=1)
            plt.stairs(self.u_hist2[i, :self.simulation_length], x_v,
                       linestyle='--', color=azul, linewidth=1)

            plt.xlabel('Time [h]', fontsize=12)
            plt.ylabel('Power [kW]', fontsize=12)
            plt.legend(['MT', 'OCCF'], loc='best', fontsize=12)

            plt.grid(True)
            plt.show()

            plt.figure()
            # plt.stairs(x_v, self.Xhist1[i, :self.simulation_length], color=rojo, linewidth=1)
            plt.stairs(self.x_hist2[i, :self.simulation_length], x_v,
                       linestyle='-.', color=azul, linewidth=1)

            plt.xlabel('Time [h]', fontsize=12)
            plt.ylabel('Energy [kWh]', fontsize=12)
            plt.legend(['MT', 'OCCF'], loc='best', fontsize=12)
            plt.grid(True)
            plt.show()
