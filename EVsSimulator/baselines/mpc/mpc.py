"""
This file contains the implementation of the MPC baseline for the profit and flexibility maximization problem.

Authors: Cesar Diaz-Londono, Stavros Orfanoudakis
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

class MPC():

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

        self.N = env.simulation_length  # Simulation length in steps
        self.t_min = env.timescale  # Time scale in minutes
        self.control_horizon = control_horizon  # prediction horizon in steps

        self.verbose = verbose

        if self.verbose:
            np.set_printoptions(linewidth=np.inf)
            print(f'Number of EVs: {self.EV_number}')
            print(f'Number of ports: {self.n_ports}')
            print(f'Simulation length: {self.N}')
            print(f'Time scale: {self.T}')
            print(f'Prediction horizon: {self.control_horizon}')

        # Assume all Chargers have the same characteristics and have only one port!!!
        assert env.charging_stations[0].n_ports == 1, "MPC baseline only works with one port per charger."
        # Maximum power per EVSE
        Pmax = env.charging_stations[0].get_max_power()

        # Assume all EVs have the same power intake characteristics, and can receive Pmax !!!
        Cx0 = np.zeros(self.EV_number)  # Initial SoC conditions [kWh] for EVs

        # Assume all EVs have enough time to reach the final SoC !!!
        Cxf = np.zeros(self.EV_number)  # Final SoC conditions [kWh] for EVs
        # Arrival time of each EV in steps
        arrival_times = np.zeros(self.EV_number, dtype=int)
        # Departure time of each EV in steps
        departure_times = np.zeros(self.EV_number, dtype=int)

        # Scheduling matrix [1, 0] (if EV is connected or not)
        self.u = np.zeros((self.n_ports, self.N + self.control_horizon + 1))
        # Matrix for the location of initial conditions
        self.x_init = np.zeros(
            (self.n_ports, self.N + self.control_horizon + 1))
        # Matrix for the location of final conditions
        self.x_final = np.zeros(
            (self.n_ports, self.N + self.control_horizon + 1))
        # Matrix with maximum powers
        self.p_max_MT = np.zeros(
            (self.n_ports, self.N + self.control_horizon + 1))
        # Matrix with minimum powers
        self.p_min_MT = np.zeros(
            (self.n_ports, self.N + self.control_horizon + 1))

        # EVs Scheduling and specs based on the EVsSimulator environment
        for index, EV in enumerate(env.EVs_profiles):

            # Assume all EVs have the same characteristics !!!
            Cx0[index] = EV.battery_capacity_at_arrival
            Cxf[index] = EV.desired_capacity
            arrival_times[index] = EV.time_of_arrival

            if EV.time_of_departure > self.N:
                departure_times[index] = self.N
            else:
                departure_times[index] = EV.time_of_departure

            ev_location = EV.location
            self.u[ev_location, arrival_times[index]                   : departure_times[index]] = 1
            self.x_init[ev_location, arrival_times[index]
                : departure_times[index]] = Cx0[index]
            self.x_final[ev_location, arrival_times[index]
                : departure_times[index]] = Cxf[index]
            self.p_max_MT[ev_location, arrival_times[index]
                : departure_times[index]] = Pmax
            self.p_min_MT[ev_location, arrival_times[index]
                : departure_times[index]] = EV.min_ac_charge_power

        if self.verbose:
            print(f'Initial SoC: {Cx0}')
            print(f'Final SoC: {Cxf}')
            print(f'Arrival times: {arrival_times}')
            print(f'Departure times: {departure_times}')
            print(f'Initial conditions: {self.x_init}')
            print(f'Final conditions: {self.x_final}')
            print(f'Pmax: {self.p_max_MT}')

        # Assume every charging station has the same energy prices
        # prices per KWh for the whole simulation
        self.prices = abs(env.charge_prices[0, :])

        # extend prices for the control horizon
        self.prices = np.concatenate(
            (self.prices, np.ones(self.control_horizon)*100000))

        self.opti_info = []
        self.x_next = self.x_init[:, 0]  # First initial condition
        self.x_hist2 = self.x_next.reshape(-1, 1)  # Save historical SoC
        self.u_hist2 = np.empty((self.n_ports, 0))     # Save historical Power
        # Save historical flexibility
        self.cap_hist = np.empty((self.n_ports, 0))

        if self.verbose:
            print(f'Prices: {self.prices}')

    def min_time(self):
        """
        This function computes the minimum time to charge an EV to its desired SoC.
        """

        # Initialization
        self.x_init = self.x_init

    def get_actions(self, t):
        """
        This function computes the MPC baseline actions for the environment.
        It is called by the environment at every step.
        """

        # reconstruct self.x_next using the environment
        counter = 0
        for charger in self.env.charging_stations:
            for ev in charger.evs_connected:
                if ev is None:
                    self.x_next[counter] = 0
                else:
                    self.x_next[counter] = ev.current_capacity
                counter += 1

        if self.verbose:
            print(f'x_next: {self.x_next}')

        na = nb = self.n_ports

        # for t in range(self.N):  # Loop for every time slot
        print(f'-------------------------------------------- \n t: {t}')
        # Station model
        Amono = np.dstack([np.diag(self.u[:, i])
                           for i in range(t, t + 1 + self.control_horizon)])
        Bmono = self.T * np.dstack([np.diag(self.u[:, i])
                                    for i in range(t, t + 1 + self.control_horizon)])
        if self.verbose:
            print(f'Amono: {Amono.shape}')
            print(f'Bmono: {Bmono.shape}')
            print(f'Amono: \n {Amono}')
            print(f'Bmono: \n {Bmono}')

        # Complete model calculation Gxx0, this is the big A in the paper
        Gxx0 = self.x_next

        #!!!!! Do we want to include the step now?? Yes
        if t == 0:
            for i in range(0, self.control_horizon-1):
                Gxx0 = np.concatenate((Gxx0, self.x_init[:, i]))
        else:
            for i in range(t, t + self.control_horizon-1):
                Gx1 = self.x_init[:, i]
                for j in range(self.n_ports):
                    if self.x_init[j, t] > 0 and self.x_init[j, t - 1] != 0:
                        Gx1[j] = self.x_next[j]
                Gxx0 = np.concatenate((Gxx0, Gx1))

        if self.verbose:
            print(f'Gxx0: {Gxx0.shape}')
            print(f'Gxx0: \n {Gxx0}')

        # Complete model calculation Gu, this is the G in the paper
        Gu = np.zeros((self.control_horizon * na,
                       self.control_horizon * nb))

        for i in range(self.control_horizon):
            Bbar = Bmono[:, :, 0]
            for j in range(i + 1):
                Abar = np.eye(self.n_ports)
                if i == j:
                    Gu[i * na: (i+1) * na, j * nb: (j+1)
                       * nb] = Bmono[:, :, j]  # H
                else:
                    for m in range(j + 1, i + 1):
                        Abar = np.dot(Abar, Amono[:, :, m])
                    Gu[i * na: (i+1) * na, j *
                        nb: (j+1) * nb] = np.dot(Abar, Bbar)  # H

                Bbar = Bmono[:, :, j]
                
        if self.verbose:
            print(f'Gu:{Gu.shape} \n {Gu}')

        # Building final SoC XF vector
        XF = np.zeros(self.control_horizon * self.n_ports)
        m = self.n_ports
        for j in range(t + 1, t + self.control_horizon + 1):
            for i in range(self.n_ports):
                m += 1
                if self.u[i, j] == 0 and self.u[i, j - 1] == 1:
                    XF[m - self.n_ports - 1] = self.x_final[i, j - 1]
                #if we want to limit SoC for v2G
                # else:
                    # XF[m - self.n_ports - 1] = minimum capacity of EVs

        # Maximum capacity of EVs
        XMAX = np.array([self.x_final[:, t + i]
                        for i in range(self.control_horizon)]).flatten()

        if self.verbose:
            print(f'XF: {XF.shape} \n {XF}')
            print(f'XMAX: {XMAX.shape} \n {XMAX}')

        # Inequality constraint
        AU = np.vstack((Gu, -Gu))
        bU = np.concatenate((np.abs(XMAX - Gxx0), -XF + Gxx0))

        if self.verbose:
            print(f'AU: {AU.shape}, BU: {bU.shape}')
            print(f'AU: \n {AU}')
            print(f'bU: \n {bU}')

        # Generate the min cost function
        # !!!! Question: Why not include the cost of the current step?
        f = []
        for i in range(self.control_horizon):
            for j in range(self.n_ports):
                f.append(self.T * self.prices[t + i]) #[t+i+1]
        f = np.array(f).reshape(-1, 1)

        # Binary variable
        BinEV = np.array([self.u[:, t + p]
                          for p in range(self.control_horizon)]).flatten()

        if self.verbose:
            print(f'f: {f.shape}, BinEV: {BinEV.shape}')
            print(f'f: \n {f}')
            print(f'BinEV: \n {BinEV}')

        # Boundaries of the power
        # LB = np.array([self.p_min_MT[j, i + t]
        #                for i in range(self.control_horizon)
        #                for j in range(self.n_ports)])
        LB = np.zeros(self.n_ports * self.control_horizon)
        UB = np.array([self.p_max_MT[j, i + t]
                       for i in range(self.control_horizon)
                       for j in range(self.n_ports)])

        if self.verbose:
            print(f'LB: {LB.shape}, \n {LB}')
            print(f'UB: {UB.shape}, \n {UB}')

        # Optimization with CVXPY
        u1 = cp.Variable(self.n_ports * self.control_horizon, name='u1')
        CapF1 = cp.Variable(
            self.n_ports * self.control_horizon, name='CapF1') # not needed for profit maxmization

        # Constraints
        constr = [AU @ u1 <= bU,
                  CapF1 <= u1, # remove for v2g
                  u1 <= np.diag(BinEV) @ (UB - CapF1), # remove
                  LB @ u1 <= CapF1, # LB cannot be positive when u1 is zero
                  CapF1 <= UB # remove for v2g
                #   u1 <= UB # new
                # LB <= u1 # new
                  ]

        # Cost function
        objective = cp.Minimize(f.T @ u1 - f.T @ CapF1)
        prob = cp.Problem(objective, constr)
        prob.solve(solver=cp.GUROBI, verbose=False)

        print("--------------------------------------------")
        if prob.status != cp.OPTIMAL:
            print(f'Objective value: {prob.status}')
            print("Optimal solution not found !!!!!")
            exit()

        u = u1.value  # Optimal power
        CapF = CapF1.value  # Optimal power
        
        if self.verbose:
            print(f'u: {u.shape} \n {u}')
            print(f'CapF: {CapF.shape} \n {CapF}')

        # Selecting the first self.n_ports power levels
        uc = u[:self.n_ports]

        # build normalized actions
        actions = np.zeros(self.n_ports)
        for i in range(self.n_ports):
            if uc[i] > 0:
                actions[i] = uc[i]/self.p_max_MT[i, t]

        if self.verbose:
            print(f'actions: {actions.shape} \n {actions}')
            
        X2 = Amono[:, :, 0] @ self.x_next + Bmono[:, :, 0] @ uc
        # print(X2)

        self.x_hist2 = np.concatenate(
            (self.x_hist2, X2.reshape(-1, 1)), axis=1)
        self.u_hist2 = np.concatenate(
            (self.u_hist2, uc.reshape(-1, 1)), axis=1)
        self.cap_hist = np.concatenate(
            (self.cap_hist, CapF[:self.n_ports].reshape(-1, 1)), axis=1)

        return actions

        # return actions

        # SoC Equations
        X2 = Amono[:, :, 0] @ self.x_next + Bmono[:, :, 0] @ uc
        self.x_next = X2

    def plot(self):
        '''
        This function plots the results of the MPC baseline in the plot folder of the run.        
        '''
        
        # Colors
        rojo = [0.6350, 0.0780, 0.1840]
        azul = [0, 0.4470, 0.7410]

        # Plot price
        x_v = np.arange(0,self.N+1,step=1)
        plt.figure()
        plt.stairs(self.prices[:self.N],x_v, color=azul, linewidth=1)
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
            x_v = np.arange(0,self.N+1,step=1)
            plt.figure()

            # plt.stairs(x_v, self.Uhist1[i, :self.N], color=rojo, linewidth=1)
            plt.stairs(self.u_hist2[i, :self.N],x_v, linestyle='--', color=azul, linewidth=1)
            
            plt.xlabel('Time [h]', fontsize=12)
            plt.ylabel('Power [kW]', fontsize=12)
            plt.legend(['MT', 'OCCF'], loc='best', fontsize=12)

            plt.grid(True)
            plt.show()

            plt.figure()
            # plt.stairs(x_v, self.Xhist1[i, :self.N], color=rojo, linewidth=1)
            plt.stairs(self.x_hist2[i, :self.N],x_v, linestyle='-.', color=azul, linewidth=1)
            
            plt.xlabel('Time [h]', fontsize=12)
            plt.ylabel('Energy [kWh]', fontsize=12)
            plt.legend(['MT', 'OCCF'], loc='best', fontsize=12)
            plt.grid(True)
            plt.show()