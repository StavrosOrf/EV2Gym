'''
This file contains the eMPC class, which is used to control the ev2gym environment using the eMPC algorithm.

Authors: Cesar Diaz-Londono, Stavros Orfanoudakis
'''


import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import numpy as np

from ev2gym.baselines.mpc.mpc import MPC


class eMPC_V2G_v2(MPC):

    def __init__(self, env, control_horizon=10, verbose=False, **kwargs):
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

    def get_action(self, env):
        """
        This function computes the MPC actions for the economic problem including V2G.
        """

        # keep looping until feasible solution is found
        while True:

            t = env.current_step
            # update transformer limits
            self.update_tr_power(t)

            # reconstruct self.x_next using the environment
            self.reconstruct_state(t)
            self.calculate_XF_V2G(t)

            # Station models: Amono and Bmono
            self.v2g_station_models(t)

            # Complete model calculation Gu, and inequality constraints Au and bu
            self.calculate_InequalityConstraints(t)

            # Set power limits
            self.set_power_limits_V2G(t)

            # Print information if verbose
            if self.verbose:
                self.print_info(t)

            # Generate the min cost function
            f = []

            for i in range(self.control_horizon):
                for j in range(self.n_ports):
                    f.append(self.T * self.ch_prices[t + i])
                    f.append(-self.T * self.disch_prices[t + i])

            f = np.array(f).reshape(-1)

            nb = self.nb
            n = self.n_ports
            h = self.control_horizon

            model = gp.Model("optimization_model")
            u = model.addMVar(nb*h,
                              vtype=GRB.CONTINUOUS,
                              name="u")  # Power

            # Binary for charging or discharging
            Zbin = model.addMVar(n*h,
                                 vtype=GRB.BINARY,
                                 name="Zbin")
            
            # Predicted SoC, SOC average, d_cyc, d_cal
            SoC = model.addMVar(n * h, vtype=GRB.CONTINUOUS, name="SoC")
            SOCav = model.addMVar(self.EV_number, vtype=GRB.CONTINUOUS, name="SOCav")
            d_cyc = model.addMVar(self.EV_number, vtype=GRB.CONTINUOUS, name="d_cyc")
            d_cal = model.addMVar(self.EV_number, vtype=GRB.CONTINUOUS, name="d_cal")


            # Constraints
            model.addConstr((self.AU @ u) <= self.bU, name="constr1")

            # Constraints for charging P
            model.addConstrs((0 <= u[j]
                              for j in range(0, nb*h, 2)), name="constr3a")

            model.addConstrs((u[j] <= self.UB[j] * Zbin[j//2]
                              for j in range(0, nb*h, 2)), name="constr3b")

            # Constraints for discharging P
            model.addConstrs((0 <= u[j]
                              for j in range(1, nb*h, 2)),
                             name="constr4a")

            model.addConstrs((u[j] <= self.UB[j]*(1-Zbin[j//2])
                              for j in range(1, nb*h, 2)),
                             name="constr4b")

            # Add the transformer constraints
            for tr_index in range(self.number_of_transformers):
                for i in range(self.control_horizon):
                    model.addConstr((gp.quicksum((u[j] - u[j+1])
                                                 for index, j in enumerate(
                        range(i*self.nb, (i+1)*self.nb, 2))
                        if self.cs_transformers[index] == tr_index) +
                        self.tr_loads[tr_index, i] +
                        self.tr_pv[tr_index, i] <=
                        self.tr_power_limit[tr_index, i]),
                        name=f'constr5_{tr_index}_t{i}')

            for tr_index in range(self.number_of_transformers):
                for i in range(self.control_horizon):
                    model.addConstr((gp.quicksum((u[j] - u[j+1])
                                                 for index, j in enumerate(
                        range(i*self.nb, (i+1)*self.nb, 2))
                        if self.cs_transformers[index] == tr_index) +
                        self.tr_loads[tr_index, i] +
                        self.tr_pv[tr_index, i] >=
                        -self.tr_power_limit[tr_index, :].max()),
                        name=f'constr5_{tr_index}_t{i}')

            # # Battery degradation modelling
            # T_event = (self.departure_times - self.arrival_times) * self.T / 24
            
            # # Battery degradation parameters:
            # zeta_0 = 4.02e-4
            # zeta_1 = 2.04e-3
            # Qacc = 11160
            # epsilon_0 = 6.23e-6
            # epsilon_1 = 1.38e-6
            # epsilon_2 = 6976
            # theta = 28
            # T_tot = 730
            
            # for i in range(self.EV_number):
            #     if self.arrival_times[i] >= t and self.departure_times[i] < t + h:  # Event within horizon h
            #         SOCav[i] = np.mean(SoC[(self.arrival_times[i] - t) * n + self.ev_locations[i]:(self.departure_times[i] - t) * n + self.ev_locations[i]])
            #         d_cal[i] = 0.75 * (epsilon_0 * SOCav[i] / self.ev_max_batt[i] - epsilon_1) * np.exp(-epsilon_2 / theta) * T_event[i] / (T_tot ** 0.25)
            #         d_cyc[i] = (zeta_0 + zeta_1 * np.mean(np.abs(SOCav[i] - SoC[(self.arrival_times[i] - t) * n + self.ev_locations[i]:(self.departure_times[i] - t) * n + self.ev_locations[i]])) / self.ev_max_batt[i]) * \
            #                 np.sum(u[(self.arrival_times[i] - t) * n + self.ev_locations[i]::nb]) * self.T / np.sqrt(Qacc)
                            
            #     elif self.arrival_times[i] < t and self.departure_times[i] <= t + h - 1 and self.departure_times[i] > t:  # Event begins before t but ends within h
            #         SOCav[i] = np.mean(np.concatenate([Xhist_e[self.ev_locations[i], self.arrival_times[i]:], SoC[self.ev_locations[i]::n]]))
            #         d_cal[i] = 0.75 * (epsilon_0 * SOCav[i] / self.ev_max_batt[i] - epsilon_1) * np.exp(-epsilon_2 / theta) * T_event[i] / (T_tot ** 0.25)
            #         d_cyc[i] = (zeta_0 + zeta_1 * np.mean(np.abs(SOCav[i] - np.concatenate([Xhist_e[self.ev_locations[i], self.arrival_times[i]:], SoC[self.ev_locations[i]::n]])) / self.ev_max_batt[i])) * \
            #                 np.sum(np.concatenate([Uhist_e2[self.ev_locations[i], self.arrival_times[i]:], Uhist_e2V[self.ev_locations[i], self.arrival_times[i]:], u[self.ev_locations[i]::nb]])) * self.T / np.sqrt(Qacc)
                            
            #     elif self.arrival_times[i] >= t and self.departure_times[i] > t + h and self.arrival_times[i] < t + h - 1:  # Event starts within h but ends after h
            #         SOCav[i] = np.mean(SoC[(self.arrival_times[i] - t) * n + self.ev_locations[i]:])
            #         T_event[i] = (t + h - self.arrival_times[i]) * self.T / 24
            #         d_cal[i] = 0.75 * (epsilon_0 * SOCav[i] / self.ev_max_batt[i] - epsilon_1) * np.exp(-epsilon_2 / theta) * T_event[i] / (T_tot ** 0.25)
            #         d_cyc[i] = (zeta_0 + zeta_1 * np.mean(np.abs(SOCav[i] - SoC[(self.arrival_times[i] - t) * n + self.ev_locations[i]:])) / self.ev_max_batt[i]) * \
            #                 np.sum(u[(self.arrival_times[i] - t) * nb + self.ev_locations[i]::nb]) * self.T / np.sqrt(Qacc)
                            
            #     elif self.arrival_times[i] < t and self.departure_times[i] > t + h:  # Event starts and ends outside h
            #         SOCav[i] = np.mean(np.concatenate([Xhist_e[self.ev_locations[i], self.arrival_times[i]:], SoC[self.ev_locations[i]::n]]))
            #         T_event[i] = (t + h - self.arrival_times[i]) * self.T / 24
            #         d_cal[i] = 0.75 * (epsilon_0 * SOCav[i] / self.ev_max_batt[i] - epsilon_1) * np.exp(-epsilon_2 / theta) * T_event[i] / (T_tot ** 0.25)
            #         d_cyc[i] = (zeta_0 + zeta_1 * np.mean(np.abs(SOCav[i] - np.concatenate([Xhist_e[self.ev_locations[i], self.arrival_times[i]:], SoC[self.ev_locations[i]::n]])) / self.ev_max_batt[i])) * \
            #                 np.sum(np.concatenate([Uhist_e2[self.ev_locations[i], self.arrival_times[i]:], Uhist_e2V[self.ev_locations[i], self.arrival_times[i]:], u[self.ev_locations[i]::nb]])) * self.T / np.sqrt(Qacc)
                            
            #     else:
            #         SOCav[i] = 0
            #         d_cyc[i] = 0
            #         d_cal[i] = 0
            

            model.setObjective(f @ u, GRB.MINIMIZE)
            model.setParam('OutputFlag', self.output_flag)
            # model.params.NonConvex = 2

            if self.MIPGap is not None:
                model.params.MIPGap = self.MIPGap

            model.params.TimeLimit = self.time_limit

            model.optimize()

            self.total_exec_time += model.Runtime

            if model.status == GRB.Status.INF_OR_UNBD or \
                    model.status == GRB.Status.INFEASIBLE:
                print(
                    f"INFEASIBLE or Unbounded - step{t} -- Relaxing SoC constraints - try {self.varch2}")
                # input("Press Enter to continue...")
                flagOut = False  # Initialize flagOut to False
                varch = 0

                # Iterate over each EV
                for i in range(n):
                    for j in range(t, t + h + 1):
                        # Detect if an EV is departing inside t+h
                        if self.x_final[i, j] == 0 and self.x_final[i, j - 1] > 0:
                            if self.verbose:
                                print(
                                    f"EV {i} is departing at {j} with {self.x_final[i, j - 1]}")
                                print(f'XFinal: {self.x_final[i, j]} ')
                                print(f'XNext: {self.x_next[i]}')
                                print(
                                    f'Diff: {self.x_final[i, j - 1] - self.p_max_MT[i, j - 1] * self.T}')

                        if self.x_final[i, j] == 0 and self.x_final[i, j - 1] > 0 and self.x_next[i] > 0:

                            # self.x_next[i] < (self.x_final[i, j - 1] - self.p_max_MT[i, j - 1] * self.T) and

                            varch += 1
                            if varch > self.varch2:
                                # Reduce the departure SoC by the maximum power it can deliver at T
                                self.x_final[i, j - 1] = self.x_final[i,
                                                                      j - 1] - self.p_max_MT[i, j - 1] * self.T
                                self.varch2 += 1
                                flagOut = True
                                break
                    if flagOut:
                        break
                    if i == n - 1:
                        self.varch2 = 0

                continue

            # calculating actions
            a = np.zeros((nb*h, 1))

            for i in range(2*self.n_ports):
                a[i] = u[i].x

            # build normalized actions
            actions = np.zeros(self.n_ports)
            if self.verbose:
                print(f'Actions:\n {a.reshape(-1,self.n_ports, 2)}')

            e = 0.001
            for i in range(0, 2*self.n_ports, 2):
                if a[i] > e and a[i + 1] > e:
                    raise ValueError(f'Charging and discharging at the same time\
                                        {i} {a[i]} {a[i+1]}')
                elif a[i] > e:
                    actions[i//2] = a[i]/self.max_ch_power[i//2]
                elif a[i + 1] > e:
                    actions[i//2] = -a[i+1]/abs(self.max_disch_power[i//2])

            if self.verbose:
                print(f'actions: {actions.shape} \n {actions}')

            # input("Press Enter to continue...")
            return actions


class eMPC_G2V_v2(MPC):
    '''
    This class implements the MPC for the G2V OCCF.
    '''

    def __init__(self, env, control_horizon=10, verbose=False, **kwargs):
        """
        Initialize the MPC baseline.

        Args:
            env: The environment to be used for the MPC baseline.
            horizon: The horizon of the MPC baseline.
            verbose: Whether to print debug information.
        """
        super().__init__(env, control_horizon, verbose)

        self.na = self.n_ports
        self.nb = self.na

    def get_action(self, env):
        """
        This function computes the MPC actions for the economic problem including G2V.
        """
        t = env.current_step
        # update transformer limits
        self.update_tr_power(t)

        # reconstruct self.x_next using the environment
        self.reconstruct_state(t)
        self.calculate_XF_G2V(t)

        # Station models: Amono and Bmono
        self.g2v_station_models(t)

        # Complete model calculation Gu, and inequality constraints Au and bu
        self.calculate_InequalityConstraints(t)

        # Set power limits
        self.set_power_limits_G2V(t)

        # Print information if verbose
        if self.verbose:
            self.print_info(t)

        # Generate the min cost function
        f = []
        for i in range(self.control_horizon):
            for j in range(self.n_ports):
                f.append(self.T * self.ch_prices[t + i])

        f = np.array(f).reshape(-1)

        nb = self.nb
        n = self.n_ports
        h = self.control_horizon

        model = gp.Model("optimization_model")
        u = model.addMVar(nb*h,
                          vtype=GRB.CONTINUOUS,
                          name="u")  # Power

        # Constraints
        model.addConstr((self.AU @ u) <= self.bU, name="constr1")

        # Add the lower bound constraints
        model.addConstr((0 <= u), name="constr2a")

        # Add the upper bound constraints
        model.addConstr((u <= self.UB), name="constr2b")

        # Add the transformer constraints
        for tr_index in range(self.number_of_transformers):
            for i in range(self.control_horizon):
                model.addConstr((gp.quicksum(u[j]
                                             for index, j in enumerate(
                                                 range(i*self.nb, (i+1)*self.nb))
                                             if self.cs_transformers[index] == tr_index) +
                                 self.tr_loads[tr_index, i] +
                                 self.tr_pv[tr_index, i] <=
                                 self.tr_power_limit[tr_index, i]),
                                name=f'constr5_{tr_index}_t{i}')

        for tr_index in range(self.number_of_transformers):
            for i in range(self.control_horizon):
                model.addConstr((gp.quicksum(u[j]
                                             for index, j in enumerate(
                                                 range(i*self.nb, (i+1)*self.nb))
                                             if self.cs_transformers[index] == tr_index) +
                                 self.tr_loads[tr_index, i] +
                                 self.tr_pv[tr_index, i] >=
                                 -self.tr_power_limit[tr_index, :].max()),
                                name=f'constr5_{tr_index}_t{i}')

        model.setObjective(f @ u, GRB.MINIMIZE)
        model.setParam('OutputFlag', self.output_flag)
        model.params.NonConvex = 2

        if self.MIPGap is not None:
            model.params.MIPGap = self.MIPGap
        model.params.TimeLimit = self.time_limit
        model.optimize()
        self.total_exec_time += model.Runtime

        if model.status == GRB.Status.INF_OR_UNBD or \
                model.status == GRB.Status.INFEASIBLE:
            print(f"INFEASIBLE (applying default actions) - step{t} !!!")
            actions = np.ones(self.n_ports) * 0  # 0.25
            return actions

        a = np.zeros((nb*h, 1))
        cap = np.zeros((nb*h, 1))

        for i in range(self.n_ports):
            a[i] = u[i].x

        if self.verbose:
            print(f'Actions:\n {a.reshape(-1,self.n_ports)}')
            print(f'CapF1:\n {cap.reshape(-1,self.n_ports)}')

        # build normalized actions
        actions = np.zeros(self.n_ports)
        for i in range(self.n_ports):
            actions[i] = a[i]/self.max_ch_power[i//2]

        if self.verbose:
            print(f'actions: {actions.shape} \n {actions}')
        # input("Press Enter to continue...")
        return actions
