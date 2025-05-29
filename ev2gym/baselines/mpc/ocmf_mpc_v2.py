'''
This file contains the implementation of the OCMF_V2G and OCMF_G2V MPC

Authors: Cesar Diaz-Londono, Stavros Orfanoudakis
'''

import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import numpy as np
import time

from ev2gym.baselines.mpc.mpc import MPC


class OCMF_V2G_v2(MPC):

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
            f2 = []
            for i in range(self.control_horizon):
                for j in range(self.n_ports):
                    f.append(self.T * self.ch_prices[t + i])
                    f.append(-self.T * self.disch_prices[t + i])

                    f2.append(self.T * self.ch_prices[t + i])
                    f2.append(self.T * self.disch_prices[t + i]*2)

            f = np.array(f).reshape(-1)
            f2 = np.array(f2).reshape(-1)

            nb = self.nb
            n = self.n_ports
            h = self.control_horizon

            model = gp.Model("optimization_model")
            u = model.addMVar(nb*h,
                            vtype=GRB.CONTINUOUS,
                            name="u")  # Power

            CapF1 = model.addMVar(nb*h,
                                vtype=GRB.CONTINUOUS,
                                name="CapF1")

            # Binary for charging or discharging
            Zbin = model.addMVar(n*h,
                                vtype=GRB.BINARY,
                                name="Zbin")

            # Constraints
            model.addConstr((self.AU @ u)  <= self.bU, name="constr1")

            # Add the lower bound constraints
            model.addConstr((0 <= CapF1), name="constr2a")

            # Add the upper bound constraints
            model.addConstr((CapF1 <= self.UB), name="constr2b")

            # Constraints for charging P
            model.addConstrs((CapF1[j] <= u[j]
                            for j in range(0, nb*h, 2)), name="constr3a")

            model.addConstrs((u[j] <= (self.UB[j]-CapF1[j]) * Zbin[j//2]
                            for j in range(0, nb*h, 2)), name="constr3b")

            # Constraints for discharging P
            model.addConstrs((CapF1[j] <= u[j]
                            for j in range(1, nb*h, 2)),
                            name="constr4a")

            model.addConstrs((u[j] <= (self.UB[j]-CapF1[j])*(1-Zbin[j//2])
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

            model.setObjective(f @ u - f2 @ CapF1, GRB.MINIMIZE)
            model.setParam('OutputFlag', self.output_flag)
            model.params.NonConvex = 2
            
            if self.MIPGap is not None:
                model.params.MIPGap = self.MIPGap
            model.params.TimeLimit = self.time_limit        
            model.optimize()
            self.total_exec_time += model.Runtime

            if self.MIPGap is not None:
                model.params.MIPGap = self.MIPGap
            model.params.TimeLimit = self.time_limit        
            model.optimize()
    
                
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


class OCMF_G2V_v2(MPC):
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
        # keep looping until feasible solution is found
        while True:
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
            if self.verbose:
                print(f'f: {f.shape}')

            nb = self.nb
            h = self.control_horizon

            model = gp.Model("optimization_model")
            u = model.addMVar(nb*h,                         
                            vtype=GRB.CONTINUOUS,
                            name="u")  # Power

            CapF1 = model.addMVar(nb*h,
                                vtype=GRB.CONTINUOUS,
                                name="CapF1")

            # Constraints
            model.addConstr((self.AU @ u)  <= self.bU, name="constr1")

            # Add the lower bound constraints
            model.addConstr((0 <= CapF1), name="constr2a")

            # Add the upper bound constraints
            model.addConstr((CapF1 <= self.UB), name="constr2b")

            # Constraints for charging P
            model.addConstr((CapF1 <= u), name="constr3a")

            # Constraints for charging P
            model.addConstr((u <= (self.UB-CapF1)), name="constr3b")

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

            model.setObjective(f @ u - f @ CapF1, GRB.MINIMIZE)
            model.setParam('OutputFlag', self.output_flag)
            model.params.NonConvex = 2
            
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
