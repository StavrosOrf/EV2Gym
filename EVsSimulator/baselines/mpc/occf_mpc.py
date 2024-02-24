'''
This file contains the implementation of the OCCF_V2G and OCCF_G2V MPC

Authors: Cesar Diaz-Londono, Stavros Orfanoudakis
'''

import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import numpy as np

from EVsSimulator.baselines.mpc.mpc import MPC


class OCCF_V2G(MPC):

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

        f = np.array(f).reshape(-1, 1)
        f2 = np.array(f2).reshape(-1, 1)

        nb = self.nb
        n = self.n_ports
        h = self.control_horizon

        model = gp.Model("optimization_model")
        u = model.addVars(range(nb*h),
                          vtype=GRB.CONTINUOUS,
                          name="u")  # Power

        CapF1 = model.addVars(range(nb*h),
                              vtype=GRB.CONTINUOUS,
                              name="CapF1")

        # Binary for charging or discharging
        Zbin = model.addVars(range(n*h),
                             vtype=GRB.BINARY,
                             name="Zbin")

        # Constraints
        model.addConstrs((gp.quicksum(self.AU[i, j] * u[j]
                                      for j in range(nb*h))
                          <= self.bU[i]
                          for i in range(nb*h)), name="constr1")  # Constraint with prediction model

        # Add the lower bound constraints
        model.addConstrs(
            (0 <= CapF1[i] for i in range(nb*h)), name="constr2a")

        # Add the upper bound constraints
        model.addConstrs(
            (CapF1[i] <= self.UB[i] for i in range(nb*h)), name="constr2b")

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

        obj_expr = gp.LinExpr()
        for i in range(nb*h):
            obj_expr.addTerms(f[i], u[i])
            obj_expr.addTerms(-f2[i], CapF1[i])

        model.setObjective(obj_expr, GRB.MINIMIZE)
        model.params.NonConvex = 2
        # model.params.MIPGap = 0.01

        # save the model
        # model.write('model.lp')
        model.optimize()

        if model.status != GRB.Status.OPTIMAL:
            print(f'Objective value: {model.status}')
            print("Optimal solution not found !!!!!")
            exit()

        a = np.zeros((nb*h, 1))
        cap = np.zeros((nb*h, 1))
        z_bin = np.zeros((n*h, 1))

        for i in range(nb*h):
            a[i] = u[i].x
            # cap[i] = CapF1[i].x

        # build normalized actions
        actions = np.zeros(self.n_ports)
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


class OCCF_G2V(MPC):
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

        f = np.array(f).reshape(-1, 1)

        nb = self.nb
        h = self.control_horizon

        model = gp.Model("optimization_model")
        u = model.addVars(range(nb*h),
                          vtype=GRB.CONTINUOUS,
                          name="u")  # Power

        CapF1 = model.addVars(range(nb*h),
                              vtype=GRB.CONTINUOUS,
                              name="CapF1")

        # Constraints
        model.addConstrs((gp.quicksum(self.AU[i, j] * u[j]
                                      for j in range(nb*h))
                          <= self.bU[i]
                          for i in range(2 * nb *h)), name="constr1")  # Constraint with prediction model

        # Add the lower bound constraints
        model.addConstrs((0 <= CapF1[i]
                          for i in range(nb*h)), name="constr2a")

        # Add the upper bound constraints
        model.addConstrs((CapF1[i] <= self.UB[i]
                          for i in range(nb*h)), name="constr2b")

        # Constraints for charging P
        model.addConstrs((CapF1[j] <= u[j]
                          for j in range(nb*h)), name="constr3a")

        # Constraints for charging P
        model.addConstrs((u[j] <= (self.UB[j]-CapF1[j])
                          for j in range(nb*h)), name="constr3b")

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

        obj_expr = gp.LinExpr()
        for i in range(nb*h):
            obj_expr.addTerms(f[i], u[i])
            obj_expr.addTerms(-f[i], CapF1[i])

        model.setObjective(obj_expr, GRB.MINIMIZE)
        model.params.NonConvex = 2
        # model.params.MIPGap = 0.01

        # model.write('model.lp')
        model.optimize()

        if model.status != GRB.Status.OPTIMAL:
            print(f'Objective value: {model.status}')
            print("Optimal solution not found !!!!!")
            exit()

        a = np.zeros((nb*h, 1))
        # cap = np.zeros((nb*h, 1))

        for i in range(self.n_ports):
            a[i] = u[i].x
            # cap[i] = CapF1[i].x

        if self.verbose:
            print(f'Actions:\n {a.reshape(-1,self.n_ports)}')
            # print(f'CapF1:\n {cap.reshape(-1,self.n_ports)}')

        # build normalized actions
        actions = np.zeros(self.n_ports)
        for i in range(self.n_ports):
            actions[i] = a[i] / self.max_ch_power[i//2]
        if self.verbose:
            print(f'actions: {actions.shape} \n {actions}')

        # input("Press Enter to continue...")
        return actions
