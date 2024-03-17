'''
This file contains the eMPC class, which is used to control the ev2gym environment using the eMPC algorithm.

Authors: Cesar Diaz-Londono, Stavros Orfanoudakis
'''


import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import numpy as np

from ev2gym.baselines.mpc.mpc import MPC


class V2GProfitMaxOracle(MPC):

    algo_name = "Optimal (Offline)"

    def __init__(self, env, verbose=False, **kwargs):
        """
        Initialize the MPC baseline.
        Args:
            env: The environment to be used for the MPC baseline.
            horizon: The horizon of the MPC baseline.
            verbose: Whether to print debug information.
        """
        control_horizon = env.simulation_length
        super().__init__(env, control_horizon, verbose)

        self.na = self.n_ports
        self.nb = 2 * self.na

        self.actions = None

    def get_action(self, env):
        """
        This function computes the MPC actions for the economic problem including V2G.
        """

        t = env.current_step

        if self.actions is not None:
            return self.actions[t, :]

        # update transformer limits using the exact values (not forecasts)
        self.update_tr_power_oracle(t)

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

        f = np.array(f).reshape(-1, 1)

        nb = self.nb
        n = self.n_ports
        h = self.control_horizon

        model = gp.Model("optimization_model")
        u = model.addVars(range(nb*h),
                          vtype=GRB.CONTINUOUS,
                          name="u")  # Power

        # Binary for charging or discharging
        Zbin = model.addVars(range(n*h),
                             vtype=GRB.BINARY,
                             name="Zbin")

        # Constraints
        model.addConstrs((gp.quicksum(self.AU[i, j] * u[j]
                                      for j in range(nb*h))
                          <= self.bU[i]
                          for i in range(nb*h)), name="constr1")  # Constraint with prediction model

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
        # for tr_index in range(self.number_of_transformers):
        #     for i in range(self.control_horizon):
        #         model.addConstr((gp.quicksum((u[j] - u[j+1])
        #                                      for index, j in enumerate(
        #                                          range(i*self.nb, (i+1)*self.nb, 2))
        #                                      if self.cs_transformers[index] == tr_index) +
        #                          self.tr_loads[tr_index, i] +
        #                          self.tr_pv[tr_index, i] <=
        #                          self.tr_power_limit[tr_index, i]),
        #                         name=f'constr5_{tr_index}_t{i}')

        obj_expr = gp.LinExpr()
        for i in range(nb*h):
            obj_expr.addTerms(f[i], u[i])

        model.setObjective(obj_expr, GRB.MINIMIZE)
        model.params.NonConvex = 2
        model.params.MIPGap = 0.01

        # model.write('model.lp')
        model.optimize()

        if model.status != GRB.Status.OPTIMAL:
            print(f'Objective value: {model.status}')
            print("Optimal solution not found !!!!!")
            exit()

        a = np.zeros((self.simulation_length, nb*h))
        # z_bin = np.zeros((n*h, 1))

        for step in range(self.simulation_length):
            for i in range(2*self.n_ports):
                a[step, i] = u[step*2*self.n_ports + i].x

        # build normalized actions
        actions = np.zeros((self.simulation_length, self.n_ports))
        if self.verbose:
            print(f'Actions:\n {a.reshape(-1,self.n_ports, 2)}')

        e = 0.001
        for step in range(self.simulation_length):
            for i in range(0, 2*self.n_ports, 2):
                if a[step, i] > e and a[step, i + 1] > e:
                    raise ValueError(f'Charging and discharging at the same time\
                                        {i} {a[step,i]} {a[step,i+1]}')
                elif a[step, i] > e:
                    actions[step, i//2] = a[step, i]/self.max_ch_power[i//2]
                elif a[step, i + 1] > e:
                    actions[step, i//2] = -a[step, i+1] / \
                        abs(self.max_disch_power[i//2])

        if self.verbose:
            print(f'actions: {actions.shape} \n {actions}')

        self.actions = actions
        
        return actions[t, :]


class V2GProfitMaxLoadsOracle(MPC):

    algo_name = "Optimal (Offline)"

    def __init__(self, env, verbose=False, **kwargs):
        """
        Initialize the MPC baseline.
        Args:
            env: The environment to be used for the MPC baseline.
            horizon: The horizon of the MPC baseline.
            verbose: Whether to print debug information.
        """
        control_horizon = env.simulation_length
        super().__init__(env, control_horizon, verbose)

        self.na = self.n_ports
        self.nb = 2 * self.na

        self.actions = None

    def get_action(self, env):
        """
        This function computes the MPC actions for the economic problem including V2G.
        """

        t = env.current_step

        if self.actions is not None:
            return self.actions[t, :]

        # update transformer limits using the exact values (not forecasts)
        self.update_tr_power_oracle(t)

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

        f = np.array(f).reshape(-1, 1)

        nb = self.nb
        n = self.n_ports
        h = self.control_horizon

        model = gp.Model("optimization_model")
        u = model.addVars(range(nb*h),
                          vtype=GRB.CONTINUOUS,
                          name="u")  # Power

        # Binary for charging or discharging
        Zbin = model.addVars(range(n*h),
                             vtype=GRB.BINARY,
                             name="Zbin")

        # Constraints
        model.addConstrs((gp.quicksum(self.AU[i, j] * u[j]
                                      for j in range(nb*h))
                          <= self.bU[i]
                          for i in range(nb*h)), name="constr1")  # Constraint with prediction model

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

        obj_expr = gp.LinExpr()
        for i in range(nb*h):
            obj_expr.addTerms(f[i], u[i])

        model.setObjective(obj_expr, GRB.MINIMIZE)
        model.params.NonConvex = 2
        model.params.MIPGap = 0.01

        # model.write('model.lp')
        model.optimize()

        if model.status != GRB.Status.OPTIMAL:
            print(f'Objective value: {model.status}')
            print("Optimal solution not found !!!!!")
            exit()

        a = np.zeros((self.simulation_length, nb*h))
        # z_bin = np.zeros((n*h, 1))

        for step in range(self.simulation_length):
            for i in range(2*self.n_ports):
                a[step, i] = u[step*2*self.n_ports + i].x

        # build normalized actions
        actions = np.zeros((self.simulation_length, self.n_ports))
        if self.verbose:
            print(f'Actions:\n {a.reshape(-1,self.n_ports, 2)}')

        e = 0.001
        for step in range(self.simulation_length):
            for i in range(0, 2*self.n_ports, 2):
                if a[step, i] > e and a[step, i + 1] > e:
                    raise ValueError(f'Charging and discharging at the same time\
                                        {i} {a[step,i]} {a[step,i+1]}')
                elif a[step, i] > e:
                    actions[step, i//2] = a[step, i]/self.max_ch_power[i//2]
                elif a[step, i + 1] > e:
                    actions[step, i//2] = -a[step, i+1] / \
                        abs(self.max_disch_power[i//2])

        if self.verbose:
            print(f'actions: {actions.shape} \n {actions}')

        self.actions = actions
        
        return actions[t, :]
