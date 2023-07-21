import torch
import pandas as pd
import numpy.random as rd
import os
import numpy as np
from pyomo.core.base.config import default_pyomo_config
from pyomo.core.base.piecewise import Bound
from pyomo.environ import *
from pyomo.opt import SolverFactory
import gurobipy as gp
from gurobipy import GRB
from gurobipy import *


def optimization_base_result(env, month, day, initial_soc):

    pv = env.data_manager.get_series_pv_data(month, day)
    price = env.data_manager.get_series_price_data(month, day)
    load = env.data_manager.get_series_electricity_cons_data(month, day)
    period = env.episode_length
# parameters
    DG_parameters = env.dg_parameters

    def get_dg_info(parameters):
        p_max = []
        p_min = []
        ramping_up = []
        ramping_down = []
        a_para = []
        b_para = []
        c_para = []

        for name, gen_info in parameters.items():
            p_max.append(gen_info['power_output_max'])
            p_min.append(gen_info['power_output_min'])
            ramping_up.append(gen_info['ramping_up'])
            ramping_down.append(gen_info['ramping_down'])
            a_para.append(gen_info['a'])
            b_para.append(gen_info['b'])
            c_para.append(gen_info['c'])
        return p_max, p_min, ramping_up, ramping_down, a_para, b_para, c_para
    p_max, p_min, ramping_up, ramping_down, a_para, b_para, c_para = get_dg_info(
        parameters=DG_parameters)
    battery_parameters = env.battery_parameters
    NUM_GEN = len(DG_parameters.keys())
    battery_capacity = env.battery.capacity
    battery_efficiency = env.battery.efficiency

    m = gp.Model("UC")

    # set variables in the system
    on_off = m.addVars(NUM_GEN, period, vtype=GRB.BINARY, name='on_off')
    gen_output = m.addVars(
        NUM_GEN, period, vtype=GRB.CONTINUOUS, name='output')
    battery_energy_change = m.addVars(period, vtype=GRB.CONTINUOUS, lb=-env.battery.max_charge,
                                      ub=env.battery.max_charge, name='battery_action')  # directly set constrains for charge/discharge
    # set constrains for exchange between external grid and distributed energy system
    grid_energy_import = m.addVars(
        period, vtype=GRB.CONTINUOUS, lb=0, ub=env.grid.exchange_ability, name='import')
    grid_energy_export = m.addVars(
        period, vtype=GRB.CONTINUOUS, lb=0, ub=env.grid.exchange_ability, name='export')
    soc = m.addVars(period, vtype=GRB.CONTINUOUS, lb=0.2, ub=0.8, name='SOC')

    # 1. add balance constrain
    m.addConstrs(((sum(gen_output[g, t] for g in range(NUM_GEN))+pv[t]+grid_energy_import[t] >= load[t] +
                 battery_energy_change[t]+grid_energy_export[t]) for t in range(period)), name='powerbalance')
    # 2. add constrain for p max pmin
    m.addConstrs((gen_output[g, t] <= on_off[g, t]*p_max[g]
                 for g in range(NUM_GEN) for t in range(period)), 'output_max')
    m.addConstrs((gen_output[g, t] >= on_off[g, t]*p_min[g]
                 for g in range(NUM_GEN) for t in range(period)), 'output_min')
    # 3. add constrain for ramping up ramping down
    m.addConstrs((gen_output[g, t+1]-gen_output[g, t] <= ramping_up[g]
                 for g in range(NUM_GEN) for t in range(period-1)), 'ramping_up')
    m.addConstrs((gen_output[g, t]-gen_output[g, t+1] <= ramping_down[g]
                 for g in range(NUM_GEN) for t in range(period-1)), 'ramping_down')
    # 4. add constrains for SOC
    m.addConstr(battery_capacity*soc[0] == battery_capacity*initial_soc+(
        battery_energy_change[0]*battery_efficiency), name='soc0')
    m.addConstrs((battery_capacity*soc[t] == battery_capacity*soc[t-1]+(
        battery_energy_change[t]*battery_efficiency)for t in range(1, period)), name='soc update')

    # set cost function
    # 1 cost of generator
    cost_gen = gp.quicksum((a_para[g]*gen_output[g, t]*gen_output[g, t]+b_para[g] *
                           gen_output[g, t]+c_para[g]*on_off[g, t])for t in range(period) for g in range(NUM_GEN))
    cost_grid_import = gp.quicksum(
        grid_energy_import[t]*price[t] for t in range(period))
    cost_grid_export = gp.quicksum(
        grid_energy_export[t]*price[t]*env.sell_coefficient for t in range(period))

    m.setObjective((cost_gen+cost_grid_import-cost_grid_export), GRB.MINIMIZE)
    m.optimize()
