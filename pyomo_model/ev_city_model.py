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


class EV_City_Math_Model():

    def __init__(self, env):
        # get data about the system

        self.sim_length = env.simulation_length
        self.n_cs = env.cs
        self.number_of_ports_per_cs = env.number_of_ports_per_cs

        # TODO: Load environment parameters of the simulation
        # !!!!!! make sure to zerrofill the data when there are no evs !!!!!!
        charge_prices = np.ones(self.n_cs, self.sim_length)  # charge prices
        discharge_prices = np.ones(self.n_cs,
                                   self.sim_length)  # discharge prices
        port_enabled = np.zeros(self.number_of_ports,
                                self.n_cs,
                                self.sim_length)  # port enabled
        port_max_power = np.ones(self.n_cs,
                                 self.sim_length)*22  # charger max power
        port_min_power = np.zeros(self.n_cs,
                                  self.sim_length)  # charger min power
        cs_efficiency = np.zeros(self.n_cs,
                                 self.sim_length)  # charger min power
        ev_max_power = np.ones(self.number_of_ports_per_cs,
                               self.n_cs,
                               self.sim_length)  # ev max power
        ev_min_power = np.zeros(self.number_of_ports_per_cs,
                                self.n_cs,
                                self.sim_length)  # ev min power

        # create model
        m = gp.Model("ev_city")

        power_output = m.addVars(self.number_of_ports,
                                 self.self.n_cs,
                                 self.sim_length,
                                 vtype=GRB.CONTINUOUS,
                                 name='output')
        # Mode variable for the charging station: 1 for Charging and 0 for discharging
        mode = m.addVars(self.self.n_cs,
                         self.sim_length,
                         vtype=GRB.Binary,
                         name='cs_mode')

        energy = m.addVars(self.number_of_ports,
                           self.self.n_cs,
                           self.sim_length,
                           vtype=GRB.CONTINUOUS,
                           name='energy')

        # battery_energy_change = m.addVars(period, vtype=GRB.CONTINUOUS, lb=-env.battery.max_charge,
        #                                   ub=env.battery.max_charge, name='battery_action')  # directly set constrains for charge/discharge

        # set constrains for exchange between external grid and distributed energy system
        # grid_energy_import = m.addVars(
        #     period, vtype=GRB.CONTINUOUS, lb=0, ub=env.grid.exchange_ability, name='import')
        # grid_energy_export = m.addVars(
        #     period, vtype=GRB.CONTINUOUS, lb=0, ub=env.grid.exchange_ability, name='export')
        # soc = m.addVars(period, vtype=GRB.CONTINUOUS,
        #                 lb=0.2, ub=0.8, name='SOC')

        # Help variable definitions

        total_power_output_per_cs = gp.quicksum(
            sum(power_output[p, :, :]) for p in range(self.number_of_ports_per_cs))

        # set cost function
        # 1 cost of generator
        # cost_gen = gp.quicksum((a_para[g]*gen_output[g, t]*gen_output[g, t]+b_para[g] *
        #                         gen_output[g, t]+c_para[g]*on_off[g, t])for t in range(period) for g in range(NUM_GEN))

        # cost_grid_import = gp.quicksum(
        #     grid_energy_import[t]*price[t] for t in range(period))
        # cost_grid_export = gp.quicksum(
        #     grid_energy_export[t]*price[t]*env.sell_coefficient for t in range(period))

        # Constrains

        # charging station binary "mode" variable constraint
        m.addConstrs(((total_power_output_per_cs[i, t] > 0 and mode[i, t] == 1) or
                      (total_power_output_per_cs[i, t]
                       <= 0 and mode[i, t] == 0)
                      for i in range(self.n_cs)
                      for t in range(self.sim_length)), name='cs_mode')
        # energy of EVs after charge/discharge constraint #TODO:check if it makes sense
        # probably remove?
        m.addConstrs((energy[p, i, t+1] == energy[p, i, t] +
                      cs_efficiency[i, t] * power_output[p, i, t]
                      for p in range(self.number_of_ports_per_cs)
                      for i in range(self.n_cs)
                      for t in range(self.sim_length-1)), name='ev_energy')
        
        # energy level of EVs
        m.addConstrs((energy[p, i, t] >= ev_min_power[p, i, t] and
                      energy[p, i, t] <= ev_max_power[p, i, t]
                      for p in range(self.number_of_ports_per_cs)
                      for i in range(self.n_cs)
                      for t in range(self.sim_length-1)), name='ev_energy')

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

        m.setObjective((cost_gen+cost_grid_import -
                       cost_grid_export), GRB.MINIMIZE)
        m.optimize()

        def get_dg_info(parameters):
            p_max = []
            for name, gen_info in parameters.items():
                p_max.append(gen_info['power_output_max'])

            return p_max, p_min, ramping_up, ramping_down, a_para, b_para, c_para


if __name__ == "__main__":

    env = None
    math_model = EV_City_Math_Model(env)
