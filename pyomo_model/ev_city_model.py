import pandas as pd
import numpy.random as rd
import os
import numpy as np
# from pyomo.core.base.config import default_pyomo_config
# from pyomo.core.base.piecewise import Bound
# from pyomo.environ import *
# from pyomo.opt import SolverFactory
import gurobipy as gp
from gurobipy import GRB
from gurobipy import *


class EV_City_Math_Model():

    def __init__(self, env):

        self.sim_length = env.simulation_length
        self.n_cs = env.cs
        self.number_of_ports_per_cs = env.number_of_ports_per_cs
        self.timescale = env.timescale
        self.dt = self.timescale / 60  # time step

        # TODO: Load environment parameters of the simulation
        # !!!!!! make sure to zerrofill the data when there are no evs !!!!!!
        print('Loading data...')
        charge_prices = np.ones(
            [self.n_cs, self.sim_length])*-1  # charge prices
        discharge_prices = np.ones([self.n_cs,
                                   self.sim_length])  # discharge prices
        port_max_power = np.ones([self.n_cs,
                                 self.sim_length]) * 22 * self.dt  # charger max power
        port_min_power = np.zeros([self.n_cs,
                                  self.sim_length]) * self.dt  # charger min power
        cs_ch_efficiency = np.zeros([self.n_cs,
                                    self.sim_length])  # charging efficiency
        cs_dis_efficiency = np.zeros([self.n_cs,
                                     self.sim_length])  # discharging efficiency
        ev_max_energy = np.ones([self.number_of_ports_per_cs,
                                self.n_cs,
                                self.sim_length])  # ev max battery capacity, 0 if no ev is there
        ev_min_energy = np.zeros([self.number_of_ports_per_cs,
                                 self.n_cs,
                                 self.sim_length])  # ev min battery capacity, 0 if no ev is there
        ev_max_ch_power = np.ones([self.number_of_ports_per_cs,
                                  self.n_cs,
                                  self.sim_length]) * self.dt  # ev max charging power, 0 if no ev is there
        ev_min_ch_power = np.ones([self.number_of_ports_per_cs,
                                  self.n_cs,
                                  self.sim_length]) * self.dt   # ev min charging power, 0 if no ev is there
        ev_max_dis_power = np.ones([self.number_of_ports_per_cs,
                                   self.n_cs,
                                   self.sim_length]) * self.dt  # ev max discharging power, 0 if no ev is there
        ev_min_dis_power = np.ones([self.number_of_ports_per_cs,
                                   self.n_cs,
                                   self.sim_length]) * self.dt  # ev min discharging power, 0 if no ev is there
        u = np.zeros([self.number_of_ports_per_cs,
                     self.n_cs,
                     self.sim_length])  # u is 0 if port is empty and 1 if port is occupied
        energy_at_arrival = np.ones([self.number_of_ports_per_cs,
                                    self.n_cs,
                                    self.sim_length])  # x when ev arrives at the port
        ev_arrival = np.zeros([self.number_of_ports_per_cs,
                               self.n_cs,
                               self.sim_length])  # 1 when an ev arrives-> power = 0 and energy = x
        t_dep = np.ones([self.number_of_ports_per_cs,
                        self.n_cs,
                        self.sim_length])  # time of departure of the ev, 0 if port is empty
        ev_des_energy = np.ones([self.number_of_ports_per_cs,
                                self.n_cs,
                                self.sim_length])  # desired energy of the ev, 0 if port is empty

        # create model
        print('Creating Gurobi model...')
        self.m = gp.Model("ev_city")

        eps = 0.0001

        power_output = self.m.addVars(self.number_of_ports_per_cs,
                                 self.n_cs,
                                 self.sim_length,
                                 vtype=GRB.CONTINUOUS,
                                 name='output')
        # Mode variable for the EV: 1 for Charging and 0 for discharging
        mode = self.m.addVars(self.number_of_ports_per_cs,
                         self.n_cs,
                         self.sim_length,
                         vtype=GRB.BINARY,
                         name='ev_mode')

        mode_cs = self.m.addVars(self.n_cs,
                            self.sim_length,
                            vtype=GRB.BINARY,
                            name='cs_mode_cs')

        energy = self.m.addVars(self.number_of_ports_per_cs,
                           self.n_cs,
                           self.sim_length,
                           lb=0,
                           vtype=GRB.CONTINUOUS,
                           name='energy')

        power_output_per_cs = self.m.addVars(self.n_cs,
                                        self.sim_length,
                                        vtype=GRB.CONTINUOUS,
                                        name='output_per_cs')

        abs_power_output_per_cs = self.m.addVars(self.n_cs,
                                            self.sim_length,
                                            vtype=GRB.CONTINUOUS,
                                            name='abs_output_per_cs')
        # Help variable definitions
        # print(power_output)
        # power_output_per_cs = power_output.sum(axis=0)
        # self.m.addConstrs((mode_cs[i,t] == power_output_per_cs[i,t] >= 0
        #              for i in range(self.n_cs)
        #              for t in range(self.sim_length)),name='mode_cs')

        total_charging_profits = gp.quicksum(
            (charge_prices[i, t]*abs_power_output_per_cs[i, t]*mode_cs[i, t])
            for i in range(self.n_cs)
            for t in range(self.sim_length))

        total_discharging_profits = gp.quicksum(
            (discharge_prices[i, t] *
             abs_power_output_per_cs[i, t]*(1-mode_cs[i, t]))
            for i in range(self.n_cs)
            for t in range(self.sim_length))

        # Constrains
        self.m.addConstrs((abs_power_output_per_cs[i, t] == abs_(power_output_per_cs[i, t])
                     for i in range(self.n_cs)
                     for t in range(self.sim_length)), name='abs_power_output_per_cs')

        # charging station total power output (sum of ports) constraint
        self.m.addConstrs((power_output_per_cs[i, t] == power_output.sum('*', i, t)
                      for i in range(self.n_cs)
                      for t in range(self.sim_length)), name='cs_power_output')

        print('1.Finished power output constraint')
        # # charging station binary "mode_cs" variable constraint
        # self.m.addConstrs(((power_output_per_cs[i, t] >= 0 and mode_cs[i, t] == 1) or
        #               (power_output_per_cs[i, t] <= 0 and mode_cs[i, t] == 0)
        #               for i in range(self.n_cs)
        #               for t in range(self.sim_length)), name='cs_mode')
        for t in range(self.sim_length):
            for i in range(self.n_cs):
                M = port_max_power[i, t]
                self.m.addLConstr(power_output_per_cs[i, t], GRB.GREATER_EQUAL,
                             (eps - M * (1 - mode_cs[i, t])), name=f"bigM1.cs.{i}.{t}")
                self.m.addLConstr(power_output_per_cs[i, t]
                             <= 0 + M * mode_cs[i, t], name=f"bigM2.cs.{i}.{t}")
                self.m.addConstr((mode_cs[i, t] == 1) >> (
                    power_output_per_cs[i, t] >= 0), name=f'cs_mode1.{i}.{t}')
                self.m.addConstr((mode_cs[i, t] == 0) >> (
                    power_output_per_cs[i, t] <= 0), name=f'cs_mode0.{i}.{t}')

        # EV binary "mode" variable constraint
        # self.m.addConstrs(((power_output[p, i, t] > 0 and mode[p, i, t] == 1) or
        #               (power_output[p, i, t] <= 0 and mode[p, i, t] == 0)
        #               for p in range(self.number_of_ports_per_cs)
        #               for i in range(self.n_cs)
        #               for t in range(self.sim_length)), name='ev_mode')
        for t in range(self.sim_length):
            for i in range(self.n_cs):
                for p in range(self.number_of_ports_per_cs):
                    M = ev_max_ch_power[p, i, t]
                    self.m.addLConstr(power_output[p, i, t], GRB.GREATER_EQUAL,
                                 (eps - M * (1 - mode[p, i, t])), name=f"bigM1.ev.{p}.{i}.{t}")
                    self.m.addLConstr(power_output[p, i, t]
                                 <= 0 + M * mode[p, i, t], name=f"bigM2.ev.{p}.{i}.{t}")
                    self.m.addConstr((mode[p, i, t] == 1) >> (
                        power_output[p, i, t] >= 0), name=f'ev_mode1.{p}.{i}.{t}')
                    self.m.addConstr((mode[p, i, t] == 0) >> (
                        power_output[p, i, t] <= 0), name=f'ev_mode0.{p}.{i}.{t}')
        print('2.Finished ev mode constraint')

        # charging station power output constraint
        self.m.addConstrs((power_output_per_cs[i, t] >= port_min_power[i, t]
                      for i in range(self.n_cs)
                      for t in range(self.sim_length)), name='cs_power_output1')
        self.m.addConstrs((power_output_per_cs[i, t] <= port_max_power[i, t]
                      for i in range(self.n_cs)
                      for t in range(self.sim_length)), name='cs_power_output2')

        # ev charging power constraint
        self.m.addConstrs((power_output[p, i, t]*mode[p, i, t]*u[p, i, t]
                      >= ev_min_ch_power[p, i, t]*mode[p, i, t]*u[p, i, t]
                      for p in range(self.number_of_ports_per_cs)
                      for i in range(self.n_cs)
                      for t in range(self.sim_length)), name='ev_ch_power1')
        self.m.addConstrs((power_output[p, i, t]*mode[p, i, t]*u[p, i, t]
                      <= ev_max_ch_power[p, i, t]*mode[p, i, t]*u[p, i, t]
                      for p in range(self.number_of_ports_per_cs)
                      for i in range(self.n_cs)
                      for t in range(self.sim_length)), name='ev_ch_power2')

        # ev discharging power constraint
        self.m.addConstrs((power_output[p, i, t] * (1-mode[p, i, t]) * u[p, i, t]
                      >= ev_max_dis_power[p, i, t] * (1-mode[p, i, t])*u[p, i, t]
                      for p in range(self.number_of_ports_per_cs)
                      for i in range(self.n_cs)
                      for t in range(self.sim_length)), name='ev_dis_power1')
        self.m.addConstrs((power_output[p, i, t] * (1-mode[p, i, t])*u[p, i, t]
                      <= ev_min_dis_power[p, i, t] * (1-mode[p, i, t])*u[p, i, t]
                      for p in range(self.number_of_ports_per_cs)
                      for i in range(self.n_cs)
                      for t in range(self.sim_length)), name='ev_dis_power2')
        print('3.Finished power constraint')

        # ev charge power if empty port constraint
        for t in range(self.sim_length):
            for i in range(self.n_cs):
                for p in range(self.number_of_ports_per_cs):
                    if u[p, i, t] == 0:
                        self.m.addLConstr(
                            (power_output[p, i, t] == 0), name=f'ev_empty_port.{p}.{i}.{t}')
                        self.m.addLConstr(
                            energy[p, i, t] == 0, name=f'ev_empty_port_energy.{p}.{i}.{t}')

        # energy of EVs after charge/discharge constraint
        for t in range(self.sim_length-1):
            for i in range(self.n_cs):
                for p in range(self.number_of_ports_per_cs):
                    if ev_arrival[p, i, t] == 1:
                        self.m.addLConstr(
                            energy[p, i, t] == energy_at_arrival[p, i, t], name=f'ev_arrival_energy.{p}.{i}.{t}')

                    if u[p, i, t] == 1:
                        self.m.addLConstr(energy[p, i, t+1] ==
                                     (energy[p, i, t] +
                                      mode[p, i, t]*cs_ch_efficiency[i, t] * power_output[p, i, t] +
                                      (1-mode[p, i, t])*cs_dis_efficiency[i, t] *
                                      power_output[p, i, t]), name=f'ev_energy.{p}.{i}.{t}')

        # energy level of EVs constraint
        self.m.addConstrs((energy[p, i, t] >= ev_min_energy[p, i, t]
                      for p in range(self.number_of_ports_per_cs)
                      for i in range(self.n_cs)
                      for t in range(self.sim_length)), name='ev_energy_level1')
        self.m.addConstrs((energy[p, i, t] <= ev_max_energy[p, i, t]
                      for p in range(self.number_of_ports_per_cs)
                      for i in range(self.n_cs)
                      for t in range(self.sim_length)), name='ev_energy_level2')
        print('4.Ev energy constraint')

        # time of departure of EVs
        for t in range(self.sim_length):
            for i in range(self.n_cs):
                for p in range(self.number_of_ports_per_cs):
                    if t_dep[p, i, t] == 1:
                        self.m.addLConstr((energy[p, i, t] >=
                                     ev_des_energy[p, i, t]),
                                     name=f'ev_departure_energy.{p}.{i}.{t}')

        # Objective function
        self.m.setObjective((total_charging_profits +
                       total_discharging_profits), GRB.MAXIMIZE)

        # print constraints
        self.m.write("model.lp")
        print(f'Starting Optimization....')
        self.m.optimize()


if __name__ == "__main__":

    env = None
    math_model = EV_City_Math_Model(env)
