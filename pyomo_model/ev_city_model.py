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
        self.timescale = env.timescale

        dt = self.timescale / 60  # time step

        # TODO: Load environment parameters of the simulation
        # !!!!!! make sure to zerrofill the data when there are no evs !!!!!!
        charge_prices = np.ones([self.n_cs, self.sim_length])  # charge prices
        discharge_prices = np.ones([self.n_cs,
                                   self.sim_length])  # discharge prices
        port_max_power = np.ones([self.n_cs,
                                 self.sim_length]) * 22 * dt  # charger max power
        port_min_power = np.zeros([self.n_cs,
                                  self.sim_length]) * dt  # charger min power
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
                                  self.sim_length]) * dt  # ev max charging power, 0 if no ev is there
        ev_min_ch_power = np.ones([self.number_of_ports_per_cs,
                                  self.n_cs,
                                  self.sim_length]) * dt   # ev min charging power, 0 if no ev is there
        ev_max_dis_power = np.ones([self.number_of_ports_per_cs,
                                   self.n_cs,
                                   self.sim_length]) * dt  # ev max discharging power, 0 if no ev is there
        ev_min_dis_power = np.ones([self.number_of_ports_per_cs,
                                   self.n_cs,
                                   self.sim_length]) * dt  # ev min discharging power, 0 if no ev is there
        u = np.zeros([self.number_of_ports_per_cs,
                     self.n_cs,
                     self.sim_length])  # u is 0 if port is empty and 1 if port is occupied
        t_dep = np.ones([self.number_of_ports_per_cs,
                        self.n_cs,
                        self.sim_length])  # time of departure of the ev, 0 if port is empty
        ev_des_energy = np.ones([self.number_of_ports_per_cs,
                                self.n_cs,
                                self.sim_length])  # desired energy of the ev, 0 if port is empty

        # create model
        m = gp.Model("ev_city")

        power_output = m.addVars(self.number_of_ports_per_cs,
                                 self.self.n_cs,
                                 self.sim_length,
                                 vtype=GRB.CONTINUOUS,
                                 name='output')
        # Mode variable for the EV: 1 for Charging and 0 for discharging
        mode = m.addVars(self.number_of_ports_per_cs,
                         self.self.n_cs,
                         self.sim_length,
                         vtype=GRB.Binary,
                         name='ev_mode')

        mode_cs = m.addVars(self.self.n_cs,
                            self.sim_length,
                            vtype=GRB.Binary,
                            name='cs_mode_cs')

        energy = m.addVars(self.number_of_ports_per_cs,
                           self.self.n_cs,
                           self.sim_length,
                           lb=0,
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

        power_output_per_cs = gp.quicksum(
            sum(power_output[p, :, :]) for p in range(self.number_of_ports_per_cs))

        total_charging_profits = gp.quicksum(
            (charge_prices[p, i, t]*abs(power_output[p, i, t])*mode_cs)
            for p in range(self.number_of_ports_per_cs)
            for i in range(self.n_cs)
            for t in range(self.sim_length))

        total_discharging_profits = gp.quicksum(
            (discharge_prices[p, i, t]*abs(power_output[p, i, t])*(1-mode_cs))
            for p in range(self.number_of_ports_per_cs)
            for i in range(self.n_cs)
            for t in range(self.sim_length))

        # Constrains

        # charging station binary "mode_cs" variable constraint
        m.addConstrs(((power_output_per_cs[i, t] > 0 and mode_cs[i, t] == 1) or
                      (power_output_per_cs[i, t] <= 0 and mode_cs[i, t] == 0)
                      for i in range(self.n_cs)
                      for t in range(self.sim_length)), name='cs_mode')

        # EV binary "mode" variable constraint
        m.addConstrs(((power_output[p, i, t] > 0 and mode[p, i, t] == 1) or
                      (power_output[p, i, t] <= 0 and mode[p, i, t] == 0)
                      for p in range(self.number_of_ports_per_cs)
                      for i in range(self.n_cs)
                      for t in range(self.sim_length)), name='ev_mode')

        # charging station power output constraint
        m.addConstrs((power_output_per_cs[i, t] >= port_min_power[i, t] and
                      power_output_per_cs[i, t] <= port_max_power[i, t]
                      for i in range(self.n_cs)
                      for t in range(self.sim_length)), name='cs_power_output')

        # ev charging power constraint
        m.addConstrs((power_output[p, i, t]*mode[p, i, t]*u[p, i, t]
                      >= ev_min_ch_power[p, i, t]*mode[p, i, t]*u[p, i, t] and
                      power_output[p, i, t]*mode[p, i, t]*u[p, i, t]
                      <= ev_max_ch_power[p, i, t]*mode[p, i, t]*u[p, i, t]
                      for p in range(self.number_of_ports_per_cs)
                      for i in range(self.n_cs)
                      for t in range(self.sim_length)), name='ev_ch_power')

        # ev discharging power constraint
        m.addConstrs((power_output[p, i, t] * (1-mode[p, i, t]) * u[p, i, t]
                      >= ev_max_dis_power[p, i, t] * (1-mode[p, i, t])*u[p, i, t] and
                      power_output[p, i, t] * (1-mode[p, i, t])*u[p, i, t]
                      <= ev_min_dis_power[p, i, t] * (1-mode[p, i, t])*u[p, i, t]
                      for p in range(self.number_of_ports_per_cs)
                      for i in range(self.n_cs)
                      for t in range(self.sim_length)), name='ev_dis_power')

        # ev charge power if empty port constraint
        m.addConstrs((power_output[p, i, t] == 0 and u[p, i, t] == 0 or u[p, i, t] == 1
                      for p in range(self.number_of_ports_per_cs)
                      for i in range(self.n_cs)
                      for t in range(self.sim_length)), name='ev_empty_port')

        # energy of EVs after charge/discharge constraint
        m.addConstrs((energy[p, i, t+1] == (energy[p, i, t] +
                      mode[p, i, t]*cs_ch_efficiency[i, t] * power_output[p, i, t] +
                      (1-mode[p, i, t])*cs_dis_efficiency[i, t] *
                      power_output[p, i, t]) and u[p, i, t] == 1
                      or (energy[p, i, t] == 0 and u[p, i, t] == 0)
                      for p in range(self.number_of_ports_per_cs)
                      for i in range(self.n_cs)
                      for t in range(self.sim_length-1)), name='ev_energy')

        # energy level of EVs constraint
        m.addConstrs((energy[p, i, t] >= ev_min_energy[p, i, t] and
                      energy[p, i, t] <= ev_max_energy[p, i, t]
                      for p in range(self.number_of_ports_per_cs)
                      for i in range(self.n_cs)
                      for t in range(self.sim_length-1)), name='ev_energy_level')

        # time of departure of EVs
        m.addConstrs((t_dep[p, i, t] == t and energy[p, i, t] >= ev_des_energy[p, i, t] or
                      t_dep[p, i, t] != t
                      for p in range(self.number_of_ports_per_cs)
                      for i in range(self.n_cs)
                      for t in range(self.sim_length-1)), name='ev_time_of_departure')

        # Objective function

        m.setObjective((total_charging_profits +
                       total_discharging_profits), GRB.MAXIMIZE)
        m.optimize()

if __name__ == "__main__":

    env = None
    math_model = EV_City_Math_Model(env)
