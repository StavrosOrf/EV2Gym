'''
===================================
Author: Stavros Orfanoudakis 2023
===================================
'''

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import pickle


class EV_City_Math_Model():
    '''
    This file contains the EV_City_Math_Model class, which is used to solve the ev_city V2G problem optimally.
    '''

    def __init__(self, sim_file_path=None):

        replay = pickle.load(open(sim_file_path, 'rb'))

        self.sim_length = replay.sim_length
        self.n_cs = replay.n_cs
        self.number_of_ports_per_cs = replay.max_n_ports
        self.n_transformers = replay.n_transformers
        self.timescale = replay.timescale
        self.dt = replay.timescale / 60  # time step
        print(f'\nGurobi MIQP solver.')
        print('Loading data...')
        charge_prices = replay.charge_prices  # Charge prices are in €/kWh
        discharge_prices = replay.discharge_prices  # Discharge prices are in €/kWh
        tra_max_power = replay.tra_max_power * self.dt
        tra_min_power = replay.tra_min_power * self.dt
        cs_transformer = replay.cs_transformer
        port_max_power = replay.port_max_power * self.dt
        port_min_power = replay.port_min_power * self.dt
        cs_ch_efficiency = replay.cs_ch_efficiency
        cs_dis_efficiency = replay.cs_dis_efficiency
        ev_max_energy = replay.ev_max_energy
        ev_min_energy = replay.ev_min_energy
        ev_max_ch_power = replay.ev_max_ch_power * self.dt
        ev_min_ch_power = replay.ev_min_ch_power * self.dt
        ev_max_dis_power = replay.ev_max_dis_power * self.dt
        ev_min_dis_power = replay.ev_min_dis_power * self.dt
        u = replay.u
        energy_at_arrival = replay.energy_at_arrival
        ev_arrival = replay.ev_arrival
        t_dep = replay.t_dep
        ev_des_energy = replay.ev_des_energy

        # create model
        print('Creating Gurobi model...')
        self.m = gp.Model("ev_city")
        self.m.setParam('OutputFlag', 0)

        energy = self.m.addVars(self.number_of_ports_per_cs,
                                self.n_cs,
                                self.sim_length,
                                vtype=GRB.CONTINUOUS,
                                name='energy')

        power_ev_dis = self.m.addVars(self.number_of_ports_per_cs,
                                      self.n_cs,
                                      self.sim_length,
                                      vtype=GRB.CONTINUOUS,
                                      name='power_ev_dis')

        power_ev_ch = self.m.addVars(self.number_of_ports_per_cs,
                                     self.n_cs,
                                     self.sim_length,
                                     vtype=GRB.CONTINUOUS,
                                     name='power_ev_ch')

        power_cs_ch = self.m.addVars(self.n_cs,
                                     self.sim_length,
                                     vtype=GRB.CONTINUOUS,
                                     name='power_cs_ch')
        power_cs_dis = self.m.addVars(self.n_cs,
                                      self.sim_length,
                                      vtype=GRB.CONTINUOUS,
                                      name='power_cs_dis')

        power_tr = self.m.addVars(self.n_transformers,
                                  self.sim_length,
                                  vtype=GRB.CONTINUOUS,
                                  name='power_tr')

        # Help variable definitions
        total_charging_costs = gp.quicksum(
            (charge_prices[i, t] * power_cs_ch[i, t])
            for i in range(self.n_cs)
            for t in range(self.sim_length))

        total_discharging_profits = gp.quicksum(
            (discharge_prices[i, t] * power_cs_dis[i, t])
            for i in range(self.n_cs)
            for t in range(self.sim_length))

        # Constrains
        print('Creating constraints...')
        # transformer power output constraint
        for t in range(self.sim_length):
            for i in range(self.n_transformers):
                # self.m.addConstr(
                    # (power_tr[i, t] == gp.quicksum(power_cs_ch[m, t] - power_cs_dis[m, t]
                    #                                for m in range(self.n_cs)
                    #                                if cs_transformer[m] == i)),
                    # name=f'power_tr.{i}.{t}')
                power_tr[i, t] = gp.quicksum(power_cs_ch[m, t] - power_cs_dis[m, t]
                                                   for m in range(self.n_cs)
                                                   if cs_transformer[m] == i)
        # for t in range(self.sim_length):
        #     for i in range(self.n_cs):
        #         power_cs_dis[i, t] = gp.quicksum(power_ev_dis[p, i, t]
        #                                          for p in range(self.number_of_ports_per_cs))
        #         power_cs_ch[i, t] = gp.quicksum(power_ev_ch[p, i, t]
        #                                         for p in range(self.number_of_ports_per_cs))

        # transformer power output constraint
        self.m.addConstrs((power_tr[i, t] <= tra_max_power[i, t]
                           for i in range(self.n_transformers)
                           for t in range(self.sim_length)), name='tr_power_limit_max')
        self.m.addConstrs((power_tr[i, t] >= tra_min_power[i, t]
                           for i in range(self.n_transformers)
                           for t in range(self.sim_length)), name='tr_power_limit_min')

        # charging station total power output (sum of ports) constraint

        self.m.addConstrs((power_cs_dis[i, t] == power_ev_dis.sum('*', i, t)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)), name='cs_power_output_dis')
        self.m.addConstrs((power_cs_ch[i, t] == power_ev_ch.sum('*', i, t)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)), name='cs_power_output_ch')

        # charging station power output constraint
        self.m.addConstrs((power_cs_dis[i, t] <= -port_min_power[i, t]
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)), name='cs_power_dis_limit_max')
        self.m.addConstrs((power_cs_dis[i, t] >= 0
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)), name='cs_power_dis_limit_min')
        self.m.addConstrs((power_cs_ch[i, t] <= port_max_power[i, t]
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)), name='cs_power_ch_limit_max')
        self.m.addConstrs((power_cs_ch[i, t] >= 0
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)), name='cs_power_ch_limit_min')

        # ev charging power constraint
        self.m.addConstrs((power_ev_ch[p, i, t] >= ev_min_ch_power[p, i, t]
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)
                           if u[p, i, t] == 1 and ev_arrival[p, i, t] == 0),
                          name='ev_ch_power_min')
        self.m.addConstrs((power_ev_ch[p, i, t] <= ev_max_ch_power[p, i, t]
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)
                           if u[p, i, t] == 1 and ev_arrival[p, i, t] == 0),
                          name='ev_ch_power_max')

        # ev discharging power constraint
        self.m.addConstrs((power_ev_dis[p, i, t] <= -ev_max_dis_power[p, i, t]
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)
                           if u[p, i, t] == 1 and ev_arrival[p, i, t] == 0),
                          name='ev_dis_power_min')
        self.m.addConstrs((power_ev_dis[p, i, t] >= ev_min_dis_power[p, i, t]
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)
                           if u[p, i, t] == 1 and ev_arrival[p, i, t] == 0),
                          name='ev_dis_power_max')

        # ev charge power if empty port constraint
        for t in range(self.sim_length):
            for i in range(self.n_cs):
                for p in range(self.number_of_ports_per_cs):
                    if u[p, i, t] == 0 or ev_arrival[p, i, t] == 1:
                        self.m.addLConstr((power_ev_ch[p, i, t] == 0),
                                          name=f'ev_empty_port_ch.{p}.{i}.{t}')
                        self.m.addLConstr((power_ev_dis[p, i, t] == 0),
                                          name=f'ev_empty_port_dis.{p}.{i}.{t}')

                    if u[p, i, t] == 0 and t_dep[p, i, t] == 0:
                        self.m.addLConstr(energy[p, i, t] == 0,
                                          name=f'ev_empty_port_energy.{p}.{i}.{t}')

        # energy of EVs after charge/discharge constraint
        for t in range(1, self.sim_length):
            for i in range(self.n_cs):
                for p in range(self.number_of_ports_per_cs):
                    if ev_arrival[p, i, t] == 1:
                        self.m.addLConstr(
                            energy[p, i, t] == energy_at_arrival[p, i, t],
                            name=f'ev_arrival_energy.{p}.{i}.{t}')

                    if u[p, i, t-1] == 1:
                        self.m.addConstr(energy[p, i, t] == (energy[p, i, t-1] +
                                         cs_ch_efficiency[i, t] * power_ev_ch[p, i, t] -
                                         cs_dis_efficiency[i, t] * power_ev_dis[p, i, t]),
                                         name=f'ev_energy.{p}.{i}.{t}')

        # energy level of EVs constraint
        self.m.addConstrs((energy[p, i, t] >= ev_min_energy[p, i, t]
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)), name='ev_energy_level_min')
        self.m.addConstrs((energy[p, i, t] <= ev_max_energy[p, i, t]
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)
                           if t_dep[p, i, t] != 1), name='ev_energy_level_max')

        # Power output of EVs constraint
        self.m.addConstrs((power_ev_dis[p, i, t] * power_ev_ch[p, i, t] == 0
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)), name='ev_power_mode')

        # # # Power output of CS constraint
        # self.m.addConstrs((power_cs_ch[i, t] * power_cs_dis[i, t] == 0
        #                    for i in range(self.n_cs)
        #                    for t in range(self.sim_length)), name='cs_power_mode')

        #time of departure of EVs
        for t in range(self.sim_length):
            for i in range(self.n_cs):
                for p in range(self.number_of_ports_per_cs):
                    if t_dep[p, i, t] == 1:
                        self.m.addLConstr((energy[p, i, t] >=
                                           ev_des_energy[p, i, t]),
                                          name=f'ev_departure_energy.{p}.{i}.{t}')

        # Objective function
        self.m.setObjective((total_charging_costs +
                            total_discharging_profits),
                            GRB.MAXIMIZE)

        # print constraints
        self.m.write("model.lp")
        print(f'Starting Optimization....')
        self.m.params.NonConvex = 2
        self.m.optimize()

        self.power_ev_ch = power_ev_ch
        self.power_ev_dis = power_ev_dis
        self.ev_max_ch_power = ev_max_ch_power
        self.ev_max_dis_power = ev_max_dis_power
        print(
            f'Is MIP?: {self.m.IsMIP}, IsMultiObj?: {self.m.IsMultiObj}, Is QCP?: {self.m.IsQCP}, Is QP?: {self.m.IsQP}')
        if self.m.status != GRB.Status.OPTIMAL:
            print(f'Optimization ended with status {self.m.status}')
            exit()


    def get_actions(self):
        '''
        This function returns the actions of the EVs in the simulation normalized to [-1, 1]
        '''

        self.actions = np.zeros([self.number_of_ports_per_cs,
                                 self.n_cs, self.sim_length])

        for t in range(self.sim_length):
            for i in range(self.n_cs):
                for p in range(self.number_of_ports_per_cs):                    
                    if self.power_ev_ch[p, i, t].x > 0:
                        self.actions[p, i, t] = self.power_ev_ch[p, i, t].x  \
                            / self.ev_max_ch_power[p, i, t]
                    elif self.power_ev_dis[p, i, t].x > 0:
                        self.actions[p, i, t] = self.power_ev_dis[p, i, t].x \
                            / self.ev_max_dis_power[p, i, t]

        return self.actions
