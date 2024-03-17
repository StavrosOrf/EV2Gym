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


class PowerTrackingErrorrMin():
    '''
    This file contains the PowerTrackingErrorrMin class, which is used to solve the ev_city V2G problem optimally.
    '''
    algo_name = 'Optimal (Offline)'        
    def __init__(self, replay_path=None, **kwargs):        
        
        replay = pickle.load(open(replay_path, 'rb'))

        self.sim_length = replay.sim_length
        self.n_cs = replay.n_cs
        self.number_of_ports_per_cs = replay.max_n_ports
        self.n_transformers = replay.n_transformers
        self.timescale = replay.timescale
        dt = replay.timescale / 60  # time step
        # print(f'\nGurobi MIQP solver.')
        # print('Loading data...')

        tra_max_amps = replay.tra_max_amps
        tra_min_amps = replay.tra_min_amps
        cs_transformer = replay.cs_transformer
        port_max_charge_current = replay.port_max_charge_current
        port_min_charge_current = replay.port_min_charge_current
        port_max_discharge_current = replay.port_max_discharge_current
        port_min_discharge_current = replay.port_min_discharge_current
        voltages = replay.voltages / 1000  # phases included in voltage

        power_setpoints = replay.power_setpoints

        cs_ch_efficiency = replay.cs_ch_efficiency
        cs_dis_efficiency = replay.cs_dis_efficiency

        ev_max_energy = replay.ev_max_energy

        ev_max_ch_power = replay.ev_max_ch_power  # * self.dt
        ev_max_dis_power = replay.ev_max_dis_power  # * self.dt
        u = replay.u
        energy_at_arrival = replay.energy_at_arrival
        ev_arrival = replay.ev_arrival
        t_dep = replay.t_dep
        # create model
        # print('Creating Gurobi model...')
        self.m = gp.Model("ev_city")
        self.m.setParam('OutputFlag', 0)
        # self.m.setParam('MIPGap', 0.2)

        # energy of EVs t timeslot t
        energy = self.m.addVars(self.number_of_ports_per_cs,
                                self.n_cs,
                                self.sim_length,
                                vtype=GRB.CONTINUOUS,
                                name='energy')

        current_ev_dis = self.m.addVars(self.number_of_ports_per_cs,
                                        self.n_cs,
                                        self.sim_length,
                                        vtype=GRB.CONTINUOUS,
                                        name='current_ev_dis')
        current_ev_ch = self.m.addVars(self.number_of_ports_per_cs,
                                       self.n_cs,
                                       self.sim_length,
                                       vtype=GRB.CONTINUOUS,
                                       name='current_ev_ch')

        act_current_ev_dis = self.m.addVars(self.number_of_ports_per_cs,
                                            self.n_cs,
                                            self.sim_length,
                                            vtype=GRB.CONTINUOUS,
                                            name='act_current_ev_dis')
        act_current_ev_ch = self.m.addVars(self.number_of_ports_per_cs,
                                           self.n_cs,
                                           self.sim_length,
                                           vtype=GRB.CONTINUOUS,
                                           name='act_current_ev_ch')

        current_cs_ch = self.m.addVars(self.n_cs,
                                       self.sim_length,
                                       vtype=GRB.CONTINUOUS,
                                       name='current_cs_ch')

        current_cs_dis = self.m.addVars(self.n_cs,
                                        self.sim_length,
                                        vtype=GRB.CONTINUOUS,
                                        name='current_cs_dis')

        omega_ch = self.m.addVars(self.number_of_ports_per_cs,
                                  self.n_cs,
                                  self.sim_length,
                                  vtype=GRB.BINARY,
                                  name='omega_ch')
        omega_dis = self.m.addVars(self.number_of_ports_per_cs,
                                   self.n_cs,
                                   self.sim_length,
                                   vtype=GRB.BINARY,
                                   name='omega_dis')

        power_cs_ch = self.m.addVars(self.n_cs,
                                     self.sim_length,
                                     vtype=GRB.CONTINUOUS,
                                     name='power_cs_ch')

        power_cs_dis = self.m.addVars(self.n_cs,
                                      self.sim_length,
                                      vtype=GRB.CONTINUOUS,
                                      name='power_cs_dis')

        power_tr_ch = self.m.addVars(self.n_transformers,
                                     self.sim_length,
                                     vtype=GRB.CONTINUOUS,
                                     name='power_tr_ch')

        power_tr_dis = self.m.addVars(self.n_transformers,
                                      self.sim_length,
                                      vtype=GRB.CONTINUOUS,
                                      name='power_tr_dis')

        current_tr_ch = self.m.addVars(self.n_transformers,
                                       self.sim_length,
                                       vtype=GRB.CONTINUOUS,
                                       name='current_tr_ch')
        current_tr_dis = self.m.addVars(self.n_transformers,
                                        self.sim_length,
                                        vtype=GRB.CONTINUOUS,
                                        name='current_tr_dis')

        power_error = self.m.addVars(self.sim_length,
                                     vtype=GRB.CONTINUOUS,
                                     name='power_error')

        total_soc = self.m.addVars(self.sim_length,
                                   vtype=GRB.CONTINUOUS,
                                   name='total_soc')

        # Constrains
        # print('Creating constraints...')
        # transformer current and power variables
        for t in range(self.sim_length):
            for i in range(self.n_transformers):
                self.m.addConstr(current_tr_ch[i, t] == gp.quicksum(current_cs_ch[m, t]
                                                                    for m in range(self.n_cs)
                                                                    if cs_transformer[m] == i))
                self.m.addConstr(current_tr_dis[i, t] == gp.quicksum(current_cs_dis[m, t]
                                                                     for m in range(self.n_cs)
                                                                     if cs_transformer[m] == i))

                self.m.addConstr(power_tr_ch[i, t] == gp.quicksum(power_cs_ch[m, t]
                                                                  for m in range(self.n_cs)
                                                                  if cs_transformer[m] == i),
                                 name=f'power_tr_ch.{i}.{t}')
                self.m.addConstr(power_tr_dis[i, t] == gp.quicksum(power_cs_dis[m, t]
                                                                   for m in range(self.n_cs)
                                                                   if cs_transformer[m] == i),
                                 name=f'power_tr_dis.{i}.{t}')

            power_error[t] = (gp.quicksum(power_tr_ch[i, t] - power_tr_dis[i, t]
                              for i in range(self.n_transformers))
                              - power_setpoints[t])**2

            # power_error[t] = gp.quicksum(power_tr_ch[i, t] - power_tr_dis[i, t]
            #                   for i in range(self.n_transformers))

        # total_error = gp.quicksum(power_error[t] for t in range(self.sim_length)
        #                           if power_error[t] > 0)

        # CS power output constraint
        self.m.addConstrs(power_cs_ch[i, t] == (current_cs_ch[i, t] * voltages[i])
                          for i in range(self.n_cs)
                          for t in range(self.sim_length))
        self.m.addConstrs(power_cs_dis[i, t] == (current_cs_dis[i, t] * voltages[i])
                          for i in range(self.n_cs)
                          for t in range(self.sim_length))

        # transformer current output constraint (circuit breaker)
        self.m.addConstrs((current_tr_ch[i, t] - current_tr_dis[i, t] <= tra_max_amps[i, t]
                           for i in range(self.n_transformers)
                           for t in range(self.sim_length)), name='tr_current_limit_max')
        self.m.addConstrs((current_tr_ch[i, t] - current_tr_dis[i, t] >= tra_min_amps[i, t]
                           for i in range(self.n_transformers)
                           for t in range(self.sim_length)), name='tr_current_limit_min')

        # charging station total current output (sum of ports) constraint
        self.m.addConstrs((current_cs_ch[i, t] == act_current_ev_ch.sum('*', i, t)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)), name='cs_ch_current_output')
        self.m.addConstrs((current_cs_dis[i, t] == act_current_ev_dis.sum('*', i, t)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)), name='cs_dis_current_output')

        # charging station current output constraint
        self.m.addConstrs((-current_cs_dis[i, t] + current_cs_ch[i, t] >= port_max_discharge_current[i]
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)), name='cs_current_dis_limit_max')
        self.m.addConstrs((-current_cs_dis[i, t] + current_cs_ch[i, t] <= port_max_charge_current[i]
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)), name='cs_curent_ch_limit_max')

        self.m.addConstrs((act_current_ev_ch[p, i, t] == current_ev_ch[p, i, t] * omega_ch[p, i, t]
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)
                           #    if u[p, i, t] == 1 and ev_arrival[p, i, t] == 0
                           ), name='act_ev_current_ch')

        self.m.addConstrs((act_current_ev_dis[p, i, t] == current_ev_dis[p, i, t] * omega_dis[p, i, t]
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)
                           #    if u[p, i, t] == 1 and ev_arrival[p, i, t] == 0
                           ), name='act_ev_current_dis')

        # ev current output constraint
        self.m.addConstrs((current_ev_ch[p, i, t] >= port_min_charge_current[i]  # * omega_ch[p, i, t]
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)
                           #    if u[p, i, t] == 1 and ev_arrival[p, i, t] == 0
                           ), name='ev_current_ch_limit_min')
        self.m.addConstrs((current_ev_dis[p, i, t] >= -port_min_discharge_current[i]  # * omega_dis[p, i, t]
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)
                           #    if u[p, i, t] == 1 and ev_arrival[p, i, t] == 0
                           ), name='ev_current_dis_limit_min')

        # ev max charging current constraint
        self.m.addConstrs((current_ev_ch[p, i, t] <= min(ev_max_ch_power[p, i, t]/voltages[i], port_max_charge_current[i])
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)
                           if u[p, i, t] == 1 and ev_arrival[p, i, t] == 0
                           ),
                          name='ev_current_ch_limit_max')

        # ev max discharging current constraint
        self.m.addConstrs((current_ev_dis[p, i, t] <= min(-ev_max_dis_power[p, i, t]/voltages[i], -port_max_discharge_current[i])
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)
                           if u[p, i, t] == 1 and ev_arrival[p, i, t] == 0
                           ),
                          name='ev_current_dis_limit_max')

        # ev charge power if empty port constraint
        for t in range(self.sim_length):
            for i in range(self.n_cs):
                for p in range(self.number_of_ports_per_cs):
                    if u[p, i, t] == 0 or ev_arrival[p, i, t] == 1:
                        #     self.m.addLConstr((act_current_ev_ch[p, i, t] == 0),
                        #                       name=f'ev_empty_port_ch.{p}.{i}.{t}')
                        #     self.m.addLConstr((act_current_ev_dis[p, i, t] == 0),
                        #                       name=f'ev_empty_port_dis.{p}.{i}.{t}')

                        self.m.addLConstr((omega_ch[p, i, t] == 0),
                                          name=f'omega_empty_port_ch.{p}.{i}.{t}')
                        self.m.addLConstr((omega_dis[p, i, t] == 0),
                                          name=f'omega_empty_port_dis.{p}.{i}.{t}')

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
                                                             act_current_ev_ch[p, i, t] * voltages[i] * cs_ch_efficiency[i, t] * dt -
                                                             act_current_ev_dis[p, i, t] * voltages[i] * cs_dis_efficiency[i, t] * dt),
                                         name=f'ev_energy.{p}.{i}.{t}')

        # energy level of EVs constraint
        self.m.addConstrs((energy[p, i, t] >= 0
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)), name='ev_energy_level_min')
        self.m.addConstrs((energy[p, i, t] <= ev_max_energy[p, i, t]
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)
                           if t_dep[p, i, t] != 1
                           ), name='ev_energy_level_max')

        # Power output of EVs constraint
        self.m.addConstrs((omega_dis[p, i, t] * omega_ch[p, i, t] == 0
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)), name='ev_power_mode_2')

        # self.m.addConstrs((omega_dis[p, i, t] + omega_ch[p, i, t] <= 0.99
        #                    for p in range(self.number_of_ports_per_cs)
        #                    for i in range(self.n_cs)
        #                    for t in range(self.sim_length)), name='ev_power_mode_2')

        # time of departure of EVs
        for t in range(self.sim_length):
            # flag = False
            # for i in range(self.n_cs):

            # total_soc[t] = gp.quicksum(energy[p, i, t]
            #                            for p in range(self.number_of_ports_per_cs)
            #                            if t_dep[p, i, t] == 1)
            # for p in range(self.number_of_ports_per_cs):
            #     # if t_dep[p, i, t] == 1:
            #     #     # total_soc[t] += energy[p, i, t-1]
            #     #     flag = True
            # input(f'Energy at departure: {t_dep[p,i,t]}')
            # self.m.addLConstr((energy[p, i, t] >= 15),
            #                 #    ev_des_energy[p, i, t]),
            #                   name=f'ev_departure_energy.{p}.{i}.{t}')
            # else:
            #     total_soc[t] = 0
            # if not flag:
            # total_soc[t] = 0

            total_soc[t] = gp.quicksum(energy[p, i, t]
                                       for p in range(self.number_of_ports_per_cs)
                                       for i in range(self.n_cs)
                                       if t_dep[p, i, t] == 1)

        # Objective function
        # self.m.setObjective( 10000000 * power_error.sum() - total_soc.sum(),
        #                     GRB.MINIMIZE)
        self.m.setObjective(power_error.sum(),
                            GRB.MINIMIZE)

        # print constraints
        # self.m.write("model.lp")
        print(f'Optimizing....')
        self.m.params.NonConvex = 2
        self.m.optimize()

        self.act_current_ev_ch = act_current_ev_ch
        self.act_current_ev_dis = act_current_ev_dis
        self.port_max_charge_current = port_max_charge_current
        self.port_max_discharge_current = port_max_discharge_current

        # total_soc_sum = 0
        # total_error_sum = 0
        # for t in range(self.sim_length):
        #     total_soc_sum += total_soc[t].x
        #     total_error_sum += power_error[t].x
        # print(f'Total SOC: {total_soc_sum:.2f} kWh, Total error: {total_error_sum:.2f} kW')
        # for t in range(self.sim_length):
        #     print(
        #         f'------------------ Time {t} ------------------------------------------- ')
        #     print(f'Power setpoint: {power_setpoints[t]:.2f} kW')
        #     for tr in range(self.n_transformers):
        #         print(
        #             f'Power: {(power_tr_ch[tr, t].x - power_tr_dis[tr, t].x):.2f} kW')
        #         print(f'Current: {(current_tr_ch[tr, t].x-current_tr_dis[tr, t].x):.2f} A')

        #     for i in range(self.n_cs):
        #         print(
        #             f'CS.{i}: {current_cs_ch[i, t].x:.2f} {current_cs_dis[i, t].x:.2f} voltage: {voltages[i]:.2f} kV')

        #         for p in range(self.number_of_ports_per_cs):

        #             print(f'Port {p} :'
        #                   f' Energy {energy[p, i, t].x:.2f} |' +
        #                   f' Ch {current_ev_ch[p, i, t].x:.2f}' +
        #                   f' o_ch {omega_ch[p, i, t].x:2.0f}  -  ' +
        #                   f' Dis {current_ev_dis[p, i, t].x:.2f}' +
        #                   f' o_dis {omega_dis[p, i, t].x:2.0f}|' +
        #                   f' u {u[p, i, t]:4.1f}')
        # print(t_dep)
        # print(
        #     f'Is MIP?: {self.m.IsMIP}, IsMultiObj?: {self.m.IsMultiObj}, Is QCP?: {self.m.IsQCP}, Is QP?: {self.m.IsQP}')
        if self.m.status != GRB.Status.OPTIMAL:
            print(f'Optimization ended with status {self.m.status}')
            exit()

        self.get_actions()

    def get_actions(self):
        '''
        This function returns the actions of the EVs in the simulation normalized to [-1, 1]
        '''

        self.actions = np.zeros([self.number_of_ports_per_cs,
                                 self.n_cs, self.sim_length])

        for t in range(self.sim_length):
            for i in range(self.n_cs):
                for p in range(self.number_of_ports_per_cs):
                    if self.act_current_ev_ch[p, i, t].x > 0:
                        self.actions[p, i, t] = self.act_current_ev_ch[p, i, t].x  \
                            / self.port_max_charge_current[i]
                    elif self.act_current_ev_dis[p, i, t].x > 0:
                        self.actions[p, i, t] = self.act_current_ev_dis[p, i, t].x \
                            / self.port_max_discharge_current[i]

        return self.actions

    def get_action(self, env, **kwargs):
        '''
        This function returns the action for the current step of the environment.
        '''

        step = env.current_step

        return self.actions[:, :, step].T.reshape(-1)
