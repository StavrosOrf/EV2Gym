
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import pickle

import time


class V2GProfitMaxOracleGB():
    '''
    This file contains the EV_City_Math_Model class, which is used to solve the ev_city V2G problem optimally.
    '''

    # Use type hints to specify the input and output dimensions of the model
    def __init__(self,
                 replay_path: str = None,

                 # General parameters
                 # number of time steps in the simulation
                 sim_length: int = 96,
                 # number of charging stations in the simulation
                 n_cs: int = 1,
                 # you can assume it is one for your simulations
                 max_n_ports: int = 1,
                 # this you can assume it is the power_limiter for all your chargers of your charging problem
                 n_transformers: int = 1,
                 # time resolution in minutes
                 timescale: int = 15,

                 # Transformer (Aggregator) parameters
                 # size(n_transformers,sim_length) transformer max current in amps
                 tra_max_amps: np.ndarray = None,
                 # size(n_transformers,sim_length) transformer min current in amps
                 tra_min_amps: np.ndarray = None,


                 # Charging station parameters
                 # size(n_cs) an id of the transformer that the charging station is connected to
                 cs_transformer: np.ndarray = None,
                 # size(n_cs) max charge current in amps
                 port_max_charge_current: np.ndarray = None,
                 # size(n_cs) min charge current in amps
                 port_min_charge_current: np.ndarray = None,
                 # size(n_cs) max discharge current in amps (should be negative value or 0)
                 port_max_discharge_current: np.ndarray = None,
                 # size(n_cs) min discharge current in amps (should be negative value or 0)
                 port_min_discharge_current: np.ndarray = None,
                 # size(n_cs) voltage in volts of the charging station
                 voltages: np.ndarray = None,
                 # size(n_cs,sim_length) charge prices in €/kWh (should be negative value)
                 charge_prices: np.ndarray = None,
                 # size(n_cs,sim_length) discharge prices in €/kWh (should be positive value)
                 discharge_prices: np.ndarray = None,
                 # size(n_cs,sim_length) charging station charging efficiency, values in  (0-1)
                 cs_ch_efficiency: np.ndarray = None,
                 # size(n_cs,sim_length) charging station discharging efficiency, values in (0-1)
                 cs_dis_efficiency: np.ndarray = None,

                # EV parameters                
                # size(number_of_ports_per_cs,n_cs,sim_length) max battery energy in kWh
                ev_max_energy: np.ndarray = None,
                # size(number_of_ports_per_cs,n_cs,sim_length) max charge power in kW
                ev_max_ch_power: np.ndarray = None,
                # size(number_of_ports_per_cs,n_cs,sim_length) max discharge power in kW (should be negative value)
                ev_max_dis_power: np.ndarray = None,
                # size(number_of_ports_per_cs,n_cs,sim_length) max energy at departure in kWh
                ev_max_energy_at_departure: np.ndarray = None,
                # size(number_of_ports_per_cs,n_cs,sim_length) 1 if port is occupied, 0 if port is empty
                u: np.ndarray = None,
                # size(number_of_ports_per_cs,n_cs,sim_length) energy at arrival in kWh
                energy_at_arrival: np.ndarray = None,
                # size(number_of_ports_per_cs,n_cs,sim_length) 1 if ev arrives, 0 if ev does not arrive
                ev_arrival: np.ndarray = None,
                # size(number_of_ports_per_cs,n_cs,sim_length) 1 if ev arrives, 0 if ev does not arrive
                t_dep: np.ndarray = None
                

                 ):

        print(f'\nGurobi MIQP solver.')
        print('Loading data...')

        if replay_path is not None:
            print("Solving using replay file")
            replay = pickle.load(open(replay_path, 'rb'))

            self.sim_length = replay.sim_length
            self.n_cs = replay.n_cs
            self.number_of_ports_per_cs = replay.max_n_ports
            self.n_transformers = replay.n_transformers
            self.timescale = replay.timescale
            dt = replay.timescale / 60  # time step

            tra_max_amps = replay.tra_max_amps
            tra_min_amps = replay.tra_min_amps
            cs_transformer = replay.cs_transformer
            port_max_charge_current = replay.port_max_charge_current
            port_min_charge_current = replay.port_min_charge_current
            port_max_discharge_current = replay.port_max_discharge_current
            port_min_discharge_current = replay.port_min_discharge_current
            voltages = replay.voltages / 1000  # phases included in voltage
            
            charge_prices = replay.charge_prices  # Charge prices are in €/kWh
            discharge_prices = replay.discharge_prices  # Discharge prices are in €/kWh
            cs_ch_efficiency = replay.cs_ch_efficiency
            cs_dis_efficiency = replay.cs_dis_efficiency
            
            ev_max_energy = replay.ev_max_energy
            ev_max_ch_power = replay.ev_max_ch_power 
            ev_max_dis_power = replay.ev_max_dis_power  
            ev_max_energy_at_departure = replay.max_energy_at_departure
            u = replay.u
            energy_at_arrival = replay.energy_at_arrival
            ev_arrival = replay.ev_arrival
            t_dep = replay.t_dep

        else:
            print("Solving using function arguments")

            self.sim_length = sim_length
            self.n_cs = n_cs
            self.number_of_ports_per_cs = max_n_ports
            self.n_transformers = n_transformers
            self.timescale = timescale
            dt = timescale / 60  # time step

            tra_max_amps = tra_max_amps
            tra_min_amps = tra_min_amps
            cs_transformer = cs_transformer
            port_max_charge_current = port_max_charge_current
            port_min_charge_current = port_min_charge_current
            port_max_discharge_current = port_max_discharge_current
            port_min_discharge_current = port_min_discharge_current
            voltages = voltages / 1000  # phases included in voltage

            charge_prices = charge_prices  # Charge prices are in €/kWh
            discharge_prices = discharge_prices
            cs_ch_efficiency = cs_ch_efficiency
            cs_dis_efficiency = cs_dis_efficiency
            
            ev_max_energy = ev_max_energy
            ev_max_ch_power = ev_max_ch_power
            ev_max_dis_power = ev_max_dis_power
            ev_max_energy_at_departure = ev_max_energy_at_departure
            u = u
            energy_at_arrival = energy_at_arrival
            ev_arrival = ev_arrival
            t_dep = t_dep
            
        # create model
        print('Creating Gurobi model...')
        self.m = gp.Model("ev_city")
        self.m.setParam('OutputFlag', 0)
        # self.m.setParam('MIPGap', 0.01)
        # self.m.setParam('TimeLimit', 60)

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

        current_tr_ch = self.m.addVars(self.n_transformers,
                                       self.sim_length,
                                       vtype=GRB.CONTINUOUS,
                                       name='current_tr_ch')
        current_tr_dis = self.m.addVars(self.n_transformers,
                                        self.sim_length,
                                        vtype=GRB.CONTINUOUS,
                                        name='current_tr_dis')

        power_cs_ch = self.m.addVars(self.n_cs,
                                     self.sim_length,
                                     vtype=GRB.CONTINUOUS,
                                     name='power_cs_ch')

        power_cs_dis = self.m.addVars(self.n_cs,
                                      self.sim_length,
                                      vtype=GRB.CONTINUOUS,
                                      name='power_cs_dis')

        costs = self.m.addVar(vtype=GRB.CONTINUOUS,
                              name='total_soc')
        # Constrains
        # transformer current and power variables
        for t in range(self.sim_length):
            for i in range(self.n_transformers):
                self.m.addConstr(current_tr_ch[i, t] == gp.quicksum(current_cs_ch[m, t]
                                                                    for m in range(self.n_cs)
                                                                    if cs_transformer[m] == i))
                self.m.addConstr(current_tr_dis[i, t] == gp.quicksum(current_cs_dis[m, t]
                                                                     for m in range(self.n_cs)
                                                                     if cs_transformer[m] == i))

        costs = gp.quicksum(act_current_ev_ch[p, i, t] * voltages[i] * cs_ch_efficiency[i, t] * dt * charge_prices[i, t] +
                            act_current_ev_dis[p, i, t] * voltages[i] *
                            cs_dis_efficiency[i, t] *
                            dt * discharge_prices[i, t]
                            for p in range(self.number_of_ports_per_cs)
                            for i in range(self.n_cs)
                            for t in range(self.sim_length))

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

        # time of departure of EVs
        for t in range(self.sim_length):
            for i in range(self.n_cs):
                for p in range(self.number_of_ports_per_cs):
                    if t_dep[p, i, t] == 1:
                        # input(f'Energy at departure: {t_dep[p,i,t]}')
                        self.m.addLConstr(energy[p, i, t] >= ev_max_energy_at_departure[p, i, t],
                                          name=f'ev_departure_energy.{p}.{i}.{t}')

        self.m.setObjective(costs,
                            GRB.MAXIMIZE)

        # print constraints
        self.m.write("model.lp")
        print(f'Optimizing...')
        self.m.params.NonConvex = 2

        self.m.optimize()

        self.act_current_ev_ch = act_current_ev_ch
        self.act_current_ev_dis = act_current_ev_dis
        self.port_max_charge_current = port_max_charge_current
        self.port_max_discharge_current = port_max_discharge_current

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
                    if self.act_current_ev_ch[p, i, t].x > 0:
                        self.actions[p, i, t] = self.act_current_ev_ch[p, i, t].x  \
                            / self.port_max_charge_current[i]
                    elif self.act_current_ev_dis[p, i, t].x > 0:
                        self.actions[p, i, t] = self.act_current_ev_dis[p, i, t].x \
                            / self.port_max_discharge_current[i]

        return self.actions


if __name__ == "__main__":
    start_time = time.time()
    replay_path = "replay_sim_2024_11_01_607878.pkl"

    # Solve the optimization problem with the given replay file 
    # or using the function arguments (if replay file is None)
    # it is faster to pass the function arguments directly in your simulations
    
    agent = V2GProfitMaxOracleGB(replay_path)

    # Get the actions of the EVs in the simulation for all time steps
    actions = agent.get_actions()
    # The actions are proportional to the maximum charge/discharge current of the charging station taking values in [-1, 1]
    # For example, if the maximum charge current of the charging station is 32 A,
    # and the action output is 0.5, then the charge current of the EV is 16 A.    
    print(actions)
    
    
    
    # actions is a numpy array of shape
    # (number_of_ports_per_cs, n_cs, sim_length)
    print(actions.shape)
    
    print(f'Execution time for 96 steps (1-day) for one EV: {time.time() - start_time:.2f} seconds')
