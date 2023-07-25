# This is a class that saves the reaply of a ev_city simulation,
# so it can be replayed later or solved optimally using the mathematical model

import os
import numpy as np
import pickle

""" These are our variables we need to save:
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

"""


class EvCityReplay():

    def __init__(self, env):

        # Create replay folder if it does not exist
        if not os.path.exists('replay'):
            os.makedirs('replay')

        self.replay_path = 'replay/replay_' + env.sim_name + '.pkl'
        self.sim_length = env.simulation_length
        self.n_cs = env.cs
        self.timescale = env.timescale
        self.score_threshold = env.score_threshold

        self.sim_date = env.sim_date
        self.simulate_grid = env.simulate_grid

        self.charge_prices = env.charge_prices
        self.discharge_prices = env.discharge_prices

        self.port_max_power = np.ones([self.n_cs, self.sim_length])
        self.port_min_power = np.ones([self.n_cs, self.sim_length])
        self.cs_ch_efficiency = np.ones([self.n_cs, self.sim_length])
        self.cs_dis_efficiency = np.ones([self.n_cs, self.sim_length])

        self.max_n_ports = 0

        for i, cs in enumerate(env.charging_stations):
            self.port_max_power[i, :] = cs.max_charge_power
            self.port_min_power[i, :] = - cs.max_discharge_power
            if cs.n_ports > self.max_n_ports:
                self.max_n_ports = cs.n_ports

        self.ev_max_energy = np.zeros([self.max_n_ports,
                                      self.n_cs,
                                      self.sim_length])  # ev max battery capacity, 0 if no ev is there
        self.ev_min_energy = np.zeros([self.max_n_ports,
                                       self.n_cs,
                                       self.sim_length])  # ev min battery capacity, 0 if no ev is there
        self.ev_max_ch_power = np.zeros([self.max_n_ports,
                                         self.n_cs,
                                         self.sim_length])  # ev max charging power, 0 if no ev is there
        self.ev_min_ch_power = np.zeros([self.max_n_ports,
                                         self.n_cs,
                                         self.sim_length])  # ev min charging power, 0 if no ev is there
        self.ev_max_dis_power = np.zeros([self.max_n_ports,
                                          self.n_cs,
                                          self.sim_length])  # ev max discharging power, 0 if no ev is there
        self.ev_min_dis_power = np.zeros([self.max_n_ports,
                                          self.n_cs,
                                          self.sim_length])  # ev min discharging power, 0 if no ev is there
        self.u = np.zeros([self.max_n_ports,
                           self.n_cs,
                           self.sim_length])  # u is 0 if port is empty and 1 if port is occupied
        self.energy_at_arrival = np.zeros([self.max_n_ports,
                                           self.n_cs,
                                           self.sim_length])  # x when ev arrives at the port
        self.ev_arrival = np.zeros([self.max_n_ports,
                                    self.n_cs,
                                    self.sim_length])  # 1 when an ev arrives-> power = 0 and energy = x
        self.t_dep = np.zeros([self.max_n_ports,
                               self.n_cs,
                               self.sim_length])  # time of departure of the ev, 0 if port is empty
        self.ev_des_energy = np.zeros([self.max_n_ports,
                                       self.n_cs,
                                       self.sim_length])  # desired energy of the ev, 0 if port is empty

        for i, ev in enumerate(env.EVs):
            port = ev.id
            cs_id = ev.location
            t_arr = ev.time_of_arrival
            original_t_dep = ev.earlier_time_of_departure
            # print(f'EV {i} is at port {port} of CS {cs_id} from {t_arr} to {original_t_dep}')

            if t_arr >= self.sim_length:
                continue            
            if original_t_dep >= self.sim_length:
                t_dep = self.sim_length
            else:
                t_dep = original_t_dep            

            self.ev_max_energy[port, cs_id, t_arr:t_dep] = ev.battery_capacity
            # self.ev_min_energy[port, cs_id, t_arr:t_dep] = ev.battery_capacity * ev.min_soc
            self.ev_max_ch_power[port, cs_id,
                                 t_arr:t_dep] = env.charging_stations[cs_id].max_charge_power
            self.ev_min_ch_power[port, cs_id,
                                 t_arr:t_dep] = 0
            self.ev_max_dis_power[port, cs_id,
                                    t_arr:t_dep] = env.charging_stations[cs_id].max_discharge_power
            self.ev_min_dis_power[port, cs_id,
                                    t_arr:t_dep] = 0
            self.u[port, cs_id, t_arr:t_dep] = 1
            self.energy_at_arrival[port, cs_id, t_arr] = ev.battery_capacity_at_arrival
            self.ev_arrival[port, cs_id, t_arr] = 1
            if original_t_dep < self.sim_length:
                self.t_dep[port, cs_id, t_dep] = 1
            self.ev_des_energy[port, cs_id, t_arr:t_dep] = ev.desired_capacity

        # print(f'u: {self.u}')
        # print(f'ev_arrival: {self.ev_arrival}')
        # print(f't_dep: {self.t_dep}')
        # print(f'ev_des_energy: {self.ev_des_energy}')
        # print(f'ev_max_energy: {self.ev_max_energy}')
        # print(f'ev_max_ch_power: {self.ev_max_ch_power}')
        # print(f'ev_max_dis_power: {self.ev_max_dis_power}')



            
