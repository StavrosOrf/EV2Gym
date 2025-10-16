'''
The following code is based on the implementation of https://github.com/ShengrenHou/RL-ADN

Cite as:
Shengren Hou, Shuyi Gao, Weijie Xia, Edgar Mauricio Salazar Duque, Peter Palensky, Pedro P. Vergara,
RL-ADN: A high-performance Deep Reinforcement Learning environment for optimal Energy Storage Systems dispatch in active distribution networks, Energy and AI,
Volume 19, 2025, 100457, ISSN 2666-5468, https://doi.org/10.1016/j.egyai.2024.100457
'''

import copy as cp
import numpy as np
import pandapower as pp
import pandas as pd
import pickle
import time

from ev2gym.models.grid_utility.grid_tensor import GridTensor
from ev2gym.models.grid_utility.grid_utils import create_pandapower_net
from ev2gym.models.data_augment import DataGenerator, get_pv_load

import pkg_resources


class PowerGrid():
    """
        Custom Environment for Power Network Management.

        The environment simulates a power network, and the agent's task is to
        manage this network by controlling the batteries attached to various nodes.

        """

    def __init__(self,
                 env_config,
                 env,
                 pv_profile=None
                 ) -> None:

        self.config = env_config
        self.env = env

        self.algorithm = self.config['pf_solver']
        self.network_info = self.config['network_info']
        self.s_base = self.network_info['s_base']

        network_bus_info = pd.read_csv(self.network_info['bus_info_file'])
        self.node_num = len((network_bus_info.NODES))

        # Conditional initialization of the distribution network based on the chosen algorithm
        if self.algorithm == "Laurent":
            # Logic for initializing with GridTensor
            self.net = GridTensor(self.network_info['bus_info_file'],
                                  self.network_info['branch_info_file'])
            self.net.Q_file = np.zeros(self.node_num-1)
            self.dense_Ybus = self.net._make_y_bus().toarray()

        elif self.algorithm == "PandaPower":
            # Logic for initializing with PandaPower
            self.net = create_pandapower_net(self.network_info)
        else:
            raise ValueError(
                "Invalid algorithm choice. Please choose 'Laurent' or 'PandaPower'.")

        assert self.config['timescale'] == 15, "Only 15 minutes timescale is supported with the simulate_grid=True !!!"

        data_generator = pkg_resources.resource_filename(
            'ev2gym', 'data/augmentor.pkl')

        with open(data_generator, "rb") as f:
            self.data_generator = CustomUnpickler(f).load()

        # self.episode_length: int = 24 * 60 / self.data_manager.time_interval
        self.episode_length = self.config['simulation_length']

        self.pv_profile = pv_profile

        # self.reset(date, None, None)

    def reset(self, date, load_data, pv_data) -> np.ndarray:
        """
        Reset the environment to its initial state and return the initial state.
        """

        hour = date.hour
        minute = date.minute
        time_slot = hour * 4 + minute // 15
        self.current_step = 0

        if load_data is not None:
            self.load_data = load_data
            self.pv_data = pv_data
        else:
            self.load_data = self.data_generator.sample_data(n_buses=self.node_num,
                                                             n_steps=self.episode_length + 24,
                                                             start_day=date.weekday(),
                                                             start_step=time_slot,
                                                             )
            # normalize active power profile data from 0 to 1
            self.load_data = (self.load_data - self.load_data.min()) / \
                (self.load_data.max() - self.load_data.min())

            self.load_data = self.load_data * self.net.p_values * self.network_info['load_multiplier']
            self.load_data = self.load_data.round(1)

            self.pv_data = get_pv_load(self.pv_profile,
                                       self.env)
            self.pv_data = self.pv_data * self.net.p_values * self.network_info['pv_scale']/100
            self.pv_data = self.pv_data.round(1)

        self.active_power = self.load_data[self.current_step,
                                           1:self.node_num].reshape(1, -1)

        self.reactive_power = self.active_power * self.net.pf
        self.reactive_power = self.reactive_power.round(1)
        self.active_power -= self.pv_data[self.current_step,
                                          1:self.node_num].reshape(1, -1)

        return self.active_power, self.reactive_power

    def step(self, actions: np.ndarray) -> tuple:
        self.active_power += actions

        self.solution = self.net.run_pf(active_power=self.active_power,
                                        reactive_power=self.reactive_power
                                        )

        v = self.solution["v"]
        v_totall = np.insert(v, 0, 1)
        vm_pu_after_control = cp.deepcopy(abs(v_totall))

        self.current_step += 1

        active_power = cp.copy(self.load_data[self.current_step, :])
        pv_data = cp.copy(self.pv_data[self.current_step, :])

        self.active_power = (active_power)[1:self.node_num].reshape(1, -1)
        self.reactive_power = self.active_power * self.net.pf
        self.reactive_power = self.reactive_power.round(1)
        self.active_power -= (pv_data)[1:self.node_num].reshape(1, -1)

        return self.active_power, self.reactive_power, vm_pu_after_control


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "ev2gym.models.data_augment"
        return super().find_class(module, name)


def power_flow_tensor_constant_power(K,
                                     L,
                                     S,
                                     v0,
                                     ts,
                                     nb,
                                     iterations,
                                     tolerance
                                     ):
    """
    Performs the tensor-based power flow calculation for constant power loads.

    Parameters:
    K (np.ndarray): Matrix K.
    L (np.ndarray): Matrix L.
    S (np.ndarray): Power values.
    v0 (np.ndarray): Initial voltage values.
    ts (int): Number of time steps.
    nb (int): Number of buses.
    iterations (int): Maximum number of iterations.
    tolerance (float): Convergence tolerance.

    Returns:
    tuple: Tuple containing the final voltage values and the number of iterations performed.
    """
    iteration = 0
    tol = np.inf
    S = S.T
    v0 = v0.T

    LAMBDA = np.zeros((nb - 1, ts)).astype(np.complex128)
    Z = np.zeros((nb - 1, ts)).astype(np.complex128)
    voltage_k = np.zeros((nb - 1, ts)).astype(np.complex128)

    while iteration < iterations and tol >= tolerance:

        # Hadamard product ( (nb-1) x ts)
        LAMBDA = np.conj(S * (1 / (v0)))  # + epsilon
        Z = K @ LAMBDA  # Matrix ( (nb-1) x ts )
        # This is a broadcasted sum dim => ( (nb-1) x ts  +  (nb-1) x 1 => (nb-1) x ts )
        voltage_k = Z + L
        tol = np.max(np.abs(np.abs(voltage_k) - np.abs(v0)))
        v0 = voltage_k
        iteration += 1

    S = S.T  # Recover the original shape of the power
    v0 = v0.T  # Recover the original shape of the power

    return v0, iteration
