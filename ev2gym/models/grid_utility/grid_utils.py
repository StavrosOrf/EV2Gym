'''
The following code is based on the implementation of https://github.com/ShengrenHou/RL-ADN from

Cite as:
Shengren Hou, Shuyi Gao, Weijie Xia, Edgar Mauricio Salazar Duque, Peter Palensky, Pedro P. Vergara,
RL-ADN: A high-performance Deep Reinforcement Learning environment for optimal Energy Storage Systems dispatch in active distribution networks, Energy and AI,
Volume 19, 2025, 100457, ISSN 2666-5468, https://doi.org/10.1016/j.egyai.2024.100457
'''
import numpy as np
from time import perf_counter

import pandapower.topology
import pandas as pd
import os
import pandapower as pp
import matplotlib.pyplot as plt
import networkx as nx
import os
import numpy as np
from ctypes import CDLL, POINTER, c_int, byref, c_bool, c_double
from numpy import ctypeslib
import sys
from time import perf_counter

def load_library():
    """
    Loads a shared library for GPU-based power flow calculations.

    Returns:
        A reference to the loaded shared library function for tensor power flow calculations.

    Description:
        This function loads a shared library (.so or .dll file) based on the operating system. It sets up
        the necessary function signatures for calling the GPU-based power flow solver. It supports Linux and Windows,
        with a specific path for the shared library file. MacOS is not currently supported.
    """
    platform = sys.platform
    my_functions = None

    if platform == "linux" or platform == "linux2":
        # linux (Hard coded in the meantime)
        so_file = r"/home/mauricio/PycharmProjects/gpu_tensorpf/shared_library_complex.so"
        my_functions = CDLL(so_file)

    elif platform == "darwin":
        raise NotImplementedError("MacOS is not currently supported.")

    elif platform == "win32":
        # TODO: Fallback if the gpu library is not found.
        # Windows.
        # os.add_dll_directory(r"C:\Users\20175334\Documents\PycharmProjects\tensorpowerflow\experiments\dll")
        # dynamic_library = r"C:\Users\20175334\Documents\PycharmProjects\tensorpowerflow\experiments\dll\shared_library_complex.dll"

        # Hardcode the directory for debugging purposes.
        os.add_dll_directory(r"C:\Users\20175334\source\repos\gpu_windows\x64\Debug")
        dynamic_library = r"C:\Users\20175334\source\repos\gpu_windows\x64\Debug\gpu_windows.dll"

        my_functions = CDLL(dynamic_library, winmode=0)
    else:
        raise ValueError("OS not recognized.")

    tensor_power_flow = my_functions.tensorPowerFlow
    tensor_power_flow.restype = None
    ctypes_dtype_complex = ctypeslib.ndpointer(np.complex64)
    tensor_power_flow.argtypes = [
                                  POINTER(ctypes_dtype_complex),  # Matrix S, dimensions: S(m x m)
                                  POINTER(ctypes_dtype_complex),  # Matrix K, dimensions: S(m x m)
                                  POINTER(ctypes_dtype_complex),  # Matrix V0, dimensions: V0(m x p)\
                                  POINTER(ctypes_dtype_complex),  # Matrix W, dimensions: W(m x 1)
                                  POINTER(c_int),  # m
                                  POINTER(c_int),  # p
                                  POINTER(c_double),  # tolerance
                                  POINTER(c_int),  # iterations
                                  POINTER(c_bool)  # convergence
                                 ]

    return tensor_power_flow


class GPUPowerFlow(object):
    """
    A class to handle GPU-based power flow calculations.

    Attributes:
        gpu_solver (function): A reference to the loaded shared library function for tensor power flow calculations.

    Description:
        This class provides an interface to perform power flow calculations using a GPU-based solver. It includes
        a method to execute the power flow calculations with the necessary data transformations and GPU function calls.
    """

    def __init__(self):
        self.gpu_solver = load_library()

    def power_flow_gpu(self,
                       K: np.ndarray,  # (m x m) == (nodes-1 x nodes-1)
                       L: np.ndarray,  # (p x 1) == (time_steps x 1) or just p
                       S: np.ndarray,  # (p x m) == (time_steps x nodes)
                       v0: np.ndarray, # (p x m) == (time_steps x nodes)
                       ts: np.ndarray,
                       nb: int,
                       iterations: int = 100,
                       tolerance: float = None):
        """
        Executes the GPU-based power flow calculation.

        Args:
            K (np.ndarray): Admittance matrix (m x m).
            L (np.ndarray): Load matrix (p x 1).
            S (np.ndarray): Power matrix (p x m).
            v0 (np.ndarray): Initial voltage matrix (p x m).
            ts (np.ndarray): Time steps.
            nb (int): Number of buses.
            iterations (int, optional): Maximum number of iterations. Defaults to 100.
            tolerance (float, optional): Convergence tolerance. If None, a default value is used.

        Returns:
            Tuple[np.ndarray, int]: The voltage solution matrix and the number of iterations taken.

        Description:
            This method wraps the GPU-based power flow solver, handling data preparation and result extraction.
            It performs power flow calculations for a given network configuration and operating conditions.
        """

        if tolerance is None:
            tolerance_gpu = 1e-10
        else:
            tolerance_gpu = tolerance ** 2  # Heuristic, this match to the CPU tolerance.

        # ======================================================================
        # Reshape/casting and making sure that the complex matrices are 32 bits.
        # S_host = S.T.copy()
        S_host = S.T
        S_host = S_host.astype(np.complex64)

        # K_host = K.copy()
        K_host = K
        K_host = K_host.astype(np.complex64)

        # V0_host = V0.T.copy()
        V0_host = v0.T
        V0_host = V0_host.astype(np.complex64)

        # W_host =  L.copy()
        W_host = L
        W_host = W_host.astype(np.complex64)

        m = int(nb - 1)
        p = int(ts)

        tolerance_gpu = float(tolerance_gpu)
        iterations = int(iterations)
        convergence = c_bool()

        # ======================================================================
        # Pointers for the dynamic library function.
        m_int = byref(c_int(m))
        p_int = byref(c_int(p))
        iterations_int = byref(c_int(iterations))
        tolerance_int = byref(c_double(tolerance_gpu))
        convergence_int = byref(convergence)

        ctypes_dtype_complex = ctypeslib.ndpointer(np.complex64)

        S_c = S_host.ravel(order="F")
        S_ca = S_c.ctypes.data_as(POINTER(ctypes_dtype_complex))

        K_c = K_host.ravel(order="F")
        K_ca = K_c.ctypes.data_as(POINTER(ctypes_dtype_complex))

        V0_c = V0_host.ravel(order="F")
        V0_ca = V0_c.ctypes.data_as(POINTER(ctypes_dtype_complex))

        W_c = W_host.ravel(order="F")
        W_ca = W_c.ctypes.data_as(POINTER(ctypes_dtype_complex))

        # start = perf_counter()
        self.gpu_solver(S_ca,
                        K_ca,
                        V0_ca,
                        W_ca,
                        m_int,
                        p_int,
                        tolerance_int,
                        iterations_int,
                        convergence_int)
        # print(f"GPU Dynamic library execution: {perf_counter() - start} sec.")
        # print(f"Convergence: {convergence.value}")
        v_solution = V0_c.reshape(m, p, order="F")
        iter_solution = iterations_int._obj.value

        # Voltage solution is a matrix with dimensions (time_steps x (n_nodes-1))-> Including the transpose.
        return v_solution.T, iter_solution

def generate_network(nodes, child=3, plot_graph=False, load_factor=2, line_factor=3):
    """
    Generates a network graph and corresponding data frames for buses and lines.

    Args:
        nodes (int): Number of nodes in the network.
        child (int, optional): Number of children for each node in the tree. Defaults to 3.
        plot_graph (bool, optional): Whether to plot the network graph. Defaults to False.
        load_factor (int, optional): Factor to scale the load values. Defaults to 2.
        line_factor (int, optional): Factor to scale the line impedance values. Defaults to 3.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Data frames for nodes and lines in the network.

    Description:
        This function generates a network graph using a full binary tree and creates pandas data frames
        for nodes and lines with their respective properties. It can also plot the network graph if required.
    """
    LINES = nodes - 1
    G = nx.full_rary_tree(child, nodes)

    if plot_graph:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        nx.draw_kamada_kawai(G, node_size=100, with_labels=True, font_size='medium', ax=ax)

    assert nodes == len(G.nodes)
    assert LINES == len(G.edges)

    # Generate a pandas dataframe
    PCT, ICT, ZCT = 1, 0, 0
    Tb, Pct, Ict, Zct = 0, PCT, ICT, ZCT
    nodes_ = pd.DataFrame(list(G.nodes), columns=["NODES"]) + 1


    active_ns = np.random.normal(50 * load_factor, scale=50, size=nodes).round(3)
    reactive_ns = (active_ns * .1).round(3)

    power = pd.DataFrame({"PD": active_ns,
                          "QD": reactive_ns})
    nodes_properties_ = pd.DataFrame(np.tile([[Tb, Pct, Ict, Zct]], (nodes, 1)),
                                     columns=["Tb", "Pct", "Ict", "Zct"])
    nodes_properties = pd.concat([power, nodes_properties_], axis=1)
    nodes_properties = nodes_properties.astype(
        {"Tb": int, "PD": float, "QD": float, "Pct": int, "Ict": int, "Zct": int})
    nodes_properties = nodes_properties[["Tb", "PD", "QD", "Pct", "Ict", "Zct"]]
    nodes_properties.loc[0] = 1, 0.0, 0.0, PCT, ICT, ZCT  # Slack
    nodes_frame = pd.concat([nodes_, nodes_properties], axis=1)

    # R, X = 0.3144, 0.054
    R, X = 0.3144 / line_factor, 0.054 / line_factor
    lines = pd.DataFrame.from_records(list(G.edges), columns=["FROM", "TO"]) + 1  # Count starts from 1
    lines_properties = pd.DataFrame(np.tile([[R, X, 0, 1, 1]], (LINES, 1)),
                                    columns=["R", "X", "B", "STATUS", "TAP"])
    lines_properties = lines_properties.astype({"R": float, "X": float, "B": int, "STATUS": int, "TAP": int})
    lines_frame = pd.concat([lines, lines_properties], axis=1)

    return nodes_frame, lines_frame

def create_pandapower_net(network_info: dict):
    """
    Creates a pandapower network from given network information.

    Args:
        network_info (dict): A dictionary containing network parameters and file paths for branch and bus information.

    Returns:
        pandapower.network: The created pandapower network.

    Description:
        This function reads network information from provided CSV files and creates a pandapower network
        with buses, lines, loads, and an external grid connection. It sets up the network for power flow analysis.
    """
    vm_pu=network_info['vm_pu']
    s_base=network_info['s_base']
    branch_info_file=network_info['branch_info_file']
    bus_info_file=network_info['bus_info_file']

    branch_info = pd.read_csv(branch_info_file, encoding='utf-8')
    bus_info = pd.read_csv(bus_info_file, encoding='utf-8')

    start = perf_counter()
    net = pp.create_empty_network()
    # Add buses
    bus_dict = {}
    for i, bus_name in enumerate(bus_info["NODES"]):
        bus_dict[bus_name] = pp.create_bus(net, vn_kv=11., name=f"Bus {bus_name}")

    # Slack
    bus_slack = bus_info[bus_info["Tb"] == 1]["NODES"].values
    assert len(bus_slack.shape) == 1 and bus_slack.shape[0] == 1, "Only one slack bus supported"
    pp.create_ext_grid(net, bus=bus_dict[bus_slack.item()], vm_pu=vm_pu, name="Grid Connection")

    # Lines
    for i, (idx, (from_bus, to_bus, res, x_react, b_susceptance)) in enumerate(
            branch_info[["FROM", "TO", "R", "X", "B"]].iterrows()):
        pp.create_line_from_parameters(net,
                                       from_bus=bus_dict[from_bus], to_bus=bus_dict[to_bus],
                                       length_km=1, r_ohm_per_km=res, x_ohm_per_km=x_react, c_nf_per_km=b_susceptance,
                                       max_i_ka=10, name=f"Line {i + 1}")

    # Loads:
    for i, node in enumerate(bus_info['NODES']):
        pp.create_load(net, bus=bus_dict[node], p_mw=0.02, q_mvar=0.0, name=f"Load")
    # print(f"Create net time: {perf_counter() - start}")

    return net

def plot_pandapower_net(net):
    """
    Plots a pandapower network.

    Args:
        net (pandapower.network): The pandapower network to be plotted.

    Description:
        This function creates a plot of the given pandapower network, showing buses, loads, PV generations,
        and lines. It uses networkx for graph representation and matplotlib for plotting.
    """
    # Create a graph from the pandapower network
    G = pandapower.topology.create_nxgraph(net, respect_switches = False)

    # Set node positions based on bus coordinates
    pos = {bus: (net.bus_geodata.at[bus, 'x'], net.bus_geodata.at[bus, 'y']) for bus in G.nodes}

    # Draw buses
    buses = [bus for bus in G.nodes if net.bus.at[bus, 'type'] == 'b']
    nx.draw_networkx_nodes(G, pos, nodelist=buses, node_color='red', node_size=200, label='Buses')

    # Draw loads
    loads = [bus for bus in G.nodes if net.bus.at[bus, 'type'] == 'l']
    nx.draw_networkx_nodes(G, pos, nodelist=loads, node_color='blue', node_size=200, label='Loads')

    # Draw PV generations
    pv_generations = [bus for bus in G.nodes if net.bus.at[bus, 'type'] == 's']
    nx.draw_networkx_nodes(G, pos, nodelist=pv_generations, node_color='green', node_size=200, label='PV Generations')

    # Draw lines
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    # Add labels to the nodes
    node_labels = {bus: bus.split(' ')[-1] for bus in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    # Add a legend
    plt.legend()

    # Display the plot
    plt.axis('off')
    plt.show()

def net_test(net):
    """
    Tests the pandapower network with different power flow algorithms.

    Args:
        net (pandapower.network): The pandapower network to be tested.

    Returns:
        bool: True if the test passes, False otherwise.

    Description:
        This function runs power flow calculations on the provided pandapower network using different algorithms
        (Newton-Raphson and BFSW). It compares the calculated voltages with a predefined solution to verify the correctness.
    """

    v_solution = [0.98965162 + 0.00180549j, 0.98060256 + 0.00337785j, 0.96828145 + 0.00704551j,
                  0.95767051 + 0.01019764j, 0.94765203 + 0.01316654j, 0.94090964 + 0.01600068j,
                  0.93719984 + 0.01754998j, 0.93283877 + 0.01937559j, 0.93073823 + 0.02026054j,
                  0.9299309 + 0.02058985j, 0.92968994 + 0.02068728j, 0.98003142 + 0.00362498j,
                  0.97950885 + 0.00385019j, 0.97936712 + 0.00391065j, 0.97935604 + 0.0039148j,
                  0.93971131 + 0.01547898j, 0.93309482 + 0.01739656j, 0.92577912 + 0.01988823j,
                  0.91988489 + 0.02188907j, 0.91475251 + 0.02362566j, 0.90888169 + 0.02596304j,
                  0.90404908 + 0.02788248j, 0.89950353 + 0.02968449j, 0.89731375 + 0.03055177j,
                  0.89647201 + 0.03088507j, 0.89622055 + 0.03098473j, 0.94032081 + 0.01625577j,
                  0.93992817 + 0.01642583j, 0.93973182 + 0.01651086j, 0.9301316 + 0.02052908j,
                  0.92952481 + 0.02079761j, 0.92922137 + 0.02093188j, 0.92912022 + 0.02097663j]
    v_solution = np.array(v_solution, dtype="complex128")

    for pf_algorithm in ["nr", "bfsw"]:
        print(f"Testing: {pf_algorithm} - Algorithm")
        start = perf_counter()
        if pf_algorithm == "bfsw":
            pp.runpp(net, algorithm=pf_algorithm, numba=False, v_debug=True, VERBOSE=False, tolerance_mva=1e-6)
            # print(f"BFSW. Iterations: {net._ppc['iterations']}. PF time: {net._ppc['et']}")
        elif pf_algorithm == "nr":
            pp.runpp(net, algorithm=pf_algorithm, numba=False, v_debug=True, VERBOSE=False, tolerance_mva=1e-6)
            v_real = net.res_bus["vm_pu"].values * np.cos(np.deg2rad(net.res_bus["va_degree"].values))
            v_img = net.res_bus["vm_pu"].values * np.sin(np.deg2rad(net.res_bus["va_degree"].values))
            v_result = v_real + 1j * v_img
            print('here starts the complex printing')
            print(v_result)
            # print(f"NR. Iterations: {net._ppc['iterations']}. PF time: {net._ppc['et']}")
        print(f"Total pf time: {perf_counter() - start}.")

        v_real = net.res_bus["vm_pu"].values * np.cos(np.deg2rad(net.res_bus["va_degree"].values))
        v_img = net.res_bus["vm_pu"].values * np.sin(np.deg2rad(net.res_bus["va_degree"].values))
        v_result = v_real + 1j * v_img
        assert np.allclose(v_result[1:], v_solution)
        print("Test OK.")

    return True

