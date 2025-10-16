'''
The following code is based on the implementation of https://github.com/ShengrenHou/RL-ADN from

Cite as:
Shengren Hou, Shuyi Gao, Weijie Xia, Edgar Mauricio Salazar Duque, Peter Palensky, Pedro P. Vergara,
RL-ADN: A high-performance Deep Reinforcement Learning environment for optimal Energy Storage Systems dispatch in active distribution networks, Energy and AI,
Volume 19, 2025, 100457, ISSN 2666-5468, https://doi.org/10.1016/j.egyai.2024.100457
'''

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, diags
from scipy.sparse.linalg import inv
from time import perf_counter
from numba import njit, set_num_threads
import warnings
from ev2gym.models.grid_utility.grid_utils import GPUPowerFlow
from ev2gym.models.grid_utility.grid_utils import generate_network
from ev2gym.models.grid_utility.numbarize import (pre_power_flow_tensor,
                                                  power_flow_tensor,
                                                  power_flow_tensor_constant_power,
                                                  pre_power_flow_sam_sequential,
                                                  power_flow_sam_sequential,
                                                  power_flow_sam_sequential_constant_power_only)
import psutil


class GridTensor:
    """
     Initializes the GridTensor object with grid parameters and data sources.

     Parameters:
     node_file_path (str): Path to the file containing node data. Default is None.
     lines_file_path (str): Path to the file containing line data. Default is None.
     s_base (int): Base apparent power in kVA for one phase. Default is 1000 kVA.
     v_base (float): Base voltage in kV for one phase. Default is 11 kV.
     iterations (int): Maximum number of iterations for power flow calculations. Default is 100.
     tolerance (float): Convergence tolerance for power flow calculations. Default is 1e-5.
     from_file (bool): Flag to indicate whether to load data from files. Default is True.
     nodes_frame (pd.DataFrame): DataFrame containing node data. Default is None.
     lines_frame (pd.DataFrame): DataFrame containing line data. Default is None.
     numba (bool): Flag to enable or disable Numba JIT compilation. Default is True.
     gpu_mode (bool): Flag to enable or disable GPU mode. Default is False.
     """

    def __init__(self,
                 node_file_path: str = None,
                 lines_file_path: str = None,
                 *,
                 s_base: int = 1000,  # kVA - 1 phase
                 v_base: float = 11,  # kV - 1 phase #11
                 iterations: int = 100,
                 tolerance: float = 1e-5,
                 from_file=True,
                 nodes_frame: pd.DataFrame = None,
                 lines_frame: pd.DataFrame = None,
                 numba=True,
                 gpu_mode=False):

        self.s_base = s_base
        # This is better loaded from the file (Extra column)
        self.v_base = v_base
        self.z_base = (self.v_base ** 2 * 1000) / self.s_base
        self.i_base = self.s_base / (np.sqrt(3) * self.v_base)

        self.iterations = iterations
        self.tolerance = tolerance

        if node_file_path is None and lines_file_path is None:

            print("no case is loaded")

        elif node_file_path is not None and lines_file_path is not None and from_file:
            self.branch_info = pd.read_csv(lines_file_path)
            self.bus_info = pd.read_csv(node_file_path)

        elif nodes_frame is not None and lines_frame is not None:
            self.branch_info = lines_frame
            self.bus_info = nodes_frame
        else:
            raise ValueError("Wrong input configuration")
                
        self.p_values = self.bus_info.iloc[:, 2].values
        self.q_values = self.bus_info.iloc[:, 3].values
        epsilon = 1e-6
        self.pf = self.q_values / (self.p_values + epsilon)
        self.pf = self.pf[1:]  # Ignore slack bus
        # print(f'p_values shape is {self.p_values.shape} {self.p_values}')
        # print(f'q_values shape is {self.q_values.shape} {self.q_values}')
        
        # exit()
        self._make_y_bus()
        self._compute_alphas()
        self.v_0 = None
        self._F_ = None
        self._W_ = None

        # Placeholder for the methods that are pre-compiled with numba.
        self._power_flow_tensor_constant_power = None
        self._pre_power_flow_tensor = None
        self._power_flow_tensor = None

        self._pre_power_flow_sam_sequential = None
        self._power_flow_sam_sequential = None
        self._power_flow_sam_sequential_constant_power_only = None

        self.is_numba_enabled = False
        self.is_gpu_enabled = False

        if np.all(self.alpha_P) and not np.any(self.alpha_Z) and not np.any(self.alpha_I):
            self.constant_power_only = True
            self.start_time_pre_pf_tensor_constant_power_only = perf_counter()

            # TODO: Change to sparse inverse.
            # Reduced version of -B^-1 (Reduced version of _F_)  TODO: check it exist .toarray()
            self._K_ = np.array(-inv(self.Ydd_sparse).todense())
            self._L_ = self._K_ @ self.Yds  # Reduced version of _W_
            self.end_time_pre_pf_tensor_constant_power_only = perf_counter()
        else:
            self.constant_power_only = False

        if numba:
            self.enable_numba()
            self.is_numba_enabled = True
        else:
            warnings.warn(
                "Numba NOT enabled. Performance is greatly reduced.", RuntimeWarning)
            self.disable_numba()
            self.is_numba_enabled = False

        if gpu_mode:
            self.gpu_solver = GPUPowerFlow()
            self.is_gpu_enabled = True

    def enable_numba(self):
        """
         Disables Numba JIT compilation, reverting to standard Python execution.
         """
        parallel = True

        self._power_flow_tensor_constant_power = power_flow_tensor_constant_power
        self._pre_power_flow_tensor = njit(
            pre_power_flow_tensor, parallel=parallel)
        self._power_flow_tensor = njit(power_flow_tensor, parallel=parallel)

        self._pre_power_flow_sam_sequential = njit(
            pre_power_flow_sam_sequential, parallel=parallel)
        self._power_flow_sam_sequential = njit(
            power_flow_sam_sequential, parallel=parallel)
        self._power_flow_sam_sequential_constant_power_only = njit(power_flow_sam_sequential_constant_power_only,
                                                                   parallel=parallel)

    def disable_numba(self):

        self._power_flow_tensor_constant_power = power_flow_tensor_constant_power
        self._pre_power_flow_tensor = pre_power_flow_tensor
        self._power_flow_tensor = power_flow_tensor

        self._pre_power_flow_sam_sequential = pre_power_flow_sam_sequential
        self._power_flow_sam_sequential = power_flow_sam_sequential
        self._power_flow_sam_sequential_constant_power_only = power_flow_sam_sequential_constant_power_only

    @classmethod
    def generate_from_graph(cls, *, nodes=100, child=2, plot_graph=True, load_factor=2, line_factor=3, **kwargs):
        """
        Generates a synthetic grid using the networkX package and returns a GridTensor object.

        Parameters:
        nodes (int): Number of nodes in the synthetic grid. Default is 100.
        child (int): Number of child nodes for each node in the grid. Default is 2.
        plot_graph (bool): Flag to plot the generated graph. Default is True.
        load_factor (int): Load factor for the grid. Default is 2.
        line_factor (int): Line factor for the grid. Default is 3.

        Returns:
        GridTensor: An instance of the GridTensor class.
        """

        nodes_frame, lines_frame = generate_network(nodes=nodes,
                                                    child=child,
                                                    plot_graph=plot_graph,
                                                    load_factor=load_factor,
                                                    line_factor=line_factor)

        return cls(node_file_path="",
                   lines_file_path="",
                   from_file=False,
                   nodes_frame=nodes_frame,
                   lines_frame=lines_frame,
                   **kwargs)

    def reset_start(self):
        """
        Resets the starting voltage values for power flow calculations to default flat start values.
        """
        # TODO
        # Flat start  #2D array
        self.v_0 = np.ones((self.nb-1, 1), dtype="complex128")

    def _set_number_of_threads(self, threads):
        """
        Sets the number of threads for parallel execution in Numba.

        Parameters:
        threads (int): Number of threads to be used.
        """
        assert isinstance(threads, int)
        assert threads <= psutil.cpu_count(), "Number of threads must be lower of cpu count."
        set_num_threads(threads)
        print(f"Number of threads set to: {threads}")

    def _make_y_bus(self) -> None:
        """
        Compute Y_bus submatrices

        For each branch, compute the elements of the branch admittance matrix where
              | Is |   | Yss  Ysd |   | Vs |
              |    | = |          | * |    |
              |-Id |   | Yds  Ydd |   | Vd |
        """

        self.nb = self.bus_info.shape[0]  # number of buses
        self.nl = self.branch_info.shape[0]  # number of lines

        sl = self.bus_info[self.bus_info['Tb'] ==
                           1]['NODES'].tolist()  # Slack node(s)

        stat = self.branch_info.iloc[:, 5]  # ones at in-service branches
        Ys = stat / ((self.branch_info.iloc[:, 2] + 1j * self.branch_info.iloc[:, 3]) / (
            self.v_base ** 2 * 1000 / self.s_base))  # series admittance
        # line charging susceptance
        Bc = stat * self.branch_info.iloc[:, 4] * \
            (self.v_base ** 2 * 1000 / self.s_base)
        tap = stat * self.branch_info.iloc[:, 6]  # default tap ratio = 1

        Ytt = Ys + 1j * Bc / 2
        Yff = Ytt / tap
        Yft = - Ys / tap
        Ytf = Yft

        # build connection matrices
        f = self.branch_info.iloc[:, 0] - 1  # list of "from" buses
        t = self.branch_info.iloc[:, 1] - 1  # list of "to" buses

        # connection matrix for line & from buses
        Cf = csr_matrix((np.ones(self.nl), (range(self.nl), f)),
                        (self.nl, self.nb))

        # connection matrix for line & to buses
        Ct = csr_matrix((np.ones(self.nl), (range(self.nl), t)),
                        (self.nl, self.nb))

        # build Yf and Yt such that Yf * V is the vector of complex branch currents injected
        # at each branch's "from" bus, and Yt is the same for the "to" bus end
        i = np.r_[range(self.nl), range(self.nl)]  # double set of row indices

        Yf = csr_matrix((np.r_[Yff, Yft], (i, np.r_[f, t])))
        Yt = csr_matrix((np.r_[Ytf, Ytt], (i, np.r_[f, t])))

        # build Ybus
        Ybus = Cf.T * Yf + Ct.T * Yt  # Full Ybus

        # Dense matrices
        # TODO
        # TODO: This takes a lot of memory. Check if I can save it always as sparse for all methods.
        self._Ybus = Ybus.toarray()
        self.Yss = csr_matrix(
            Ybus[sl[0] - 1, sl[0] - 1], shape=(len(sl), len(sl))).toarray()
        # TODO: Here assume the slack is the first one?
        self.Ysd = np.array(Ybus[0, 1:].toarray())
        self.Yds = self.Ysd.T
        # TODO: This consumes a huge amount of memory
        self.Ydd = np.array(Ybus[1:, 1:].toarray())

        self._Ybus_sparse = Ybus
        self.Yss_sparse = csr_matrix(
            Ybus[sl[0] - 1, sl[0] - 1], shape=(len(sl), len(sl)))
        # TODO: Here assume the slack is the first one?
        self.Ysd_sparse = Ybus[0, 1:]
        self.Yds_sparse = csc_matrix(self.Ysd.T)
        self.Ydd_sparse = Ybus[1:, 1:]

        return Ybus

    def _compute_alphas(self):
        """
        Computes alpha values for different load types in the grid. Assume P-1 and Z,I=0
        """
        self.alpha_P = 1
        self.alpha_I = 0
        self.alpha_Z = 0

        self.flag_all_constant_impedance_is_zero = not np.any(self.alpha_Z)
        self.flag_all_constant_current_is_zero = not np.any(self.alpha_I)
        self.flag_all_constant_powers_are_ones = np.all(self.alpha_P)

    def _check_2d_to_1d(self, active_power, reactive_power):
        """
        Checks and converts 2D power matrices to 1D vectors if applicable.

        Parameters:
        active_power (np.ndarray): Active power matrix.
        reactive_power (np.ndarray): Reactive power matrix.

        Returns:
        tuple: Tuple containing active and reactive power as 1D vectors.
        """

        assert active_power.ndim == reactive_power.ndim, "Active and reactive power must have same dimension."

        if (active_power.ndim == 2 and active_power.shape[0] == 1) and \
                (reactive_power.ndim == 2 and reactive_power.shape[0] == 1):
            active_power = active_power.flatten()
            reactive_power = reactive_power.flatten()
        elif (active_power.ndim == 2 and active_power.shape[0] != 1) and \
                (reactive_power.ndim == 2 and reactive_power.shape[0] != 1):
            raise ValueError(
                "Active and reactive power tensors must have only one time step.")

        assert active_power.ndim == 1, "Array should be one dimensional."
        assert reactive_power.ndim == 1, "Array should be one dimensional."
        assert len(active_power) == len(reactive_power) == self.nb - \
            1, "All load nodes must have power values."

        return active_power, reactive_power

    def _compute_chunks(self, DIMENSION_BOUND, n_nodes, n_steps):
        """
        Computes chunks for processing based on dimension bounds and grid parameters.

        Parameters:
        DIMENSION_BOUND (int): The upper bound for the dimension of the matrices.
        n_nodes (int): Number of nodes in the grid.
        n_steps (int): Number of time steps for the simulation.

        Returns:
        list: Indices for slicing the power consumption array.
                Breaks the n_steps in chunks so it can fit in memory.
        The ideas is that n_nodes * n_steps cannot be bigger than DIMENSION_BOUND
        DIMENSION_BOUND is a empirically found value (should vary due to the computer's RAM).

        Return:
            idx: list: All the ts indices to slice the power consumption array.
                e.g., idx = [0, 1000, 2000, 2500]. 2500 time step requested, chunked in 1000 time steps (last item is
                the reminder: 2500-2000=500).
        """

        # DIMENSION_BOUND =  500 * 5_000
        # n_nodes = 4999
        # n_steps = 3000

        TS_MAX = DIMENSION_BOUND // n_nodes
        if n_steps > TS_MAX:  # Chunk it
            (quotient, reminder) = divmod(n_steps, TS_MAX)
            idx = [i * TS_MAX for i in range(quotient + 1)]

            if reminder != 0:
                idx = idx + [idx[-1] + reminder]
            # if reminder == 0:
            #     idx = idx + [idx[-1] + TS_MAX]

        else:  # The requested amount of TS is lower than the bound. So, everything is ok
            idx = [0, n_steps]

        # print(idx)

        return idx

    def _make_big_sparse_matrices(self, S_nom, Ydd_sparse, Yds_sparse):
        """
        Creates large sparse matrices for solving the sparse tensor power flow problem.

        Parameters:
        S_nom (np.ndarray): Nominal power values.
        Ydd_sparse (sparse matrix): Sparse Ydd matrix.
        Yds_sparse (sparse matrix): Sparse Yds matrix.

        Returns:
        tuple: Tuple containing the big M matrix and H vector as sparse matrices.
        """

        n_steps = S_nom.shape[0]
        n_nodes = S_nom.shape[1]

        M = -diags(1 / np.conj(S_nom[0, :])).dot(Ydd_sparse).asformat("coo")
        H = diags(1 / np.conj(S_nom[0, :])).dot(Yds_sparse).asformat("coo")

        # First iteration of M-matrix and H-vector
        idx_1_M_col = M.col
        idx_1_M_row = M.row
        M_1_data = M.data

        idx_1_H_row = H.row
        H_1_data = H.data

        # Placeholder for M-matrix
        idx_col_M_temp = []
        idx_row_M_temp = []
        M_data_temp = []

        # Placeholder for H-vector
        idx_row_H_temp = []
        H_data_temp = []
        if n_steps > 1:
            times_multiplying = []
            for ii in range(1, n_steps):
                start_multiply = perf_counter()
                M_temp = - \
                    diags(1 / np.conj(S_nom[ii, :])
                          ).dot(Ydd_sparse).asformat("coo")
                H_temp = diags(
                    1 / np.conj(S_nom[ii, :])).dot(Yds_sparse).asformat("coo")

                idx_col_M_temp.append(M_temp.col + ii * n_nodes)
                idx_row_M_temp.append(M_temp.row + ii * n_nodes)
                M_data_temp.append(M_temp.data)

                idx_row_H_temp.append(H_temp.row + ii * n_nodes)
                H_data_temp.append(H_temp.data)

                times_multiplying.append(perf_counter() - start_multiply)

            M_big_idx_col = np.hstack([idx_1_M_col, np.hstack(idx_col_M_temp)])
            M_big_idx_row = np.hstack([idx_1_M_row, np.hstack(idx_row_M_temp)])
            M_big_idx_data = np.hstack([M_1_data, np.hstack(M_data_temp)])

            H_big_idx_row = np.hstack([idx_1_H_row, np.hstack(idx_row_H_temp)])
            H_big_idx_col = np.zeros(H_big_idx_row.shape[0], dtype=np.int32)
            H_big_data = np.hstack([H_1_data, np.hstack(H_data_temp)])

            M_big = csr_matrix(
                (M_big_idx_data, (M_big_idx_row, M_big_idx_col)))
            H_big = csr_matrix(
                (H_big_data, (H_big_idx_row, H_big_idx_col)), shape=(n_steps * n_nodes, 1))

        else:
            M_big = M
            H_big = H

        return M_big, H_big

    def reshape_tensor(sefl, tensor_array):
        """
        Reshapes a tensor array for power flow calculations.

        Parameters:
        tensor_array (np.ndarray): The tensor array to be reshaped.

        Returns:
        tuple: Reshaped tensor array and its original shape.
        """
        original_shape = tensor_array.shape
        tau = np.prod(original_shape[:-1])
        # This reshapes in place (No new memory use)
        tensor_array.shape = (tau, original_shape[-1])

        return tensor_array, original_shape

    def run_pf(self,
               active_power: np.ndarray = None,
               reactive_power: np.ndarray = None,
               flat_start: bool = True,
               start_value: np.ndarray = None,
               tolerance: float = 1e-6,
               algorithm: str = "tensor",
               sparse_solver: str = "scipy"):
        """
        Computes the power flow on the grid for the active and reactive power matrices/tensor.

        Parameters:
        -----------
            active_power: np.ndarray: Real array/tensor type np.float64. The dimension of the tensor should be
                            of the form (a x b x ... x m), where m is the number of buses minus the slack bus.
                            e.g., m = nbus - 1. The values of the array are in kilo-Watt [kW].
            reactive_power: np.ndarray: The array/tensor has the same characteristics than the active_power parameter.
            flat_start: bool: Flag to indicate if ta flat start should be use. This is currently the default. All
                            values of voltage starts at (1.0 + j0.0) p.u.
            start_value: np.ndarray: Array/Tensor with the same dimension os active_power parameter. It indicates the
                            warm start voltage values for the iterative algorithm. This array is in complex number
                            of type np.complex128.
            algorithm: str: Algorithm to be used for the power flow. The options are:
                            "hp-tensor":  Tensor-sparse power flow. The sparse solver is defined by the "sparse_solver"
                                        parameter.
                            "tensor":  Tensor-dense power flow.
                            "sequential": Power flow for only one instance of consumption. i.e., active_power and
                                        reactive_power is a 1-D vector.
                            "hp":  Power flow for only one instance of consumption, but using sparse matrices. The
                                        sparse solver is defined by the "sparse_solver" parameter.
                            "sam":  SAM (Successive Approximation Method).
            sparse_solver: str: Sparse solver algorithm to be used to solve a problem of the type Ax = b. The options
                            available are:
                            "scipy":  Sparse solver from scipy.
                            "pardiso":  Sparse solver from the Intel MKL library. Using this solver, we can factorize
                                        the matrix A only one time in a separate step. (from the equation Ax=b).
                                        This means that the iterative solution of the power flow is much faster.
        Return:
        -------
            solution: dict: This is a dictionary with the voltage solutions and different times to

            solution = {"v": Solution of voltage in complex numbers with the shape of the active_power array.
                        "time_pre_pf": Time in seconds to compute the power flow before the iterative for loop.
                        "time_pf":  Time in seconds to compute the power flow inside the iterative for loop.
                        "time_algorithm": Total time algorithm. time_algorithm = time_pre_pf + time_pf
                        "iterations": Total number of iterations to converge.

                        "convergence": Boolean indicating: True: Algorithm converged, False: it didn't.
                        "iterations_log": NOT USED.
                        "time_pre_pf_log": NOT USED.
                        "time_pf_log": NOT USED.
                        "convergence_log": NOT USED.
                        }
        """

        is_tensor = False
        if active_power is not None and reactive_power is not None:
            assert active_power.shape == reactive_power.shape, "Active and reactive power arrays must have the " \
                                                               "same shape."
            original_shape = active_power.shape

            if active_power.ndim > 2:  # Reshape form N-D to 2-D:
                active_power, original_shape = self.reshape_tensor(
                    active_power)
                reactive_power, _ = self.reshape_tensor(reactive_power)
                is_tensor = True

        self.P_file = active_power
        kwargs = dict()
        if algorithm == "hp":  # Same as hp-tensor but receive 1-D vectors
            pf_algorithm = self.run_pf_tensor_hp_laurent
            kwargs.update(solver=sparse_solver)
        elif algorithm == "sam":
            pf_algorithm = self.run_pf_sam_sequential
        elif algorithm == "sequential":  # Same as tensor but receive 1-D vectors
            pf_algorithm = self.run_pf_tensor
        elif algorithm == "tensor":
            pf_algorithm = self.run_pf_tensor
        elif algorithm == "hp-tensor":
            pf_algorithm = self.run_pf_tensor_hp_laurent
        elif algorithm == "gpu-tensor":
            pf_algorithm = self.run_pf_tensor
            kwargs.update(compute="gpu")
        else:
            raise ValueError("Incorrect power flow algorithm selected")

        solutions = pf_algorithm(active_power=self.P_file,  # 2-D Array
                                 reactive_power=reactive_power,  # 2-D Array
                                 flat_start=flat_start,
                                 start_value=start_value,
                                 tolerance=tolerance,
                                 **kwargs)

        if is_tensor:  # Solutions from a 2-D array to an N-D array.
            solutions["v"].shape = original_shape
            active_power.shape = original_shape
            reactive_power.shape = original_shape

        return solutions

    def run_pf_tensor(self,
                      active_power: np.ndarray,
                      reactive_power: np.ndarray = None,
                      *,
                      start_value=None,
                      iterations: int = 100,
                      tolerance: float = 1e-6,
                      flat_start: bool = True,
                      compute: str = "cpu") -> dict:
        if (active_power is not None) and (reactive_power is not None):
            # print('ok')
            assert len(
                active_power.shape) == 2, "Array must be two dimensional."
            assert len(
                reactive_power.shape) == 2, "Array must be two dimensional."
            assert active_power.shape[1] == reactive_power.shape[1] == self.nb - \
                1, "All nodes must have power values."
        else:
            # active_power = self.P_file[np.newaxis, :
            reactive_power = self.Q_file[np.newaxis, :]
            # print('zhong')

        self.ts_n = active_power.shape[0]  # Time steps to be simulated
        if flat_start:
            self.v_0 = np.ones((self.ts_n, self.nb - 1)) + 1j * \
                np.zeros((self.ts_n, self.nb - 1))  # Flat star
        v0_solutions = []
        total_time_pre_pf_all = []
        total_time_pf_all = []
        total_time_algorithm_all = []
        iterations_all = []
        flag_convergence_all = []
        flag_convergence_bool_all = True

        # Vector with all active power except slack
        active_power_pu = active_power / self.s_base
        # Vector with all reactive power except slack
        reactive_power_pu = reactive_power / self.s_base
        
        S_nom = active_power_pu + 1j * reactive_power_pu  # (ts x nodes)

        n_steps = S_nom.shape[0]
        n_nodes = S_nom.shape[1]

        if compute == "cpu":
            # print("CPU Solver selected")
            self._power_flow_tensor_solver = self._power_flow_tensor_constant_power
        elif compute == "gpu" and self.is_gpu_enabled is False:
            warnings.warn("GPU library not found, falling back to CPU.")
            self._power_flow_tensor_solver = self._power_flow_tensor_constant_power
        elif compute == "gpu" and self.is_gpu_enabled is True:
            # print("GPU Solver selected")
            self._power_flow_tensor_solver = self.gpu_solver.power_flow_gpu

        if compute == "cpu":
            # 5_000 x 10_000 did work. Empirical value for my machine
            DIMENSION_BOUND = 500 * 100_000
        else:
            # 5_000 x 15_000 did work. Empirical value for my machine
            DIMENSION_BOUND = 500 * 125_000

        idx = self._compute_chunks(
            DIMENSION_BOUND, n_nodes=n_nodes, n_steps=n_steps)
        n_chunks = len(idx) - 1

        # t = trange(n_chunks, desc='Chunk', leave=False)
        for ii in range(n_chunks):
            # t.set_description(f"Chunk: {ii + 1} of {n_chunks}", refresh=True)

            ts_chunk = idx[ii + 1] - idx[ii]  # Size of the chunk

            self.v_0 = np.ones((ts_chunk, self.nb - 1)) + 1j * \
                np.zeros((ts_chunk, self.nb - 1))  # Flat start

            S_chunk = S_nom[idx[ii]:idx[ii + 1]]

            if self.constant_power_only:
                start_time_pre_pf = self.start_time_pre_pf_tensor_constant_power_only
                # No pre-computing (Already done when creating the object)
                end_time_pre_pf = self.end_time_pre_pf_tensor_constant_power_only

                start_time_pf = perf_counter()
                self.v_0, t_iterations = self._power_flow_tensor_solver(K=self._K_,
                                                                        L=self._L_,
                                                                        S=S_chunk,
                                                                        v0=self.v_0,
                                                                        ts=ts_chunk,
                                                                        nb=self.nb,
                                                                        iterations=iterations,
                                                                        tolerance=tolerance)
                end_time_pf = perf_counter()

            else:
                # raise ValueError("This should not be running")
                start_time_pre_pf = perf_counter()
                self._F_, self._W_ = self._pre_power_flow_tensor(flag_all_constant_impedance_is_zero=self.flag_all_constant_impedance_is_zero,
                                                                 flag_all_constant_current_is_zero=self.flag_all_constant_current_is_zero,
                                                                 flag_all_constant_powers_are_ones=self.flag_all_constant_powers_are_ones,
                                                                 ts_n=ts_chunk,
                                                                 nb=self.nb,
                                                                 S_nom=S_chunk,
                                                                 alpha_Z=self.alpha_Z,
                                                                 alpha_I=self.alpha_I,
                                                                 alpha_P=self.alpha_P,
                                                                 Yds=self.Yds,
                                                                 Ydd=self.Ydd)
                end_time_pre_pf = perf_counter()

                start_time_pf = perf_counter()
                self.v_0, t_iterations = self._power_flow_tensor(_F_=self._F_,
                                                                 _W_=self._W_,
                                                                 v_0=self.v_0,
                                                                 ts_n=ts_chunk,
                                                                 nb=self.nb,
                                                                 iterations=iterations,
                                                                 tolerance=tolerance)
                end_time_pf = perf_counter()

            if t_iterations == iterations:
                flag_convergence = False
                warnings.warn("Power flow did not converge.")
            else:
                flag_convergence = True

            total_time_pre_pf = end_time_pre_pf - start_time_pre_pf
            total_time_pf = end_time_pf - start_time_pf
            total_time_algorithm = total_time_pre_pf + total_time_pf

            total_time_pre_pf_all.append(total_time_pre_pf)
            total_time_pf_all.append(total_time_pf)
            total_time_algorithm_all.append(total_time_algorithm)
            iterations_all.append(t_iterations)
            flag_convergence_all.append(flag_convergence)
            flag_convergence_bool_all = flag_convergence_bool_all & flag_convergence

            v0_solutions.append(self.v_0.copy())

        self.v_0 = np.vstack(v0_solutions)

        solution = {"v": self.v_0,  # 2D-Vector. Solution of voltage in complex numbers
                    "time_pre_pf": sum(total_time_pre_pf_all),
                    "time_pf":  sum(total_time_pf_all),
                    "time_algorithm":  sum(total_time_algorithm_all),
                    "iterations": np.floor(np.mean(iterations_all)),

                    "convergence": flag_convergence_bool_all,
                    "iterations_log": iterations_all,
                    "time_pre_pf_log": total_time_pre_pf_all,
                    "time_pf_log": total_time_pf_all,
                    "convergence_log": flag_convergence_all
                    }

        return solution

    def run_pf_sam_sequential(self,
                              active_power: np.ndarray = None,
                              reactive_power: np.ndarray = None,
                              flat_start: bool = True,
                              start_value: np.array = None):
        """
        Single time step power flow with numba performance increase.
        This is the implementation of [1], algorithm called SAM (Successive Approximation Method)

        V[k+1] = B^{-1} ( A[k] @ V[k]^{*}  - C - D[k])

        Where:
        A[k] = np.diag(\alpha_p \odot V[k]^{* -2} * S_n^{*}), \odot == Hadamard product, * == complex conjugate
        B = np.diag(\alpha_z \odot S_n^{*}) + Y_dd
        C = Y_ds @ V_s + \alpha_i \odot S_n^{*}
        D[k] = 2 \alpha_p \odot V[k]^{* -1} \odot S_n^{*}

        Please note that for constant power only. i.e., \alpha_p = 1, \alpha_i = 0, \alpha_z = 0.
        The matrices reduces to:

        A[k] = np.diag(V[k]^{* -2} * S_n^{*}), \odot == Hadamard product, * == complex conjugate
        B = Y_dd
        C = Y_ds @ V_s
        D[k] = 2 V[k]^{* -1} \odot S_n^{*}

        [1] Juan S. Giraldo, Oscar Danilo Montoya, Pedro P. Vergara, Federico Milano, "A fixed-point current injection
            power flow for electric distribution systems using Laurent series", Electric Power Systems Research,
            Volume 211, 2022. https://doi.org/10.1016/j.epsr.2022.108326.

        """

        if (active_power is not None) and (reactive_power is not None):
            active_power, reactive_power = self._check_2d_to_1d(
                active_power, reactive_power)
        else:  # Default case
            active_power = self.P_file
            reactive_power = self.Q_file

        if flat_start:
            # TODO
            self.v_0 = np.ones(
                (self.nb - 1, 1), dtype="complex128")  # 2D-Vector
        elif start_value is not None:
            # TODO: Check the dimensions of the flat start
            self.v_0 = start_value  # User's start value

        # Vector with all active power except slack
        active_power_pu = active_power / self.s_base
        # Vector with all reactive power except slack
        reactive_power_pu = reactive_power / self.s_base
        S_nom = (active_power_pu + 1j * reactive_power_pu).reshape(-1, )

        if self.constant_power_only:
            start_time_pre_pf = perf_counter()
            # No precomputing, the minimum matrix multiplication is done in the initialization of the object.
            end_time_pre_pf = perf_counter()

            start_time_pf = perf_counter()
            V, iteration = self._power_flow_sam_sequential_constant_power_only(B_inv=-self._K_,
                                                                               C=self.Yds.flatten(),
                                                                               v_0=self.v_0,
                                                                               s_n=S_nom,
                                                                               iterations=self.iterations,
                                                                               tolerance=self.tolerance)
            end_time_pf = perf_counter()

        else:
            start_time_pre_pf = perf_counter()
            B_inv, C, S_nom = self._pre_power_flow_sam_sequential(active_power,  # TODO: Change the input to S_nom
                                                                  reactive_power,
                                                                  s_base=self.s_base,
                                                                  alpha_Z=self.alpha_Z,
                                                                  alpha_I=self.alpha_I,
                                                                  Yds=self.Yds,
                                                                  Ydd=self.Ydd,
                                                                  nb=self.nb)
            end_time_pre_pf = perf_counter()

            start_time_pf = perf_counter()
            V, iteration = self._power_flow_sam_sequential(B_inv,
                                                           C,
                                                           v_0=self.v_0,
                                                           s_n=S_nom,
                                                           alpha_P=self.alpha_P,
                                                           iterations=self.iterations,
                                                           tolerance=self.tolerance)
            end_time_pf = perf_counter()

        if iteration == self.iterations:
            flag_convergence = False
        else:
            flag_convergence = True

        total_time_pre_pf = end_time_pre_pf - start_time_pre_pf
        total_time_pf = end_time_pf - start_time_pf
        total_time_algorithm = total_time_pre_pf + total_time_pf

        solution = {"v": V.flatten(),  # 1D-Vector. Solution of voltage in complex numbers
                    "time_pre_pf": total_time_pre_pf,
                    "time_pf": total_time_pf,
                    "time_algorithm": total_time_algorithm,
                    "iterations": iteration,
                    "convergence": flag_convergence, }

        return solution

    def line_currents(self, volt_solutions=None):
        raise NotImplementedError
