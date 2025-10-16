'''
The following code is based on the implementation of https://github.com/ShengrenHou/RL-ADN from

Cite as:
Shengren Hou, Shuyi Gao, Weijie Xia, Edgar Mauricio Salazar Duque, Peter Palensky, Pedro P. Vergara,
RL-ADN: A high-performance Deep Reinforcement Learning environment for optimal Energy Storage Systems dispatch in active distribution networks, Energy and AI,
Volume 19, 2025, 100457, ISSN 2666-5468, https://doi.org/10.1016/j.egyai.2024.100457
'''
from numba import prange
import numpy as np


def pre_power_flow_sam_sequential(active_power,
                                  reactive_power,
                                  s_base,
                                  alpha_Z,
                                  alpha_I,
                                  Yds,
                                  Ydd,
                                  nb
                                  ):
    """
     Prepares the matrices for the SAM sequential power flow method.

     Parameters:
     active_power (np.ndarray): Array of active power values.
     reactive_power (np.ndarray): Array of reactive power values.
     s_base (float): Base power value for per-unit conversion.
     alpha_Z (np.ndarray): Array of constant impedance values.
     alpha_I (np.ndarray): Array of constant current values.
     Yds (np.ndarray): Admittance matrix between slack and load buses.
     Ydd (np.ndarray): Admittance matrix between load buses.
     nb (int): Number of buses in the network.

     Returns:
     tuple: Tuple containing B_inv, C, and S_nom matrices used in the SAM sequential power flow method.
     """
    active_power_pu = active_power / s_base  # Vector with all active power except slack
    reactive_power_pu = reactive_power / s_base  # Vector with all reactive power except slack

    S_nom = (active_power_pu + 1j * reactive_power_pu).reshape(-1, )
    if not np.any(alpha_Z):  # \alpha_z is 0
        B_inv = np.linalg.inv(Ydd)
    else:
        # TODO: Assert that the shapoes of alpha_Z and S_nom are the same
        B = np.diag(np.multiply(alpha_Z, np.conj(S_nom))) + Ydd
        B_inv = np.linalg.inv(B)

    if not np.any(alpha_I):  # all \alpha_i are 0
        C = Yds  # Constant
    else:
        C = Yds + np.multiply(alpha_I, np.conj(S_nom)).reshape(nb - 1, 1)  # Constant

    return B_inv, C, S_nom

def power_flow_sam_sequential(B_inv,
                              C,
                              v_0,
                              s_n,
                              alpha_P,
                              iterations,
                              tolerance,
                              ):
    """
    Performs the SAM sequential power flow calculation.

    Parameters:
    B_inv (np.ndarray): Inverse of matrix B.
    C (np.ndarray): Matrix C.
    v_0 (np.ndarray): Initial voltage values.
    s_n (np.ndarray): Power values.
    alpha_P (np.ndarray): Array of constant power values.
    iterations (int): Maximum number of iterations.
    tolerance (float): Convergence tolerance.

    Returns:
    tuple: Tuple containing the final voltage values and the number of iterations performed.
    """
    iteration = 0
    tol = np.inf
    while (iteration < iterations) & (tol >= tolerance):
        # Update matrices A and D:
        A = np.diag(alpha_P * (1 / np.conj(v_0) ** 2) * np.conj(s_n))
        D = 2 * alpha_P * (1 / np.conj(v_0)) * np.conj(s_n)

        v = B_inv @ (A @ np.conj(v_0) - C - D)
        tol = np.max(np.abs(np.abs(v) - np.abs(v_0)))
        v_0 = v  # Voltage at load buses
        iteration += 1

    return v_0, iteration  # Solution of voltage in complex numbers


def power_flow_sam_sequential_constant_power_only(B_inv,
                                                  C,
                                                  v_0,
                                                  s_n,
                                                  iterations,
                                                  tolerance,
                                                  ):
    """
    Performs the SAM sequential power flow calculation for constant power loads only.

    Parameters:
    B_inv (np.ndarray): Inverse of matrix B.
    C (np.ndarray): Matrix C.
    v_0 (np.ndarray): Initial voltage values.
    s_n (np.ndarray): Power values.
    iterations (int): Maximum number of iterations.
    tolerance (float): Convergence tolerance.

    Returns:
    tuple: Tuple containing the final voltage values and the number of iterations performed.
    """
    iteration = 0
    tol = np.inf
    while (iteration < iterations) & (tol >= tolerance):
        # Update matrices A and D:
        A = np.diag((1 / np.conj(v_0) ** 2) * np.conj(s_n))
        D = 2 * (1 / np.conj(v_0)) * np.conj(s_n)
        # D = D.reshape(-1, 1)

        v = B_inv @ (A @ np.conj(v_0) - C - D)
        tol = np.max(np.abs(np.abs(v) - np.abs(v_0)))
        v_0 = v  # Voltage at load buses
        iteration += 1

    return v_0, iteration  # Solution of voltage in complex numbers


def pre_power_flow_tensor(flag_all_constant_impedance_is_zero,
                          flag_all_constant_current_is_zero,
                          flag_all_constant_powers_are_ones,
                          ts_n,
                          nb,
                          S_nom,
                          alpha_Z,
                          alpha_I,
                          alpha_P,
                          Yds,
                          Ydd):
    """
    Prepares the matrices for the tensor-based power flow method.

    Parameters:
    flag_all_constant_impedance_is_zero (bool): Flag indicating if all constant impedances are zero.
    flag_all_constant_current_is_zero (bool): Flag indicating if all constant currents are zero.
    flag_all_constant_powers_are_ones (bool): Flag indicating if all constant powers are one.
    ts_n (int): Number of time steps.
    nb (int): Number of buses.
    S_nom (np.ndarray): Nominal power values.
    alpha_Z (np.ndarray): Array of constant impedance values.
    alpha_I (np.ndarray): Array of constant current values.
    alpha_P (np.ndarray): Array of constant power values.
    Yds (np.ndarray): Admittance matrix between slack and load buses.
    Ydd (np.ndarray): Admittance matrix between load buses.

    Returns:
    tuple: Tuple containing matrices _F_2, _W_2 used in the tensor power flow method.
    """

    if not flag_all_constant_impedance_is_zero:
        _alpha_z_power = np.multiply(np.conj(S_nom), alpha_Z)  # (ts x nodes)
    else:
        _alpha_z_power = np.zeros((ts_n, nb - 1))  # (ts x nodes)

    if not flag_all_constant_current_is_zero:
        _alpha_i_power = np.multiply(np.conj(S_nom), alpha_I)  # (ts x nodes)
    else:
        _alpha_i_power = np.zeros((ts_n, nb - 1))  # (ts x nodes)

    if flag_all_constant_powers_are_ones:
        _alpha_p_power = np.conj(S_nom)  # (ts x nodes)
    else:
        _alpha_p_power = np.multiply(np.conj(S_nom), alpha_P)  # (ts x nodes)

    _B_inv2 = np.zeros((ts_n, nb - 1, nb - 1), dtype="complex128")
    _F_2 = np.zeros((ts_n, nb - 1, nb - 1), dtype="complex128")
    _W_2 = np.zeros((ts_n, nb - 1), dtype="complex128")

    _C2 = _alpha_i_power + Yds.reshape(-1)  # (ts x nodes)  Sum is broadcasted to all rows of _alpha_i_power

    for i in prange(ts_n):
        _B_inv2[i] = np.linalg.inv(np.diag(_alpha_z_power[i]) + Ydd)
        _F_2[i] = -_B_inv2[i] * _alpha_p_power[i].reshape(1, -1)  # Broadcast multiplication
        _W_2[i] = (-_B_inv2[i] @ _C2[i].reshape(-1, 1)).reshape(-1)

    return _F_2, _W_2

def power_flow_tensor(_F_,
                      _W_,
                      v_0,
                      ts_n,
                      nb,
                      iterations,
                      tolerance,
                      ):
    """
    Performs the tensor-based power flow calculation.

    Parameters:
    _F_ (np.ndarray): Matrix F.
    _W_ (np.ndarray): Matrix W.
    v_0 (np.ndarray): Initial voltage values.
    ts_n (int): Number of time steps.
    nb (int): Number of buses.
    iterations (int): Maximum number of iterations.
    tolerance (float): Convergence tolerance.

    Returns:
    tuple: Tuple containing the final voltage values and the number of iterations performed.
    """
    iteration = 0
    tol = np.inf
    while (iteration < iterations) & (tol >= tolerance):
        v_recp_conj = np.reciprocal(np.conj(v_0))
        RT2 = np.zeros((ts_n, nb - 1), dtype="complex128")
        for i in prange(ts_n):  # This is critical as it makes a lot of difference
            RT2[i] = _F_[i] @ v_recp_conj[i]
        v = _W_ + RT2
        tol = np.max(np.abs(np.abs(v) - np.abs(v_0)))
        v_0 = v
        iteration += 1

    return v_0, iteration



def power_flow_tensor_constant_power_numba_parallel_True(K,
                                         L,
                                         S,
                                         v0,
                                         ts,
                                         nb,
                                         iterations,
                                         tolerance
                                         ):
    """
    Performs the tensor-based power flow calculation with constant power loads, optimized for parallel execution.

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
    while (iteration < iterations) & (tol >= tolerance):
        v = np.zeros((ts, nb - 1), dtype="complex128")  # TODO: Test putting this outside of while loop
        for i in prange(ts):
            v[i] = (K @ (np.conj(S[i]) * (1 / np.conj(v0[i]))).reshape(-1, 1) + L).T
        tol = np.max(np.abs(np.abs(v) - np.abs(v0)))
        v0 = v  # Voltage at load buses
        iteration += 1

    return v0, iteration


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
    
    # print(f'------numbarize-----')
    # print(f'S: {S}')
    # print(f'K shape: {K.shape}')
    # print(f'L shape: {L.shape}')
    # print(f'S shape: {S.shape}')
    # print(f'v0 shape: {v0.shape}')
    # print(f'voltage_k shape: {voltage_k.shape}')    
    
    while iteration < iterations and tol >= tolerance:
        # safe_v0 = np.where(np.abs(v0) < epsilon, epsilon, v0)
        # v0 = np.nan_to_num(v0, nan=0.0, posinf=0.0, neginf=0.0)
        # S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        # assert not np.isinf(S).any(), "There are inf values in the S values"
        LAMBDA = np.conj(S * (1 / (v0)))  # Hadamard product ( (nb-1) x ts)
        Z = K @ LAMBDA  # Matrix ( (nb-1) x ts )
        voltage_k = Z + L  # This is a broadcasted sum dim => ( (nb-1) x ts  +  (nb-1) x 1 => (nb-1) x ts )
        tol = np.max(np.abs(np.abs(voltage_k) - np.abs(v0)))
        v0 = voltage_k
        iteration += 1

    S = S.T  # Recover the original shape of the power
    v0 = v0.T  # Recover the original shape of the power

    return v0, iteration


def power_flow_tensor_constant_power_new(K,
                                         L,
                                         S,
                                         v0,
                                         ts,
                                         nb,
                                         iterations,
                                         tolerance
                                         ):
    """
    A new version of the tensor-based power flow calculation for constant power loads, supporting parallel execution.

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
    # S = S.T
    # v0 = v0.T

    LAMBDA = np.zeros((nb - 1, ts)).astype(np.complex128)
    Z = np.zeros((nb - 1, ts)).astype(np.complex128)
    voltage_k = np.zeros((nb - 1, ts)).astype(np.complex128)

    voltage_k = voltage_k.T
    W = L.ravel()

    while iteration < iterations and tol >= tolerance:
        LAMBDA = np.conj(S.T * (1 / v0.T))  # Hadamard product ( (nb-1) x ts)
        Z = K @ LAMBDA  # Matrix ( (nb-1) x ts )
        Z = Z.T
        for j in prange(ts): # This is a brodcasted sum ( (nb-1) x ts  +  (nb-1) x 1 => (nb-1) x ts )
            voltage_k[j] = Z[j] + W

        tol = np.max(np.abs(np.abs(voltage_k) - np.abs(v0)))
        v0 = voltage_k
        iteration += 1

    return v0, iteration