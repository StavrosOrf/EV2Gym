import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.spaces import MultiDiscrete, Discrete

import numpy as np


class BinaryAction(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """
    Clip the continuous action within the valid :class:`Box` observation space bound.
    """

    def __init__(self, env: gym.Env):
        """A wrapper for clipping continuous actions within the valid bound.

        Args:
            env: The environment to apply the wrapper
        """
        assert isinstance(env.action_space, Box)

        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)

        self.min_action = np.zeros(env.action_space.shape)

        epsilon = 1e-4
        counter = 0
        for cs in env.charging_stations:
            n_ports = cs.n_ports
            for i in range(n_ports):
                self.min_action[counter] = cs.min_charge_current / \
                    cs.max_charge_current + epsilon

                counter += 1

    def action(self, action: np.ndarray) -> np.ndarray:
        """ 
        If action[i] > 0 then action[i] = 1 else action[i] = min_action[i]

        Args:
            action: The action to clip

        Returns:
            The clipped action
        """

        return np.where(action > 0.5, 1, self.min_action)


class ThreeStep_Action(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """
    Clip the continuous action within the valid :class:`Box` observation space bound.
    """

    def __init__(self, env: gym.Env):
        """
        Args:
            env: The environment to apply the wrapper
        """
        assert isinstance(env.action_space, Box)

        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)

        self.min_action = np.zeros(env.action_space.shape)

        epsilon = 1e-4
        counter = 0
        for cs in env.charging_stations:
            n_ports = cs.n_ports
            for i in range(n_ports):
                self.min_action[counter] = cs.min_charge_current / \
                    cs.max_charge_current + epsilon

                counter += 1

    def action(self, action: np.ndarray) -> np.ndarray:
        """ 
        If action[i] == 0 then action[i] = 0
        elif action[i] == -1 then action[i] = self.min_action
        else action[i] = 1

        Args:
            action: The action to clip

        Returns:
            The clipped action
        """

        return np.where(action == 0, 0, np.where(action == 1, self.min_action, 1))


class ThreeStep_Action_DiscreteActionSpace(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """
    Clip the continuous action within the valid :class:`Box` observation space bound.
    """

    def __init__(self, env: gym.Env):
        """
        Args:
            env: The environment to apply the wrapper
        """
        assert isinstance(env.action_space, Box)

        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)

        num_actions = env.action_space.shape[0]
        env.action_space = MultiDiscrete([3]*env.action_space.shape[0])

        print(f"Action Space: {env.action_space}")

        self.min_action = np.zeros(num_actions)

        epsilon = 1e-4
        counter = 0
        for cs in env.charging_stations:
            n_ports = cs.n_ports
            for i in range(n_ports):
                self.min_action[counter] = cs.min_charge_current / \
                    cs.max_charge_current + epsilon

                counter += 1

    def action(self, action: np.ndarray) -> np.ndarray:
        """ 
        If action[i] == 0 then action[i] = 0
        elif action[i] == -1 then action[i] = self.min_action
        else action[i] = 1

        Args:
            action: The action to clip

        Returns:
            The clipped action
        """

        return np.where(action == 0, 0, np.where(action == 1, self.min_action, 1))


def mask_fn(env: gym.Env) -> np.ndarray:
    """
    Create a mask for the action space to mask the actions that are not available.
    For example, if an EV is not connected to a charging station, then the action to charge the EV is not available.
    """

    mask = np.ones((env.action_space.nvec.shape[0], 3))

    counter = 0
    for cs in env.charging_stations:
        for EV in cs.evs_connected:
            if EV is None:
                mask[counter, 1:] = [0] * (env.action_space.nvec[counter] - 1)
            counter += 1

    return mask


class Rescale_RepairLayer(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    '''
    This class is used to rescale the actions to the valid range of the charging stations
    and proportionally increase or decrease the power of the EVs to meet the power setpoint.
    '''

    def __init__(self, env: gym.Env):
        """
        Args:
            env: The environment to apply the wrapper
        """
        assert isinstance(env.action_space, Box)

        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)

        self.verbose = False
        self.env = env
        # find average charging power of the simulation

        epsilon = 1e-4
        self.threshold = 0  # kW

        # initialize the min_action list
        self.min_action = np.zeros(env.action_space.shape)
        self.max_cs_power = np.zeros(env.action_space.shape)

        assert (env.number_of_ports_per_cs ==
                1), "This class is only implemented for one port per charging station"

        for i, cs in enumerate(env.charging_stations):
            self.min_action[i] = cs.min_charge_current / \
                cs.max_charge_current + epsilon
            self.max_cs_power[i] = cs.get_max_power()

        if self.verbose:
            print(f'Min action: {self.min_action}')

        self.number_of_ports_per_cs = env.number_of_ports_per_cs
        # list with the ids of EVs that were already served in this round
        self.ev_buffer = []
        self.min_power = []
        self.max_power = []

        self.occupied_ports = [0] * env.number_of_ports

    def update_ev_buffer(self, env) -> None:
        '''
        This function updates the EV buffer list with the EVs that are currently parked by adding or removing them.
        '''
        counter = 0
        # iterate over all ports
        for cs in env.charging_stations:
            for port in range(cs.n_ports):
                if cs.evs_connected[port] is not None:
                    if cs.evs_connected[port].get_soc() < 1:

                        if counter not in self.ev_buffer:
                            self.ev_buffer.insert(0, counter)
                            min_power = max(cs.get_min_charge_power(
                            ), cs.evs_connected[port].min_ac_charge_power)
                            self.min_power.insert(0, min_power)
                            max_power = min(
                                cs.get_max_power(), cs.evs_connected[port].max_ac_charge_power)
                            self.max_power.insert(0, max_power)

                            self.occupied_ports[counter] = 1
                    else:
                        if counter in self.ev_buffer:
                            # find index of the EV in the buffer
                            index = self.ev_buffer.index(counter)
                            self.ev_buffer.remove(counter)
                            self.min_power.pop(index)
                            self.max_power.pop(index)
                            self.occupied_ports[counter] = 0

                else:
                    if counter in self.ev_buffer:
                        index = self.ev_buffer.index(counter)
                        self.ev_buffer.remove(counter)
                        self.min_power.pop(index)
                        self.max_power.pop(index)
                        self.occupied_ports[counter] = 0

                counter += 1

    def calculate_total_power(self, action: np.ndarray) -> float:
        '''
        This function calculates the total power that is requested by the EVs in the buffer list.
        '''
        total_power = 0
        for i in self.ev_buffer:
            total_power += np.clip(action[i] * self.max_cs_power[i], self.min_power[self.ev_buffer.index(
                i)], self.max_power[self.ev_buffer.index(i)])

        return total_power

    def rescale_actions(self, actions, min_action):
        """
        Rescale actions from interval (0, 1) to (min_action, 1) for each corresponding action.

        Parameters:
            actions (numpy array or list): Actions in the interval (0, 1).
            min_action (numpy array or list): Minimum value of the new interval for each action.

        Returns:
            numpy array: Rescaled actions in the interval (min_action, 1) for each corresponding action.
        """
        actions = np.array(actions)  # Ensure actions is a numpy array
        min_action = np.array(min_action)  # Ensure min_action is a numpy array
        if len(actions) != len(min_action):
            raise ValueError(
                "actions and min_action must be of the same length")

        # Apply the rescaling transformation element-wise
        rescaled_actions = actions * (1 - min_action) + min_action
        return rescaled_actions

    def action(self, action: np.ndarray) -> np.ndarray:

        # this function returns the action list based on the round robin algorithm

        # in W
        power_setpoint = self.env.power_setpoints[self.env.current_step]

        # rescale actions from interval (0,1) to interval (min_action,1) for every charger

        if self.verbose:
            print("-------------------RR Correction Layer-------------------")
            print(
                f'Power setpoint: {power_setpoint:.2f} kW | Step: {self.env.current_step}/{self.env.simulation_length}')
            print(f'  actions:       { [round(a, 2) for a in action]}')
            print(
                f' min action:     { [round(a, 2) for a in self.min_action]}')

        # rescaled actions
        action = self.rescale_actions(action, self.min_action)

        if self.verbose:
            print(f'Rescaled actions:{[round(a, 2) for a in action]}')

        # get currently parked EVs
        self.update_ev_buffer(self.env)

        current_action_power = self.calculate_total_power(action)
        total_power_potential = sum(self.min_power)

        if self.verbose:
            print(f'Current action power: {current_action_power:.2f} kW')
            print(f'EV buffer: {self.ev_buffer}')
            print(f'Min power: {[round(a, 2) for a in self.min_power]}')
            print(f'Max power: {[round(a, 2) for a in self.max_power]}')

        if current_action_power < power_setpoint:

            total_power_potential = current_action_power

            # Calculate the deficit in power that needs to be increased
            power_deficit = power_setpoint - total_power_potential + self.threshold

            # Calculate the proposed power for each EV based on current action
            proposed_power = np.zeros(len(self.ev_buffer))

            for index, i in enumerate(self.ev_buffer):
                # print(f'EV: {i} | Action: {action[i]} | Max power: {self.max_power[index]}')
                # clamp proposed power to the minimum and maximum power for each EV
                proposed_power[index] = np.clip(
                    action[i] * self.max_cs_power[i], self.min_power[index], self.max_power[index])

            if self.verbose:
                print(f"Proposed power before increase: {proposed_power}")

            # Calculate the available power range (up to the maximum power)
            ev_power_range = [self.max_power[i] - proposed_power[i]
                              for i in range(len(self.ev_buffer))]
            total_power_range = sum(ev_power_range)

            # If the total power range is zero (all EVs already at max power), no adjustment is needed
            if total_power_range > 0:
                # Proportional increase factor based on available power range
                increase_factor = min(1, power_deficit / total_power_range)

                # Increase the power of each EV proportionally up to their max_power
                for i in range(len(self.ev_buffer)):
                    power_to_increase = ev_power_range[i] * increase_factor
                    new_power = proposed_power[i] + power_to_increase

                    # Ensure the new power does not exceed the max power for each EV
                    proposed_power[i] = min(new_power, self.max_power[i])

            # After adjustment, calculate the new total power
            total_power_potential = sum(proposed_power)

            # Reflect the changes to the actions by scaling them between (min_power, 1)
            new_actions = []
            for i in range(len(self.ev_buffer)):
                # Scale the proposed power between (min_power[i], max_power[i])
                new_actions.append(proposed_power[i] / self.max_cs_power[i])

            if self.verbose:
                print(f"New proposed power for each EV: {proposed_power}")
                print(f'Final power used: {total_power_potential:.2f} kW')
                print(f"New actions: {[round(a, 2) for a in new_actions]}")

            # Update the actions for each EV in the action dictionary
            for i in range(len(self.ev_buffer)):
                action[self.ev_buffer[i]] = new_actions[i]

            if self.verbose:
                print(
                    f"Final actions vector: {[round(a, 2) for a in action * self.occupied_ports]}")

            return action * self.occupied_ports

        elif current_action_power + self.threshold > power_setpoint:

            total_power_potential = current_action_power

            # Calculate the excess power that needs to be reduced
            excess_power = total_power_potential - power_setpoint + self.threshold

            proposed_power = np.zeros(len(self.ev_buffer))

            # Calculate the proposed power for each EV
            for index, i in enumerate(self.ev_buffer):
                # print(f'EV: {i} | Action: {action[i]} | Max power: {self.max_power[index]}')
                proposed_power[index] = np.clip(
                    action[i] * self.max_cs_power[i], self.min_power[index], self.max_power[index])

            if self.verbose:
                print(
                    f"Proposed power before adjustment: {[round(p, 2) for p in proposed_power]}")

            # Proportionally reduce the charging power of all selected EVs
            ev_power_range = [p - min_p for p,
                              min_p in zip(proposed_power, self.min_power)]
            total_power_range = sum(ev_power_range)

            if total_power_range > 0:
                # Proportional reduction factor based on available power range
                reduction_factor = min(1, excess_power / total_power_range)

                # Reduce the power of each EV proportionally up to their min_power
                for i in range(len(self.ev_buffer)):
                    power_to_reduce = ev_power_range[i] * reduction_factor
                    new_power = proposed_power[i] - power_to_reduce

                    # Ensure the new power does not go below the minimum power for each EV
                    proposed_power[i] = max(new_power, self.min_power[i])

            # After initial adjustment, check if the new total power is still lower than the power setpoint
            total_power_potential = sum(proposed_power)
            remaining_deficit = power_setpoint - total_power_potential + self.threshold

            if remaining_deficit > 0:  # If there is still a deficit in power, try to increase
                for i in range(len(self.ev_buffer)):
                    if remaining_deficit <= 0:
                        break  # If we've increased the power to meet the setpoint, stop
                    # Try increasing power, but not above max_power
                    increaseable_amount = self.max_power[i] - proposed_power[i]
                    increase = min(remaining_deficit, increaseable_amount)
                    proposed_power[i] += increase
                    remaining_deficit -= increase

            # Final recalculation of total power after all adjustments
            total_power_potential = sum(proposed_power)

            # Reflect the changes to the actions by scaling them between (min_power, 1)
            new_actions = []
            for i in range(len(self.ev_buffer)):
                new_actions.append(proposed_power[i] / self.max_cs_power[i])

            if self.verbose:
                print(
                    f"-- Final proposed power: {[round(p, 2) for p in proposed_power]}")
                print(f'Final power used: {total_power_potential:.2f} kW')
                print(f"New actions: {[round(a, 2) for a in new_actions]}")

            for i in range(len(self.ev_buffer)):
                action[self.ev_buffer[i]] = new_actions[i]

            if self.verbose:
                print(
                    f"Final actions vector: {[round(a, 2) for a in action * self.occupied_ports]}")

            return action * self.occupied_ports
        else:
            # no change needed

            if self.verbose:
                print(f'Final power used: {current_action_power:.2f} kW')

            return action * self.occupied_ports


class MinMax_RepairLayer(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    '''
    This class is used to rescale the actions to the valid range of the charging stations
    and to give maximum power to EVs with the highest power setpoint, and minimum power to EVs with the lowest power setpoint.

    In case of draw, the EV with the highest SOC will be given the maximum power.
    '''

    def __init__(self, env, verbose=False,  **kwargs):
        raise NotImplementedError("MinMax_RepairLayer is not implemented yet!!!!")
        self.verbose = verbose
        self.env = env
        # find average charging power of the simulation
        epsilon = 1e-4
        self.max_cs_power = np.zeros(env.action_space.shape)

        for i, cs in enumerate(env.charging_stations):
            self.min_action = cs.min_charge_current / cs.max_charge_current + epsilon
            self.max_cs_power[i] = cs.get_max_power()

        self.number_of_ports_per_cs = env.number_of_ports_per_cs
        # list with the ids of EVs that were already served in this round
        self.ev_buffer = []
        self.min_power = []
        self.max_power = []

    def get_env(self):
        return self.env
    
    def rescale_actions(self, actions, min_action):
        """
        Rescale actions from interval (0, 1) to (min_action, 1) for each corresponding action.

        Parameters:
            actions (numpy array or list): Actions in the interval (0, 1).
            min_action (numpy array or list): Minimum value of the new interval for each action.

        Returns:
            numpy array: Rescaled actions in the interval (min_action, 1) for each corresponding action.
        """
        actions = np.array(actions)  # Ensure actions is a numpy array
        min_action = np.array(min_action)  # Ensure min_action is a numpy array
        if len(actions) != len(min_action):
            raise ValueError(
                "actions and min_action must be of the same length")

        # Apply the rescaling transformation element-wise
        rescaled_actions = actions * (1 - min_action) + min_action
        return rescaled_actions

    def update_ev_buffer(self, env) -> None:
        '''
        This function updates the EV buffer list with the EVs that are currently parked by adding or removing them.
        '''
        counter = 0
        # iterate over all ports
        for cs in env.charging_stations:
            for port in range(cs.n_ports):
                if cs.evs_connected[port] is not None:
                    if cs.evs_connected[port].get_soc() < 1:

                        if counter not in self.ev_buffer:
                            self.ev_buffer.insert(0, counter)
                            min_power = max(cs.get_min_charge_power(
                            ), cs.evs_connected[port].min_ac_charge_power)
                            self.min_power.insert(0, min_power)
                            max_power = min(
                                cs.get_max_power(), cs.evs_connected[port].max_ac_charge_power)
                            self.max_power.insert(0, max_power)
                    else:
                        if counter in self.ev_buffer:
                            # find index of the EV in the buffer
                            index = self.ev_buffer.index(counter)
                            self.ev_buffer.remove(counter)
                            self.min_power.pop(index)
                            self.max_power.pop(index)

                else:
                    if counter in self.ev_buffer:
                        index = self.ev_buffer.index(counter)
                        self.ev_buffer.remove(counter)
                        self.min_power.pop(index)
                        self.max_power.pop(index)
                counter += 1

    def action(self, action: np.ndarray) -> np.ndarray:

        # this function returns the action list based on the round robin algorithm

        power_setpoint = self.env.power_setpoints[self.env.current_step]  # in W

        if self.verbose:
            print("-------------------MinMax RepairLayer -------------------")
            print(f'Power setpoint: {power_setpoint:.2f} kW')

        # get currently parked EVs
        self.update_ev_buffer(self.env)

        if self.verbose:
            print(f'EV buffer: {self.ev_buffer}')
            print(f'Min power: {self.min_power}')
            print(f'Max power: {self.max_power}')

        total_power_potential = sum(self.min_power)

        evs_to_charge = []
        temp_ev_buffer = self.ev_buffer.copy()
        counter = 0
        for EV in temp_ev_buffer:
            next_power = self.max_power[temp_ev_buffer.index(EV)] - \
                self.min_power[temp_ev_buffer.index(EV)]

            if total_power_potential > power_setpoint:
                break
            total_power_potential += next_power
            counter += 1

        # get the EVs to charge in this round
        evs_to_charge = self.ev_buffer[:counter]
        min_power = self.min_power[:counter]
        max_power = self.max_power[:counter]

        self.ev_buffer = self.ev_buffer[counter:]
        self.min_power = self.min_power[counter:]
        self.max_power = self.max_power[counter:]

        self.ev_buffer.extend(evs_to_charge)
        self.min_power.extend(min_power)
        self.max_power.extend(max_power)

        # create action list

        if self.verbose:
            print(f'Final power used: {total_power_potential}')

        action_list = np.ones(self.env.number_of_ports) * self.min_action

        # set the action for the EVs to charge
        for i, ev in enumerate(evs_to_charge):

            if i == len(evs_to_charge) - 1 and total_power_potential >= power_setpoint:
                if total_power_potential - power_setpoint < 0:
                    break
                action_list[ev] = 1 - (total_power_potential -
                                       power_setpoint) / self.max_cs_power[ev]
            else:
                action_list[ev] = 1

        if self.verbose:
            print(f'Evs to charge: {evs_to_charge}')
            print(f'Action list: {action_list}')
        return action_list