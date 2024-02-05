# This file contains the reward function for the RL agent
# Users can create their own reward function here or in their own file using the same structure as below

def Squared_efficiency_and_satisfaction_balance_reward(env, *args):
    # Penalty for deviation from power setpoints
    # Reward for charging near to charging power potential
    error = min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) - \
        env.current_power_setpoints[env.current_step-1]
    
    charging_efficiency_reward = 0
    if env.charge_power_potential[env.current_step-1] > 0:
        efficiency_ratio = env.current_power_setpoints[env.current_step-1] / env.charge_power_potential[env.current_step-1]
        charging_efficiency_reward = min(efficiency_ratio, 1)
    
    reward = - error**2 + charging_efficiency_reward * 100 
    return reward

def Linear_efficiency_and_satisfaction_balance_reward(env, *args):
    # Penalty for deviation from power setpoints
    # Reward for charging near to charging power potential
    error = min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) - \
        env.current_power_setpoints[env.current_step-1]
    
    charging_efficiency_reward = 0
    if env.charge_power_potential[env.current_step-1] > 0:
        efficiency_ratio = env.current_power_setpoints[env.current_step-1] / env.charge_power_potential[env.current_step-1]
        charging_efficiency_reward = min(efficiency_ratio, 1)
    
    reward = - error + charging_efficiency_reward * 100
    return reward


def SquaredTrackingErrorRewardwithPSPpenalty(env,*args):
    # This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    # The reward is negative, and there is a penalty when the power setpoints are lower than the current power setpoints
    # There is also a penalty for not charging when there is charge power potential
    reward = 0
    penalty_charging = 0
    penalty_setpoint = 0
    error = min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) - \
        env.current_power_setpoints[env.current_step-1]
    
    if env.power_setpoints[env.current_step-1] < env.current_power_setpoints[env.current_step-1]:
        penalty_setpoint = (env.current_power_setpoints[env.current_step-1]-env.power_setpoints[env.current_step-1])
    else:
        penalty_setpoint = 0

    if env.charge_power_potential[env.current_step-1] != 0 and env.current_power_setpoints[env.current_step-1] == 0: 
        penalty_charging = (env.charge_power_potential[env.current_step-1]-env.current_power_setpoints[env.current_step-1])
    else:
        penalty_charging = 0
    
    reward = - error**2 - penalty_charging - penalty_setpoint
    return reward

def LinearTrackingErrorRewardwithPSPpenalty(env,*args):
    # This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    # The reward is negative, and there is a penalty when the power setpoints are lower than the current power setpoints
    # There is also a penalty for not charging when there is charge power potential
    reward = 0
    penalty_charging = 0
    penalty_setpoint = 0
    error = abs(min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) - \
        env.current_power_setpoints[env.current_step-1])
    
    if env.power_setpoints[env.current_step-1] < env.current_power_setpoints[env.current_step-1]:
        penalty_setpoint = - (env.current_power_setpoints[env.current_step-1]-env.power_setpoints[env.current_step-1])
    else:
        penalty_setpoint = 0

    if env.charge_power_potential[env.current_step-1] != 0 and env.current_power_setpoints[env.current_step-1] == 0: 
        penalty_charging = (env.charge_power_potential[env.current_step-1]-env.current_power_setpoints[env.current_step-1])
    else:
        penalty_charging = 0
    
    reward = - error - penalty_charging - penalty_setpoint
    return reward
    

def SquaredTrackingErrorwithEqualPenalty(env,*args):
    # This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    # Different than the SquaredTrackingErrorRewardwithPenalty,
    # this reward function penalizes equally when the charge power potential is higher than the current power setpoints

    error = min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) - \
        env.current_power_setpoints[env.current_step-1]
    
    if env.charge_power_potential[env.current_step-1] != 0 and env.current_power_setpoints[env.current_step-1] == 0: 
        penalty = env.charge_power_potential[env.current_step-1] - env.current_power_setpoints[env.current_step-1]
    else:
        penalty = 0
    
    reward = - error**2 - penalty
    return reward

def LinearTrackingErrorwithEqualPenalty(env,*args):
    # This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    # Different than the SquaredTrackingErrorRewardwithPenalty,
    # this reward function penalizes equally when the charge power potential is higher than the current power setpoints

    error = abs(min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) - \
        env.current_power_setpoints[env.current_step-1])
    
    if env.charge_power_potential[env.current_step-1] != 0 and env.current_power_setpoints[env.current_step-1] == 0: 
        penalty = env.charge_power_potential[env.current_step-1] - env.current_power_setpoints[env.current_step-1]
    else:
        penalty = 0
    
    reward = - error - penalty
    return reward

def SquaredTrackingErrorwithPenaltyandPriorityChargingReward(env,*args):
    # This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    # The reward is negative, there is an equal penalty for not charging when there is charge power potential
    # There is also a reward for priority charging
    # The reward should be changed!!!! Doesn't work

    error = min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) - \
        env.current_power_setpoints[env.current_step-1]
        
    priority_charging_reward = 0
    for cs in env.charging_stations:
        for EV in cs.evs_connected:
            if EV is not None: 
                if EV.get_soc() < 0.5:  # If SoC is below 50%, prioritize charging
                    priority_charging_reward += (0.5 - EV.get_soc())  # Increase reward for lower SoC
                    priority_charging_reward *= 100  # scaling factor
                else:
                    priority_charging_reward = 0
            else:
                priority_charging_reward = 0
            
    if env.charge_power_potential[env.current_step-1] != 0 and env.current_power_setpoints[env.current_step-1] == 0: 
        penalty = env.charge_power_potential[env.current_step-1] - env.current_power_setpoints[env.current_step-1]
    else:
        penalty = 0
    
    reward = - error**2 - penalty + priority_charging_reward
    return reward


def SquaredTrackingErrorReward(env,*args):
    # This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    # The reward is negative
    
    reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
        env.current_power_setpoints[env.current_step-1])**2
    
    return reward

def SquaredTrackingErrorRewardWithPenalty(env,*args):
    # This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    # The reward is negative
    # If the EV is not charging, the reward is penalized
    
    if env.current_power_setpoints[env.current_step-1] == 0 and env.charge_power_potential[env.current_step-2] != 0:
        reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
            env.current_power_setpoints[env.current_step-1])**2 - 100
    else:
        reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
            env.current_power_setpoints[env.current_step-1])**2
    
    return reward

def SimpleReward(env,*args):
    # This reward function does not consider the charge power potential
    
    reward = - (env.power_setpoints[env.current_step-1] - env.current_power_setpoints[env.current_step-1])**2
    
    return reward

def MinimizeTrackerSurplusWithChargeRewards(env,*args):
    # This reward function minimizes the tracker surplus and gives a reward for charging
    
    reward = 0
    if env.power_setpoints[env.current_step-1] < env.current_power_setpoints[env.current_step-1]:
            reward -= (env.current_power_setpoints[env.current_step-1]-env.power_setpoints[env.current_step-1])

    reward += env.current_power_setpoints[env.current_step-1]/75
    
    return reward




# Previous reward functions for testing
#############################################################################################################
        # reward = total_costs  # - 0.5
        # print(f'total_costs: {total_costs}')
        # print(f'user_satisfaction_list: {user_satisfaction_list}')
        # for score in user_satisfaction_list:
        #     reward -= 100 * (1 - score)

        # Punish invalid actions (actions that try to charge or discharge when there is no EV connected)
        # reward -= 2 * (invalid_action_punishment/self.number_of_ports)

        # reward = min(2, 1 * 4 * self.cs / (0.00001 + (
        #     self.power_setpoints[self.current_step-1] - self.current_power_setpoints[self.current_step-1])**2))

        # this is the new reward function
        # reward = min(2, 1/((min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #           self.current_power_setpoints[self.current_step-1])**2 + 0.000001))

        # new_*10*charging
        # if self.power_setpoints[self.current_step-1] < self.current_power_setpoints[self.current_step-1]:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_setpoints[self.current_step-1])
        # else:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_setpoints[self.current_step-1])*10

        # new_1_equal
        # if self.power_setpoints[self.current_step-1] < self.current_power_setpoints[self.current_step-1]:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_setpoints[self.current_step-1])
        # else:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_setpoints[self.current_step-1])

        # new_0.1
        # if self.power_setpoints[self.current_step-1] < self.current_power_setpoints[self.current_step-1]:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_setpoints[self.current_step-1])**2
        # else:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_setpoints[self.current_step-1])*0.1

        # new_reward squared
        # if self.power_setpoints[self.current_step-1] < self.current_power_setpoints[self.current_step-1]:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #           self.current_power_setpoints[self.current_step-1])**2
        # else:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #           self.current_power_setpoints[self.current_step-1])

        # for score in user_satisfaction_list:
        #     reward -= 100 * (1 - score)

        # for tr in self.transformers:
        #     if tr.current_amps > tr.max_current:
        #         reward -= 1000 * abs(tr.current_amps - tr.max_current)
        #     elif tr.current_amps < tr.min_current:
        #         reward -= 1000 * abs(tr.current_amps - tr.min_current)

        # reward -= 100 * (tr.current_amps < tr.min_amps)
        #######################################################################################################
        # squared tracking error
        # reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #            self.current_power_setpoints[self.current_step-1])**2

        # best reward so far
        ############################################################################################################
        # if self.power_setpoints[self.current_step-1] < self.current_power_setpoints[self.current_step-1]:
        #     reward -= (self.current_power_setpoints[self.current_step-1]-self.power_setpoints[self.current_step-1])

        # reward += self.current_power_setpoints[self.current_step-1]/75
        ############################################################################################################
        # normalize reward to -1 1
        # reward = reward/1000
        # reward = (100 +reward) / 1000
        # print(f'reward: {reward}')

        # reward -= 2 * (invalid_action_punishment/self.number_of_ports)
        # reward /= 100
        # reward = (100 +reward) / 1000
        # print(f'current_power_setpoints: {self.current_power_setpoints[self.current_step-1]}')

        # return reward