'''This file contains various example reward functions for the RL agent. Users can create their own reward function here or in their own file using the same structure as below
'''

import math

def SquaredTrackingErrorReward(env,*args):
    '''This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    The reward is negative'''
    
    reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
        env.current_power_usage[env.current_step-1])**2
        
    return reward

def SqTrError_TrPenalty_UserIncentives(env, _, user_satisfaction_list, *args):
    ''' This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    It penalizes transofrmers that are overloaded    
    The reward is negative'''
    
    tr_max_limit = env.transformers[0].max_power[env.current_step-1]
    
    reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1],tr_max_limit) -
        env.current_power_usage[env.current_step-1])**2
            
    for tr in env.transformers:
        reward -= 100 * tr.get_how_overloaded()
        
    for score in user_satisfaction_list:
        reward -= 1000 * (1 - score)
                    
    return reward

def ProfitMax_TrPenalty_UserIncentives(env, total_costs, user_satisfaction_list, *args):
    
    reward = total_costs
    
    for tr in env.transformers:
        reward -= 100 * tr.get_how_overloaded()                        
    
    for score in user_satisfaction_list:        
        reward -= 100 * math.exp(-10*score)
        
    return reward

def SquaredTrackingErrorRewardWithPenalty(env,*args):
    ''' This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    The reward is negative
    If the EV is not charging, the reward is penalized
    '''
    if env.current_power_usage[env.current_step-1] == 0 and env.charge_power_potential[env.current_step-2] != 0:
        reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
            env.current_power_usage[env.current_step-1])**2 - 100
    else:
        reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
            env.current_power_usage[env.current_step-1])**2
    
    return reward

def SimpleReward(env,*args):
    '''This reward function does not consider the charge power potential'''
    
    reward = - (env.power_setpoints[env.current_step-1] - env.current_power_usage[env.current_step-1])**2
    
    return reward

def MinimizeTrackerSurplusWithChargeRewards(env,*args):
    ''' This reward function minimizes the tracker surplus and gives a reward for charging '''
    
    reward = 0
    if env.power_setpoints[env.current_step-1] < env.current_power_usage[env.current_step-1]:
            reward -= (env.current_power_usage[env.current_step-1]-env.power_setpoints[env.current_step-1])**2

    reward += env.current_power_usage[env.current_step-1] #/75
    
    return reward

def profit_maximization(env, total_costs, user_satisfaction_list, *args):
    ''' This reward function is used for the profit maximization case '''
    
    reward = total_costs
    
    for score in user_satisfaction_list:
        # reward -= 100 * (1 - score)
        reward -= 100 * math.exp(-10*score)
    
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
        #     self.power_setpoints[self.current_step-1] - self.current_power_usage[self.current_step-1])**2))

        # this is the new reward function
        # reward = min(2, 1/((min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #           self.current_power_usage[self.current_step-1])**2 + 0.000001))

        # new_*10*charging
        # if self.power_setpoints[self.current_step-1] < self.current_power_usage[self.current_step-1]:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])
        # else:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])*10

        # new_1_equal
        # if self.power_setpoints[self.current_step-1] < self.current_power_usage[self.current_step-1]:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])
        # else:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])

        # new_0.1
        # if self.power_setpoints[self.current_step-1] < self.current_power_usage[self.current_step-1]:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])**2
        # else:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #             self.current_power_usage[self.current_step-1])*0.1

        # new_reward squared
        # if self.power_setpoints[self.current_step-1] < self.current_power_usage[self.current_step-1]:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #           self.current_power_usage[self.current_step-1])**2
        # else:
        #     reward -= (min(self.power_setpoints[self.current_step-1], self.charge_power_potential[self.current_step-1]) -
        #           self.current_power_usage[self.current_step-1])

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
        #            self.current_power_usage[self.current_step-1])**2

        # best reward so far
        ############################################################################################################
        # if self.power_setpoints[self.current_step-1] < self.current_power_usage[self.current_step-1]:
        #     reward -= (self.current_power_usage[self.current_step-1]-self.power_setpoints[self.current_step-1])

        # reward += self.current_power_usage[self.current_step-1]/75
        ############################################################################################################
        # normalize reward to -1 1
        # reward = reward/1000
        # reward = (100 +reward) / 1000
        # print(f'reward: {reward}')

        # reward -= 2 * (invalid_action_punishment/self.number_of_ports)
        # reward /= 100
        # reward = (100 +reward) / 1000
        # print(f'current_power_usage: {self.current_power_usage[self.current_step-1]}')

        # return reward