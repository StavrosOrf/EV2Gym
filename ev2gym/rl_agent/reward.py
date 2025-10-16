'''This file contains various example reward functions for the RL agent. Users can create their own reward function here or in their own file using the same structure as below
'''

import math
import numpy as np

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

def V2G_grid_full_reward(env, total_costs, user_satisfaction_list, *args):

    reward = total_costs
    
    verbose = False
    
    if verbose:
        print(f'!!! Costs: {total_costs}')
    
    user_costs = 0
    for ev in env.departing_evs:
        if verbose:
            print(f'!!! EV: {ev.current_capacity} | {ev.desired_capacity}')
        user_costs += -(ev.current_capacity - ev.desired_capacity)**2
    
    if verbose:
        print(f'!!! User Satisfaction Penalty: {user_costs}')

    current_step = env.current_step - 1
    v_m = env.node_voltage[:, current_step]

    loss_v = np.minimum(np.zeros_like(v_m), 0.05 - np.abs(1-v_m)).sum()
    return reward + 1000 * loss_v + user_costs


def V2G_grid_simple_reward(env, total_costs, user_satisfaction_list, *args):

    current_step = env.current_step - 1
    v_m = env.node_voltage[:, current_step]

    loss_v = np.minimum(np.zeros_like(v_m), 0.05 - np.abs(1-v_m)).sum()

    return 1000 * loss_v

def V2G_profitmax(env, total_costs, user_satisfaction_list, *args):

    reward = total_costs
    
    # verbose = False
    
    # if verbose:
    #     print(f'!!! Costs: {total_costs}')
    
    user_costs = 0
    for ev in env.departing_evs:
        # if verbose:
        #     print(f'!!! EV: {ev.current_capacity} | {ev.desired_capacity}')
        if ev.desired_capacity > ev.current_capacity:
            # user_costs += -(ev.current_capacity - ev.desired_capacity)**2
            user_costs += -100 * (ev.desired_capacity - ev.current_capacity)        
    
    # if verbose:
    #     print(f'!!! User Satisfaction Penalty: {user_costs}')

    # current_step = env.current_step - 1
    # v_m = env.node_voltage[:, current_step]

    # loss_v = np.minimum(np.zeros_like(v_m), 0.05 - np.abs(1-v_m)).sum()
    return (reward + user_costs)


def V2G_costs_simple(env, total_costs, user_satisfaction_list, *args):

    reward = total_costs
    return reward

def V2G_profitmaxV2(env, total_costs, user_satisfaction_list, *args):

    reward = total_costs
    
    verbose = False
    
    if verbose:
        print("\n=---- Reward Calculation V2G ProfitMax V2 ----=")
        print(f'!!! Costs: {total_costs}')
    
    user_costs = 0
    
    linear = False
    if linear:
        cost_multiplier = 0.1
    else:
        cost_multiplier = 0.05
    
    for cs in env.charging_stations:
        for ev in cs.evs_connected:
            if ev is not None:
                min_steps_to_full = (ev.desired_capacity - ev.current_capacity) / \
                    (ev.max_ac_charge_power/(60/env.timescale))
                
                
                departing_step = ev.time_of_departure - env.current_step
                
                cost = 0
                if min_steps_to_full > departing_step:                    
                    min_capacity_at_time = ev.desired_capacity - ((departing_step+1) * ev.max_ac_charge_power/(60/env.timescale))
                    
                    if linear:
                        cost = cost_multiplier*(min_capacity_at_time - ev.current_capacity)
                    else:
                        cost = cost_multiplier*(min_capacity_at_time - ev.current_capacity)**2
                        
                    user_costs += - cost
                
                if verbose:
                    if min_steps_to_full > departing_step:                    
                        print(f'-!EV: {min_capacity_at_time} | {ev.current_capacity} | {ev.desired_capacity} | {min_steps_to_full:.3f} | {departing_step} | cost {(cost):.3f}') 
                    else:
                        print(f'- EV: {ev.current_capacity} | {ev.desired_capacity} | {min_steps_to_full:.3f} | {departing_step} | cost {(cost):.3f}')
                
    for ev in env.departing_evs:
        if ev.desired_capacity > ev.current_capacity:            
            if verbose:
                print(f'!!! EV: {ev.current_capacity} | {ev.desired_capacity} | costs: {-cost_multiplier*(ev.desired_capacity - ev.current_capacity)**2}')
                
            if linear:
                user_costs += -cost_multiplier * (ev.desired_capacity - ev.current_capacity)
            else:
                user_costs += -cost_multiplier * (ev.desired_capacity - ev.current_capacity)**2
            
    if verbose:
        print(f'!!! User Satisfaction Penalty: {user_costs}')
        print("=-"*25)
    
    return (reward + user_costs)

def Grid_V2G_profitmaxV2(env, total_costs, user_satisfaction_list, *args):

    reward = total_costs
    
    verbose = False
    
    if verbose:
        print("\n=---- Reward Calculation V2G ProfitMax V2 ----=")
        print(f'!!! Costs: {total_costs}')
    
    user_costs = 0
    
    linear = False
    if linear:
        cost_multiplier = 0.1
    else:
        cost_multiplier = 0.05
    
    for cs in env.charging_stations:
        for ev in cs.evs_connected:
            if ev is not None:
                min_steps_to_full = (ev.desired_capacity - ev.current_capacity) / \
                    (ev.max_ac_charge_power/(60/env.timescale))
                
                
                departing_step = ev.time_of_departure - env.current_step
                
                cost = 0
                if min_steps_to_full > departing_step:                    
                    min_capacity_at_time = ev.desired_capacity - ((departing_step+1) * ev.max_ac_charge_power/(60/env.timescale))
                    
                    if linear:
                        cost = cost_multiplier*(min_capacity_at_time - ev.current_capacity)
                    else:
                        cost = cost_multiplier*(min_capacity_at_time - ev.current_capacity)**2
                        
                    user_costs += - cost
                
                if verbose:
                    if min_steps_to_full > departing_step:                    
                        print(f'-!EV: {min_capacity_at_time} | {ev.current_capacity} | {ev.desired_capacity} | {min_steps_to_full:.3f} | {departing_step} | cost {(cost):.3f}') 
                    else:
                        print(f'- EV: {ev.current_capacity} | {ev.desired_capacity} | {min_steps_to_full:.3f} | {departing_step} | cost {(cost):.3f}')
                
    for ev in env.departing_evs:
        if ev.desired_capacity > ev.current_capacity:            
            if verbose:
                print(f'!!! EV: {ev.current_capacity} | {ev.desired_capacity} | costs: {-cost_multiplier*(ev.desired_capacity - ev.current_capacity)**2}')
                
            if linear:
                user_costs += -cost_multiplier * (ev.desired_capacity - ev.current_capacity)
            else:
                user_costs += -cost_multiplier * (ev.desired_capacity - ev.current_capacity)**2
    
    current_step = env.current_step - 1
    v_m = env.node_voltage[:, current_step]
    loss_v = np.minimum(np.zeros_like(v_m), 0.05 - np.abs(1-v_m)).sum()
        
    if verbose:
        print(f'!!! User Satisfaction Penalty: {user_costs}')
        print(f'!!! loss_v: {loss_v}')
        print(f'!!! m*loss_v: {50_000*loss_v}')
        print("=-"*25)
        
    return reward + user_costs + 50_000 * loss_v

def pst_V2G_profitmaxV2(env, total_costs, user_satisfaction_list, *args):

    reward = total_costs
    
    verbose = False
    
    if verbose:
        print("\n=---- Reward Calculation V2G ProfitMax V2 ----=")
        print(f'!!! Costs: {total_costs}')
    
    user_costs = 0
    
    linear = False
    if linear:
        cost_multiplier = 0.1
    else:
        cost_multiplier = 0.05
    
    for cs in env.charging_stations:
        for ev in cs.evs_connected:
            if ev is not None:
                min_steps_to_full = (ev.desired_capacity - ev.current_capacity) / \
                    (ev.max_ac_charge_power/(60/env.timescale))
                
                
                departing_step = ev.time_of_departure - env.current_step
                
                cost = 0
                if min_steps_to_full > departing_step:                    
                    min_capacity_at_time = ev.desired_capacity - ((departing_step+1) * ev.max_ac_charge_power/(60/env.timescale))
                    
                    if linear:
                        cost = cost_multiplier*(min_capacity_at_time - ev.current_capacity)
                    else:
                        cost = cost_multiplier*(min_capacity_at_time - ev.current_capacity)**2
                        
                    user_costs += - cost
                
                if verbose:
                    if min_steps_to_full > departing_step:                    
                        print(f'-!EV: {min_capacity_at_time} | {ev.current_capacity} | {ev.desired_capacity} | {min_steps_to_full:.3f} | {departing_step} | cost {(cost):.3f}') 
                    else:
                        print(f'- EV: {ev.current_capacity} | {ev.desired_capacity} | {min_steps_to_full:.3f} | {departing_step} | cost {(cost):.3f}')
                
    for ev in env.departing_evs:
        if ev.desired_capacity > ev.current_capacity:            
            if verbose:
                print(f'!!! EV: {ev.current_capacity} | {ev.desired_capacity} | costs: {-cost_multiplier*(ev.desired_capacity - ev.current_capacity)**2}')
                
            if linear:
                user_costs += -cost_multiplier * (ev.desired_capacity - ev.current_capacity)
            else:
                user_costs += -cost_multiplier * (ev.desired_capacity - ev.current_capacity)**2
    
    pst_violation = 0
    if env.power_setpoints[env.current_step-1] < env.current_power_usage[env.current_step-1]:
        pst_violation += (env.power_setpoints[env.current_step-1] - env.current_power_usage[env.current_step-1])  
               
        
    return reward + user_costs + 1000 * pst_violation