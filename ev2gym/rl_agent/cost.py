'''
This file contains the cost  and reward functions for the EV2Gym safety environment.
'''
import math



def transformer_overload_usrpenalty_cost(env,total_costs, user_satisfaction_list, *args):
    """
    Returns the cost of a transformer overload.
    """
    cost = 0
    for tr in env.transformers:
        cost += 100 * tr.get_how_overloaded()                        
    
    for score in user_satisfaction_list:        
        cost += 100 * math.exp(-10*score)        
    return cost



def ProfitMax_TrPenalty_UserIncentives_safety(env, total_costs, user_satisfaction_list, *args):
    
    reward = total_costs
        
    return reward