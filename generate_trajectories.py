import pickle
import torch
import numpy as np

import numpy.random as rd
from torch.nn.modules import loss

from gym_env import ev_city

from tqdm import tqdm

generate_trajectories = False
MONTHS_LEN = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

trajectory_list = []
trajectories_number = 10000000
generate_optimal_trajectories = False

if generate_optimal_trajectories:
    file_name = 'optimal_trajectories_new.pkl'
else:
    file_name = 'random_trajectories_new.pkl'

args = Arguments()
args.agent = AgentDDPG()
agent_name = f'{args.agent.__class__.__name__}'
args.agent.cri_target = True
args.env = ESSEnv()

args.init_before_training(if_main=True)
'''init agent and environment'''
agent = args.agent
env = args.env
agent.init(
    args.net_dim, env.state_space.shape[0], env.action_space.shape[0], args.learning_rate, args.if_per_or_gae)
agent.state = env.reset()

for counter in tqdm(range(trajectories_number)):    
    with torch.no_grad():
        if generate_optimal_trajectories:

            month = np.random.randint(1, 13)  # here we choose 12 month
            day = np.random.randint(1, MONTHS_LEN[month-1]-1)

            initial_soc = round(np.random.uniform(0.2, 0.8), 2)
            # print(f'month:{month}, day:{day}, initial_soc:{initial_soc}')

            # print(initial_soc)
            base_result = optimization_base_result(
                env, month, day, initial_soc)

            # base_result = base_result.iloc[i]
            # extract actions
            actions = []
            for i in range(len(base_result)):
                action = [0, 0, 0, 0]

                if i == 0:
                    action[0] = (base_result.iloc[i]['soc'] -
                                 initial_soc) / (env.battery.max_charge / env.battery.capacity)
                    action[1] = base_result.iloc[i]['gen1'] / \
                        env.dg1.ramping_up
                    action[2] = base_result.iloc[i]['gen2'] / \
                        env.dg2.ramping_up
                    action[3] = base_result.iloc[i]['gen3'] / \
                        env.dg3.ramping_up
                else:
                    action[0] = (base_result.iloc[i]['soc'] -
                                 base_result.iloc[i-1]['soc']) / (env.battery.max_charge / env.battery.capacity)
                    action[1] = (base_result.iloc[i]['gen1'] -
                                 base_result.iloc[i-1]['gen1']) / env.dg1.ramping_up
                    action[2] = (base_result.iloc[i]['gen2'] -
                                 base_result.iloc[i-1]['gen2']) / env.dg2.ramping_up
                    action[3] = (base_result.iloc[i]['gen3'] -
                                 base_result.iloc[i-1]['gen3']) / env.dg3.ramping_up

                actions.append(action)

            if np.max(actions) > 1.1 or np.min(actions) < -1.1:
                print('action out of range!')
                print(np.max(actions))
                print(np.min(actions))
                exit(0)

            trajectory = agent.explore_env_opt_actions(
                env, args.target_step, actions, day, month, initial_soc)

        else:

            trajectory = agent.explore_env(env, args.target_step)

        trajectory_i = {"observations": [],
                        "actions": [], "rewards": [], "dones": []}

        for state_s in trajectory:
            trajectory_i["observations"].append(state_s[0])
            trajectory_i["actions"].append(state_s[1][2:6])

            reward_mode = 'normal'
            if reward_mode == 'return_to_go':
                trajectory_i["rewards"].append(
                    sum(trajectory_i["rewards"]) + state_s[1][0])
            else:
                trajectory_i["rewards"].append(state_s[1][0])
            trajectory_i["dones"].append(state_s[1][1])

        trajectory_i["observations"] = np.array(
            trajectory_i["observations"])
        trajectory_i["actions"] = np.array(trajectory_i["actions"])
        trajectory_i["rewards"] = np.array(trajectory_i["rewards"])
        trajectory_i["dones"] = np.array(trajectory_i["dones"])
        # print(trajectory_i)
        trajectory_list.append(trajectory_i)
        
        if counter % 1000000 == 0:
            print(f'counter:{counter}')
            f = open(file_name, 'wb')
            # source, destination
            pickle.dump(trajectory_list, f)
            f.close()


print("====================================")
print(trajectory_list[0])
print(len(trajectory_list))
print('Finished trajectory generating!')

f = open(file_name, 'wb')
# source, destination
pickle.dump(trajectory_list, f)
f.close()