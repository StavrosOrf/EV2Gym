import numpy as np
import torch
from evsim import ev_city
import os
from icecream import ic

def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
        exp_prefix,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        n_test_episodes=10,
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    test_rewards = []
    test_stats = []
    highest_opt_ratio = np.NINF

    number_of_charging_stations = 1
    n_transformers = 1
    prices = "static"

    eval_replay_path = "./replay/" + \
        f'{number_of_charging_stations}cs_{n_transformers}tr_{prices}_prices/'
    eval_replay_files = [f for f in os.listdir(
        eval_replay_path) if os.path.isfile(os.path.join(eval_replay_path, f))]
    
    global_target_return = target_return

    for test_cycle in range(n_test_episodes):

        if test_cycle == 0:                        
            save_plots = True
        else:
            save_plots = False

        env = ev_city.EVCity(load_from_replay_path= eval_replay_path + 
                                    eval_replay_files[test_cycle],
                                    load_ev_from_replay=True,
                                    load_prices_from_replay=True,
                                    save_replay=False,
                                    save_plots=save_plots,
                                    simulation_length=max_ep_len,
                                    extra_sim_name=exp_prefix,)        

        state = env.reset()
        if mode == 'noise':
            state = state + np.random.normal(0, 0.1, size=state.shape)

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)        

        ep_return = global_target_return        
        target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        sim_states = []

        episode_return, episode_length = 0, 0
        for t in range(max_ep_len):

            # add padding
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()

            state, reward, done, stats = env.step(action)

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            if mode != 'delayed':
                pred_return = target_return[0,-1] - (reward/scale)
            else:
                pred_return = target_return[0,-1]
            target_return = torch.cat(
                [target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

            episode_return += reward
            episode_length += 1

            if done:
                test_stats.append(stats)
                test_rewards.append(episode_return)
                break
        
    stats = {}
    for key in test_stats[0].keys():
        stats[key] = np.mean([test_stats[i][key]
                                for i in range(len(test_stats))])
        
    # get all values of a key in a list
    opt_profits = [1 - ((test_stats[i]['opt_profits'] - test_stats[i]['total_profits']) / 
                        abs(test_stats[i]['opt_profits'])) \
                            for i in range(len(test_stats))]                        

    #drop key 'opt_profits' from dict stats
    stats.pop('opt_profits')    
    stats.pop('ev_spawn_rate')    

    ic(opt_profits)
    for ind in range(n_test_episodes):
        if np.mean(opt_profits) > highest_opt_ratio and test_stats[ind]['average_user_satisfaction'] == 1:
            highest_opt_ratio = np.mean(opt_profits)
            # agent.save_checkpoint(timestep, memory, run_name+"_best")
            # time_last_checkpoint = time.time()
            # logger.info('Saved model at {}'.format(time.strftime(
            #     '%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

    stats['mean_opt_ratio'] = np.mean(opt_profits)
    stats['std_opt_ratio'] = np.std(opt_profits)
    stats['highest_opt_ratio'] = highest_opt_ratio
    stats['mean_test_return'] = np.mean(test_rewards)

    return stats #, episode_length