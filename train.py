import argparse
import logging
import os
import random
import time

import gym
import numpy as np
# import roboschool
from gym_env import ev_city

import torch
import wandb

from utils.arg_parser import arg_parser

from ddpg import DDPG
from utils.noise import OrnsteinUhlenbeckActionNoise
from utils.replay_memory import ReplayMemory, Transition
from wrappers.normalized_actions import NormalizedActions

# Create logger
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

# Libdom raises an error if this is not set to true on Mac OSX
# see https://github.com/openai/spinningup/issues/16 for more information
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using {}".format(device))

if __name__ == "__main__":

    args = arg_parser()

    # Define the directory where to save and load models
    checkpoint_dir = args.save_dir + args.env
    # name the run accordign to time
    if args.name:
        run_name = args.name
    else:
        run_name = 'r_' + time.strftime("%Y%m%d-%H%M%S")    

    # Create the env
    # env = NormalizedActions(env)
    #     
    log_to_wandb = args.wandb
    verbose = False
    n_transformers = args.transformers
    number_of_charging_stations = args.cs
    steps = args.steps  # 288 steps = 1 day with 5 minutes per step
    timescale = args.timescale  # (5 minutes per step)
    score_threshold = args.score_threshold  # 1
    static_prices = args.static_prices
    static_ev_spawn_rate = args.static_ev_spawn_rate
    n_test_cycles = args.n_test_cycles

    replay_path = "replay/replay_ev_city_150_2023-09-08_11-44.pkl"
    replay_path = None

    args.env = 'evcity-v0'

    gym.register(id='evcity-v0', entry_point='gym_env.ev_city:EVCity')
    
    env = ev_city.EVCity(cs=number_of_charging_stations,
                         number_of_ports_per_cs=2,
                         number_of_transformers=n_transformers,
                         static_ev_spawn_rate=True,
                         load_ev_from_replay=False,
                         load_prices_from_replay=False,
                         static_prices=static_prices,
                         load_from_replay_path=replay_path,
                         empty_ports_at_end_of_simulation=True,
                         generate_rnd_game=True,
                         simulation_length=steps,
                         timescale=timescale,
                         score_threshold=score_threshold,
                         save_plots=False,
                         save_replay=False,
                         verbose=verbose,)

    # Set random seed for all used libraries where possible
    # env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Define and build DDPG agent
    hidden_size = tuple(args.hidden_size)
    agent = DDPG(args.gamma,
                 args.tau,
                 hidden_size,
                 env.observation_space.shape[0],
                 env.action_space,
                 checkpoint_dir=checkpoint_dir
                 )

    if static_prices:
        prices = "static"
    else:
        prices = "dynamic"

    if static_ev_spawn_rate:
        ev_spawn_rate = "static"
    else:
        ev_spawn_rate = "dynamic"

    # Check if replay folder exists
    eval_replay_path = "./replay/" + \
        f'{number_of_charging_stations}cs_{n_transformers}tr_{prices}_prices/'
    assert os.path.exists(
        eval_replay_path), "Evaluation Replay folder does not exist"
    # count number of files in replay folder
    eval_replay_files = [f for f in os.listdir(
        eval_replay_path) if os.path.isfile(os.path.join(eval_replay_path, f))]
    assert len(
        eval_replay_files) > 0, "No replay files found in evaluation replay folder"

    print(f'Found {len(eval_replay_files)} replay files in {eval_replay_path}')
    if n_test_cycles > len(eval_replay_files):
        n_test_cycles = len(eval_replay_files)
        print(f'Number of test cycles set to {n_test_cycles} due to the number of replay files found')

    if log_to_wandb:
        wandb.init(
            name=run_name,
            group=f'{number_of_charging_stations}cs_{n_transformers}tr_{prices}_prices_{ev_spawn_rate}_ev_spawn_rate',
            project='EVsSimulator',
            config={"batch_size": args.batch_size,
                    "replay_size": args.replay_size,
                    "gamma": args.gamma,
                    "tau": args.tau,
                    "noise_stddev": args.noise_stddev,
                    "hidden_size": args.hidden_size,
                    "n_test_cycles": args.n_test_cycles,
                    "seed": args.seed,
                    "score_threshold": score_threshold,
                    "timescale": timescale,
                    "steps": steps,
                    "number_of_charging_stations": number_of_charging_stations,
                    "replay_path": replay_path,
                    "n_transformers": n_transformers,
                    "static_prices": static_prices,
                    "static_ev_spawn_rate": static_ev_spawn_rate
                    }
        )
        wandb.watch(agent.actor)
        wandb.watch(agent.critic)

    # Initialize replay memory
    memory = ReplayMemory(int(args.replay_size))

    # Initialize OU-Noise
    nb_actions = env.action_space.shape[-1]
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                            sigma=float(args.noise_stddev) * np.ones(nb_actions))

    # Define counters and other variables
    start_step = 0
    # timestep = start_step
    if args.load_model:
        # Load agent if necessary
        start_step, memory = agent.load_checkpoint()
    timestep = start_step // 10000 + 1
    rewards, policy_losses, value_losses, mean_test_rewards = [], [], [], []
    epoch = 0
    t = 0
    time_last_checkpoint = time.time()

    highest_profits = -np.inf

    # Start training
    logger.info('Train agent on {} env'.format({args.env}))
    logger.info('Doing {} timesteps'.format(args.timesteps))
    logger.info('Start at timestep {0} with t = {1}'.format(timestep, t))
    logger.info('Start training at {}'.format(
        time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

    while timestep <= args.timesteps:
        ou_noise.reset()
        epoch_return = 0

        print(f'Epoch: {epoch} timestep: {timestep}')
        state = torch.Tensor([env.reset()]).to(device)

        while True:
            env.save_plots = False
            if args.render_train:
                env.render()
            action = agent.calc_action(state, ou_noise)
            # print(f'action: {action}')
            next_state, reward, done, stats = env.step(action.cpu().numpy()[0])
        
            timestep += 1
            epoch_return += reward

            mask = torch.Tensor([done]).to(device)
            reward = torch.Tensor([reward]).to(device)
            next_state = torch.Tensor([next_state]).to(device)
            # print(f'State: {next_state}')
            # next_state = next_state / torch.norm(next_state)

            memory.push(state, action, mask, next_state, reward)

            state = next_state

            epoch_value_loss = 0
            epoch_policy_loss = 0

            if len(memory) > args.batch_size:
                transitions = memory.sample(args.batch_size)
                # Transpose the batch
                # (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
                batch = Transition(*zip(*transitions))

                # Update actor and critic according to the batch
                value_loss, policy_loss = agent.update_params(batch)

                epoch_value_loss += value_loss
                epoch_policy_loss += policy_loss

            if done:
                break

        rewards.append(epoch_return)
        value_losses.append(epoch_value_loss)
        policy_losses.append(epoch_policy_loss)

        if log_to_wandb:
            wandb.log({'epoch/return': epoch_return,
                       'epoch/ev_served': stats['total_ev_served'],
                       'epoch/profits': stats['total_profits'],
                       'epoch/energy_charged': stats['toal_energy_charged'],
                       'epoch/energy_discharged': stats['total_energy_discharged'],
                       'epoch/user_satisfaction': stats['average_user_satisfaction'],
                       'epoch/value_loss': epoch_value_loss,
                       'epoch/policy_loss': epoch_policy_loss,
                       'epoch/ev_spawn_rt': stats['ev_spawn_rate']})

        # Test every 10th episode (== 1e4) steps for a number of test_epochs epochs
        if timestep >= 5000 * t:
            t += 1
            test_rewards = []
            test_stats = []
            for test_cycle in range(n_test_cycles):

                # load evaluation enviroments from replay files               
                
                if test_cycle == 0:                        
                    save_plots = True
                else:
                    save_plots = False
                eval_env = ev_city.EVCity(load_from_replay_path= eval_replay_path + 
                                          eval_replay_files[test_cycle],
                                          load_ev_from_replay=True,
                                          load_prices_from_replay=True,
                                          save_replay=False,
                                          save_plots=save_plots,
                                          simulation_length=steps,)

                state = torch.Tensor([eval_env.reset()]).to(device)
                test_reward = 0
                while True:                    

                    # Selection without noise
                    action = agent.calc_action(state)

                    next_state, reward, done, stats = eval_env.step(
                        action.cpu().numpy()[0])
                    test_reward += reward

                    if test_cycle == 0:
                        print('Action', action.detach().cpu().numpy(), reward)
                        print('State', next_state)

                    next_state = torch.Tensor([next_state]).to(device)
                    state = next_state

                    if done:
                        test_stats.append(stats)
                        break
                test_rewards.append(test_reward)

            mean_test_rewards.append(np.mean(test_rewards))

            # average the results of the test cycles
            stats = {}
            for key in test_stats[0].keys():
                stats[key] = np.mean([test_stats[i][key]
                                     for i in range(len(test_stats))])
                
            # get all values of a key in a list
            opt_profits = [1 - ((test_stats[i]['opt_profits'] - test_stats[i]['total_profits']) / 
                                abs(test_stats[i]['opt_profits'])) \
                                 for i in range(len(test_stats))]                        

            print(opt_profits)
            for ind in range(args.n_test_cycles):
                if test_stats[ind]['total_profits'] > highest_profits and test_stats[ind]['average_user_satisfaction'] == 1:
                    highest_profits = test_stats[ind]['total_profits']
                    agent.save_checkpoint(timestep, memory, run_name+"_best")
                    time_last_checkpoint = time.time()
                    logger.info('Saved model at {}'.format(time.strftime(
                        '%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

            if log_to_wandb:
                wandb.log({'test/mean_test_return': mean_test_rewards[-1],
                           'test/total_ev_served': stats['total_ev_served'],
                           'test/total_profits': stats['total_profits'],
                           'test/toal_energy_charged': stats['toal_energy_charged'],
                           'test/total_energy_discharged': stats['total_energy_discharged'],
                           'test/average_user_satisfaction': stats['average_user_satisfaction'],
                           'test/higher_profits': highest_profits,
                           'test/mean_opt_ratio': np.mean(opt_profits),
                           'test/std_opt_ratio': np.std(opt_profits),})
                            

            logger.info("Epoch: {}, current timestep: {}, last reward: {}, "
                        "mean reward: {}, mean test reward {}".format(epoch,
                                                                      timestep,
                                                                      rewards[-1],
                                                                      np.mean(
                                                                          rewards[-10:]),
                                                                      np.mean(test_rewards)))
        epoch += 1

    agent.save_checkpoint(timestep, memory, run_name+"_last")

    logger.info('Saved model at endtime {}'.format(
        time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
    logger.info('Stopping training at {}'.format(
        time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
    env.close()
