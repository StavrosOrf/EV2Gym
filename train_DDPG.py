  
import os
import time
import pkg_resources

import gym
import numpy as np
from EVsSimulator.ev_city import EVsSimulator

import torch
import wandb
import yaml

from EVsSimulator.utilities.arg_parser import arg_parser

from EVsSimulator.baselines.DDPG.ddpg import DDPG
from EVsSimulator.baselines.DDPG.noise import OrnsteinUhlenbeckActionNoise
from EVsSimulator.baselines.DDPG.replay_memory import ReplayMemory, Transition

from EVsSimulator.rl_agent.reward import SquaredTrackingErrorRewardWithPenalty, SquaredTrackingErrorReward, \
    Squared_efficiency_and_satisfaction_balance_reward, Linear_efficiency_and_satisfaction_balance_reward, \
    SquaredTrackingErrorRewardwithPSPpenalty, LinearTrackingErrorRewardwithPSPpenalty, SquaredTrackingErrorwithEqualPenalty, \
    LinearTrackingErrorwithEqualPenalty, SquaredTrackingErrorwithPenaltyandPriorityChargingReward
    
from EVsSimulator.rl_agent.state import BusinessPSTwithMoreKnowledge


# Libdom raises an error if this is not set to true on Mac OSX
# see https://github.com/openai/spinningup/issues/16 for more information
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    args = arg_parser()

    # Define the directory where to save and load models
    checkpoint_dir = args.save_dir + args.env
    # name the run accordign to time
    if args.name:
        run_name = args.name
    else:
        run_name = 'r_' + time.strftime("%Y%m%d-%H%M%S")

    run_name += "_DDPG"
    # Create the env
    # env = NormalizedActions(env)

    config = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)

    log_to_wandb = args.wandb
    verbose = False

    number_of_charging_stations = config["number_of_charging_stations"]
    n_transformers = config["number_of_transformers"]
    steps = config["simulation_length"]
    timescale = config["timescale"]

    n_test_cycles = args.n_test_cycles

    replay_path = None

    gym.register(id=args.env, entry_point='gym_env.ev_city:EVsSimulator')
    
    
    ###### Set the config file here ######
    config_file = "EVsSimulator/example_config_files/BusinessPST_config.yaml"
    
    ####### Set the reward function here #######
    #reward_function = SquaredTrackingErrorwithPenaltyandPriorityChargingReward

    reward_function_name = os.getenv("REWARD_FUNCTION", "LinearTrackingErrorwithEqualPenalty")

    # Map environment variable values to actual reward function classes
    reward_functions = {
    "SquaredTrackingErrorRewardWithPenalty": SquaredTrackingErrorRewardWithPenalty,
    "SquaredTrackingErrorReward": SquaredTrackingErrorReward,
    "Squared_efficiency_and_satisfaction_balance_reward": Squared_efficiency_and_satisfaction_balance_reward,
    "Linear_efficiency_and_satisfaction_balance_reward": Linear_efficiency_and_satisfaction_balance_reward,
    "SquaredTrackingErrorRewardwithPSPpenalty": SquaredTrackingErrorRewardwithPSPpenalty,
    "LinearTrackingErrorRewardwithPSPpenalty": LinearTrackingErrorRewardwithPSPpenalty,
    "SquaredTrackingErrorwithEqualPenalty": SquaredTrackingErrorwithEqualPenalty,
    "LinearTrackingErrorwithEqualPenalty": LinearTrackingErrorwithEqualPenalty,
    "SquaredTrackingErrorwithPenaltyandPriorityChargingReward": SquaredTrackingErrorwithPenaltyandPriorityChargingReward
}

    # Select the reward function based on the environment variable
    reward_function = reward_functions.get(reward_function_name, LinearTrackingErrorwithEqualPenalty)
    
    ####### Set the State function here #######
    state_function = BusinessPSTwithMoreKnowledge

    env = EVsSimulator(config_file=config_file,
                         generate_rnd_game=True,
                         save_plots=False,
                         save_replay=False,
                         reward_function=reward_function,
                         state_function=state_function,
                         )

    # Set random seed for all used libraries where possible
    # seed = np.random.randint(0, 1000000)
    # env.seed(seed)
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

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

    # Check if replay folder exists
    eval_replay_path = "./replay/" + \
        f'{number_of_charging_stations}cs_{n_transformers}tr/'
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
        print(
            f'Number of test cycles set to {n_test_cycles} due to the number of replay files found')

    if log_to_wandb:
        wandb.init(
            name=run_name,
            group=f'DDPG_{number_of_charging_stations}cs_{n_transformers}tr',
            project='EVsSimulator',
            config={"batch_size": args.batch_size,
                    "replay_size": args.replay_size,
                    "gamma": args.gamma,
                    "tau": args.tau,
                    "noise_stddev": args.noise_stddev,
                    "hidden_size": args.hidden_size,
                    "n_test_cycles": args.n_test_cycles,
                    "seed": args.seed,
                    "timescale": timescale,
                    "steps": steps,
                    "number_of_charging_stations": number_of_charging_stations,
                    "replay_path": replay_path,
                    "n_transformers": n_transformers,
                    "config_file": args.config_file,
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

    highest_opt_ratio = np.inf
    best_trackking_error = np.inf

    # Main training loop
    while timestep <= args.timesteps:
        ou_noise.reset()
        epoch_return = 0

        print(f'Epoch: {epoch} | Timestep: {timestep}')
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
            wandb.log({'train/return': epoch_return,
                       'train/ev_served': stats['total_ev_served'],
                       'train/energy_charged': stats['total_energy_charged'],
                       'train/energy_discharged': stats['total_energy_discharged'],
                       'train/user_satisfaction': stats['average_user_satisfaction'],
                       'train/tracking_error': stats['tracking_error'],
                       'train/power_tracker_violation': stats['power_tracker_violation'],
                       'train/energy_user_satisfaction': stats['energy_user_satisfaction']/100,
                       'train/transformer_overload': stats['total_transformer_overload'],
                       'tran/total_profits': stats['total_profits'],
                       'train/value_loss': epoch_value_loss,
                       'train/policy_loss': epoch_policy_loss})

        # Test every 10th episode (== 1e4) steps for a number of test_epochs epochs
        if timestep >= 10000 * t:
            print(f'Testing at timestep {timestep}')
            t += 1
            test_rewards = []
            test_stats = []
            for test_cycle in range(n_test_cycles):

                # load evaluation enviroments from replay files

                if test_cycle == 0:
                    save_plots = True
                else:
                    save_plots = False
                eval_env = EVsSimulator(config_file=config_file,
                                          load_from_replay_path=eval_replay_path +
                                          eval_replay_files[test_cycle],
                                          save_replay=False,
                                          generate_rnd_game=True,
                                          save_plots=save_plots,
                                          extra_sim_name=run_name,
                                          reward_function=reward_function,
                                          state_function=state_function)

                state = torch.Tensor([eval_env.reset()]).to(device)
                test_reward = 0
                while True:

                    # Selection without noise
                    action = agent.calc_action(state)

                    next_state, reward, done, stats = eval_env.step(
                        action.cpu().numpy()[0])
                    test_reward += reward

                    # if test_cycle == 0:
                    #     print('Action', action.detach().cpu().numpy(), reward)
                    #     print('State', next_state)

                    next_state = torch.Tensor([next_state]).to(device)
                    state = next_state

                    if done:
                        test_stats.append(stats)
                        # print(stats)
                        break
                test_rewards.append(test_reward)

            mean_test_rewards.append(np.mean(test_rewards))

            # average the results of the test cycles
            stats = {}
            for key in test_stats[0].keys():
                stats[key] = np.mean([test_stats[i][key]
                                     for i in range(len(test_stats))])
            
                       
            if stats['tracking_error'] < best_trackking_error:
                best_trackking_error = stats['tracking_error']
                agent.save_checkpoint(timestep, memory, run_name+"_best")
                time_last_checkpoint = time.time()

            if log_to_wandb:
                wandb.log({'test/mean_test_return': mean_test_rewards[-1],
                           'test/total_ev_served': stats['total_ev_served'],
                              'test/total_profits': stats['total_profits'],
                           'test/total_energy_charged': stats['total_energy_charged'],
                           'test/total_energy_discharged': stats['total_energy_discharged'],
                           'test/average_user_satisfaction': stats['average_user_satisfaction'],
                           #    'test/highest_opt_ratio': highest_opt_ratio,
                           'test/opt_tracking_error': stats['opt_tracking_error'],
                           'test/opt_energy_user_satisfaction': stats['opt_energy_user_satisfaction']/100,
                           'test/opt_power_tracker_violation': stats['opt_power_tracker_violation'],
                           'test/tracking_error': stats['tracking_error'],
                           'test/power_tracker_violation': stats['power_tracker_violation'],
                           'test/energy_user_satisfaction': stats['energy_user_satisfaction']/100,
                           'test/transformer_overload': stats['total_transformer_overload'],
                           #    'test/std_opt_ratio': np.std(opt_profits),
                           })

            print(f'Testing at timestep {timestep}, Mean test return: {mean_test_rewards[-1]}')
        epoch += 1

    agent.save_checkpoint(timestep, memory, run_name+"_last")

    env.close()
