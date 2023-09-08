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
from torch.utils.tensorboard import SummaryWriter

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

# Parse given arguments
# gamma, tau, hidden_size, replay_size, batch_size, hidden_size are taken from the original paper
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="RoboschoolInvertedPendulumSwingup-v1",
                    help="the environment on which the agent should be trained "
                         "(Default: RoboschoolInvertedPendulumSwingup-v1)")
parser.add_argument("--render_train", default=False, type=bool,
                    help="Render the training steps (default: False)")
parser.add_argument("--render_eval", default=True, type=bool,
                    help="Render the evaluation steps (default: False)")
parser.add_argument("--load_model", default=False, type=bool,
                    help="Load a pretrained model (default: False)")
parser.add_argument("--save_dir", default="./saved_models/",
                    help="Dir. path to save and load a model (default: ./saved_models/)")
parser.add_argument("--seed", default=0, type=int,
                    help="Random seed (default: 0)")
parser.add_argument("--timesteps", default=1e6, type=int,
                    help="Num. of total timesteps of training (default: 1e6)")
parser.add_argument("--batch_size", default=128, type=int,
                    help="Batch size (default: 64; OpenAI: 128)")
parser.add_argument("--replay_size", default=1e5, type=int,
                    help="Size of the replay buffer (default: 1e6; OpenAI: 1e5)")
parser.add_argument("--gamma", default=0.99,
                    help="Discount factor (default: 0.99)")
parser.add_argument("--tau", default=0.001,
                    help="Update factor for the soft update of the target networks (default: 0.001)")
parser.add_argument("--noise_stddev", default=0.3, type=int,
                    help="Standard deviation of the OU-Noise (default: 0.2)")
parser.add_argument("--hidden_size", nargs=2, default=[64, 64], type=tuple,
                    help="Num. of units of the hidden layers (default: [400, 300]; OpenAI: [64, 64])")
parser.add_argument("--n_test_cycles", default=1, type=int,
                    help="Num. of episodes in the evaluation phases (default: 10; OpenAI: 20)")
args = parser.parse_args()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using {}".format(device))

if __name__ == "__main__":

    # Define the directory where to save and load models
    checkpoint_dir = args.save_dir + args.env
    # name the run accordign to time
    writer = SummaryWriter('runs/r_' + time.strftime("%Y%m%d-%H%M%S"))

    # Create the env
    kwargs = dict()
    # if args.env == 'RoboschoolInvertedPendulumSwingup-v1':
    #     # 'swingup=True' must be passed as an argument
    #     # See pull request 'https://github.com/openai/roboschool/pull/192'
    #     kwargs['swingup'] = True
    # elif args.env == 'evcity':

    # env = gym.make(args.env, **kwargs)
    # env = NormalizedActions(env)

    verbose = False
    n_transformers = 1
    number_of_charging_stations = 1
    steps = 150  # 288 steps = 1 day with 5 minutes per step
    timescale = 5  # (5 minutes per step)
    score_threshold = 0  # [0,1] 1 means fully charged, 0 means empty
    save_plots = True
    # replay_path = "replay/replay_ev_city_288_2023-09-08_09-33.pkl"
    replay_path = "replay/replay_ev_city_150_2023-09-08_11-44.pkl"

    args.env = 'evcity-v0'

    gym.register(id='evcity-v0', entry_point='gym_env.ev_city:EVCity')

    env = ev_city.EVCity(cs=number_of_charging_stations,
                         number_of_ports_per_cs=2,
                         number_of_transformers=n_transformers,
                         load_ev_from_replay=True,
                         load_prices_from_replay=True,
                         load_from_replay_path=replay_path,
                         empty_ports_at_end_of_simulation=True,
                         generate_rnd_game=True,
                         simulation_length=steps,
                         timescale=timescale,
                         score_threshold=score_threshold,
                         save_plots=False,
                         save_replay=False,
                         verbose=verbose,)

    # Define the reward threshold when the task is solved (if existing) for model saving
    reward_threshold = gym.spec(args.env).reward_threshold if gym.spec(
        args.env).reward_threshold is not None else np.inf

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

    # Start training
    logger.info('Train agent on {} env'.format({args.env}))
    logger.info('Doing {} timesteps'.format(args.timesteps))
    logger.info('Start at timestep {0} with t = {1}'.format(timestep, t))
    logger.info('Start training at {}'.format(
        time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

    while timestep <= args.timesteps:
        ou_noise.reset()
        epoch_return = 0

        print(f'Epoch: {epoch} timestep: {timestep} ')

        # print(env.reset().shape)
        # state = np.array(env.reset().reshape(-1))
        # state = torch.Tensor(state).to(device)
        # print(f'state shape: {state.shape}'')
        state = torch.Tensor([env.reset()]).to(device)
        # Normalize the state
        state = state / torch.norm(state)
        # print(f'state: {state}')

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
            next_state = next_state / torch.norm(next_state)

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
        writer.add_scalar('epoch/return', epoch_return, epoch)
        #  stats = {'total_ev_served': total_ev_served,
        #              'total_profits': total_profits,
        #              'toal_energy_charged': toal_energy_charged,
        #              'total_energy_discharged': total_energy_discharged,
        #              'average_user_satisfaction': average_user_satisfaction}
        writer.add_scalar('epoch/ev_served', stats['total_ev_served'], epoch)
        writer.add_scalar('epoch/profits', stats['total_profits'], epoch)
        writer.add_scalar('epoch/energy_charged',
                          stats['toal_energy_charged'], epoch)
        writer.add_scalar('epoch/energy_discharged',
                          stats['total_energy_discharged'], epoch)
        writer.add_scalar('epoch/user_satisfaction',
                          stats['average_user_satisfaction'], epoch)

        # Test every 10th episode (== 1e4) steps for a number of test_epochs epochs
        if timestep >= 5000 * t:
            t += 1
            test_rewards = []
            for _ in range(args.n_test_cycles):
                state = torch.Tensor([env.reset()]).to(device)
                test_reward = 0
                while True:
                    if args.render_eval:
                        # env.render()
                        env.save_plots = True

                    # Selection without noise
                    action = agent.calc_action(state)

                    next_state, reward, done, stats = env.step(
                        action.cpu().numpy()[0])
                    test_reward += reward

                    next_state = torch.Tensor([next_state]).to(device)

                    state = next_state
                    if done:
                        break
                test_rewards.append(test_reward)

            mean_test_rewards.append(np.mean(test_rewards))

            for name, param in agent.actor.named_parameters():
                writer.add_histogram(
                    name, param.clone().cpu().data.numpy(), epoch)
            for name, param in agent.critic.named_parameters():
                writer.add_histogram(
                    name, param.clone().cpu().data.numpy(), epoch)

            writer.add_scalar('test/mean_test_return',
                              mean_test_rewards[-1], epoch)
            writer.add_scalar('test/total_ev_served',
                              stats['total_ev_served'], epoch)
            writer.add_scalar('test/total_profits', stats['total_profits'], epoch)
            writer.add_scalar('test/toal_energy_charged', stats['toal_energy_charged'], epoch)
            writer.add_scalar('test/total_energy_discharged', stats['total_energy_discharged'], epoch)
            writer.add_scalar('test/average_user_satisfaction', stats['average_user_satisfaction'], epoch)

            logger.info("Epoch: {}, current timestep: {}, last reward: {}, "
                        "mean reward: {}, mean test reward {}".format(epoch,
                                                                      timestep,
                                                                      rewards[-1],
                                                                      np.mean(
                                                                          rewards[-10:]),
                                                                      np.mean(test_rewards)))

            # Save if the mean of the last three averaged rewards while testing
            # is greater than the specified reward threshold
            # TODO: Option if no reward threshold is given
            if np.mean(mean_test_rewards[-3:]) >= reward_threshold:
                agent.save_checkpoint(timestep, memory)
                time_last_checkpoint = time.time()
                logger.info('Saved model at {}'.format(time.strftime(
                    '%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

        epoch += 1

    agent.save_checkpoint(timestep, memory)
    logger.info('Saved model at endtime {}'.format(
        time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
    logger.info('Stopping training at {}'.format(
        time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
    env.close()
