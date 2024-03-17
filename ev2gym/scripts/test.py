import argparse
import logging
import os

import gym
import numpy as np
import roboschool
import torch

from ev2gym.baselines.DDPG.ddpg import DDPG
# from wrappers import NormalizedActions

# Create logger
logger = logging.getLogger('test')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Libdom raises an error if this is not set to true on Mac OSX
# see https://github.com/openai/spinningup/issues/16 for more information
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Parse given arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="RoboschoolInvertedPendulumSwingup-v1",
                    help="Env. on which the agent should be trained")
parser.add_argument("--render", default="True", help="Render the steps")
parser.add_argument("--seed", default=0, help="Random seed")
parser.add_argument("--save_dir", default="./saved_models/", help="Dir. path to load a model")
parser.add_argument("--episodes", default=100, help="Num. of test episodes")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Some parameters, which are not saved in the network files
gamma = 0.99  # discount factor for reward (default: 0.99)
tau = 0.001  # discount factor for model (default: 0.001)
hidden_size = (400, 300)  # size of the hidden layers (Deepmind: 400 and 300; OpenAI: 64)

if __name__ == "__main__":

    logger.info("Using device: {}".format(device))

    # Create the env
    kwargs = dict()
    if args.env == 'RoboschoolInvertedPendulumSwingup-v1':
        # 'swingup=True' must be passed as an argument
        # See pull request 'https://github.com/openai/roboschool/pull/192'
        kwargs['swingup'] = True
    env = gym.make(args.env, **kwargs)
    # env = NormalizedActions(env)

    # Setting rnd seed for reproducibility
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_dir = args.save_dir + args.env

    agent = DDPG(gamma,
                 tau,
                 hidden_size,
                 env.observation_space.shape[0],
                 env.action_space,
                 checkpoint_dir=checkpoint_dir
                 )

    agent.load_checkpoint()

    # Load the agents parameters
    agent.set_eval()

    for _ in range(args.episodes):
        step = 0
        returns = list()
        state = torch.Tensor([env.reset()]).to(device)
        episode_return = 0
        while True:
            if args.render:
                env.render()

            action = agent.calc_action(state, action_noise=None)
            q_value = agent.critic(state, action)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            episode_return += reward

            state = torch.Tensor([next_state]).to(device)

            step += 1

            if done:
                logger.info(episode_return)
                returns.append(episode_return)
                break

    mean = np.mean(returns)
    variance = np.var(returns)
    logger.info("Score (on 100 episodes): {} +/- {}".format(mean, variance))
