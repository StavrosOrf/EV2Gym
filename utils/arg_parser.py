import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="ev-city-v0",
                        help="the environment on which the agent should be trained ")
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
    parser.add_argument("--timesteps", default=5*1e6, type=int,
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
    parser.add_argument("--n_test_cycles", default=10, type=int,
                        help="Num. of episodes in the evaluation phases (default: 10; OpenAI: 20)")
    parser.add_argument("--wandb", default=True, type=bool,
                        help="Enable logging to wandb (default: True)")

    # Envirioned specific arguments
    parser.add_argument("--cs", default=1, type=int,
                        help="Num. of CS (default: 1)")
    parser.add_argument("--transformers", default=1, type=int,
                        help="Num. of Transformers (default: 1)")
    parser.add_argument("--ports", default=2, type=int,
                        help="Num. of Ports per CS (default: 2)")
    parser.add_argument("--steps", default=150, type=int,
                        help="Num. of steps (default: 150)")
    parser.add_argument("--timescale", default=5, type=int,
                        help="Timescale (default: 5)")
    parser.add_argument("--score_threshold", default=1, type=int,
                        help="Score threshold (default: 1)")
    parser.add_argument("--static_prices", default=True, type=bool,
                        help="Static prices (default: True)")
    parser.add_argument("--static_ev_spawn_rate", default=True, type=bool,
                        help="Static ev spawn rate (default: True)")

    return parser.parse_args()