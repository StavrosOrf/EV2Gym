import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="ev-city-v0",
                        help="the environment on which the agent should be trained ")
    parser.add_argument("--name",type=str, help="name of the experiment")
    parser.add_argument("--render_train", default=False, type=bool,
                        help="Render the training steps (default: False)")
    parser.add_argument("--render_eval", default=True, type=bool,
                        help="Render the evaluation steps (default: False)")
    parser.add_argument("--load_model", default=False, type=bool,
                        help="Load a pretrained model (default: False)")
    parser.add_argument("--save_dir", default="./saved_models/",
                        help="Dir. path to save and load a model (default: ./saved_models/)")
    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed (default: 0)")
    parser.add_argument("--timesteps", default=10*1e6, type=int,
                        help="Num. of total timesteps of training (default: 1e6)")
    parser.add_argument("--batch_size", default=512, type=int,#128
                        help="Batch size (default: 64; OpenAI: 128)")
    parser.add_argument("--replay_size", default=1e5, type=int,
                        help="Size of the replay buffer (default: 1e6; OpenAI: 1e5)")
    parser.add_argument("--gamma", default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--tau", default=0.001,
                        help="Update factor for the soft update of the target networks (default: 0.001)")
    parser.add_argument("--noise_stddev", default=0.3, type=int,
                        help="Standard deviation of the OU-Noise (default: 0.2)")
    parser.add_argument("--hidden_size", nargs=2, default=[256, 256], type=tuple,
                        help="Num. of units of the hidden layers (default: [400, 300]; OpenAI: [64, 64])")    
    parser.add_argument("--wandb", default=True, type=bool,
                        help="Enable logging to wandb (default: True)")

    # Environment specific arguments
    parser.add_argument("--config_file", default="ev2gym/example_config_files/PublicPST.yaml",
    # parser.add_argument("--config_file", default="ev2gym/example_config_files/V2G_MPC.yaml",
    # parser.add_argument("--config_file", default="ev2gym/example_config_files/V2GProfitPlusLoads.yaml",
                        help="Path to the config file (default: config_files/config.yaml)")
    parser.add_argument("--n_test_cycles", default=50, type=int,
                        help="Num. of episodes in the evaluation phases (default: 10; OpenAI: 20)")

    # Generate trajectories specific arguments
    parser.add_argument("--n_trajectories", default=200_000, type=int,
                        help="Num. of trajectories to generate (default: 10)")
    
    #
    parser.add_argument("--dataset", default="RR", type=str)
    parser.add_argument("--save_opt_trajectories", default=True, type=bool,
                        help="Save Optimal trajectories (default: False)")
    

    return parser.parse_args()
