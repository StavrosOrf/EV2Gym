try:
    from gymnasium.envs.registration import register
except (ModuleNotFoundError, ImportError):
    try:
        from gym.envs.registration import register
    except (ModuleNotFoundError, ImportError):
        raise ImportError("Neither 'gymnasium' nor 'gym' is installed; please install one to register environments.")

register(
    id='EV2Gym-v1',
    entry_point='ev2gym.models.ev2gym_env:EV2Gym',
    kwargs={'config_file': 'ev2gym/example_config_files/PublicPST.yaml'}
)