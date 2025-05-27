import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.spaces import MultiDiscrete, Discrete
from gymnasium.core import WrapperObsType, ActType, ObsType

from ev2gym.rl_agent.state import PublicPST

from copy import deepcopy
import numpy as np


class FailedActionCommunication(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """
    This wrapper is used to simulate the failure of the communication channel between the agent and the charging stations.
    If the communication fails, the agent will not be able to send any action to the charging stations, therefore the action will be equal to the previous one.
    """

    def __init__(self, env: gym.Env, p_fail: float = 0.1):
        """
        Args:
            env: The environment to apply the wrapper
        """
        assert isinstance(env.action_space, Box)
        assert p_fail >= 0 and p_fail <= 1, "p_fail must be between 0 and 1"

        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)
        
        # generate random floats sized as the action space * timesteps
        self.random = np.random.rand(env.unwrapped.action_space.shape[0], env.unwrapped.simulation_length)
        self.p_fail = p_fail
        self.previous_actions_list = np.zeros(env.unwrapped.action_space.shape[0])
        self.env = env
        
        self.verbose = False

    def action(self, action: np.ndarray) -> np.ndarray:
        step = self.env.unwrapped.current_step
                
            
        
        new_action = np.where(self.random[:,step] < self.p_fail,
                              self.previous_actions_list,
                              action)        
        
        
        if self.verbose:
            print(f'----------------Noisy Communication--------------------')
            print(f"Step: {step}")
            print(f"Action: {action}")
            print(f"Previous actions: {self.previous_actions_list}")            
            print(f"New action: {new_action}")
            
            if not np.array_equal(action, new_action):
                input(f"Communication failed")
                
            
        self.previous_actions_list = new_action.copy()      
        
        return new_action
        
class DelayedObservation(
    gym.ObservationWrapper[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """
    This wrapper is used to simulate the delay in the observation of the environment.
    If the observation is delayed, the agent will receive the observation from the previous timestep.    
    """

    def __init__(self, env: gym.Env[ObsType, ActType], p_delay: float = 0.1):
        
        gym.utils.RecordConstructorArgs.__init__(self, p_delay=p_delay)
        gym.ObservationWrapper.__init__(self, env)

        assert p_delay >= 0 and p_delay <= 1, "p_fail must be between 0 and 1"
        assert env.unwrapped.observation_space.shape is not None                
        
        assert env.unwrapped.state_function is PublicPST, \
            f"The state function must be PublicPST for this wrapper to work. It was found to be: {env.unwrapped.state_function}"        
        
        if env.unwrapped.state_function is PublicPST:
            self.GNN_state = False
        else:
            self.GNN_state = True
                
        self.num_of_actions = env.unwrapped.action_space.shape[0]
        self.simulation_length = env.unwrapped.simulation_length
        
        self.random = np.random.rand(self.num_of_actions, self.simulation_length)
        
        self.timescale = env.unwrapped.timescale
        
        # self.observation_space = gym.spaces.Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=env.unwrapped.observation_space.shape,
        #     dtype=np.float32,
        # )
        
        if self.GNN_state:
            self.previous_obs_list = None
            self.actual_previous_obs_list = None
        else:            
            self.previous_obs_list = np.zeros(env.unwrapped.observation_space.shape[0])            
            self.actual_previous_obs_list = np.zeros(env.unwrapped.observation_space.shape[0])
            
        self.verbose = False
        self.p_delay = p_delay

        

    def observation(self, observation: ObsType) -> WrapperObsType:
        """Normalises the observation using the running mean and variance of the observations."""

        step = self.env.unwrapped.current_step
        
        not_communicated_energy_usage = 0                  
        
        if self.verbose:
            print(f'----------------Delayed Observation--------------------')
            print(f"Step: {step}")          
            
            if self.GNN_state:
                if self.previous_obs_list is not None:
                    print(f"Previous Observation: {self.previous_obs_list.ev_features}")
                print(f"Observation: {observation.ev_features}")
                print(f'Action Mapper: {observation.action_mapper}')
                print(f"env_features: {observation.env_features}")
            else:  
                print(f"Previous Observation: {self.previous_obs_list}")
                print(f"Observation: {observation}")                  
        
        if self.GNN_state:
            temp_observation = deepcopy(observation)
            
            if self.previous_obs_list is None:
                self.previous_obs_list = deepcopy(observation)
                
            
            for i, ev_features in enumerate(observation.ev_features):
                action_index = observation.action_mapper[i]
                if self.random[action_index, step] < self.p_delay:
                    if action_index in self.previous_obs_list.action_mapper:
                        # Find the index of the action in the previous observation
                        prev_index = self.previous_obs_list.action_mapper.index(action_index)                        
                        
                        if self.verbose:
                            print(f"Communication failed for action {action_index}, \
                                  features {observation.ev_features[i, 1]}, \
                                  prev: {self.actual_previous_obs_list.ev_features[prev_index, 1]}")
                        
                        not_communicated_energy_usage += observation.ev_features[i, 1] - self.actual_previous_obs_list.ev_features[prev_index, 1]
                        observation.ev_features[i, 1] = self.previous_obs_list.ev_features[prev_index, 1]
                        
                    else:
                        continue
                                                    
            observation.env_features[0,4] -= not_communicated_energy_usage * 60/self.timescale                        
            self.previous_obs_list = deepcopy(observation)
            self.actual_previous_obs_list = deepcopy(temp_observation)
            
            
        else:
            temp_observation = deepcopy(observation)
            for i in range(self.num_of_actions):
                if observation[3+i*3] == 0:
                    continue
                if self.random[i, step] < self.p_delay:
                    not_communicated_energy_usage += observation[4+i*3] - self.actual_previous_obs_list[4+i*3]
                    observation[4+i*3] = self.previous_obs_list[4+i*3] 

            observation[2] -= not_communicated_energy_usage * 60/self.timescale
            self.previous_obs_list = deepcopy(observation)
            self.actual_previous_obs_list = deepcopy(temp_observation)
        
        if self.verbose:
            
            print(f"Energy not communicated: {not_communicated_energy_usage}")
            if self.GNN_state:
                print(f"---New Observation---")
                print(f"Observation: {observation.ev_features}")
                print(f'Action Mapper: {observation.action_mapper}')
                print(f"env_features: {observation.env_features}")         
            else:
                print(f"---New Observation---")
                print(f"Observation: {observation}")
                
        if self.GNN_state:
            assert (observation.env_features[0,4] >= -5), "Power cannot be negative (in PST)"
            observation.env_features[0,4] = max(0, observation.env_features[0,4])
        else:
            assert (observation[2] >= -5), "Power cannot be negative (in PST)"
            observation[2] = max(0, observation[2])
            
        # input("Press Enter to continue...")
        
        return observation

                