'''
===================================
Author: Stavros Orfanoudakis 2023
===================================
'''
import numpy as np


class Transformer():
    """
    Transformer class for the ev_city environment
    """

    def __init__(self,
                 id,  # unique identifier of the transformer
                 max_current=100,  # The maximum capacity of the transformer in A
                 min_current=-100,  # (can be negative for discharging)
                 cs_ids=[],  # the charging stations connected to the transformer
                 timescale=5):

        self.id = id
        self.max_current = max_current
        self.min_current = min_current
        self.timescale = timescale
        self.cs_ids = cs_ids

        self.current_amps = 0

    def step(self, amps):
        '''
        Update current power of the transformer
        '''
        self.current_amps += amps

    def is_overloaded(self):
        '''
        Check if the transformer is overloaded

        Returns:
            - True if the transformer is overloaded
            - False if the transformer is not overloaded
        '''
        e = 0.0001
        if self.current_amps > self.max_current + e \
                or self.current_amps < self.min_current - e:

            return True
        else:
            return False

    def get_state(self, scenario):
        '''
        Returns the state of the transformer
        '''
        if scenario == "PowerSetpointTracking":
            state = [
                self.max_current/100,
                self.min_current/100
            ]
            #  self.cs_ids] #TODO check observation space if needed

            return np.hstack(state)
        
        elif scenario == "PST_business":
            state = [
                self.max_current/100,
                self.min_current/100
            ]
            

            return np.hstack(state)
        else:
            raise NotImplementedError

    def __str__(self) -> str:
        return f'  - Transformer {self.id}:  {self.min_current:.1f} / ' +\
            f'{self.current_amps:5.1f} /{self.max_current:5.1f} A' +\
            f'\tCSs: {self.cs_ids}'
