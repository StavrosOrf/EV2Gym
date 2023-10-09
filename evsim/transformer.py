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
                 max_power=1000,  # The maximum capacity of the transformer in kWh
                 min_power=-200,  # (can be negative for discharging)
                 cs_ids=[],  # the charging stations connected to the transformer
                 efficiency=1, timescale=5):

        self.id = id
        self.max_power = max_power
        self.min_power = min_power
        self.efficiency = efficiency
        self.timescale = timescale
        self.cs_ids = cs_ids

        self.current_power = 0
        

    def step(self, power):
        '''
        Update current power of the transformer
        '''
        self.current_power += power

    def is_overloaded(self):
        '''
        Check if the transformer is overloaded

        Returns:
            - True if the transformer is overloaded
            - False if the transformer is not overloaded
        '''
        e = 0.0001
        if self.current_power > self.max_power + e  * self.timescale / 60 \
            or self.current_power < self.min_power - e * self.timescale / 60:
            
            return True
        else:
            return False

    def get_state(self):
        '''
        Returns the state of the transformer
        '''
        state = [self.current_power,
                 self.max_power,
                 self.min_power ]
                #  self.cs_ids] #TODO check observation space if needed 

        return np.hstack(state)

    def __str__(self) -> str:
        return f'  - Transformer {self.id}:  {self.min_power*self.timescale / 60:.1f} / ' +\
            f'{self.current_power:5.1f} /{self.max_power*self.timescale / 60:5.1f} kWh' +\
            f'\tCSs: {self.cs_ids}'
