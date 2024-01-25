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
                 max_current=150,  # The maximum capacity of the transformer in A
                 min_current=0,  # (can be negative for discharging)
                 cs_ids=[],  # the charging stations connected to the transformer
                 timescale=5,
                 standard_transformer_loading=None,
                 ):

        self.id = id
        self.max_current = max_current
        self.min_current = min_current
        self.standard_transformer_loading = standard_transformer_loading
        self.timescale = timescale
        self.cs_ids = cs_ids

        self.current_amps = 0
        self.current_power = 0

    def step(self, amps, power):
        '''
        Update current power of the transformer
        '''
        self.current_amps += amps
        self.current_power += power

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

    def get_how_overloaded(self):
        '''
        Check how overloaded the transformer is

        Returns:
            - a amps value if the transformer is overloaded
        '''
        if self.is_overloaded():
            return np.abs(self.current_amps - self.max_current)
        else:
            return 0

    def __str__(self) -> str:
        return f'  - Transformer {self.id}:  {self.min_current:.1f} / ' +\
            f'{self.current_amps:5.1f} /{self.max_current:5.1f} A' +\
            f'\tCSs: {self.cs_ids}'
