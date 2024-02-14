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
                 max_current=250,  # The maximum capacity of the transformer in A
                 max_power=100,  # The maximum power of the transformer in kW
                 max_power_or_current_mode='current',  # 'current' or 'power'
                 cs_ids=[],  # the charging stations connected to the transformer
                 inflexible_transformer_loading=None,
                 demand_response=None,
                 ): 
        """
        Initialize the transformer

        :param id: unique identifier of the transformer
        :type id: int
        :param max_current: The maximum capacity of the transformer in A, defaults to 150
        :type max_current: int, optional
        :param min_current: The minimum capacity of the transformer in A, defaults to 0
        :type min_current: int, optional
        :param cs_ids: the charging stations connected to the transformer, defaults to []
        :type cs_ids: list, optional
        :param timescale: the timescale of the simulation, defaults to 5
        :type timescale: int, optional

        """

        self.id = id
        self.max_current = max_current
        self.min_current = -max_current
        self.max_power = max_power
        self.min_power = -max_power
        self.max_power_or_current_mode = max_power_or_current_mode
        self.inflexible_transformer_loading = inflexible_transformer_loading
        self.demand_response = demand_response
        self.cs_ids = cs_ids

        self.current_amps = 0
        self.current_power = 0

    def reset(self, step) -> None:
        '''
        Reset the current power of the transformer
        '''

        if self.inflexible_transformer_loading is not None:
            self.current_power = self.inflexible_transformer_loading[step]
            self.current_amps = (self.current_power * 1000) / 400
        else:
            self.current_amps = 0
            self.current_power = 0

    def step(self, amps, power) -> None:
        '''
        Update current power of the transformer
        '''
        self.current_amps += amps
        self.current_power += power

    def is_overloaded(self) -> bool:
        '''
        Check if the transformer is overloaded

        Returns:
            - True if the transformer is overloaded
            - False if the transformer is not overloaded
        '''
        e = 0.0001

        if self.max_power_or_current_mode == 'power':
            if self.current_power > self.max_power + e \
                    or self.current_power < self.min_power - e:
                return True
        else:
            if self.current_amps > self.max_current + e \
                    or self.current_amps < self.min_current - e:

                return True

        return False
    
    def get_how_overloaded(self) -> float:
        '''
        Check how overloaded the transformer is

        Returns:
            - a amps value if the transformer is overloaded
        '''
        if self.is_overloaded():
            if self.max_power_or_current_mode == 'power':
                return np.abs(self.current_power - self.max_power)
            else:
                return np.abs(self.current_amps - self.max_current)
        else:
            return 0

    def __str__(self) -> str:        
        if self.max_power_or_current_mode == 'power':
            return f'  - Transformer {self.id}:  {self.min_power:.1f} / ' +\
                f'{self.current_power:5.1f} /{self.max_power:5.1f} kW' +\
                f'\tCSs: {self.cs_ids}'
        else:
            return f'  - Transformer {self.id}:  {self.min_current:.1f} / ' +\
                f'{self.current_amps:5.1f} /{self.max_current:5.1f} A' +\
                f'\tCSs: {self.cs_ids}'
