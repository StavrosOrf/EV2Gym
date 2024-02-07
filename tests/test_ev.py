"""
This file is used to assess the battery drgradation model.

"""
import sys
import numpy as np
import os
sys.path.append(os.path.realpath('../'))

from EVsSimulator.EVsSimulator.models.ev import EV



def battery_degradation_test():
    '''
    This function print the battery degradation due to calendar aging and cycling 
    as a parameter of the charging power (x-axis) and the required energy per day (y-axis).
    '''
    

    timescale = 15  # 1 hour
    simulation_length = 96 * 1  # 1 day
    # initialzing an EV

    amps_list = np.arange(6, 32, 2)
    required_energy_list = np.arange(5, 50, 5)
    
    calendar_degradation = np.zeros((len(amps_list), len(required_energy_list)))
    cycling_degradation = np.zeros((len(amps_list), len(required_energy_list)))
        
    
    for i, amps in enumerate(amps_list):
        
        for j, required_energy in enumerate(required_energy_list):
            
            ev = EV(id=0,
                    location=0,
                    battery_capacity_at_arrival=50-required_energy,
                    battery_capacity=50,
                    time_of_arrival=0,
                    time_of_departure=simulation_length-1,
                    transition_soc=0.9999,
                    timescale=timescale,
                    )

            for t in range(simulation_length):                                
                current_energy, actual_current = ev.step(amps=amps,
                                                        voltage=230,
                                                        phases=3,
                                                        type='AC')
                print(f'Battery capacity: {ev.get_soc():.2f},' +
                    f'Controller current: {amps:.2f},' +
                    f' Actual: {actual_current:.2f} A | {current_energy:.2f} kW')
                
            d_cal, d_cyc = ev.get_battery_degradation()
            
            # print(f'degradation due to calendar aging: {d_cal} \n' +
            #     f'degradation due to cycling: {d_cyc}')
            calendar_degradation[i, j] = d_cal
            cycling_degradation[i, j] = d_cyc
            
    #plot a 3d graph of the battery degradation as a function of the charging power and the required energy per day
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(amps_list, required_energy_list)
    ax.plot_surface(X, Y, calendar_degradation, label='calendar degradation')
    ax.plot_surface(X, Y, cycling_degradation, label='cycling degradation')
    ax.set_xlabel('Charging power [A]')
    ax.set_ylabel('Required energy per day [kWh]')
    ax.set_zlabel('Battery degradation')
    plt.show()
    

if __name__ == "__main__":
    battery_degradation_test()
