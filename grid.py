#This file implements functions that simulate the grid of the simulations using PandaPower

import pandapower as pp
import numpy as np

from pandapower.plotting.plotly.mapbox_plot import set_mapbox_token
set_mapbox_token('pk.eyJ1Ijoic29yZmFub3VkYWtpcyIsImEiOiJjbGs2ejQycGcwM3dwM21ydDd2dGc1dG1oIn0.dJ9E-1u53SZfDH0xgVaBkw')

from pandapower.plotting.plotly import pf_res_plotly

net = pp.networks.create_cigre_network_mv(with_der="pv_wind")
net = pp.networks.create_cigre_network_lv()

#TODO: Initiallizing the test case with the charging stations 
#get c random numbers between 0 and len(net.bus)
c = 10
random_numbers = np.random.randint(0, len(net.bus), c)

#add c storage units to the grid
for i in random_numbers:       
    pp.create_storage(net, bus=13, p_mw=-0.1, max_e_mwh=1000, soc_percent=50, type="bat")

#TODO: connect each charging station to a bus -> indirectly connect to a transformer
print(net.storage)
# print(net.storage[0])


#TODO: run in every step of the simulation -> def step()

#update with new charging actions
net.storage['p_mw'][0] = 0.5
print(net.storage)
#solve the power flow
pp.runpp(net,numba=False)

#solve the dc power flow
# pp.rundcpp(net)

print(net.bus)
print(net.line)
print(net.trafo)
# print(net.res_bus)



#Get informatino about the overaloading and the lines from here
print(net.res_trafo.loading_percent.round(3).astype(str))
print(net.res_line)
print(net.res_trafo)
#visualize the grid using plotly
# pp.plotting.plotly.simple_plotly(net)

print(random_numbers)
pf_res_plotly(net)
                   



