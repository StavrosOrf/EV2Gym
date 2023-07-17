#This file implements functions that simulate the grid of the simulations using PandaPower

import pandapower as pp

net = pp.networks.create_cigre_network_mv(with_der=False)

#print bus information
print(net.bus)

print(net.line)

print(net.trafo)

#solve the power flow
pp.runpp(net,numba=False)

#print the results
print(net.res_bus)
print(net.bus)

#solve the optimal power flow
pp.runopp(net,numba=False)

#print the results
print(net.res_bus)
print(net.gen)
print(net.ext_grid)


