# This file implements functions that simulate the grid of the simulations using PandaPower

from pandapower.plotting.plotly import pf_res_plotly
import pandapower as pp
import numpy as np

from pandapower.plotting.plotly.mapbox_plot import set_mapbox_token
set_mapbox_token(
    'pk.eyJ1Ijoic29yZmFub3VkYWtpcyIsImEiOiJjbGs2ejQycGcwM3dwM21ydDd2dGc1dG1oIn0.dJ9E-1u53SZfDH0xgVaBkw')


net = pp.networks.create_cigre_network_mv(with_der="pv_wind")
net = pp.networks.create_cigre_network_mv(with_der="all")
# net = pp.networks.create_cigre_network_lv()
# net = pp.networks.case14()

timeslot_duration = 5  # minutes
simulation_duration = 10  # * timeslot_duration minutes

# TODO: Initiallizing the test case with the charging stations
# get c random numbers between 0 and len(net.bus)
c = 2

np.random.seed(65)
random_numbers = np.random.randint(0, len(net.bus), c)

# add c storage units to the grid
for i in random_numbers:
    pp.create_storage(net, bus=13, p_mw=0, max_e_mwh=1000,
                      soc_percent=50, type="bat")

# TODO: connect each charging station to a bus -> indirectly connect to a transformer
print(net.storage)
# print(net.storage[0])

#generate random load profiles for each load fo the simulation duration
rnd_data = np.random.rand(len(net.load), simulation_duration)

# TODO: run in every step of the simulation -> def step()

# update with new charging actions
net.storage['p_mw'][0] = 0.5
net.storage['soc_percent'][0] = 65


print(net.storage)
# solve the power flow
pp.runpp(net, numba=False)

# TODO: replace with REAL load profiles
# iterate over net.load rows and change the values
for i in range(len(net.load)):
    if i == 1:
        net.load.p_mw[i] = 0.2
        net.load.q_mvar[i] = 0.2

# TODO: replace with REAL static generation (pv, wind) profiles
for i in range(len(net.sgen)):
    if i == 1:
        net.sgen.p_mw[i] = 0.2
        net.sgen.q_mvar[i] = 1.0

# TODO: replace with REAL generation profiles
for i in range(len(net.sgen)):
    if i == 1:
        net.gen.p_mw[i] = 0.2
        net.gen.vm_pu[i] = 1.0

# TODO: replace with REAL ext_grid profile
for i in range(len(net.ext_grid)):
    net.ext_grid.vm_pu[i] = 1.0
    net.ext_grid.va_degree[i] = 0.0

# solve the dc power flow
# pp.rundcpp(net)

print(net.bus)
print(net.line)
print(net.trafo)
# print(net.res_bus)


# Get informatino about the overaloading and the lines from here
print(net.res_line)
print(net.res_trafo)
print(net.res_trafo.loading_percent.round(3).astype(str))
# visualize the grid using plotly
# pp.plotting.plotly.simple_plotly(net)

print(f'Overloaded lines: {net.res_line[net.res_line.loading_percent > 100]}')
pp.plotting.to_html(net, "./test.html")

print(random_numbers)
# pf_res_plotly(net)

# simple plot of net
pp.plotting.simple_plot(net, plot_loads=True, plot_sgens=True)
