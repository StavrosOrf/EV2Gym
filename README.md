![logo](https://github.com/stavrosgreece/EVsSimulator/assets/17108978/4852f3c4-3371-4561-8e8f-661acbaf2a01)

# EVsSimulator
A realistic V2X Simulation Environment for large scale EV charging optimization!

<!-- Bullet points with all the benefits -->
## Features

* The simulator can be used to evaluate any type of algorithm to gain insights into its efficiency.
* The “gym environment” can readily support the development of RL algorithms.
* Uses only open-source data.
* Replays of simulations are saved and can be solved optimally using the Gurobi Solver.
* Easy to incorporate additional functionality for any use-case.
* Does not simulate the grid yet, but groups EV chargers at the level of the transformer/ parking lot, etc, so extra functionality can be easily added.


Focused on **realistic** parameters and **fully customizable**:
* Transformer models
  * Max Current
* Charging Stations models
  * Min and Max charge/discharge power/ Current
  * Voltage and phases, AC or DC
  * Charge and discharge efficiency
  * List of connected transformers
* Electric Vehicles models
  * Connected charging station and port
  * Min and Max battery energy level
  * Time of arrival and departure
  * Energy at arrival/ desired energy at departure
  * Min and Max current /power levels
  * Constant-Current/ Constant-Voltage load-curve option 


## Data sources
* The number and the topology of Transformers, Charging stations, and Electric Vehicles are parameterizable.
* Charging/ Discharging prices are based on historical day-ahead prices.
* EV spawn rate, time of stay, and energy required are based on realistic distributions ElaadNL,time, day, month and year.
* EV and Charger characteristics are based on real EVs and chargers existing in NL.

