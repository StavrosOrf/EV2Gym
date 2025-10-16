import pandas as pd
import numpy as np
from multicopula import EllipticalCopula
import pickle
import time

import matplotlib.pyplot as plt


class DataGenerator:
    def __init__(self):

        self._fit_load_profiles()

    def _fit_load_profiles(self):
        """
        Load active power data from the data manager.
        """
        data_path = "./ev2gym/data/original_train_data.csv"

        df = pd.read_csv(data_path, parse_dates=['date_time'])
        print(f'df initial shape is {df.shape}')

        # Drop the "price" column and any columns related to renewable generation.
        # cols_to_drop = [col for col in df.columns
        #                 if col.startswith('price') or col.startswith('renewable_active_power')]
        cols_to_drop = [col for col in df.columns
                        if col.startswith('price')]
        df.drop(columns=cols_to_drop, inplace=True)

        df['date_time'] = pd.to_datetime(df['date_time'])

        # Extract the date and time from the date_time column.
        df['day'] = df['date_time'].dt.day_of_week
        df.drop(columns=['date_time'], inplace=True)
        df['timestep'] = df.index % 96

        print(f'df shape is {df.values.shape}')
        # df = df[:96*7]
        new_df = pd.DataFrame()
        print(f'number of days is {int(len(df.values)//96)}')

        for j in range(int(len(df.values)//96)):

            for i in range(1, 34):
                day = df.values[j*96, -2].astype(int)
                active_power_96_node_i = df.values[j *
                                                   96:(j+1)*96, i]
                # - df.values[j*96:(j+1)*96, i+34]
                # active_power_96_node_i = -df.values[j*96:(j+1)*96, i+34]
                entry = {'day': [day],
                         }

                for k in range(96):
                    entry[f'active_power_{k}'] = active_power_96_node_i[k]

                new_df = pd.concat(
                    [new_df, pd.DataFrame(entry)], ignore_index=True)

        print(f'new_df shape is {new_df.shape}')

        dataset = new_df.values.T
        print(f'dataset shape is {dataset.shape}')

        self.copula_model = EllipticalCopula(dataset)
        self.copula_model.fit()

    def sample_data(self,
                    n_buses: int,
                    n_steps: int,
                    start_day: int,
                    start_step: int = 0,
                    ):

        data = np.zeros((int(np.ceil((start_step + n_steps)/96))*96, n_buses))

        for j in range(int(np.ceil((start_step + n_steps)/96))):
            day = (start_day + j) % 7
            
            # for node in range(1, n_buses):
            while True:
                augmented_data = self.copula_model.sample(n_buses,
                                                        conditional=True,
                                                        variables={
                                                            'x1': day,
                                                            },
                                                        )                    
                if not np.isnan(augmented_data).any() and not np.isinf(augmented_data).any():
                    data[j*96:(j+1)*96, :] = augmented_data
                    break

        return data[start_step:start_step+n_steps, :]

def get_pv_load(data, env):

    dataset_starting_date = '2019-01-01 00:00:00'
    simulation_length = env.simulation_length + 24
    simulation_date = env.sim_starting_date.strftime('%Y-%m-%d %H:%M:%S')

    # find year of the data
    year = int(dataset_starting_date.split('-')[0])
    # replace the year of the simulation date with the year of the data
    simulation_date = f'{year}-{simulation_date.split("-")[1]}-{simulation_date.split("-")[2]}'

    simulation_index = data[data['date'] == simulation_date].index[0]

    # select the data for the simulation date
    data = data[simulation_index:simulation_index+simulation_length]
    
    return data['electricity'].values.reshape(-1, 1)

if __name__ == "__main__":

    # Code for fitting the copula model and saving it to a file so it can be quickly loaded later.

    augmentor = DataGenerator()
    pickle.dump(augmentor, open('augmentor.pkl', 'wb'))

    augmentor = pickle.load(open('augmentor.pkl', 'rb'))

    start_time = time.time()    
    augmented_data = augmentor.sample_data(n_buses=34,
                                           n_steps=96*5,
                                           start_day=5,
                                           start_step=0,
                                           )
    print(f'Elapsed time is {time.time() - start_time}')
    
    # plot the data
    # plt.plot(augmented_data[:,11:16])
    plt.plot(augmented_data)
    # plt.legend([f'Node {i}' for i in range(1, 6)])
    plt.show()
