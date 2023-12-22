import pandas as pd

import commons

# Generate simulation IDs and values
simulation_ids = ['simulation_' + str(i) for i in range(10)]
values = range(101, 150)

# Create the DataFrame
data = {'simulation_id': [], 'value': [], 'output': []}
for sim_id in simulation_ids:
    for value in values:
        data['simulation_id'].append(sim_id)
        data['value'].append(value)
        data['output'].append(value*5)

sim_df = pd.DataFrame(data)
windows = commons.create_windows(sim_df, 10, 5, ["simulation_id", "value"], ["output"])
print("xxx")