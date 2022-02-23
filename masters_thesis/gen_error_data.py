"""
calculate the difference between state and target and saves it to a new key
state_and_error, which is a horizontal stack of state and error
"""
import numpy as np
from abr_analyze import DataHandler

db_name = 'llp_pd'
train_data = '100_linear_targets' #  90820 temporal data points
dat = DataHandler(db_name, database_dir='data/databases')
data = dat.load(
    save_location=train_data,
    parameters=dat.get_keys(train_data)#['state', 'target']
)
print('Available keys: ', dat.get_keys(train_data))
print('State shape: ', data['state'].shape)
print(' Target shape: ', data['target'].shape)
state_err = np.hstack((data['state'], np.subtract(data['target'], data['state'])))
print('Stacked state-error shape: ', state_err.shape)
data['state_and_error'] = state_err
dat.save(
    save_location=train_data,
    data=data,
    overwrite=True
)
