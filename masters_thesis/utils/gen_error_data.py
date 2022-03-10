"""
calculate the difference between state and target and saves it to a new key
state_and_error, which is a horizontal stack of state and error
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from abr_analyze import DataHandler

view = False
if len(sys.argv) > 1:
    view = sys.argv[1]


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

labs = [
    'x', 'y', 'z', 'dx', 'dy' , 'dz', 'a', 'b' , 'g', 'da', 'db', 'dg',
    'ex', 'ey', 'ez', 'edx', 'edy', 'edz', 'ea', 'eb' , 'eg', 'eda', 'edb', 'edg'
]

plt.figure(figsize=(12, 12))
for ii in range(0, min(state_err.shape)):
    plt.subplot(8, 3, ii+1)
    plt.title(f"{labs[ii]}")
    plt.plot(state_err.T[ii])
plt.show()



if not view:
    dat.save(
        save_location=train_data,
        data=data,
        overwrite=True
    )
