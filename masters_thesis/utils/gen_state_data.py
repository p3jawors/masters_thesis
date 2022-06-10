#TODO update multiplier to properly normalize errors based on a set max
# keeping vectors together (xyz scale together etc)
# you're too tired right now to get it going so multiplying error by 30
# for now since that gets your close
"""
calculate the difference between state and target and saves it to a new key
state_and_error, which is a horizontal stack of state and error
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from abr_analyze import DataHandler

def plot_thing(thing):
    plt.figure(figsize=(12, 12))
    for ii in range(0, min(thing.shape)):
        plt.subplot(8, 3, ii+1)
        plt.title(f"{labs[ii]}")
        plt.plot(data['time'], thing.T[ii])
        # plt.plot(data['state_and_error-max_0_1-normalized'].T[ii])
    plt.show()


view = False
if len(sys.argv) > 1:
    view = sys.argv[1]

# db_name = 'llp_pd_e'
# database_dir = 'data/databases'
# train_data = '100_linear_targets_faster'

db_name = 'llp_pd_d'
database_dir = 'data/databases'
train_data = '9999_linear_targets_faster' #  90820 temporal data points
dat = DataHandler(
    db_name=db_name,
    database_dir=database_dir
)
data = dat.load(
    save_location=train_data,
    parameters=dat.get_keys(train_data)#['state', 'target']
)
# MULTIPLIER = 1#30
print('Available keys: ', dat.get_keys(train_data))
print('State shape: ', data['state'].shape)
print(' Target shape: ', data['target'].shape)
# state_err = MULTIPLIER * np.hstack((data['state'], np.subtract(data['target'], data['state'])))
# print('Stacked state-error shape: ', state_err.shape)
# # data['state_and_error_30x'] = state_err

labs = [
    'x', 'y', 'z', 'dx', 'dy' , 'dz', 'a', 'b' , 'g', 'da', 'db', 'dg',
    'ex', 'ey', 'ez', 'edx', 'edy', 'edz', 'ea', 'eb' , 'eg', 'eda', 'edb', 'edg'
]
new_state = np.empty(data['state'].shape)
for ii in range(0, data['state'].shape[1]):
    dim = data['state'][:, ii]
    new_state[:, ii] = (dim - np.mean(dim))/np.amax(abs(dim-np.mean(dim)))
    print('---------')
    print('DIM: ', ii)
    print('MEAN: ', np.mean(dim))
    print('SCALE: ', np.amax(abs(dim-np.mean(dim))))
    new_state[:, ii] = (dim - np.mean(dim))/np.amax(abs(dim-np.mean(dim)))

data['mean_shift_abs_max_scale_state'] = new_state
if view:
    plot_thing(new_state)

# max_err = 0.1
# data['state_and_error-max_0_1-normalized'] = np.clip(
#     data['state_and_error'], -max_err, max_err)/max_err
    # np.subtract(data['target'], data['state']), -max_err, max_err)/max_err

# normalize error vectors
# xyz_err = data['target'].T[:3] - data['state'].T[:3]
# norm_err = np.linalg.norm(xyz_err, axis=0)
# print(xyz_err)
# print(norm_err)
# print(np.linalg.norm(xyz_err/norm_err))
# max_err = np.amax(xyz_err)
# norm_err = xyz_err / max_err



if not view:
    print(f"saving new state data to {train_data}")
    dat.save(
        save_location=train_data,
        data=data,
        overwrite=True
    )
