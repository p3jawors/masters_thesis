"""
Test LDN ability to represent activities and our state predictions
"""
from masters_thesis.utils.eval_utils import calc_ldn_repr_err
import matplotlib.pyplot as plt
import numpy as np
from masters_thesis.utils import plotting
from abr_analyze import DataHandler

db_name = 'llp_pd'
dat = DataHandler(db_name, database_dir='../data/databases')
data = dat.load(save_location='train/param_set_0003', parameters=['activities', 'z'])
act = data['activities']
predicted_z = data['z']

print('shape: ', act.shape)
n_steps = 10000
# raise Exception
z = predicted_z[:n_steps, :]
output_labs = ['x', 'y', 'z']

# z = act[:n_steps, 0]
# z = z[:, np.newaxis]
# output_labs = ['neuron0']

dt = 0.01
label = 'activity_repr'
# T = 5
# t = np.arange(0, T, dt)
# F = 1
# z = np.sin(F * t*(np.pi*2))
# z = z[: , np.newaxis]

qvals = [10]
theta = 0.1
theta_p = [0.5, 1.0]
error, zhats = calc_ldn_repr_err(z, qvals, theta, theta_p, dt=dt, return_zhat=True)
# TODO update this to subplot based on m dimensionality (n outputs)
plotting.plot_ldn_repr_error(error, theta, theta_p, z, dt, zhats, output_labs, label)
