"""
Test LDN ability to represent activities and our state predictions
"""
from masters_thesis.utils.eval_utils import calc_ldn_repr_err
import matplotlib.pyplot as plt
import numpy as np
from masters_thesis.utils import plotting
from abr_analyze import DataHandler

# dat = DataHandler('codebase_test_set', 'data/databases')
# data = dat.load(save_location='sin5t', parameters=['time', 'state'])
# # z = data['state']
# z = data['state'][:, 0]
# z = z[:, np.newaxis]

n_steps = 10000
db_name = 'llp_pd'
dat = DataHandler(db_name, database_dir='data/databases')
data = dat.load(save_location='train/param_set_0003', parameters=['activities', 'z'])
act = data['activities']
predicted_z = data['z']

z = predicted_z[:n_steps, :]
output_labs = ['x', 'y', 'z']

z = act[:n_steps, :2]
# z = z[:, np.newaxis]
# output_labs = ['neuron0']
output_labs = None

dt = 0.01
label = 'activity_repr'

qvals = [50]
theta = 1
theta_p = [1]
error, zhats = calc_ldn_repr_err(z, qvals, theta, theta_p, dt=dt, return_zhat=True)
# TODO update this to subplot based on m dimensionality (n outputs)
plotting.plot_ldn_repr_error(error, theta, theta_p, z, dt, zhats, output_labs, label)
