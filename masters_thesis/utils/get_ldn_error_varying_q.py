"""
Test LDN ability to represent activities and our state predictions
"""
from masters_thesis.utils.eval_utils import calc_ldn_err_vs_q
import matplotlib.pyplot as plt
import numpy as np
from masters_thesis.utils import plotting
from abr_analyze import DataHandler
import sys

# dat = DataHandler('codebase_test_set', 'data/databases')
# data = dat.load(save_location='sin5t', parameters=['time', 'state'])
# # z = data['state']
# z = data['state'][:, 0]
# z = z[:, np.newaxis]

n_steps = 10000
db_name = 'llp_pd'
dat = DataHandler(db_name, database_dir='data/databases')
data = dat.load(save_location='train/param_set_0003', parameters=['activities', 'Z', 'q'])
act = data['activities']
predicted_z = data['Z']
q = data['q']

data2 = dat.load(save_location='100_linear_targets', parameters=['state'])
state = data2['state']

to_show = 'activities'
if len(sys.argv) > 1:
    to_show = str(sys.argv[1])

dt = 0.01
theta = 1
theta_p = [1]

if to_show in ['Z', 'prediction']:
    z = predicted_z[:n_steps, :]
    prediction_dim_labs = []
    for jj in ['x', 'y', 'z']:
        for ii in range(0, q):
            prediction_dim_labs.append(f"{jj}_{ii}")
    label = 'LDN of Prediction'
    qvals = [5]
    error, zhats = calc_ldn_err_vs_q(
        z, qvals, theta, theta_p, dt=dt, return_zhat=True)
    plotting.plot_ldn_repr_error(
        error, theta, theta_p, z, dt, zhats, prediction_dim_labs,
        max_rows=3,
        folder='data/presentation_figures/',
        save_name='LDN_prediction_Z')

elif to_show in ['act', 'activities', 'neurons', 'spikes']:
    z = act[:n_steps, :3]
    prediction_dim_labs = ['neuron0', 'neuron1', 'neuron2']
    label = 'Neural Activities'
    qvals = [6]

    error, zhats = calc_ldn_err_vs_q(
        z, qvals, theta, theta_p, dt=dt, return_zhat=True)

    plotting.plot_ldn_repr_error(
        error=error,
        theta=theta,
        theta_p=theta_p,
        z=z,
        dt=dt,
        zhats=zhats,
        prediction_dim_labs=prediction_dim_labs,
        max_rows=3,
        folder='data/presentation_figures/',
        save_name='LDN_prediction_activities'
    )

elif to_show in ['state', 'context', 'c']:
    z = state[:n_steps, :3]
    prediction_dim_labs = ['x', 'y', 'z']
    label = 'Input Context and GT'
    qvals = [4]
    error, zhats = calc_ldn_err_vs_q(
        z, qvals, theta, theta_p, dt=dt, return_zhat=True)

    # print(zhats['4'].shape)
    plotting.plot_ldn_repr_error(
        error, theta, theta_p, z, dt, zhats, prediction_dim_labs)#, label)


