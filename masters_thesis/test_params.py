import numpy as np
import nengo
import sys

from abr_analyze import DataHandler
from utils import decode_ldn_data, calc_shifted_error
import os
from plotting import plot_pred, plot_error
from train import run_model

experiment_id = 'train/param_set_0000'
dt = 0.01
n_neurons = 2000
theta = 1.0
theta_p = np.linspace(dt, theta, int(theta/dt))
print(theta_p)
print(len(theta_p))
seed = 47

# Load training data
db_name = 'llp_pd'
train_data = '100_linear_targets' #  90820 temporal data points
print(f"Current dir: {os.getcwd()}")
dat = DataHandler(db_name, database_dir='data/databases')
data = dat.load(
    save_location=train_data,
    parameters=['time', 'state', 'ctrl']
)
n_data_pts = 10000
data['time'] = data['time'][:n_data_pts]
data['state'] = data['state'][:n_data_pts]
data['ctrl'] = data['ctrl'][:n_data_pts]

print("Running with default parameters")

params = {}
# x, y, z, dx, dy, dz, a, b, g, da, db, dg
# run_model appends ctrl to the context dims as input
params['context_dims'] = (0, 1, 2, 6, 7, 8) # xyz
params['q_a'] = 10
params['q_p'] = 7
params['q'] = 7
params['learning_rate'] = 0.000012834580078171056

size_in = len(params['context_dims']) + 4

# NOTE scaling learning rate by dt here, and llp class scales by 1/n_neurons
# save_data = run_model(
#     dt=dt,
#     n_neurons=n_neurons,
#     size_in=size_in,
#     q=params['q'],
#     q_a=params['q_a'],
#     q_p=params['q_p'],
#     theta=theta,
#     learning_rate=params['learning_rate']*dt,
#     seed=seed,
#     context_dims=params['context_dims'],
#     data=data
# )
#
# # no need to save nni data
# save_data['train_data'] = train_data
# dat.save(
#     save_location=experiment_id,
#     data=save_data,
#     overwrite=True
# )

# TODO coment when training to view live results
save_data = dat.load(
    save_location=experiment_id,
    parameters=['Z']
)

zhat = decode_ldn_data(
    Z=save_data['Z'],
    q=params['q'],
    theta=theta,
    theta_p=theta_p
)
errors = calc_shifted_error(
    z=data['state'][:, :3], #  xyz
    zhat=zhat,
    dt=dt,
    theta_p=theta_p
)
errors = np.absolute(errors)
print('ERRORS: ', errors.shape)

plot_error(theta_p=theta_p, errors=errors, dt=dt)

# plot_pred(
#     time=data['time'],
#     z=data['state'][:, :3],
#     zhat=zhat[:, -1, :],
#     theta_p=[max(theta_p)],
#     size_out=3,
#     gif_name='llp_default.gif',
#     animate=True
# )
