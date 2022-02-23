import numpy as np
import nengo
import sys

from abr_analyze import DataHandler
from utils import decode_ldn_data, calc_shifted_error
import os
from plotting import plot_pred, plot_error
from train import run_model

if len(sys.argv) > 1:
    # pass in False to just reload the test results
    rerun = eval(sys.argv[1])
else:
    rerun = True

animate = True
experiment_id = 'train/param_set_0002'
notes = (
"""
- clean u, gravity offset removed, clipped at 250, and normalized
- using radius of sqrt(size_in)
"""
)
dt = 0.01
n_neurons = 2000
theta = 1.0
# theta_p = np.linspace(dt, 0.5*theta, int(theta/dt))
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
    parameters=['time', 'state', 'ctrl', 'state_and_error', 'clean_u']
)
n_data_pts = 30000
data['time'] = data['time'][:n_data_pts]
data['state'] = data['state'][:n_data_pts]
data['ctrl'] = data['ctrl'][:n_data_pts]
data['state_and_error'] = data['state_and_error'][:n_data_pts]
data['clean_u'] = data['clean_u'][:n_data_pts]

print("Running with default parameters")

params = {}
# x, y, z, dx, dy, dz, a, b, g, da, db, dg
# run_model appends ctrl to the context dims as input
params['c_dims'] = (12, 13, 14, 20)#, 3, 4, 5) # xyz
params['z_dims'] = (12, 13, 14)
params['q_a'] = 10
params['q_p'] = 8
params['q'] = 8
params['learning_rate'] = 0.000012860959190751539
params['n_neurons'] = 2000

size_in = len(params['c_dims']) + 4

# NOTE scaling learning rate by dt here, and llp class scales by 1/n_neurons
if rerun:
    save_data = run_model(
        dt=dt,
        n_neurons=params['n_neurons'],
        size_in=size_in,
        q=params['q'],
        q_a=params['q_a'],
        q_p=params['q_p'],
        theta=theta,
        learning_rate=params['learning_rate']*dt,
        seed=seed,
        c_dims=params['c_dims'],
        z_dims=params['z_dims'],
        data=data,
        radius=np.sqrt(len(params['c_dims']))
    )

    save_data['notes'] = notes
    # no need to save nni data
    save_data['train_data'] = train_data
    dat.save(
        save_location=experiment_id,
        data=save_data,
        overwrite=True
    )

else:
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
print("\nTODOOO!!!! \n\n NEED TO REMOVE HARDCODING FOR ERROR CALC DIMS\n\n")
errors = calc_shifted_error(
    z=data['state_and_error'][:, 12:15], #  xyz
    zhat=zhat,
    dt=dt,
    theta_p=theta_p
)
errors = np.absolute(errors)
print('ERRORS: ', errors.shape)

plot_error(theta_p=theta_p, errors=errors, dt=dt)

plot_pred(
    time=data['time'],
    # z=data['state'][:, :3],
    z=data['state_and_error'][:, 12:15], #  xyz
    zhat=zhat[:, -1, :],
    theta_p=[max(theta_p)],
    size_out=3,
    gif_name=f"{experiment_id.split('/')[-1]}.gif",
    animate=animate
)
