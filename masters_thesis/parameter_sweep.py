import numpy as np
import os
import nengo
import nni
import sys

from abr_analyze import DataHandler
from utils import decode_ldn_data, calc_shifted_error
from plotting import plot_pred, plot_error
from train import run_model

dt = 0.01
theta = 1.0
theta_p = np.linspace(dt, theta, int(theta/dt))
print(theta_p)
print(len(theta_p))
seed = 47

run_nni = False
n_data_pts = 10000
if len(sys.argv) > 1:
    if 'nni' in sys.argv[1]:
        run_nni = True
    if len(sys.argv) > 2:
        n_data_pts = int(sys.argv[2])

# Load training data
db_name = 'llp_pd'
train_data = '100_linear_targets' #  90820 temporal data points
# print(f"Current dir: {os.getcwd()}")
dat = DataHandler(db_name, database_dir='../data/databases') #  since nni is run from nni_scripts folder, go back a dir
data = dat.load(
    save_location=train_data,
    parameters=['time', 'state', 'ctrl', 'state_and_error', 'clean_u']
)
data['time'] = data['time'][:n_data_pts]
data['state'] = data['state'][:n_data_pts]
data['ctrl'] = data['ctrl'][:n_data_pts]
data['state_and_error'] = data['state_and_error'][:n_data_pts]
data['clean_u'] = data['clean_u'][:n_data_pts]

if run_nni:
    # Load nni params
    experiment_id = nni.get_experiment_id()
    params = nni.get_next_parameter()
    cid = nni.get_sequence_id()
    print(f'--Starting nni trial: {experiment_id} | {cid}--')
else:
    experiment_id = 'train/default'
    print("Running with default parameters")

    params = {}
    # x, y, z, dx, dy, dz, a, b, g, da, db, dg
    # run_model appends ctrl to the context dims as input
    # params['c_dims'] = (0, 1, 2, 8)#, 3, 4, 5) # xyz
    params['c_dims'] = (12, 13, 14, 20)#, 3, 4, 5) # xyz
    params['q_a'] = 10
    params['q_p'] = 8
    params['q'] = 8
    params['learning_rate'] = 0.000012860959190751539
    # params['n_neurons'] = 2000

size_in = len(params['c_dims']) + 4
params['z_dims'] = (12, 13, 14)

# NOTE scaling learning rate by dt here, and llp class scales by 1/n_neurons
save_data = run_model(
    dt=dt,
    n_neurons=2000,#params['n_neurons'],
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

if not run_nni:
    # no need to save nni data
    save_data['train_data'] = train_data
    dat.save(
        save_location=experiment_id,
        data=save_data,
        overwrite=True
    )

# TODO coment when training to view live results
# save_data = dat.load(
#     save_location=experiment_id,
#     parameters=['Z']
# )

zhat = decode_ldn_data(
    Z=save_data['Z'],
    q=params['q'],
    theta=theta,
    theta_p=theta_p
)
print("\nTODOOO!!!! \n\n NEED TO REMOVE HARDCODING FOR ERROR CALC DIMS\n\n")
errors = calc_shifted_error(
    # z=data['state'][:, :3], #  xyz
    z=data['state_and_error'][:, 12:15], #  xyz
    zhat=zhat,
    dt=dt,
    theta_p=theta_p
)
print('ERRORS: ', errors.shape)
errors = np.absolute(errors)
error = sum(sum(sum(errors)))
print('final error: ', error)

if run_nni:
    nni.report_final_result(error)
else:
    plot_error(theta_p=theta_p, errors=errors)

    plot_pred(
        time=data['time'],
        z=data['state'][:, :3],
        zhat=zhat,
        theta_p=theta_p,
        size_out=3,
        gif_name='llp_default.gif',
        animate=False
    )
