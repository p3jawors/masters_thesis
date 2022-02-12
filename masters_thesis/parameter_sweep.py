import numpy as np
import nengo
import nni

from abr_analyze import DataHandler
from utils import decode_ldn_data, calc_shifted_error
from plotting import plot_pred, plot_error
from train import run_model

db_name = 'llp_pd'
train_data = '100_targets_0000'
test_name = 'train/0000'

dat = DataHandler(db_name)
data = dat.load(
    save_location=train_data,
    parameters=['time', 'state', 'ctrl']
)

dt = 0.001
n_neurons = 1000
size_in = 7
# x, y, z, dx, dy, dz, a, b, g, da, db, dg
# run_model appends ctrl to the context dims as input
context_dims = (0, 1, 2) # xyz
q_a = 6
q_p = 6
q = 6
theta = 0.1
theta_p = [0.05, 0.1]
learning_rate = 5e-5*dt
seed = 0

# save_data = run_model(
#     dt=dt,
#     n_neurons=n_neurons,
#     size_in=size_in,
#     q=q,
#     q_a=q_a,
#     q_p=q_p,
#     theta=theta,
#     learning_rate=learning_rate,
#     seed=seed,
#     context_dims=context_dims,
#     data=data
# )
# save_data['train_data'] = train_data
#
# dat.save(
#     save_location=test_name,
#     data=save_data
# )

# # TODO coment when training to view live results
save_data = dat.load(
    save_location=test_name,
    parameters=['Z']
)

zhat = decode_ldn_data(
    Z=save_data['Z'],
    q=q,
    theta=theta,
    theta_p=theta_p
)

errors = calc_shifted_error(
    z=data['state'][:, :3], #  xyz
    zhat=zhat,
    t=data['time'], # TODO make sure this is the same as sim.trange()
    theta_p=theta_p
)

plot_error(theta_p=theta_p, errors=errors)

plot_pred(
    time=data['time'],
    z=data['state'][:, :3],
    zhat=zhat,
    theta_p=theta_p,
    size_out=3,
    gif_name='llp_test.gif',
    animate=False
)
