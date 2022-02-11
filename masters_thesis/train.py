from network import LLP
import numpy as np
import nni
from abr_analyze import DataHandler
import nengo
from utils import decode_ldn_data, calc_shifted_error
from plotting import plot_pred, plot_error

db_name = 'llp_pd'
train_data = '100_targets_0000'
train_params = ['time', 'state', 'ctrl']
test_name = 'train/0000'

dat = DataHandler(db_name)
data = dat.load(
    save_location=train_data,
    parameters=train_params,
)

dt = 0.001
n_neurons = 1000
size_in = 7
q_a = 6
q_p = 6
q = 6
theta = 0.1
theta_p = [0.05, 0.1]
learning_rate = 5e-5*dt
seed = 0
#x, y, z, dx, dy, dz, a, b, g, da, db, dg
context_dims = (0, 1, 2)

# def run_model(data, n_neurons, size_in, q_a, q_p, q, theta, learning_rate, seed, dt):
model = nengo.Network()
with model:
    n_pts = len(data['time'])

    llp = LLP(
            n_neurons=n_neurons,
            size_in=size_in,
            size_out=3, #  predict xyz
            q_a=q_a,
            q_p=q_p,
            q=q,
            theta=theta,
            learning=True,
            K=learning_rate,
            seed=seed,
    )

    def input_func(t):
        index = int((t-dt)/dt)
        state = np.take(data['state'][index], context_dims)
        u = data['ctrl'][index]
        c = np.hstack((state, u)).tolist()
        return c

    input_node = nengo.Node(
        input_func, size_out=size_in, label='input')

    nengo.Connection(input_node, llp.c, synapse=None)
    nengo.Connection(input_node[:3], llp.z, synapse=None)

    Z_probe = nengo.Probe(llp.Z, synapse=None)

#NOTE uncomment when actually training to save data
sim = nengo.Simulator(model)
with sim:
    sim.run(dt*n_pts)

save_data = {}
save_data['Z'] = sim.data[Z_probe]
save_data['n_neurons'] = n_neurons
save_data['size_in'] = size_in
save_data['q_a'] = q_a
save_data['q_p'] = q_p
save_data['q'] = q
save_data['theta'] = theta
save_data['learning_rate'] = learning_rate
save_data['seed'] = seed
save_data['context_dims'] = context_dims
save_data['train_data'] = train_data
save_data['train_params'] = train_params

dat.save(
    save_location=test_name,
    data=save_data
)

# # TODO coment when training to view live results
# save_data = dat.load(
#     save_location=test_name,
#     parameters=['Z']
# )

zhat = decode_ldn_data(
    z=data['state'][:, :3], #  xyz
    Z=save_data['Z'],
    t=data['time'], # TODO make sure this is the same as sim.trange()
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
