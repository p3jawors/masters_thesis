import numpy as np
import nengo
from network import LLP

def run_model(
    dt, n_neurons, size_in, q, q_a, q_p, theta, learning_rate,
    seed, c_dims, z_dims, data, radius):
    """
    data needs to have keys for: state, ctrl
    """
    model = nengo.Network()
    with model:
        n_pts = len(data['state'])
        print(f"{n_pts=}")
        print(f"{data['state'].shape}")

        llp = LLP(
                n_neurons=n_neurons,
                size_in=size_in,
                size_out=len(z_dims), #  predict xyz
                q_a=q_a,
                q_p=q_p,
                q=q,
                theta=theta,
                learning=True,
                K=learning_rate,
                seed=seed,
                radius=radius
        )

        def input_func(t):
            index = int((t-dt)/dt)
            state = np.take(data['state_and_error'][index], c_dims)
            # u = data['ctrl'][index]
            u = data['clean_u'][index]
            c = np.hstack((state, u)).tolist()
            return c

        input_node = nengo.Node(
            input_func, size_out=size_in, label='c node')

        nengo.Connection(input_node, llp.c, synapse=None)

        def z_func(t):
            index = int((t-dt)/dt)
            z = np.take(data['state_and_error'][index], z_dims)
            return z

        z_node = nengo.Node(
            z_func, size_out=len(z_dims), label='z_node')

        nengo.Connection(z_node, llp.z, synapse=None)

        Z_probe = nengo.Probe(llp.Z, synapse=None)

    sim = nengo.Simulator(model, dt=dt)
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
    save_data['c_dims'] = c_dims
    save_data['z_dims'] = z_dims

    return save_data
