import numpy as np
import nengo
from network import LLP

def run_model(dt, n_neurons, size_in, q, q_a, q_p, theta, learning_rate, seed, context_dims, data, size_out=3):
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
    save_data['context_dims'] = context_dims

    return save_data
