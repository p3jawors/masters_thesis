"""
Run an LLP model on given input data. Creates the connections
for c and z nodes. c func takes c_dims from data['state_and_error'] and
stacks data['clean_u']. z func takes z_dims from data['state_and_error'].
Activities can be optionally saved with the record_activities param.

Returns dictionary of predictions Z
"""
#TODO
"""
Update to take in state and u data, avoid having to index into data
"""
import numpy as np
import nengo
from network import LLP
from learn_dyn_sys.network import LearnDynSys

def run_model(c_state, z_state, control, dt, record_activities=False, **llp_args):

    # dt, n_neurons, size_in, q, q_a, q_p, theta, learning_rate, radius,
    # seed, state_data, control_data, record_activities=False):
    """

    """
    if c_state.ndim == 1:
        c_state = c_state[:, np.newaxis]
    if z_state.ndim == 1:
        z_state = z_state[:, np.newaxis]
    if control.ndim == 1:
        control = control[:, np.newaxis]

    model = nengo.Network()
    with model:
        n_pts = len(c_state)
        print(f"{n_pts=}")

        if llp_args['model_type'] == 'mine':
            # scaling factor to better align with other model
            llp_params['learning_rate'] *= dt
            llp = LLP(**llp_args)
        if llp_args['model_type'] == 'other':
            llp = LearnDynSys(
                size_c=llp_args['size_in'],
                size_z=llp_args['size_out'],
                q=llp_args['q'],
                theta=llp_args['theta'],
                n_neurons=llp_args['n_neurons'],
                learning_rate=llp_args['learning_rate'],
                neuron_type=llp_args['neuron_model']()
            )

        def input_func(t):
            index = int((t-dt)/dt)
            state = c_state[index]
            u = control[index]
            c = np.hstack((state, u)).tolist()
            return c

        input_node = nengo.Node(
            input_func, size_out=c_state.shape[1]+control.shape[1], label='c node')

        nengo.Connection(input_node, llp.c, synapse=None)

        def z_func(t):
            index = int((t-dt)/dt)
            z = z_state[index]
            return z

        z_node = nengo.Node(
            z_func, size_out=z_state.shape[1], label='z_node')

        nengo.Connection(z_node, llp.z, synapse=None)

        Z_probe = nengo.Probe(llp.Z, synapse=None)
        if record_activities:
            activity_probe = nengo.Probe(llp.ens.neurons, synapse=None)

    sim = nengo.Simulator(model, dt=dt)
    with sim:
        sim.run(dt*n_pts)

    results = {}
    if record_activities:
        results['activities'] = sim.data[activity_probe]
    results['Z'] = sim.data[Z_probe]
    # save_data['n_neurons'] = n_neurons
    # save_data['size_in'] = size_in
    # save_data['q_a'] = q_a
    # save_data['q_p'] = q_p
    # save_data['q'] = q
    # save_data['theta'] = theta
    # save_data['learning_rate'] = learning_rate
    # save_data['seed'] = seed
    # save_data['c_dims'] = c_dims
    # save_data['z_dims'] = z_dims
    # save_data['dt'] = dt
    # save_data['radius'] = radius

    return results
