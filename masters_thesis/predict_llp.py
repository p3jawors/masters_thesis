"""
Run an LLP model on given input data. Creates the connections
for c and z nodes. c func takes c_dims from data['state_and_error'] and
stacks data['clean_u']. z func takes z_dims from data['state_and_error'].
Activities can be optionally saved with the record_activities param.

Returns dictionary of predictions Z
"""
import numpy as np
import nengo
from network import LLP
from learn_dyn_sys.network import LearnDynSys

def run_model(c_state, z_state, dt, record_activities=False, ens_args=None, **llp_args):
    """

    """
    # HACK FOR WEIGHTS
    # with np.load('weights.npz') as data:
    #     llp_args['decoders'] = np.reshape(
    #         data['weights'].T,
    #         (llp_args['n_neurons'], llp_args['q'], z_state.shape[1])
    #     )
    if c_state.ndim == 1:
        c_state = c_state[:, np.newaxis]
    if z_state.ndim == 1:
        z_state = z_state[:, np.newaxis]

    model = nengo.Network()
    with model:
        n_pts = len(c_state)

        if llp_args['model_type'] == 'mine':
            # scaling factor to better align with other model
            # also used to account for different timesteps as
            # this the LLP is implemented in a nengo node so it
            # has to be accounted for manually
            llp_args['learning_rate'] *= dt
            del llp_args['model_type']
            llp = LLP(
                ens_args=ens_args,
                **llp_args,
            )
        elif llp_args['model_type'] == 'other':
            llp = LearnDynSys(
                size_c=llp_args['size_c'],
                size_z=llp_args['size_z'],
                q=llp_args['q'],
                theta=llp_args['theta'],
                n_neurons=llp_args['n_neurons'],
                learning_rate=llp_args['learning_rate'],
                neuron_type=llp_args['neuron_model'](),
                **ens_args,
            )

        def input_func(t):
            index = int((t-dt)/dt)
            c = c_state[index]
            return c

        input_node = nengo.Node(
            input_func, size_out=c_state.shape[1], label='c node')

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

    return results
