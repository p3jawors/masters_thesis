import numpy as np
import nengo
from learn_dyn_sys.network import LearnDynSys
from abr_analyze import DataHandler
from llp import LLP

n_neurons = 1000
freq = 5
learning_rate = 5e-5
t_delays = np.linspace(0, 0.1, 5)
q = 6

dat = DataHandler('codebase_test_set', 'data/databases')
data = dat.load(save_location='sin5t', parameters=['time', 'state', 'control'])
dt = 0.01
model = nengo.Network()
with model:
    model.config[nengo.Connection].synapse = None

    def stim_func(t):
        # return np.sin(t*2*np.pi*freq), np.cos(t*2*np.pi*freq)
        return data['state'][int(t/dt)%data['state'].shape[0]]
    c = nengo.Node(stim_func)

    z = nengo.Node(None, size_in=1)
    nengo.Connection(c[0], z)

    #===LearnDynSys
    learn = LearnDynSys(
        n_neurons=n_neurons,
        size_c=2,
        size_z=1,
        q=q,
        theta=np.max(t_delays),
        learning_rate=learning_rate
    )
    nengo.Connection(z, learn.z, synapse=None)
    nengo.Connection(c, learn.c, synapse=None)

    display = nengo.Node(None, size_in=1+len(t_delays), label='theirs')
    nengo.Connection(z, display[0])
    nengo.Connection(learn.Z, display[1:],
                     transform=learn.get_weights_for_delays(t_delays/learn.theta))


    #===Pawels implementation
    llp = LLP(
            n_neurons=n_neurons,
            size_c=2,
            size_z=1,
            q_a=q,
            q_p=q,
            q=q,
            theta=np.max(t_delays),
            learning_rate=learning_rate*dt,
            seed=0,
            verbose=True,
    )

    nengo.Connection(z, llp.z, synapse=None)
    nengo.Connection(c, llp.c, synapse=None)

    display_2 = nengo.Node(None, size_in=1+len(t_delays), label='mine')
    nengo.Connection(z, display_2[0])
    nengo.Connection(llp.Z, display_2[1:],
                     transform=learn.get_weights_for_delays(t_delays/learn.theta))


    # Difference between implementations
    diff = nengo.Node(size_in=1+len(t_delays))
    nengo.Connection(display, diff, synapse=None)
    nengo.Connection(display_2, diff, synapse=None, transform=-1)
