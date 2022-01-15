import numpy as np
import nengo
from llp import LLP
from ldn import LDN

freq = 5
learning_rate = 5e-5
t_delays = np.linspace(0, 0.1, 5)
q = 6

model = nengo.Network()
with model:
    model.config[nengo.Connection].synapse = None

    def stim_func(t):
        return np.sin(t*2*np.pi*freq), np.cos(t*2*np.pi*freq)
    c = nengo.Node(stim_func)

    z = nengo.Node(None, size_in=1)
    nengo.Connection(c[0], z)

    llp = LLP(
            n_neurons=1000,
            size_in=2,
            size_out=1,
            q_a=q,
            q_p=q,
            q=q,
            theta=np.max(t_delays),
            dt=0.001,
            learning=True,
            K=learning_rate,
            seed=0,
            verbose=True,
            output_r=t_delays
    )

    f = 5

    nengo.Connection(c, llp.input, synapse=None)
    nengo.Connection(z, llp.z, synapse=None)

    display = nengo.Node(None, size_in=1+len(t_delays))
    nengo.Connection(z, display[0])
    nengo.Connection(llp.Z, display[1:],
                     transform=LDN(q=q, theta=np.max(t_delays)).get_weights_for_delays(t_delays/np.max(t_delays)))


