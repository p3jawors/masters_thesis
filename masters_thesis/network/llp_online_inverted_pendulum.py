import numpy as np
import sys
import nengo
from abr_analyze import DataHandler
import matplotlib.pyplot as plt
from llp import LLP
from ldn import LDN
from masters_thesis.utils import eval_utils, plotting

n_neurons = 10000
freq = 5
# learning_rate = 5e-4
# t_delays = np.linspace(0, 0.1, 5)
t_delays = [0.03]
learning_rates = {
    '0.01': (1e-3, 10),
    '0.02': (5e-4, 10),
    '0.03': (3.5e-4, 10),
    # '0.04': (3e-4, 40),
}
learning_rate = learning_rates[str(max(t_delays))][0]
q = learning_rates[str(max(t_delays))][1]
# q = 10

dat = DataHandler('codebase_test_set', 'data/databases')
# data = dat.load(save_location='sin5t', parameters=['time', 'state', 'control'])
data = dat.load(save_location='inverted_pendulum_pid', parameters=['time', 'state', 'vel', 'control'])
# data = dat.load(save_location='sin(t**2)', parameters=['time', 'state', 'control'])
dt = 0.001
model = nengo.Network()
state_scale = 1.57
vel_scale = 60
control_scale = 100
with model:
    model.config[nengo.Connection].synapse = None

    def stim_func(t):
        # return np.sin(freq*t), np.cos(t*2*np.pi*freq)
        # return [np.sin(t+np.sin(t)), (np.cos(t) + 1)*np.cos(t+np.sin(t))]
        # print(data['state'])
        # print(data['state'][0])
        # print(data['state'][0].shape[0])
        return [
            # data['state'][0][int(t/dt)%data['state'].shape[0]]/state_scale,
            # data['state'][1][int(t/dt)%data['state'].shape[0]]/vel_scale,
            data['state'][int(t/dt)%data['state'].shape[0]]/state_scale,
            data['vel'][int(t/dt)%data['vel'].shape[0]]/vel_scale,
            data['control'][int(t/dt)%data['control'].shape[0]]/control_scale,
        ]
    c = nengo.Node(stim_func)

    z = nengo.Node(None, size_in=1)
    nengo.Connection(c[0], z)

    llp = LLP(
            n_neurons=n_neurons,
            size_c=3,
            size_z=1,
            q_a=q,
            q_p=q,
            q=q,
            theta=np.max(t_delays),
            learning_rate=learning_rate*dt,
            seed=0,
            verbose=True,
            radius=np.sqrt(3)
    )

    nengo.Connection(z, llp.z, synapse=None)
    nengo.Connection(c, llp.c, synapse=None)

    display = nengo.Node(None, size_in=1+len(t_delays))
    nengo.Connection(z, display[0])
    nengo.Connection(llp.Z, display[1:],
                     transform=LDN(q=q, theta=np.max(t_delays)).get_weights_for_delays(t_delays/np.max(t_delays)))
                     # transform=LDN(q=q, theta=np.max(t_delays)).get_weights_for_delays(0))
    Z = nengo.Probe(llp.Z, synapse=None)
    z = nengo.Probe(z, synapse=None)
    zhat = nengo.Probe(display[1:], synapse=None)
    learning_error = nengo.Probe(llp.learning_rule_error, synapse=None)

if len(sys.argv) > 1:
    sim = nengo.Simulator(model, dt=dt)
    with sim:
        sim.run(29.95)

    errors = eval_utils.calc_shifted_error(
        z=sim.data[z],
        zhat=sim.data[zhat][:, :, np.newaxis],
        dt=dt,
        theta_p=t_delays
    )
    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.title('LLP Decoded Prediction')
    plt.xlabel('Time [sec]')
    plt.plot(sim.trange(), sim.data[z])
    plt.plot(sim.trange(), sim.data[zhat])
    plt.subplot(212)
    plt.title('RMSE of Decoded Prediction')
    plt.xlabel('Time [sec]')
    plt.plot(sim.trange(), errors[:, :, 0])
    # plt.subplot(313)
    # plt.plot(sim.trange(), sim.data[learning_error])
    plt.tight_layout()
    plt.show()

    plotting.plot_pred_1D_with_zoom(
        time=sim.trange(),
        z=sim.data[z],
        zhat=sim.data[zhat],
        theta_p=t_delays,
        xlims=[
            [0, 30],
            [0, 2],
            [24.5, 26.5]
        ],
        ylims=[
            [-1, 1.25],
            [-1, 1.25],
            [0.16, 0.325]
        ],
    )
    # print('ERROR SHAPE: ', sim.data[learning_error].shape)
