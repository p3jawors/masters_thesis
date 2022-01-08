# Import numpy and matplotlib
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import nengo
from nengo.processes import WhiteSignal
class IdealDelay(nengo.synapses.Synapse):
    def __init__(self, delay):
        super().__init__()
        self.delay = delay

    def make_state(self, shape_in, shape_out, dt, dtype=None, y0=None):
        return {}

    def make_step(self, shape_in, shape_out, dt, rng, state):
        # buffer the input signal based on the delay length
        buffer = deque([0] * int(self.delay / dt))

        def delay_func(t, x):
            buffer.append(x.copy())
            return buffer.popleft()

        return delay_func

class LDN:

    def __init__(self, n_neurons, q, theta, Tau, delay, eval_points=None, target=None, show_ldn=True, seed=0):
        # Calculate Pade approximants
        A = np.zeros((q, q))
        B = np.zeros((q, 1))
        for i in range(q):
            B[i] = (-1.)**i * (2*i+1)
            for j in range(q):
                A[i,j] = (2*i+1)*(-1 if i<j else (-1.)**(i-j+1))
        A = A / theta
        B = B / theta
        self.Ap = Tau*A + np.eye(A.shape[0])
        self.Bp = Tau*B
        # NOTE
        # A, B, _, _, _ = cont2discrete((A, B, C, D), dt=dt, method="zoh")
        self.q = q
        self.theta = theta
        self.delay = delay
        self.r = self.delay/self.theta

        sim_t = 100  # length of simulation
        seed = 0  # fixed for deterministic results
        rms = 0.3
        freq = 2

        model = nengo.Network(seed=seed)
        with model:
            # NOTE can be node, or ens with linear neurons
            lgdr_polynomials = nengo.Ensemble(
                n_neurons=n_neurons,
                dimensions=self.q
            )
            # stim = nengo.Node(WhiteSignal(7, high=5), size_out=1)
            stim = nengo.Node(
                output=nengo.processes.WhiteSignal(
                    high=freq, period=sim_t, rms=rms, y0=0, seed=seed
                )
            )
            stim_p = nengo.Probe(stim, synapse=Tau)

            nengo.Connection(stim, lgdr_polynomials, transform=self.Bp, synapse=Tau)
            # nengo.Connection(stim, lgdr_polynomials, transform=self.Bp, synapse=None)
            nengo.Connection(lgdr_polynomials, lgdr_polynomials, transform=self.Ap, synapse=Tau)
            # nengo.Connection(lgdr_polynomials, lgdr_polynomials, transform=self.Ap, synapse=0)
            lgdr_polynomials_probe = nengo.Probe(lgdr_polynomials, synapse=Tau)

            ideal_delay = nengo.Probe(stim, synapse=IdealDelay(self.delay))

            def decode_lgdr(t, x):
                def Pi(i):
                    xhat = 0
                    for j in range(0, i+1):
                        # NOTE is choose, not dot product
                        # top factorial / bottom factorial * (same thing inverted)
                        xhat += math.comb(i, j) * math.comb(i+j, j) * (-self.r)**j
                        # print('------')
                        # print('dot: ', dot)
                        # print('scaled: ', scaled)
                        # print('xhat: ', xhat)

                    xhat *= (-1)**i

                    return xhat

                u_delayed = 0
                for ii, lgdr_poly in enumerate(x):
                    u_delayed += Pi(i=ii) * lgdr_poly

                return u_delayed

            ldn = nengo.Node(decode_lgdr, size_in=self.q, size_out=1, label='LDN_out')
            nengo.Connection(lgdr_polynomials, ldn, synapse=None)
            out_p = nengo.Probe(ldn, synapse=Tau)

        sim = nengo.Simulator(model)
        with sim:
            sim.run(4)

        if show_ldn:
            plt.figure()
            plt.title('LMU Internal Representation')
            plt.ylabel('Represented Value')
            plt.xlabel('Time [sec]')
            plt.plot(sim.trange(), sim.data[lgdr_polynomials_probe])

        plt.figure()
        plt.title(f"Ideal vs LDN delay({self.delay}sec)")
        plt.plot(sim.trange(), sim.data[stim_p], label='input')
        plt.plot(sim.trange(), nengo.synapses.Lowpass(0.1).filt(sim.data[ideal_delay]), label='ideal delay')
        plt.plot(sim.trange(), sim.data[out_p], label='ldn delay')
        plt.legend()
        plt.show()
        # xGT = np.ones(len(sim.trange()))
        # xGT[-int(len(sim.trange())/2):] *= -1
        #
        # plt.figure()
        # plt.title('LMU Decision')
        # plt.ylabel('1Hz vs 2Hz Decision')
        # plt.xlabel('Time [sec]')
        # plt.plot(sim.trange(), sim.data[decision_probe], label='Decision')
        # plt.plot(sim.trange(), xGT, label='Ground Truth')
        # plt.legend()
        # plt.show()

        # return sim.data[decision_probe]

LDN(n_neurons=5000, Tau=0.1, q=8, theta=1.0, delay=0.5)
