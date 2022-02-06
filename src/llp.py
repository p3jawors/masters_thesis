"""
"""
from ldn import LDN, DecodeLDN
import itertools
from numpy.polynomial.legendre import Legendre
import nengo
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

blue = '\033[94m'
green = '\033[92m'
red = '\033[91m'
endc = '\033[0m'

class LLP(nengo.Network):
    def __init__(
            self, n_neurons=1000, size_in=1, size_out=1, q_a=6, q_p=6, q=6, theta=1.0,
            dt=0.001, learning=True, decoders=None, K=5e-8, seed=0, verbose=False, theta_p=None):

        self.theta = theta
        self.learning = learning
        K = dt*K/n_neurons  #NOTE Terrys implementation had this scaling
        # K = K/n_neurons
        q_r = q

        if verbose:
            print(f"{q=}")
            print(f"{q_a=}")
            print(f"{q_r=}")
            print(f"{q_p=}")

        # stored sizes of nodes for cleaner code
        shapes = {
                'A': (n_neurons, q_a),
                'M': (q_p, size_out, q),
                'Q': (q_a, q, q_p, q_r),
                'S': (q_r, q_r),
                'z': size_out,
                'Z': (size_out, q),
                'd': (q_a, q_r),
                'QS': (q, q_a, q_p, q_r),
                'MQS': (q_a, size_out, q_r),
                'zd': (q_a, q_r, size_out),
                'D': (n_neurons, q_r, size_out),
        }
        sizes = {}
        for key in shapes:
            sizes[key] = np.prod(shapes[key])

        model = nengo.Network()
        # starting decoders uniformly sampled
        # TODO remove this from constuctor to get variable outputs (diff weights)
        if decoders is None:
            self.decoders = np.zeros((n_neurons, q, size_out))
            # decoders = nengo.dists.distributions.Uniform(low=0, high=0.1).sample(n_neurons, q, size_out)
        else:
            self.decoders = decoders

        with model:
            self.input = nengo.Node(size_in=size_in, size_out=size_in, label='c')
            self.z = nengo.Node(size_in=size_out, size_out=size_out, label='z')

            # Our neurons that will predict the future on their output connections
            neurons = nengo.Ensemble(
                    n_neurons=n_neurons,
                    dimensions=size_in,
                    #radius=np.sqrt(size_in),
                    # neuron_type=nengo.RectifiedLinear(),
                    neuron_type=nengo.LIFRate(),
                    label='neurons')
            nengo.Connection(self.input, neurons, synapse=None)

            ldn_a = nengo.Node(LDN(theta=self.theta, q=q_a, size_in=n_neurons), label='ldn_activities')
            nengo.Connection(neurons.neurons, ldn_a, synapse=None)

            # Constants of learning rule
            """
            Legendre Dimensionality
            q = input
            a = activities
            p = prediction
            r = result, r == q
            """
            Q = self.generate_quad_integrals(q_a, q, q_p, q_r)
            S = self.generate_scaling_diagonal(q)
            if verbose:
                print('Q: ', Q.shape)
                print('S: ', S.shape)
            QS = np.einsum("qapr, Rr->qapr", Q, S) # TODO try using non repeating indices
            d = self.generate_delta_identity(q_a, q_r)


            def llp_learning_rule(t, x):
                """
                the learning rule is essentially the delta rule, but in the Legendre domain

                dD = -KA(MQS - zd)

                    dD: the change in decoders for the legendre domain decoders
                        shape: n_neurons*q*output_dims
                    -K: learning rate
                    A: legendre space activities
                        shape: n_neurons*q_a
                    M: legendre space representation of our legendre space predictions of
                        the future theta time
                        shape: q_p*output_dims*q
                    Q: 4D tensor of constants pulled out of learning rule integral
                        precomputed in generate_quad_integrals()
                        shape: q_a*q_p*q*q
                    S: scaling factor
                        precomputed diagonal matrix
                        shape: q*q
                    z: our ground truth value for the future theta q values
                        shape: output_dims
                    d: identity function when q_a=q
                        shape: q_a*q

                """
                # TODO figure out indexing into input vector
                A = x[:sizes['A']]
                A = np.reshape(A, shapes['A'])

                M = x[sizes['A'] : sizes['A']+sizes['M']]
                M = np.reshape(M, shapes['M'])

                z = x[sizes['A']+sizes['M'] : sizes['A']+sizes['M']+sizes['z']]

                a = x[-n_neurons:]

                if self.learning:
                    # zd = np.einsum("mq, aq->maq", z, d)
                    zd = np.einsum("m, ar->mar", z, d)
                    MQS = np.einsum("pmq, qapr->mar", M, QS)
                    error = np.subtract(MQS,  zd)
                    dD = -K * np.einsum("Na, mar->Nrm", A, error)
                    self.decoders += dD
                    # decoders = np.ravel(self.decoders).tolist()
                    if verbose and t < dt:
                        print('z: ', z.shape)
                        print('d: ', d.shape)
                        print('zd: ', zd.shape)
                        print('M: ', M.shape)
                        print('QS: ', QS.shape)
                        print('MQS: ', MQS.shape)
                        print('MQS-zd: ', error.shape)
                        print('dD: ', dD.shape)
                        print('decoders: ', self.decoders.shape)

                y = np.einsum("N, Nqm->qm", a, self.decoders)
                if verbose and t < dt:
                    print('output: ', y.shape)
                return np.ravel(y).tolist()

            # our prediction of the legendre coefficients that predict the future theta of our input
            self.Z = nengo.Node(
                    llp_learning_rule,
                    size_in=sizes['A'] + sizes['M'] + sizes['z'] + n_neurons,
                    size_out=sizes['Z'],
                    label='Z (learn)'
            )

            # store our predictions in an LDN for use in the learning rule
            # this is an ldn storing the values of an ldn
            ldn_Z = nengo.Node(LDN(theta=self.theta, q=q_p, size_in=q*size_out), label='ldn_Z')
            nengo.Connection(self.Z, ldn_Z, synapse=None)

            # == Input to learning rule ==
            # LDN of activities
            nengo.Connection(
                ldn_a,
                self.Z[:sizes['A']],
                synapse=None)

            # LDN of predictions
            nengo.Connection(
                ldn_Z,
                self.Z[sizes['A'] : sizes['A']+sizes['M']],
                synapse=0)

            # the current state of the dimension we are trying to predict the future window of
            nengo.Connection(
                self.z,
                self.Z[sizes['A']+sizes['M'] : sizes['A']+sizes['M']+sizes['z']],
                synapse=None)

            # our current neural activity
            nengo.Connection(
                neurons.neurons,
                self.Z[-n_neurons:],
                synapse=None)


            # print('LDN Decode out: ', LDN(theta=self.theta, q=q_p, size_in=1).get_weights_for_delays(np.asarray(theta_p)/self.theta).shape)
            # print(f'Want to decode {size_out} values')
            # print(f'Encoding with {q_p} legendre coefficients')
            # print(f"Shape of Z is {shapes['Z']}")
            if theta_p is not None:
                self.zhat = nengo.Node(size_out=size_out*len(theta_p), size_in=size_out*len(theta_p))
                nengo.Connection(self.Z, self.zhat,
                    transform=np.tile(
                        LDN(theta=self.theta, q=q_p, size_in=1).get_weights_for_delays(np.asarray(theta_p)/self.theta), size_out)
                )


    # NOTE: credit LLP patent for this function
    def generate_quad_integrals(self, q_a, q, q_p, q_r):
        def quad(i, j, m, n):
            li, lj, lm, ln = (Legendre([0] * k + [1]) for k in (i, j, m, n))
            L = (li * lj * lm * ln).integ()
            # the desired result is (L(1) - L(-1)) / 2 (as this is for non-shifted Legendre)
            #  but since L(1) == -L(-1), this is just L(1)
            return L(1)
        qs = [q, q_a, q_p, q_r]
        w = np.zeros((qs[0], qs[1], qs[2], qs[3]))
        for i in range(qs[0]):
            for j in range(i, qs[1]):
                for m in range(j, qs[2]):
                    for n in range(m, qs[3]):
                        # skip indices guranteed to be 0
                        if (i+j+m-n >= 0) and ((i+j+m-n) % 2 == 0):
                            v = quad(i, j, m, n)
                            for index in itertools.permutations([i, j, m, n]):
                                # TODO catch these properly with somthing like filter()
                                try:
                                    w[index] = v
                                except Exception as e:
                                    pass
        return w


    def generate_scaling_diagonal(self, q):
        S = np.zeros((q, q))
        for i in range(0, q):
            S[i, i] = 2*i + 1
        print('S: ', S)
        return S

    def generate_delta_identity(self, q_a, q_p):
        # TODO is this the right order?
        i = min(q_a, q_p)
        d = np.zeros((q_a, q_p))
        for ii in range(0, i):
            d[ii, ii] = 1
        print('delta: ', d)
        return d


model = nengo.Network()
with model:
    freq = 5
    learning_rate = 5e-5
    theta_p = np.linspace(0, 0.1, 5)
    q = 6
    size_out = 1
    theta = max(theta_p)
    dt = 0.001

    llp = LLP(
            n_neurons=2000,
            size_in=2,
            size_out=size_out,
            q_a=q,
            q_p=q,
            q=q,
            theta=theta,
            dt=dt,
            learning=True,
            decoders=None,
            K=learning_rate,
            seed=0,
            verbose=True,
            theta_p=theta_p
    )

    context = nengo.Node(
            lambda t: [np.sin(t*np.pi*2*freq), np.cos(t*np.pi*2*freq)],
            size_in=0, size_out=2)

    nengo.Connection(context, llp.input, synapse=None)
    nengo.Connection(context[0], llp.z, synapse=None)

    z_probe = nengo.Probe(llp.z, synapse=None)
    zhat_probe = nengo.Probe(llp.zhat, synapse=None)


if __name__ == '__main__':
    sim = nengo.Simulator(model)
    with sim:
        sim.run(10)

    animate = True
    window = theta*5
    step = dt*10

    plt.figure()
    plt.title('Predictions over time')
    plt.plot(sim.trange(), sim.data[z_probe])
    plt.plot(sim.trange(), sim.data[zhat_probe])
    plt.legend(['z'] + [str(tp) for tp in theta_p])

    plt.figure()
    axs = []
    for ii in range(0, size_out):
        axs.append(plt.subplot(ii+1, 1, size_out))
        axs[ii].plot(sim.trange(), sim.data[zhat_probe])

        plt.gca().set_prop_cycle(None)
        for pred in theta_p:
            axs[ii].plot(sim.trange()-pred, sim.data[z_probe].T[ii], linestyle='--')

        axs[ii].legend(
            ['zhat at: ' + str(round(tp, 3)) for tp in theta_p]
            + ['z shifted: ' + str(round(tp, 3)) for tp in theta_p],
            loc=1)

    if animate:
        start = 0.0
        stop = window
        ss = 0
        filenames = []
        while stop <= sim.trange()[-1]:
            for ax in axs:
                ax.set_xlim(start, stop)
            filename = f".cache/img_{ss:08d}.jpg"
            filenames.append(filename)
            plt.savefig(filename)
            start += step
            stop += step
            ss += 1

        with imageio.get_writer('llp.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                os.remove(filename)
    plt.show()
