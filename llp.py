"""
"""
from ldn import LDN, DecodeLDN
import itertools
from numpy.polynomial.legendre import Legendre
import nengo
import numpy as np
import matplotlib.pyplot as plt

blue = '\033[94m'
green = '\033[92m'
red = '\033[91m'
endc = '\033[0m'

class LLP(nengo.Network):
    def __init__(
            self, n_neurons=1000, dimensions=1, q_a=5, q_p=4, q=6, theta=1,
            dt=0.001, learning=True, decoders=None, K=1e-6):
        self.theta = theta
        self.learning = learning
        q_r = q
        print(f"{q=}")
        print(f"{q_a=}")
        print(f"{q_r=}")
        print(f"{q_p=}")

        # stored sizes of nodes for cleaner code
        shapes = {
                'A': (n_neurons, q_a),
                'M': (q_p, dimensions, q),
                'Q': (q_a, q, q_p, q_r),
                'S': (q_r, q_r),
                'z': dimensions,
                'd': (q_a, q_r),
                'QS': (q_a, q_p), # NOTE This is wrong should be 4D
                'MQS': (q_a, dimensions, q),
                'zd': (q_a, q_r, dimensions),
                'D': (n_neurons, q_r, dimensions),
                'Z': (dimensions, q)
        }
        sizes = {}
        for key in shapes:
            sizes[key] = np.prod(shapes[key])

        model = nengo.Network()
        # starting decoders uniformly sampled
        # TODO remove this from constuctor to get variable outputs (diff weights)
        if decoders is None:
            self.decoders = np.zeros((n_neurons, q, dimensions))
            # decoders = nengo.dists.distributions.Uniform(low=0, high=0.1).sample(n_neurons, q, dimensions)
        else:
            self.decoders = decoders

        with model:
            self.input = nengo.Node(size_in=dimensions, size_out=dimensions, label='input')
            # Our context held in an LDN
            ldn_c = nengo.Node(LDN(theta=theta, q=q, size_in=dimensions), label='ldn_context')
            nengo.Connection(self.input, ldn_c, synapse=None)

            # Our neurons that will predict the future on their output connections
            neurons = nengo.Ensemble(
                    n_neurons=n_neurons,
                    dimensions=q*dimensions,
                    radius=np.sqrt(q*dimensions),
                    neuron_type=nengo.RectifiedLinear(),
                    label='neurons')
            nengo.Connection(ldn_c, neurons, synapse=None)

            ldn_a = nengo.Node(LDN(theta=theta, q=q_a, size_in=n_neurons), label='ldn_activities')
            nengo.Connection(neurons.neurons, ldn_a, synapse=None)

            # Constants of learning rule
            """
            Legendre Dimensionality
            q = input
            a = activities
            p = prediction
            r = intermediate in learning rule, r == q
            """
            Q = self.generate_quad_integrals(q_a, q, q_p, q_r)
            S = self.generate_scaling_diagonal(q)
            print('Q: ', Q.shape)
            print('S: ', S.shape)
            QS = np.einsum("qapr, rr->qapr", Q, S) # TODO try using non repeating indices
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
                    print('z: ', z.shape)
                    print('d: ', d.shape)
                    # zd = np.einsum("mq, aq->maq", z, d)
                    zd = np.einsum("m, ar->mar", z, d)
                    print('zd: ', zd.shape)
                    print('M: ', M.shape)
                    print('QS: ', QS.shape)
                    MQS = np.einsum("pmq, qapr->mar", M, QS)
                    print('MQS: ', MQS.shape)
                    error = np.subtract(MQS,  zd)
                    print('MQS-zd: ', error.shape)
                    dD = -K * np.einsum("Na, mar->Nrm", A, error)
                    print('dD: ', dD.shape)
                    print('decoders: ', self.decoders.shape)
                    self.decoders += dD
                    # decoders = np.ravel(self.decoders).tolist()

                y = np.einsum("N, Nqm->qm", a, self.decoders)
                print('output: ', y.shape)
                return np.ravel(y).tolist()

            # prediction in legendre space, stored in an LDN
            z = nengo.Node(
                    llp_learning_rule,
                    size_in=sizes['A'] + sizes['M'] + sizes['z'] + n_neurons,
                    size_out=sizes['Z'],
                    label='Z'
            )

            # store our predictions for use in the learning rule
            ldn_Z = nengo.Node(LDN(theta=theta, q=q_p, size_in=q*dimensions), label='ldn_Z')
            nengo.Connection(z, ldn_Z, synapse=None)

            # Input to learning rule
            nengo.Connection(
                ldn_a,
                z[:sizes['A']],
                synapse=None)

            nengo.Connection(
                ldn_Z,
                z[sizes['A'] : sizes['A']+sizes['M']],
                synapse=0)

            nengo.Connection(
                self.input, # TODO this should actually be size of output since we might use more dims of input to predict the output
                z[sizes['A']+sizes['M'] : sizes['A']+sizes['M']+sizes['z']],
                synapse=None)

            # input to forward pass in legendre space
            nengo.Connection(
                neurons.neurons,
                z[-n_neurons:],
                synapse=None)


            prediction_nodes = []
            theta_ps = [0.1, 0.25, 0.5, 0.75, 0.9]
            # prediction_probes = []
            for r in theta_ps:
                prediction_nodes.append(nengo.Node(DecodeLDN(q, r), label=f'z(r={r})'))
                nengo.Connection(z, prediction_nodes[-1], synapse=None)
                # prediction_probes.append(nengo.Probe(prediction_nodes[-1].output, synapse=0.01))

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
                                # TODO catch these properly
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
    llp = LLP()
    input_node = nengo.Node(
        output=nengo.processes.WhiteSignal(
            high=3, period=3, rms=0.25, y0=0, seed=3),
        label=f"stim",
    )
    nengo.Connection(input_node, llp.input, synapse=None)

sim = nengo.Simulator(model)
with sim:
    sim.run(0.001)
