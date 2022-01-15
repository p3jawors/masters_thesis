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
            self, n_neurons=1000, size_in=1, size_out=1, q_a=6, q_p=6, q=6, theta=1.0,
            dt=0.001, learning=True, decoders=None, K=5e-8, seed=0, verbose=False, output_r=None):

        self.theta = theta
        self.learning = learning
        K = dt*K/n_neurons  #NOTE Terrys implementation had this scaling
        # K = K/n_neurons
        q_r = q

        if output_r is None:
            output_r = self.theta*np.array([0.25, 0.5, 0.75])

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
            self.input = nengo.Node(size_in=size_in, size_out=size_in, label='input')
            self.z = nengo.Node(size_in=size_out, size_out=size_out, label='z')
            # self.gt = nengo.Node(size_in=1, size_out=1, label='gt')

            # Our neurons that will predict the future on their output connections
            neurons = nengo.Ensemble(
                    n_neurons=n_neurons,
                    dimensions=size_in,
                    #radius=np.sqrt(size_in),
                    # neuron_type=nengo.RectifiedLinear(),
                    neuron_type=nengo.LIFRate(),
                    label='neurons')
            nengo.Connection(self.input, neurons, synapse=None)

            ldn_a = nengo.Node(LDN(theta=theta, q=q_a, size_in=n_neurons), label='ldn_activities')
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
            ldn_Z = nengo.Node(LDN(theta=theta, q=q_p, size_in=q*size_out), label='ldn_Z')
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


            # prediction_node = nengo.Node(None, size_in=len(output_r) + 1, label='Predictions')
            #
            # # TODO generalize the connection sizes, currently assuming dim 0 is the state we are predicting
            # nengo.Connection(self.z, prediction_node[0])
            # nengo.Connection(
            #         Z,
            #         prediction_node[1:],
            #         transform=LDN(q=q, theta=self.theta).get_weights_for_delays(output_r))


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
    theta = 0.5
    output_r = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    llp = LLP(
            n_neurons=1000,
            size_in=2,
            size_out=1,
            q_a=6,
            q_p=6,
            q=6,
            theta=theta,
            dt=0.001,
            learning=True,
            decoders=None,
            K=5e-5,
            seed=0,
            verbose=True,
            output_r=output_r
    )

    f = 5
    input_node = nengo.Node(
            lambda t: [np.sin(t*np.pi*2*f), np.cos(t*np.pi*2*f)],
            size_in=0, size_out=2)

    # gt_node = nengo.Node(
    #         lambda t: [np.sin((t+0.25)*np.pi*2*f)],
    #         size_in=0, size_out=1)

    nengo.Connection(input_node, llp.input, synapse=None)
    nengo.Connection(input_node[0], llp.z, synapse=None)
    # nengo.Connection(gt_node, llp.gt, synapse=None)

# sim = nengo.Simulator(model)
# with sim:
#     sim.run(0.001)
