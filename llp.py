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

# NOTE: credit LLP patent for this function
# TODO update to use q, q_a, and q_p
def generate_quad_integrals(q):
    def quad(i, j, m, n):
        li, lj, lm, ln = (Legendre([0] * k + [1]) for k in (i, j, m, n))
        L = (li * lj * lm * ln).integ()
        # the desired result is (L(1) - L(-1)) / 2 (as this is for non-shifted Legendre)
        #  but since L(1) == -L(-1), this is just L(1)
        return L(1)  

    w = np.zeros((q, q, q, q))
    for i in range(q):
        for j in range(i, q):
            for m in range(j, q):
                for n in range(m, q):
                    # skip indices guranteed to be 0
                    if (i+j+m-n >= 0) and ((i+j+m-n) % 2 == 0):
                        v = quad(i, j, m, n)
                        for index in itertools.permutations([i, j, m, n]):
                            w[index] = v
    return w


def generate_scaling_diagonal(q):
    S = np.zeros((q, q))
    for i in range(0, q):
        S[i, i] = 2*i + 1
    return S

def generate_delta_identity(q_a, q_p):
    # TODO is this the right order?
    i = min(q_a, q_p)
    d = np.zeros((q_a, q_p))
    for ii in range(0, i):
        d[ii, ii] = 1
    return d

class LLP(nengo.Network):
    def __init__(self):#, n_neurons=1000, dimensions=1, q=6, theta=1, dt=0.001)
        n_neurons = 1000
        dimensions = 1
        theta = 1
        dt = 0.001
        # Legendre dimensionality
        q = 6
        q_a = 6
        q_p = 6
        # get the number of steps in our window, this tells us how much data to cut
        # off the start for our GT, and how much to cut off the end for our state data
        window_size = int(theta/dt)

        # Generate or load training data
        # state_data = np.load("training_2022_01_03-12_10_57.npz")['q']
        data = np.sin(np.arange(0, np.pi*10, dt))#[:, np.newaxis]

        # Pass input to LDN to generate the Legendre coefficient to train off
        # We are trying to predict theta into the future, so remove the first theta seconds
        # from the target data, and the last theta seconds for the eval points
        state = LDN(theta=theta, q=q).apply(data[:-window_size][:, np.newaxis])
        gt_predictions = LDN(theta=theta, q=q).apply(data[window_size:][:, np.newaxis])

        model = nengo.Network()
        # true to adjust encoders
        # TODO add weight saving and loading
        train = True
        # starting decoders uniformly sampled
        # decoders = nengo.dists.distributions.Uniform(low=0, high=0.1).sample(n_neurons, q, dimensions)
        self.decoders = np.zeros((n_neurons, q, dimensions))
        with model:
            def input_func(t, x):
                return np.sin(t)
                # return data[(int(t/dt))%len(data)]
            input_node = nengo.Node(input_func, size_in=dimensions, size_out=dimensions, label='input')

            def gt_func(t):
                return gt_predictions[(int(t/dt))%len(data)]
            gt_node = nengo.Node(gt_func, size_in=0, size_out=dimensions*q, label='gt')

            # Our context held in an LDN
            ldn_c = nengo.Node(LDN(theta=theta, q=q, size_in=dimensions), label='ldn_c')
            nengo.Connection(input_node, ldn_c, synapse=None)

            # Our neurons that will predict the future on their output connections
            # TODO when dimensions is no longer 1, turn this into an ensemble array so
            # that each input dim has it's own ensemble representing the respective
            # q legendre coefficients
            neurons = nengo.Ensemble(
                    n_neurons=n_neurons,
                    dimensions=q*dimensions,
                    # neuron_type=nengo.Direct(),
                    label='neurons')
            nengo.Connection(ldn_c, neurons)
            # dummy_out = nengo.Node(size_in=q, size_out=q, label='dummy_out')
            # nengo.Connection(neurons, dummy_out)

            ldn_a = nengo.Node(LDN(theta=theta, q=q_a, size_in=n_neurons), label='ldn_a')
            nengo.Connection(neurons.neurons, ldn_a)

            # Constants of learning rule
            Q = generate_quad_integrals(q)
            S = generate_scaling_diagonal(q)
            # q = input legendre q
            # a = q_a
            # p = q_p
            # r = q_r = q
            QS = np.einsum("qapr, qr->ap", Q, S)
            d = generate_delta_identity(q_a, q_p)
            K = 1e-6


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
                learning = True
                # TODO figure out indexing into input vector
                A = x[:n_neurons*q_a]
                A = np.reshape(A, (n_neurons, q_a))

                M = x[n_neurons*q_a:n_neurons*q_a+q_p*q*dimensions]
                M = np.reshape(M, (q_p, dimensions, q))

                z = x[n_neurons*q_a+q_p*q*dimensions:n_neurons*q_a+q_p*q*dimensions + dimensions]
                # z = x[n_neurons*q_a+q_p*q*dimensions:n_neurons*q_a+q_p*q*dimensions + dimensions*q]
                # z = np.reshape(z, (dimensions, q))

                if learning:
                    # print('z: ', z.shape)
                    # print('d: ', d.shape)
                    # zd = np.einsum("mq, aq->maq", z, d)
                    zd = np.einsum("m, aq->maq", z, d)
                    # print('zd: ', zd.shape)
                    # print('M: ', M.shape)
                    # print('QS: ', QS.shape)
                    MQS = np.einsum("pmq, ap->maq", M, QS)
                    # print('MQS: ', MQS.shape)
                    error = MQS - zd
                    # print('MQS-zd: ', error.shape)
                    dD = -K * np.einsum("Na, maq->Nqm", A, error)
                    # print('dD: ', dD.shape)
                    self.decoders += dD

                y = np.einsum("Nq, Nqm->qm", A, self.decoders)
                # print(y.shape)
                return np.ravel(y.tolist())

            # prediction in legendre space, stored in an LDN
            z = nengo.Node(
                    llp_learning_rule,
                    # size_in=n_neurons*q_a + q_p*q*dimensions + dimensions*q,
                    size_in=n_neurons*q_a + q_p*q*dimensions + dimensions,
                    size_out=dimensions*q,
                    label='z')

            # store our predictions for use in the learning rule
            ldn_z = nengo.Node(LDN(theta=theta, q=q_p, size_in=q*dimensions), label='ldn_z')
            nengo.Connection(z, ldn_z, synapse=None)

            # where our learning will happen
            nengo.Connection(
                ldn_a,
                z[:n_neurons*q_a])

            nengo.Connection(
                ldn_z,
                z[n_neurons*q_a:n_neurons*q_a + q_p*q*dimensions])

            nengo.Connection(
                # gt_node,
                input_node,
                z[n_neurons*q_a+q_p*q*dimensions:n_neurons*q_a+q_p*q*dimensions + dimensions])


            prediction_nodes = []
            theta_ps = [0.1, 0.25, 0.5, 0.75, 0.9]
            # prediction_probes = []
            for r in theta_ps:
                prediction_nodes.append(nengo.Node(DecodeLDN(q, r), label=f'LDN_out_{r}'))
                # nengo.Connection(gt_node, prediction_nodes[-1], synapse=None)
                nengo.Connection(z, prediction_nodes[-1], synapse=None)
                # prediction_probes.append(nengo.Probe(prediction_nodes[-1].output, synapse=0.01))

            # p_input = nengo.Probe(input_node)
            # p_ldn_c = nengo.Probe(ldn_c)
            # p_z = nengo.Probe(z, synapse=0.01)
        # plt.figure(figsize=(12,4))
        # plt.plot(sim.trange(), sim.data[p_ldn_c], label='ldn_c')
        # plt.plot(sim.trange(), sim.data[p_z], label='output', linestyle='--')
        # plt.legend()
        # plt.show()
        #
model = nengo.Network()
with model:
    llp = LLP()

sim = nengo.Simulator(model)
with sim:
    sim.run(0.001)
