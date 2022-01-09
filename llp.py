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
    def __init__(self, n_neurons=1000, dimensions=1, q=6, theta=1, dt=0.001):
        dimensions = 1
        if dimensions != 1:
            raise NotImplementedError
        self.theta = theta
        # Legendre dimensionality
        q_a = q
        q_p = q
        q_r = q

        # stored sizes of nodes for cleaner code
        shapes = {
                'A': (n_neurons, q_a),
                'M': (q_p, dimensions, q),
                'Q': (q_a, q, q_p, q_r),
                'S': (q_r, q_r),
                'z': dimensions,
                'd': (q_a, q_r),
                'QS': (q_a, q_p),
                'MQS': (q_a, dimensions, q),
                'zd': (q_a, q_r, dimensions),
                'D': (n_neurons, q_r, dimensions),
                'Z': (dimensions, q)
        }
        sizes = {}
        for key in shapes:
            sizes[key] = np.prod(shapes[key])

        model = nengo.Network()
        # true to adjust encoders
        # TODO add weight saving and loading
        train = True
        # starting decoders uniformly sampled
        # decoders = nengo.dists.distributions.Uniform(low=0, high=0.1).sample(n_neurons, q, dimensions)
        self.decoders = np.zeros((n_neurons, q, dimensions))
        with model:
            # TODO fix this for dimensions > 1
            input_node = nengo.Node(
                output=nengo.processes.WhiteSignal(
                    high=3, period=3, rms=0.25, y0=0, seed=3),
                label=f"stim",
                # size_in=dimensions,
                # size_out=dimensions
            )

            # def gt_func(t):
            #     return input_func(t+self.theta)
            # gt_node = nengo.Node(gt_func, size_in=0, size_out=dimensions*q, label='gt')

            # Our context held in an LDN
            ldn_c = nengo.Node(LDN(theta=theta, q=q, size_in=dimensions), label='ldn_context')
            nengo.Connection(input_node, ldn_c, synapse=None)

            # Our neurons that will predict the future on their output connections
            # TODO when dimensions is no longer 1, turn this into an ensemble array so
            # that each input dim has it's own ensemble representing the respective
            # q legendre coefficients
            neurons = nengo.Ensemble(
                    n_neurons=n_neurons,
                    dimensions=q*dimensions,
                    radius=np.sqrt(q*dimensions),
                    neuron_type=nengo.RectifiedLinear(),
                    label='neurons')
            nengo.Connection(ldn_c, neurons)

            ldn_a = nengo.Node(LDN(theta=theta, q=q_a, size_in=n_neurons), label='ldn_activities')
            nengo.Connection(neurons.neurons, ldn_a)

            # Constants of learning rule
            Q = generate_quad_integrals(q)
            S = generate_scaling_diagonal(q)
            """
            Legendre Dimensionality
            q = input
            a = activities
            p = prediction
            r = intermediate in learning rule, r == q
            """
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
                # TODO figure out indexing into input vector
                A = x[:sizes['A']]
                A = np.reshape(A, shapes['A'])

                M = x[sizes['A'] : sizes['A']+sizes['M']]
                M = np.reshape(M, shapes['M'])

                z = x[sizes['A']+sizes['M'] : sizes['A']+sizes['M']+sizes['d']]

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
                decoders = np.ravel(self.decoders).tolist()
                return decoders

            def forward(t, x):
                A = x[:n_neurons]
                # A = np.reshape(A, shapes['A'])
                decoders = x[n_neurons:]
                decoders = np.reshape(decoders, shapes['D'])

                y = np.einsum("N, Nqm->qm", A, decoders)
                # print(y.shape)
                return np.ravel(y.tolist())

            # prediction in legendre space, stored in an LDN
            learning_rule = nengo.Node(
                    llp_learning_rule,
                    size_in=sizes['A'] + sizes['M'] + sizes['z'],
                    size_out=sizes['D'],
                    label='learning_rule'
            )

            z = nengo.Node(
                    forward,
                    size_in=n_neurons + sizes['D'],
                    size_out=sizes['Z'],
                    label='Z'
            )

            # store our predictions for use in the learning rule
            ldn_Z = nengo.Node(LDN(theta=theta, q=q_p, size_in=q*dimensions), label='ldn_Z')
            nengo.Connection(z, ldn_Z, synapse=None)

            # Input to learning rule
            nengo.Connection(
                ldn_a,
                learning_rule[:sizes['A']])

            nengo.Connection(
                ldn_Z,
                learning_rule[sizes['A'] : sizes['A']+sizes['M']])

            nengo.Connection(
                input_node,
                learning_rule[sizes['A']+sizes['M']:])

            # input to forward pass in legendre space
            nengo.Connection(
                neurons.neurons,
                z[:n_neurons])

            nengo.Connection(
                learning_rule,
                z[n_neurons:])


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
