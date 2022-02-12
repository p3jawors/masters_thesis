"""
"""
from network.ldn import LDN
import itertools
from numpy.polynomial.legendre import Legendre
import nengo
import numpy as np

blue = '\033[94m'
green = '\033[92m'
red = '\033[91m'
endc = '\033[0m'

class LLP(nengo.Network):
    # Constants of learning rule
    """
    Parameters
    ----------
    n_neurons: int
        the number of neurons to predict the legendre coefficients
    size_in: int
        the dimensionality of the context signal
    size_out: int
        the dimensionality of the context signal to be predicted
    q: int
        number of legendre polynomials for the prediction
    q_a: int
        number of legendre polynomials for storing neural activity
    q_p: int
        number of legendre polynomials for storing our legendre coefficients
        of our prediction
    theta: float
        number of seconds into the future to predict
    learning: bool
        toggles learning on and off
    decoders: float array, Optional (Default: np.zeros)
        the decoders to scale our activities by to get our q legendre coefficients
    K: float
        learning rate
    verbose:
        True for status print outs
    theta_p: list of floats, Optional (Default: None)
        Times into the future to decode predictions for.
        When not None, will create nodes for each theta p. Each node will decode
        the m state at times theta_p from the q legendre coefficients
    neuron_model: nengo neuron model, Optional (Default: nengo.LIFRate)
        nengo neuron model to use for prediction
    **ens_params: dict
        extra ensemble parameters
    """

    def __init__(
            self, n_neurons=1000, size_in=1, size_out=1, q_a=6, q_p=6, q=6, theta=1.0,
            learning=True, decoders=None, K=5e-8, verbose=False, theta_p=None,
            neuron_model=nengo.LIFRate, **ens_params):

        # if neuron_model is None:
        #     if verbose:
        #         print('Using default neuron model of nengo.LIFRate')
        #     neuron_model = nengo.LIFRate

        self.theta = theta
        self.learning = learning
        K = K/n_neurons
        # for clarity, q_r is used even though q_r == q
        q_r = q

        # Calculate constant portions of learning rule
        Q = self.generate_quad_integrals(q_a, q, q_p, q_r)
        S = self.generate_scaling_diagonal(q)
        QS = np.einsum("qapr, Rr->qapr", Q, S)
        d = self.generate_delta_identity(q_a, q_r)

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
        if verbose:
            print("---SHAPES---")
            for key, val in shapes.items():
                print(f"- {key}: {val}")
            print(f"{q=}")
            print(f"{q_a=}")
            print(f"{q_r=}")
            print(f"{q_p=}")

        # stored sizes of nodes for cleaner code
        sizes = {}
        for key in shapes:
            sizes[key] = np.prod(shapes[key])

        model = nengo.Network()
        if decoders is None:
            print('No decoders passed in, starting from zeros')
            self.decoders = np.zeros((n_neurons, q, size_out))
        else:
            self.decoders = decoders

        with model:
            # Our context, and state we are predicting the future window of
            self.c = nengo.Node(size_in=size_in, size_out=size_in, label='c')
            self.z = nengo.Node(size_in=size_out, size_out=size_out, label='z')

            # Our neurons that will predict the future on their output connections
            neurons = nengo.Ensemble(
                    n_neurons=n_neurons,
                    dimensions=size_in,
                    neuron_type=neuron_model(),
                    label='neurons',
                    **ens_params)
            nengo.Connection(self.c, neurons, synapse=None)

            ldn_a = nengo.Node(LDN(theta=self.theta, q=q_a, size_in=n_neurons), label='ldn_activities')
            nengo.Connection(neurons.neurons, ldn_a, synapse=None)

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
                a = x[-n_neurons:]

                if self.learning:
                    A = x[:sizes['A']]
                    A = np.reshape(A, shapes['A'])

                    M = x[sizes['A'] : sizes['A']+sizes['M']]
                    M = np.reshape(M, shapes['M'])

                    z = x[sizes['A']+sizes['M'] : sizes['A']+sizes['M']+sizes['z']]

                    zd = np.einsum("m, ar->mar", z, d)
                    MQS = np.einsum("pmq, qapr->mar", M, QS)
                    error = np.subtract(MQS,  zd)
                    dD = -K * np.einsum("Na, mar->Nrm", A, error)
                    self.decoders += dD

                y = np.einsum("N, Nqm->qm", a, self.decoders)
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


            # For convenience, can output the legendre decoded output at various theta_p values
            if theta_p is not None:
                self.zhat = nengo.Node(size_out=size_out*len(theta_p), size_in=size_out*len(theta_p))
                for ii in range(0, size_out):
                    nengo.Connection(
                        self.Z[ii*q_p:(ii+1)*q_p], self.zhat[ii*len(theta_p):(ii+1)*len(theta_p)],
                        transform=LDN(theta=self.theta, q=q_p, size_in=1).get_weights_for_delays(
                            np.asarray(theta_p)/self.theta)
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
        return S

    def generate_delta_identity(self, q_a, q_p):
        # TODO is this the right order?
        i = min(q_a, q_p)
        d = np.zeros((q_a, q_p))
        for ii in range(0, i):
            d[ii, ii] = 1
        return d
