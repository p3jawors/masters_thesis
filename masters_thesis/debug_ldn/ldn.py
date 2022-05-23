"""
This notebook shows how to build an LMU (Legendre Memory Unit) using Nengo,
and explores some thing that we can do with it.

First, let's build the linear part of an LMU. This is just a linear differential
equation of the form where  and  are carefully chosen so that  will represent
the past history of  over some window . The way it does this is to encode the
past history using Legendre polynomials. This fact will let us decode information
about  out of . We call this component a Legendre Delay Network (LDN).

To implement this in Nengo, we define a nengo.Process. This is arbitrary Python
code that will be called every timestep during the nengo simulation. Since this
is going to be run on a computer with a discrete time step, we also discretize
the system using the standard zero-order-hold approach (this makes the discrete
simulation be very very close to the ideal continuous result, without forcing us
to go with a very small time step).
"""
import scipy.linalg
from scipy.special import legendre
import nengo
import numpy as np
import math

blue = '\033[94m'
green = '\033[92m'
red = '\033[91m'
endc = '\033[0m'


class LDN(nengo.Process):
    def __init__(self, theta, q, size_in=1):
        self.q = q              # number of internal state dimensions per input
        self.theta = theta      # size of time window (in seconds)
        self.size_in = size_in  # number of inputs

        # Do Aaron's math to generate the matrices A and B so that
        #  dx/dt = Ax + Bu will convert u into a legendre representation over a window theta
        #  https://github.com/arvoelke/nengolib/blob/master/nengolib/synapses/analog.py#L536
        A = np.zeros((q, q))
        B = np.zeros((q, 1))
        for i in range(q):
            B[i] = (-1.)**i * (2*i+1)
            for j in range(q):
                A[i,j] = (2*i+1)*(-1 if i<j else (-1.)**(i-j+1))
        self.A = A / theta
        self.B = B / theta

        super().__init__(default_size_in=size_in, default_size_out=q*size_in)

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        state = np.zeros((self.q, self.size_in))

        # Handle the fact that we're discretizing the time step
        #  https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
        Ad = scipy.linalg.expm(self.A*dt)
        Bd = np.dot(np.dot(np.linalg.inv(self.A), (Ad-np.eye(self.q))), self.B)

        # this code will be called every timestep
        def step_legendre(t, x, state=state):
            state[:] = np.dot(Ad, state) + np.dot(Bd, x[None, :])
            return state.T.flatten()
        return step_legendre

    def get_weights_for_delays(self, r):
        # compute the weights needed to extract the value at time r
        # from the network (r=0 is right now, r=1 is theta seconds ago)
        r = np.asarray(r)
        m = np.asarray([legendre(i)(2*r - 1) for i in range(self.q)])
        return m.reshape(self.q, -1).T

