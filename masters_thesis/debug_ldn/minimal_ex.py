"""
Test LDN ability to represent activities and our state predictions
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import nengo
from ldn import LDN

# Creates a sin and cos wave with a set frequency
def stim_func(t, freq=5):
    try:
        if int(sys.argv[1]) == 1:
            return np.sin(t*2*np.pi*freq)
        elif int(sys.argv[1]) == 2:
            return [np.sin(t*2*np.pi*freq), np.cos(t*2*np.pi*freq)]
    except IndexError as e:
        print("Pass either 1 or 2 as a sys.argv input to define input dimensionality")
        raise e

# generate a 5Hz sin wave
dt = 0.01
t = np.arange(0, 1, dt)
z = np.asarray(stim_func(t)).T
if z.ndim == 1:
    z = z[:, np.newaxis]

steps = z.shape[0]
m = z.shape[1]

# list of q values to evaluate
q = 10
theta = 0.1
theta_p = 0.1

print(f"encoding ldn with {q=}")
model = nengo.Network()
with model:
    ldn = nengo.Node(LDN(theta=theta, q=q, size_in=z.shape[1]), label='ldn')

    def in_func(t):
        return z[int(t/dt - dt)]

    in_node = nengo.Node(in_func, size_in=None, size_out=z.shape[1])
    nengo.Connection(in_node, ldn, synapse=None)
    # our ldn representation of in_func
    Z = nengo.Probe(ldn, synapse=None)

sim = nengo.Simulator(network=model, dt=dt)
with sim:
    sim.run(steps*dt)

Z = sim.data[Z]
transform = LDN(theta=theta, q=q, size_in=1).get_weights_for_delays(theta_p/theta)

zhat = []
# get the decoded value at each timestep
for _Z in Z:
    _Z = _Z.reshape((m, q)).T
    result = np.squeeze(np.dot(transform, _Z))
    zhat.append(result)
zhat = np.asarray(zhat)
print(f"LDN(t) shape: {_Z.shape}")
print(f"Transform(theta_p/theta) shape: {transform.shape}")
print(f"Dot prod shape:(transform, _Z): {result.shape}")
print(f"{zhat.shape=}")
plt.figure()
plt.plot(z, label='z')
plt.gca().set_prop_cycle(None)
plt.plot(zhat, label='zhat', linestyle='--')
plt.legend()
plt.show()
