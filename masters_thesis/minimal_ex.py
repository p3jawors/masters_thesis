"""
Test LDN ability to represent activities and our state predictions
"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import nengo
from masters_thesis.network.ldn import LDN

# Creates a sin and cos wave with a set frequency
def stim_func(t, freq=5):
    return [np.sin(t*2*np.pi*freq), np.cos(t*2*np.pi*freq)]
    # return np.sin(t*2*np.pi*freq)

# generate a 5Hz sin wave
dt = 0.01
t = np.arange(0, 1, dt)
z = np.asarray(stim_func(t)).T
if z.ndim == 1:
    z = z[:, np.newaxis]

steps = z.shape[0]
# get the dimensionality of the output we want decoded from the ldn
m = z.shape[1]

# list of q values to evaluate
qvals = [10]
# the length of window we are trying to store [sec]
theta = 0.1
# the list of points in the window to decode out [sec]
theta_p = [0.1]
theta_p = np.asarray(theta_p)

# ======================================================================================= DECODE LDN REPR ERROR
results = {}
zhats = {}

for q in qvals:
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
        sim.run(z.shape[0]*dt)

    # =================================================================================== DECODE LDN DATA
    Z = sim.data[Z]
    # get the transform to extract our state given q, theta, and theta_p
    transform = LDN(theta=theta, q=q, size_in=1).get_weights_for_delays(theta_p/theta)

    zhat = []
    # get the decoded value at each timestep
    for _Z in tqdm(Z):
        _Z = _Z.reshape((q, m))
        print(f"{_Z.shape=}")
        print(f"{transform.shape=}")
        print(f"{np.dot(transform, _Z).shape}")
        a = np.dot(transform, _Z[:, 0])
        b = np.dot(transform, _Z[:, 1])
        new = np.array([a, b]).T
        # new = new[:, np.newaxis].T
        print('NEW: ', new.shape)
        # zhat.append(np.dot(transform, _Z))
        zhat.append(new)

    zhat = np.asarray(zhat)
    # =================================================================================================================================


    # ============================================================================== CALC SHIFTED ERROR
    assert z.shape[0] == zhat.shape[0]
    assert z.shape[1] == zhat.shape[2]

    errors = np.zeros((steps, len(theta_p), m))
    model = 'ldn'
    # step through each dim being predicted
    for dim in range(0, m):
        # step through each theta_p being decoded out of the theta window
        for tp_index, _theta_p in enumerate(theta_p):
            theta_steps = int(_theta_p/dt)
            for step in range(0, steps):
                if model == 'llp':
                    # stop at the last theta seconds, since we won't have the future theta
                    # seconds of ground truth to compare to
                    if step < steps - theta_steps:
                        diff = z[step + theta_steps, dim] - zhat[step, tp_index, dim]
                        errors[step, tp_index, dim] = diff

                elif model == 'ldn':
                    # shift forward by theta since we can't say what happened theta seconds
                    # ago before theta seconds pass
                    if step > theta_steps:
                        diff = z[step - theta_steps, dim] - zhat[step, tp_index, dim]
                        errors[step, tp_index, dim] = diff

    errors = np.asarray(errors)
    # =================================================================================================================================

    print(f"{q=} error shape: {np.asarray(errors).shape}")
    results[f"{q}"] = errors
    zhats[f"{q}"] = zhat

# ================================================================================================== PLOT LDN REPR ERROR
error = results
label = None
folder = 'Figures'
max_rows = 4
"""
Parameters
----------
error: dict
    keys are q value
    values are error (steps, len(theta_p), m)
theta_p: list
    list of theta_p values in seconds
"""
if not os.path.exists(folder):
    os.makedirs(folder)

def get_shifted_t(theta_p, dt, steps, direction='forward'):
    """
    """
    if direction == 'backward':
        # shift backward (used for z to match zhat)
        shifted_t = np.arange(0-theta_p, z.shape[0]*dt-theta_p, dt)
    elif direction == 'forward':
        # shift forward (used for z to match zhat)
        shifted_t = np.arange(0+theta_p, z.shape[0]*dt+theta_p, dt)
    else:
        print(f"{direction} is not valid, choose 'ldn' or 'llp'")
    return shifted_t


if label is None:
    label = 'testfig'

for key in zhats:
    # output dimensionality in meat space
    mm = zhats[key].shape[2]
    break

jj = min(mm, max_rows)
kk = int(np.ceil(mm/jj))

for key in error:
    # print(f"{key}: {error[key].shape}")
    for ii in range(0, len(theta_p)):
        plt.figure(figsize=(12,9))
        for ll in range(0, mm):
            plt.subplot(jj, kk, ll+1)

            plt.title(f"q={key} | theta_p={theta_p[ii]}")

            steps = z.shape[0]
            shifted_t = get_shifted_t(theta_p[ii], dt, steps, direction='forward')
            t = np.arange(0, z.shape[0]*dt, dt)

            # plt.plot(t, error[key][:, ii, mm-1], label=f'error_{theta_p[ii]}')
            plt.plot(t, z[:, mm-1], label='z')
            plt.plot(shifted_t, z[:, mm-1], label=f'z shift>>{theta_p[ii]}')
            plt.plot(t, zhats[key][:, ii, mm-1], label=f'zhat_{theta_p[ii]}', linestyle='--')
            plt.legend(loc=1)
            plt.xlabel('Time [sec]')
    plt.savefig(f"{folder}/{label}_q={key}-tp={theta_p[ii]}.jpg")

plt.show()
