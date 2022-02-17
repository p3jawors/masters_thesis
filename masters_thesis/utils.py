import numpy as np
import nengo
from tqdm import tqdm
from masters_thesis.network.ldn import LDN

def calc_ldn_repr_err(z, qvals, theta, theta_p, dt=0.01):
    """
    Shows error of representation of decoded LDN at theta_p values, vary
    q used in LDN representation.

    Parameters
    ----------
    z: float array (steps, m)
        state to be predicted

    """
    results = []
    for q in qvals:
        model = nengo.Network()
        with model:
            ldn = nengo.Node(LDN(theta=theta, q=q, size_in=z.shape[1]), label='ldn')

            def in_func(t):
                return z[int(t/dt - dt)]

            in_node = nengo.Node(in_func, size_in=None, size_out=z.shape[1])

            nengo.Connection(in_node, ldn, synapse=None)
            Z = nengo.Probe(ldn, synapse=None)
        sim = nengo.Simulator(network=model, dt=dt)
        with sim:
            sim.run(z.shape[0]*dt)

        zhat = decode_ldn_data(
            Z=sim.data[Z],
            q=q,
            theta=theta,
            theta_p=theta_p
        )

        errors = calc_shifted_error(
            z=z,
            zhat=zhat,
            dt=dt,
            theta_p=theta_p,
            model='ldn'
        )

        results.append(sum(sum(errors)))

    return results


def decode_ldn_data(Z, q, theta, theta_p=None):
    """
    Parameters
    ----------
    Z: float array(steps, m*q)
        prediction of state Z in legendre domain
    q: int
        legendre dimensionality
    theta: float
        prediction horizon length [sec]
    theta_p: float array, Optional (Default: None)
        The times to extract from the legendre predictions
        if None, we will output the prediction at theta.
    """
    # print(f"{Z.shape=}")
    # print(f"{q=}")
    # print(f"{theta=}")
    # print(f"{theta_p.shape=}")
    m = int(Z.shape[1]/q)
    # print(f"{m=}")
    if theta_p is None:
        theta_p = [theta]
    theta_p = np.asarray(theta_p)

    # shape (len(theta_p), q)
    transform = LDN(theta=theta, q=q, size_in=1).get_weights_for_delays(theta_p/theta)
    # print(f"{transform.shape=}")
    zhat = []
    for _Z in tqdm(Z):
        _Z = _Z.reshape((q, m))
        zhat.append(np.dot(transform, _Z))
        # print(f"{_Z.shape=}")

    return np.asarray(zhat)


def calc_shifted_error(z, zhat, dt, theta_p, model='llp'):
    """
    Returns the difference between zhat and z shifted by the
    corresponding theta_p. Error is return in the same shape as
    zhat (steps, len(theta_p), m)
    Parameters
    ----------
    z: float array (steps, m)
        state to be predicted
    zhat: float array (steps, len(theta_p), m)
        predicted state in world space
    dt: float
        time step
    theta_p: float array
        the times into the future zhat predictions are in [sec]
    model: string, Optional (Default: 'llp')
        'llp' to get shifted error for llp. In this case we shift our
        ground truth forward in time
        'ldn' to get shifted error for ldn. In this case we shift our
        ground truth backward in time
    """
    print(z.shape)
    print(zhat.shape)
    steps = z.shape[0]
    m = z.shape[1]
    assert z.shape[0] == zhat.shape[0]
    assert z.shape[1] == zhat.shape[2]

    errors = np.empty((steps-int(max(theta_p)/dt), len(theta_p), m))
    for dim in range(0, m):
        for step in range(0, steps-int(max(theta_p)/dt)): #  can't get ground truth at time n so remove the last max theta_p steps
            for tp_index, _theta_p in enumerate(theta_p):
                if model == 'llp':
                    diff = z[step + int(_theta_p/dt), dim] - zhat[step, tp_index, dim]
                elif model == 'ldn':
                    # start at max theta_p steps in
                    diff = z[int(max(theta_p)/dt) + step - int(_theta_p/dt), dim] - zhat[int(max(theta_p)/dt) + step, tp_index, dim]

                errors[step, tp_index, dim] = diff

    return np.asarray(errors)
