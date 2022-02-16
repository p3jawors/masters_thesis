import numpy as np
from tqdm import tqdm
from masters_thesis.network.ldn import LDN

def calc_ldn_repr_err(z, qrange, theta, theta_p):
    """
    Shows error of representation of decoded LDN at theta_p values, vary
    q used in LDN representation.
    """
    raise NotImplementedError

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


def calc_shifted_error(z, zhat, dt, theta_p):
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
    """
    # print(f"{z.shape=}")
    # print(f"{zhat.shape=}")
    steps = z.shape[0]
    m = z.shape[1]
    assert z.shape[0] == zhat.shape[0]
    assert z.shape[1] == zhat.shape[2]

    errors = np.empty((steps-int(max(theta_p/dt)), len(theta_p), m))
    # min_horizon = int(min(theta_p)/dt)
    # max_horizon = int(max(theta_p)/dt) + 1 #  to include this value due to np indexing being [start, end)
    for dim in range(0, m):
        for step in range(0, steps-int(max(theta_p/dt))): #  can't get ground truth at time n so remove the last max theta_p steps
            for tp, _theta_p in enumerate(theta_p):
                # theta_steps = int(_theta_p/dt)
                # print(f"{_theta_p=}")
                # print(f"{theta_steps}")
                # print(z[step:step+theta_steps, dim].shape)
                # print(zhat[step, :, dim].shape)
                # print(f"{step=}+{int(min(theta_p)/dt)=}")
                # print(f"{int(max(theta_p)/dt)=}")
                # step_error = np.linalg.norm((np.squeeze(z[step+int(min(theta_p)/dt):step+1+int(max(theta_p)/dt), dim]) - np.squeeze(zhat[step, :, dim])))
                diff = z[step + int(_theta_p/dt), dim] - zhat[step, tp, dim]
                errors[step, tp, dim] = diff

    # for ii, _theta_p in enumerate(theta_p):
    #     offset = int(_theta_p/dt)
    #     # print(f"{_theta_p=}")
    #     # print(f"{offset=}")
    #     # print(f"{z.shape=}")
    #     # print(f"{z[offset:].shape=}")
    #     # print(f"{zhat.shape=}")
    #     # print(f"{zhat[:-offset, ii, :].shape=}")
    #     error = np.linalg.norm((z[offset:] - zhat[:-offset, ii, :]), axis=1)
    #     # print(f"{error.shape=}")
    #     errors.append(error)
    return np.asarray(errors)

# def old_func():
#     animate = True
#     window = theta*5
#     step = dt*10
#
#     plt.figure()
#     plt.title('Predictions over time')
#     plt.plot(sim.trange(), sim.data[z_probe])
#     plt.plot(sim.trange(), sim.data[zhat_probe])
#     plt.legend(['z'] + [str(tp) for tp in theta_p])
#
#     plt.figure()
#     axs = []
#     for ii in range(0, size_out):
#         axs.append(plt.subplot(ii+1, 1, size_out))
#         axs[ii].plot(sim.trange(), sim.data[zhat_probe])
#
#         plt.gca().set_prop_cycle(None)
#         for pred in theta_p:
#             axs[ii].plot(sim.trange()-pred, sim.data[z_probe].T[ii], linestyle='--')
#
#         axs[ii].legend(
#             ['zhat at: ' + str(round(tp, 3)) for tp in theta_p]
#             + ['z shifted: ' + str(round(tp, 3)) for tp in theta_p],
#             loc=1)
#
#     if animate:
#         start = 0.0
#         stop = window
#         ss = 0
#         filenames = []
#         while stop <= sim.trange()[-1]:
#             for ax in axs:
#                 ax.set_xlim(start, stop)
#             filename = f".cache/img_{ss:08d}.jpg"
#             filenames.append(filename)
#             plt.savefig(filename)
#             start += step
#             stop += step
#             ss += 1
#
#         with imageio.get_writer('llp.gif', mode='I') as writer:
#             for filename in filenames:
#                 image = imageio.imread(filename)
#                 writer.append_data(image)
#                 os.remove(filename)
#     plt.show()
#
#
#
