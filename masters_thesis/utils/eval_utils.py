# import matplotlib
# matplotlib.use('TkAgg')

import numpy as np
import nengo
from tqdm import tqdm
from masters_thesis.network.ldn import LDN
import matplotlib.pyplot as plt

def calc_nni_err(errors):
    """
    Input is results from calc_shifted_error()
    Output is a single value that can be used for optimization
    """
    errors = np.absolute(errors)
    error = sum(sum(sum(errors)))
    return error


def calc_ldn_repr_err(z, qvals, theta, theta_p, dt=0.01, return_zhat=False):
    """
    Shows error of representation of decoded LDN at theta_p values, vary
    q used in LDN representation.

    Parameters
    ----------
    z: float array (steps, m)
        state to be predicted

    """
    results = {}
    if return_zhat:
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

        print(f"{q=} error shape: {np.asarray(errors).shape}")
        # results.append(sum(sum(errors)))
        results[f"{q}"] = errors

        if return_zhat:
            zhats[f"{q}"] = zhat

    # return np.asarray(results)
    if return_zhat:
        return results, zhats
    else:
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
        _Z = _Z.reshape((m, q)).T
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
    if not isinstance(theta_p, (list, np.ndarray)):
        theta_p = [theta_p]
    print('calc shifted error')
    print('z shape: ', z.shape)
    print('zhat shape: ', zhat.shape)
    steps = z.shape[0]
    m = z.shape[1]
    assert z.shape[0] == zhat.shape[0]
    assert z.shape[1] == zhat.shape[2]

    # errors = np.empty((steps-int(max(theta_p)/dt), len(theta_p), m))
    errors = np.zeros((steps, len(theta_p), m))
    for dim in range(0, m):
        for tp_index, _theta_p in enumerate(theta_p):
            theta_steps = int(_theta_p/dt)
            for step in range(0, steps):#-int(_theta_p/dt)): #  can't get ground truth at time n so remove the last max theta_p steps
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

    return np.asarray(errors)

def get_mean_and_range(data):
    print(f"{data.shape=}")
    plt.figure(figsize=(12,18))
    means = []
    stds = []
    ranges = []
    for ii in range(0, data.shape[0]):
        print(ii)
        lab = f"q{ii}"
        _t = np.arange(0, len(data[ii]))*0.001
        _mean = np.mean(data[ii])
        means.append(_mean)
        _variance = np.var(data[ii])
        _std = np.std(data[ii])
        stds.append(_std)
        _range = max(max(data[ii])-_mean, abs(min(data[ii])-_mean))
        ranges.append(_range)
        # print(f'{_mean=}')
        # print(f'{_variance=}')
        # print(f'{_range=}')
        # print(f'{_std=}')
        # print(f'{_std**2=}')
        plt.subplot(data.shape[0],1,ii+1)
        plt.title(lab)
        plt.plot(_t, data[ii])
        plt.hlines(_mean, _t[0], _t[-1], label='mean', color='r', linestyle='--')
        plt.fill_between(_t, _std+_mean, _mean-_std, alpha=0.5, label='1 std=68.2%', color='tab:purple')
        plt.fill_between(_t, 2*_std+_mean, _mean-2*_std, alpha=0.3, label='2 std=95.4%', color='tab:purple')
        plt.fill_between(_t, 3*_std+_mean, _mean-3*_std, alpha=0.1, label='3 std=99.8%', color='tab:purple')
        plt.hlines(_mean+_variance, _t[0], _t[-1], label='+variance', color='y', linestyle='--')
        plt.hlines(_mean-_variance, _t[0], _t[-1], label='-variance', color='y', linestyle='--')
        plt.hlines(_range, _t[0], _t[-1], label='+range', color='g', linestyle='--')
        plt.hlines(-_range, _t[0], _t[-1], label='-range',  color='g', linestyle='--')
        plt.legend(loc=1)

    print(f"{means=}")
    print(f"{stds=}")
    print(f"{2*stds=}")
    print(f"{3*stds=}")
    # np.savetxt('data/dq_mean.txt', means)
    # np.savetxt('data/dq_1std.txt', stds)
    # np.savetxt('data/dq_2std.txt', 2*np.array(stds))
    # np.savetxt('data/dq_3std.txt', 3*np.array(stds))
    # np.savetxt('data/dq_range.txt', 3*np.array(ranges))
    plt.show()

def gen_BLWN(T, dt, rms, limit, seed, sigma, debug=False):
    """
    Generates a bandwidth limited white noise signal
    """
    np.random.seed(seed)
    ts = np.arange(0, T, dt)
    N = ts.size
    # only generate half the points since we're making it symmetric about zero
    n_coeffs = int(N/2)+1

    # gernerate bool list of invalid frequencies below nyquist
    possible_coeffs = np.arange(0, int((T/dt)/2)+1)
    #print('possible: ', possible_coeffs)
    invalid_coeffs = possible_coeffs > limit
    #print('invalid: ', invalid_coeffs)

    # generate random real and imaginary values
    X_w = 1j * np.random.normal(scale=sigma, size=(n_coeffs,))
    X_w += np.random.normal(scale=sigma, size=(n_coeffs,))
    X_w[0] = 0
    X_w[-1] = X_w[-1].real
    X_w[invalid_coeffs] = 0
    # Stack our frequencies to create the negative entries
    """
    np.fft.ifft wants order...
    a[0] should contain the zero frequency term,
    a[1:n//2] should contain the positive-frequency terms,
    a[n//2 + 1:] should contain the negative-frequency terms, starting from most negative
    """
    #X_w = np.hstack((X_w, -np.flip(X_w[1:])))
    X_w = np.hstack((X_w, np.conj(np.flip(X_w[1:]))))
    #print('xw: ', X_w)
    x_t = np.real(np.fft.ifft(X_w))

    rmsp, scale = RMSP(x_t, T, rms, debug=False)
    x_t *= scale
    X_w *= scale
    rmsp, _ = RMSP(x_t, T, debug=debug)

    return x_t, X_w

def RMSP(x_t, T, des_rmsp=None, debug=False, dt=0.001):
    #rmsp = np.sqrt(1/T * sum(np.square(x_t)))*dt

    sig = 0
    for ii in range(0, len(x_t)):
        sig += x_t[ii]**2
    sig /=T
    sig *= dt
    rmsp = np.sqrt(sig)

    if debug:
        print('Root Mean Square Power: ', rmsp)
    if des_rmsp is not None:
        scale = des_rmsp / rmsp
        return rmsp, scale
    return rmsp, 1

def clipOutliers(arr, outlierConstant=1.7):
    """
    Default of 1.7 gives 3 sigma for normally distributed data
    """
    #<script src="https://gist.github.com/vishalkuo/f4aec300cf6252ed28d3.js"></script>
    clean_arr = np.empty(arr.shape)
    assert arr.shape[0] > arr.shape[1], ("Time should be the 0th dimension")

    for ii in range(0, arr.shape[1]):
        upper_quartile = np.percentile(arr[:, ii], 75)
        lower_quartile = np.percentile(arr[:, ii], 25)
        IQR = (upper_quartile - lower_quartile) * outlierConstant
        quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
        clean_arr[:, ii] = np.clip(arr[:, ii], quartileSet[1], quartileSet[0])
        # resultList = []
        # for y in a.tolist():
        #     if y >= quartileSet[0] and y <= quartileSet[1]:
        #         resultList.append(y)
        # return resultList
    return clean_arr
