import numpy as np
from utils import decode_ldn_data, calc_shifted_error
import timeit
import matplotlib.pyplot as plt
"""
Plots the time it takes to decode m outputs from q legendre coefficients
and the time it takes to calculate the error from a reference.

Most values are randomly generated since it is the timing that is of importance.

The idea is to use this with your sim parameters to see what values are reasonable
to use in the context of realtime control.

Of the following parameters, pass one as a list to get a plot with two subplots.
The Y values will be the two timing values listed above, and x will be the list.

Parameters
----------
dt: the timestep to use in sim
steps in sim: when testing for purposes of calculating errors over sim time in
    parameter sweeps
steps_to_predict: the number of steps into the future we want to predict. The actual
    time between values does not matter as the math doesn't change
m: dimensions to predict
q: number of legendre polynomials used to predict m
"""
def gen_timing_plots(params):
    for key, val in params.items():
        if isinstance(val, (list, np.ndarray)):
            plot_key = key
            break

    decode_times = []
    error_times = []

    for ii in range(0, len(params[plot_key])):
        # dt
        if isinstance(params['dt'], (list, np.ndarray)):
            dt = params['dt'][ii]
        else:
            dt = params['dt']

        # steps_in_sim
        if isinstance(params['steps_in_sim'], (list, np.ndarray)):
            steps_in_sim = params['steps_in_sim'][ii]
        else:
            steps_in_sim = params['steps_in_sim']

        # steps_to_predict
        if isinstance(params['steps_to_predict'], (list, np.ndarray)):
            steps_to_predict = params['steps_to_predict'][ii]
        else:
            steps_to_predict = params['steps_to_predict']

        # m
        if isinstance(params['m'], (list, np.ndarray)):
            m = params['m'][ii]
        else:
            m = params['m']

        # q
        if isinstance(params['q'], (list, np.ndarray)):
            q = params['q'][ii]
        else:
            q = params['q']

        theta = steps_to_predict * dt
        theta_p = np.linspace(dt, theta, steps_to_predict)

        decode_times.append(time_decode(steps_in_sim, m, q, theta_p))
        error_times.append(time_err_calc(steps_in_sim, m, theta_p, dt))

    plt.figure(figsize=(8,4))

    plt.subplot(121)
    plt.title('Time to decode Legendre polynomials')
    plt.ylabel('Time [sec]')
    plt.scatter(params[plot_key], decode_times)
    legend = []
    for key, val in params.items():
        legend.append(f"{key}={val}")
    legend = '\n'.join(legend)
    plt.legend(legend)
    plt.xlabel(f"{plot_key}\n{legend}")

    plt.subplot(122)
    plt.title('Time to calculate errors')
    plt.ylabel('Time [sec]')
    plt.xlabel(f"{plot_key}\n{legend}")
    plt.scatter(params[plot_key], error_times)
    plt.legend(legend)

    plt.show()

    return decode_times, error_times



def time_decode(steps_in_sim, m, q, theta_p):
    """
    Assumes theta = max(theta_p)
    """
    start_decode = timeit.default_timer()
    zhat = decode_ldn_data(
        Z=np.random.uniform(low=0, high=2, size=(steps_in_sim, m*q)),
        q=q,
        theta=max(theta_p),
        theta_p=theta_p
    )
    time_to_decode = timeit.default_timer() - start_decode
    return time_to_decode

def time_err_calc(steps_in_sim, m, theta_p, dt):
    start_err_calc = timeit.default_timer()
    errors = calc_shifted_error(
        z=np.random.uniform(
            low=0, high=2, size=(steps_in_sim, m)),
        zhat=np.random.uniform(
            low=0, high=2, size=(steps_in_sim, int(len(theta_p)), m)),
        dt=dt,
        theta_p=theta_p
    )
    time_to_calc_err = timeit.default_timer() - start_err_calc
    return time_to_calc_err

if __name__ == '__main__':
    params = {}
    params['dt'] = 0.01
    params['steps_in_sim'] = 1
    params['steps_to_predict'] = np.arange(10, 500) #100
    params['m'] = 3
    params['q'] = 6 #np.arange(1, 12, 1)

    decode_times, error_times = gen_timing_plots(params=params)
