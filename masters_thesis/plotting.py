# import matplotlib
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm
# import seaborn as sb
import imageio
import os
import numpy as np

def plot_pred(
        time, z, zhat, theta_p, size_out, gif_name, animate=True, window=None,
        step=None, save=False):

    if zhat.ndim == 2:
        zhat = zhat[:, np.newaxis, :]

    # plt.figure()
    # # print(z.shape)
    # # print(zhat.shape)
    # for tt, _theta_p in enumerate(theta_p):
    #     plt.subplot(len(theta_p), 1, tt+1)
    #     plt.title(f"{_theta_p} prediction")
    #     plt.xlabel("Time[sec]")
    #     plt.plot(time-_theta_p, z, label='z shifted by {_theta_p}')
    #     # plt.plot(time, z, label='z shifted by {_theta_p}')
    #     plt.plot(time, np.squeeze(zhat[:, tt, :]), label='zhat', linestyle='--')
    #     plt.legend()

    plt.figure()#figsize=(12,0))
    axs = []
    labels = ['x', 'y', 'z']
    for ii in range(0, size_out):
        axs.append(plt.subplot(size_out, 1, ii+1))
        axs[ii].plot(time, np.squeeze(zhat[:, :, ii]))

        plt.gca().set_prop_cycle(None)
        for pred in theta_p:
            axs[ii].plot(time-pred, z.T[ii], linestyle='--')
            # axs[ii].plot(time, z.T[ii], linestyle='--')

        axs[ii].legend(
            [f'{labels[ii]}hat at: ' + str(round(tp, 3)) for tp in theta_p]
            + [f'{labels[ii]} shifted: ' + str(round(tp, 3)) for tp in theta_p],
            loc=1)
    if save:
        plt.savefig('llp_prediction_over_time.jpg')

    if animate:
        print('Generating gif...')
        if window is None:
            window = max(theta_p)*2
        if step is None:
            step = np.mean(np.diff(time))*100
        if not os.path.exists('.cache'):
            os.makedirs('.cache')
        start = 0.0
        stop = window
        ss = 0
        filenames = []
        print('time; ', time[-1])
        while stop <= time[-1]:
            for ax in axs:
                ax.set_xlim(start, stop)
            filename = f".cache/img_{ss:08d}.jpg"
            filenames.append(filename)
            plt.savefig(filename)
            start += step
            stop += step
            ss += 1

        if not os.path.exists('Figures'):
            os.makedirs('Figures')

        with imageio.get_writer(f"Figures/{gif_name}", mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                os.remove(filename)
    plt.show()

def plot_error(theta_p, errors, dt, output_labs=('X', 'Y', 'Z'), theta=None, save=False, label=''):
    """
    Parameters
    ----------
    theta: float Optional (Default: max(theta_p))
        size of window we are predicted
    theta_p: float array
        the times into the future zhat predictions are in [sec]
    errors: float array
        the errors returns from utils.calc_shifted_error (steps, len(theta_p), m),
        where m is the number of output dims
    """
    time = np.linspace(0, errors.shape[0]*dt, errors.shape[0])

    if theta is None:
        theta = max(theta_p)

    if 1 < len(theta_p) < 10:
        plt.figure(figsize=(8,8))
        for ii in range(0, len(theta_p)):
            plt.subplot(len(theta_p), 1, ii+1)
            plt.title(f"{theta_p[ii]} prediction 2norm error")
            plt.plot(errors[:, ii, :])
        plt.show()

        if save:
            plt.savefig(f'{label}2norm_over_time.jpg')

    if len(theta_p) > 1:
        # Plot a 3d plot for each xyz output dim
        # fig = plt.figure()
        fig = plt.figure(figsize=(8,8))
        axs = []
        for ii in range(0, errors.shape[2]):
            axs.append(plt.subplot(1, errors.shape[2], ii+1, projection='3d'))
            plt.xlabel('Time [sec]')
            plt.ylabel('Theta P [sec]')
            plt.title(f'{output_labs[ii]} Error')
            X, Y = np.meshgrid(time, theta_p)
            surf = axs[ii].plot_surface(X, Y, errors[:, :, ii].T,
                    cmap=cm.coolwarm, linewidth=0, antialiased=False)
            axs[ii].zaxis.set_major_locator(LinearLocator(10))
            axs[ii].zaxis.set_major_formatter('{x:.02f}')
            fig.colorbar(surf, shrink=0.5, aspect=5)
        # plt.show()

        if save:
            plt.savefig(f'{label}3d_error_heat_map.jpg')


        # Plot a 2d heat map of the above
        fig = plt.figure()
        axs = []
        for ii in range(0, errors.shape[2]+1):
            axs.append(plt.subplot(errors.shape[2]+1, 1, ii+1))
            if ii < errors.shape[2]:
                plt.xlabel('Time [sec]')
                plt.ylabel('Theta P [sec]')
                plt.title(f'{output_labs[ii]} Error | theta={theta}')
                X, Y = np.meshgrid(time, theta_p)
                axs[ii].pcolormesh(X, Y, errors[:, :, ii].T)
            else:
                plt.xlabel('Time [sec]')
                plt.ylabel('Theta P [sec]')
                plt.title(f'2norm Error | theta={theta}')
                X, Y = np.meshgrid(time, theta_p)
                axs[ii].pcolormesh(X, Y, np.linalg.norm(errors, axis=2).T)

            # surf = axs[ii].plot_surface(X, Y, errors[:, :, ii].T,
            #         cmap=cm.coolwarm, linewidth=0, antialiased=False)
            # axs[ii].zaxis.set_major_locator(LinearLocator(10))
            # axs[ii].zaxis.set_major_formatter('{x:.02f}')
            # fig.colorbar(surf, shrink=0.5, aspect=5)

        if save:
            plt.savefig(f'{label}2d_error_heat_map.jpg')


        # Plot avg error over time, averaging over theta_p
        plt.figure(figsize=(8,12))
        axs = []
        for ii in range(0, errors.shape[2]+1):
            axs.append(plt.subplot(errors.shape[2]+1, 1, ii+1))
            if ii < errors.shape[2]:
                plt.title("Error over Time")
                plt.xlabel('Time [sec]')
                plt.ylabel(f'{output_labs[ii]} Mean Error Over Theta_P')
                axs[ii].plot(time, np.mean(errors[:, :, ii], axis=1))
            else:
                plt.title("Error over Time")
                plt.xlabel('Time [sec]')
                plt.ylabel(f'2norm Error of Mean Over Theta_P')
                axs[ii].plot(time, np.linalg.norm(np.mean(errors, axis=1), axis=1))

        if save:
            plt.savefig(f'{label}error_over_time_avg_tp.jpg')

    # Plot avg error over time, showing each theta_p line
    plt.figure(figsize=(8,12))
    axs = []
    for ii in range(0, errors.shape[2]+1):
        axs.append(plt.subplot(errors.shape[2]+1, 1, ii+1))
        for tp, _theta_p in enumerate(theta_p):
            alpha = 1 - tp/len(theta_p)
            if ii < errors.shape[2]:
                plt.title("Error over Time, Varying Theta_p")
                plt.xlabel('Time [sec]')
                plt.ylabel(f'{output_labs[ii]} Mean Error Over Theta_P')
                axs[ii].plot(time, errors[:, tp, ii], alpha=alpha, label=f"{_theta_p}")#, c='r')
            else:
                plt.title("Error over Time")
                plt.xlabel('Time [sec]')
                plt.ylabel(f'2norm Error of Mean Over Theta_P')
                # axs[ii].plot(time, np.linalg.norm(np.mean(errors, axis=1), axis=1))
                axs[ii].plot(time, np.linalg.norm(errors[:, tp, :], axis=1), alpha=alpha, label=f"{_theta_p}")#, c='r')
        plt.legend()

    # plt.show()

    if save:
        plt.savefig(f'{label}error_over_time.jpg')

    # Plot avg error over theta_p, averaging over time
    if len(theta_p) > 1:
        plt.figure(figsize=(8,12))
        axs = []
        for ii in range(0, errors.shape[2]):
            axs.append(plt.subplot(errors.shape[2], 1, ii+1))
            plt.title("Error over Theta_P")
            plt.xlabel('Theta_P [sec]')
            plt.ylabel(f'{output_labs[ii]} Mean Error Over Time')
            axs[ii].plot(theta_p, np.mean(errors[:, :, ii], axis=0))

        if save:
            plt.savefig(f'{label}error_over_tp.jpg')

    if not save:
        plt.show()

def plot_ldn_repr_error(error, theta_p, z, dt, zhats, output_labs=('X', 'Y', 'Z'), label=None):
    """
    Parameters
    ----------
    error: dict
        keys are q value
        values are error (steps, len(theta_p), m)
    theta_p: list
        list of theta_p values in seconds
    """
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

    for key in error:
        # print(f"{key}: {error[key].shape}")
        for ii in range(0, len(theta_p)):
            plt.figure(figsize=(12,9))
            plt.title(f"q={key} | theta_p={theta_p[ii]}")

            steps = z.shape[0]
            shifted_t = get_shifted_t(theta_p[ii], dt, steps, direction='forward')
            t = np.arange(0, z.shape[0]*dt, dt)

            plt.plot(t, error[key][:, ii, :], label=f'error_{theta_p[ii]}')
            plt.plot(t, z, label='z')
            plt.plot(shifted_t, z, label=f'z shift>>{theta_p[ii]}')
            plt.plot(t, zhats[key][:, ii, :], label=f'zhat_{theta_p[ii]}', linestyle='--')
            plt.legend()
            plt.savefig(f"{label}_q={key}-tp={theta_p[ii]}.jpg")

        plotting.plot_error(
            theta_p=theta_p, errors=error[key], dt=dt, output_labs=output_labs,
            theta=theta, save=True, label=f"q={key}_")
