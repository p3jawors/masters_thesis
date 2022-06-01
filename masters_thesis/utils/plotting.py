from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import imageio
import os
import numpy as np
from masters_thesis.utils.eval_utils import decode_ldn_data

def plot_x_vs_xhat(x, xhat):
    """
    Weird plot comparing (x, x), against (x, xhat).
    The idea is the x axis is the target value to be
    represented, and the y axis is the representation
    of it. The perfect result would be a straight line
    with a slope of 1.
    """
    plt.figure(figsize=(6, 8))
    plt.subplot(211)
    plt.title('Network Value Decoding')
    plt.xlabel('Value to Represent')
    plt.ylabel('Decoded Attempt')
    plt.plot(x, x, label='ideal')
    plt.plot(x, xhat, linestyle='--', label='decoded')
    plt.legend()

    rmse = RMSE(x=x, xhat=xhat)

    plt.subplot(212)
    plt.title('Error Representing Values')
    plt.xlabel('Value to Represent')
    plt.ylabel(r'$Error (x-\^x)$')
    plt.plot(x, x-xhat, label=f"RMSE: {rmse}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_prediction_vs_gt(tgt, decoded, q, theta, theta_p, theta_steps=None, z_state=None, xlim=None, save=False, savename='pred_vs_gt.jpg', show=True, label=''):
    """
    Plots predictions of legendre coefficients against GT,
    and their decoded values given theta and theta_p

    NOTE: input has to already be shifted in time if comparing ldn/llp predictions
    ie: this function does NOT do time shifting
    """
    plt.figure(figsize=(12,12))
    for ii in range(0, tgt.shape[1]):
        if ii == 0:
            plt.title(f"Prediction vs GT in Legendre: {label}")
        plt.subplot(tgt.shape[1], 1, ii+1)
        plt.plot(tgt[:, ii], label='target')
        # plt.gca().set_prop_cycle(None)
        plt.plot(decoded[:, ii], linestyle='--', label='decoded')
    plt.tight_layout()

    plt.figure(figsize=(20,12))
    zhat_GT = decode_ldn_data(
        Z=tgt,
        q=q,
        theta=theta,
        theta_p=theta_p
    )
    zhat_pred = decode_ldn_data(
        Z=decoded,
        q=q,
        theta=theta,
        theta_p=theta_p
    )

    for ii in range(0, zhat_GT.shape[2]):
        for jj in range(0, zhat_GT.shape[1]):
            plt.subplot(zhat_GT.shape[2], zhat_GT.shape[1], ii*(zhat_GT.shape[1]) + jj+1)
            if ii == 0:
                plt.title(f"Prediction vs GT Decoded\ntheta={theta} | theta_p={theta_p[jj]}\n{label}")
            if jj == 0:
                plt.ylabel(f"dim_{ii}")
            plt.plot(zhat_GT[:, jj, ii], label='gt Z decoded', c='c')
            # plt.gca().set_prop_cycle(None)
            plt.plot(zhat_pred[:, jj, ii], linestyle='--', c='r', label='predicted Z decoded')
            if z_state is not None:
                plt.plot(z_state[:, ii], linestyle='-', c='k', label='z actual')
                if theta_steps is not None:
                    plt.plot(z_state[int(theta_steps):, ii], linestyle='--', c='k', label='z actual shifted theta_p')
            plt.legend()
            if xlim is not None:
                plt.xlim(xlim[0], xlim[1])
            plt.grid(True)

    if save:
        plt.savefig(savename)
        print(f'Saved pred_vs_gt figure to {savename}')

    if show:
        plt.show()


def plot_pred(
        time, z, zhat, theta_p, size_out, gif_name='gif', animate=False, window=None,
        step=None, save=False, folder='Figures'):

    if not os.path.exists(folder):
        os.makedirs(folder)

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
        axs[ii].plot(time, z.T[ii], 'k', linestyle='-', label=f"{labels[ii]}(t)")

        for pred in theta_p:
            axs[ii].plot(time-pred, z.T[ii], linestyle='-', label=f"{labels[ii]}(t+{pred})")

        plt.gca().set_prop_cycle(None)
        for pp, pred in enumerate(theta_p):
            axs[ii].plot(time, np.squeeze(zhat[:, pp, ii]), linestyle='--', label=f"{labels[ii]}hat(t, {pred})")
            # axs[ii].plot(time, z.T[ii], linestyle='--')
            axs[ii].set_ylim(2*np.amin(z.T[ii]), 2*np.amax(z.T[ii]))

        # axs[ii].legend(
        #     f'{labels[ii]}(t)'
        #     + [f'{labels[ii]}hat(t, ' + str(round(tp, 3)) for tp in theta_p + ')']
        #     + [f'{labels[ii]}(t+' + str(round(tp, 3)) for tp in theta_p + ')'],
        #     loc=1)
        axs[ii].legend(loc=1)
    if save:
        plt.savefig(f'{folder}/llp_prediction_over_time.jpg')

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

        with imageio.get_writer(f"{folder}/{gif_name}", mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                os.remove(filename)
    plt.show()

def plot_error_subplot_theta_p(theta, theta_p, errors, dt, prediction_dim_labs=('X', 'Y', 'Z'), save=False, label='', folder='Figures'):
    """
    Parameters
    ----------
    theta: float Optional (Default: max(theta_p))
        size of window we are predicting
    theta_p: float array
        the times into the future zhat predictions are in [sec]
    errors: float array
        the errors returns from utils.calc_shifted_error (steps, len(theta_p), m),
        where m is the number of output dims
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    time = np.linspace(0, errors.shape[0]*dt, errors.shape[0])

    if theta is None:
        theta = max(theta_p)

    plt.figure(figsize=(8,8))
    for ii in range(0, len(theta_p)):
        plt.subplot(len(theta_p), 1, ii+1)
        plt.title(f"{theta_p[ii]} prediction 2norm error")
        plt.plot(errors[:, ii, :])
    if save:
        plt.savefig(f'{folder}{label}2norm_over_time.jpg')
    plt.show()

def plot_error_3d_surf(theta, theta_p, errors, dt, prediction_dim_labs=('X', 'Y', 'Z'), save=False, label='', folder='Figures'):
    """
    Parameters
    ----------
    theta: float Optional (Default: max(theta_p))
        size of window we are predicting
    theta_p: float array
        the times into the future zhat predictions are in [sec]
    errors: float array
        the errors returns from utils.calc_shifted_error (steps, len(theta_p), m),
        where m is the number of output dims
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    time = np.linspace(0, errors.shape[0]*dt, errors.shape[0])

    # Plot a 3d plot for each xyz output dim
    # fig = plt.figure()
    fig = plt.figure(figsize=(8,8))
    axs = []
    for ii in range(0, errors.shape[2]):
        axs.append(plt.subplot(1, errors.shape[2], ii+1, projection='3d'))
        plt.xlabel('Time [sec]')
        plt.ylabel('Theta P [sec]')
        plt.title(f'{prediction_dim_labs[ii]} Error')
        X, Y = np.meshgrid(time, theta_p)
        surf = axs[ii].plot_surface(X, Y, errors[:, :, ii].T,
                cmap=cm.coolwarm, linewidth=0, antialiased=False)
        axs[ii].zaxis.set_major_locator(LinearLocator(10))
        # axs[ii].zaxis.set_major_formatter('{x:.02f}')
        fig.colorbar(surf, shrink=0.5, aspect=5)

    if save:
        plt.savefig(f'{folder}/{label}3d_error_heat_map.jpg')
    plt.show()

def plot_error_heatmap_subplot_dims(theta, theta_p, errors, dt, prediction_dim_labs=('X', 'Y', 'Z'), save=False, label='', folder='Figures'):
    """
    Parameters
    ----------
    theta: float Optional (Default: max(theta_p))
        size of window we are predicting
    theta_p: float array
        the times into the future zhat predictions are in [sec]
    errors: float array
        the errors returns from utils.calc_shifted_error (steps, len(theta_p), m),
        where m is the number of output dims
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    time = np.linspace(0, errors.shape[0]*dt, errors.shape[0])

    # Plot a 2d heat map of the above
    fig = plt.figure()
    axs = []
    for ii in range(0, errors.shape[2]+1):
        axs.append(plt.subplot(errors.shape[2]+1, 1, ii+1))
        if ii < errors.shape[2]:
            plt.xlabel('Time [sec]')
            plt.ylabel('Theta P [sec]')
            plt.title(f'{prediction_dim_labs[ii]} Error | theta={theta}')
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
        plt.savefig(f'{folder}/{label}2d_error_heat_map.jpg')
    plt.show()

def plot_mean_thetap_error_subplot_dims(
        theta, theta_p, errors, dt, prediction_dim_labs=('X', 'Y', 'Z'), save=False, label='', folder='Figures',
        fig=None, axs=None, show=None, all_constants=None):
    """
    Parameters
    ----------
    theta: float Optional (Default: max(theta_p))
        size of window we are predicting
    theta_p: float array
        the times into the future zhat predictions are in [sec]
    errors: float array
        the errors returns from utils.calc_shifted_error (steps, len(theta_p), m),
        where m is the number of output dims
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    time = np.linspace(0, errors.shape[0]*dt, errors.shape[0])

    # Plot avg error over time, averaging over theta_p
    if fig is None:
        fig = plt.figure(figsize=(8,12))
        fig.minimum_y = 0
    if axs is None:
        axs = []
        gen_axs = True
    else:
        gen_axs = False
    for ii in range(0, errors.shape[2]+1):
        if gen_axs:
            axs.append(plt.subplot(errors.shape[2]+1, 1, ii+1))
        if ii < errors.shape[2]:
            plt.title("Error over Time")
            plt.xlabel('Time [sec]')
            plt.ylabel(f'{prediction_dim_labs[ii]} Mean Error Over Theta_P')
            y = np.mean(errors[:, :, ii], axis=1)
            # axs[ii].axhline(np.mean(np.mean(errors[:, :, ii], axis=1)), linestyle='--', c='k')
            if ii == 0:
                axs[ii].plot(time, y, label=label)
            else:
                axs[ii].plot(time, y)

            if ii == 2:
                fig.minimum_y = min(fig.minimum_y, np.amin(y))
        else:
            plt.title("Error over Time")
            plt.xlabel('Time [sec]')
            plt.ylabel(f'2norm Error of Mean Over Theta_P')
            axs[ii].plot(time, np.linalg.norm(np.mean(errors, axis=1), axis=1))

        axs[ii].legend(loc='center left', bbox_to_anchor=(1, 0.75), fontsize=8)
        if all_constants is not None and ii==2:#errors.shape[2]:
            # def print_nested(d, indent=0):
            #     for key, value in d.items():
            #         if isinstance(value, dict):
            #             print('\t' * indent + str(key) + ': ')
            #             print_nested(value, indent+1)
            #         else:
            #             print('\t' * indent + str(key) + f': {value}')
            #
            # print_nested(all_constants)

            def dict_nested2str(d, indent=4, _recursive_call=False):
                str_dict = ''
                if _recursive_call:
                    internal_indent = indent
                else:
                    internal_indent = 0
                # print('internal: ', internal_indent)
                for key, value in d.items():
                    if isinstance(value, dict):
                        str_dict += '\n' + ' ' * internal_indent + str(key) + ': '
                        # str_dict += str(key) + ": "
                        # str_dict += '-woah-' + str(value)
                        str_dict += dict_nested2str(value, indent*2, _recursive_call=True)
                    else:
                        str_dict += '\n' + ' ' * internal_indent + str(key) + f': {value}'
                return str_dict

            axs[ii].text(
                max(time)+20, fig.minimum_y*2,
                ('Constant Parameters\n'
                +'___________________\n'
                + dict_nested2str(all_constants)),
                fontsize=8
            )
            plt.subplots_adjust(right=0.6)

    # plt.tight_layout()
    if save:
        plt.savefig(f'{folder}/{label}error_over_time_avg_tp.jpg')
    if show:
        plt.show()

    return fig, axs



def plot_alpha_theta_p_error_subplot_dims(
        theta, theta_p, errors, dt, prediction_dim_labs=('X', 'Y', 'Z'), save=False, label='', folder='Figures'):
    """
    Parameters
    ----------
    theta: float Optional (Default: max(theta_p))
        size of window we are predicting
    theta_p: float array
        the times into the future zhat predictions are in [sec]
    errors: float array
        the errors returns from utils.calc_shifted_error (steps, len(theta_p), m),
        where m is the number of output dims
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    time = np.linspace(0, errors.shape[0]*dt, errors.shape[0])

    # Plot avg error over time, showing each theta_p line
    plt.figure(figsize=(8,12))
    axs = []
    for ii in range(0, errors.shape[2]+1):
        axs.append(plt.subplot(errors.shape[2]+1, 1, ii+1))
        for tp, _theta_p in enumerate(theta_p):
            alpha = 1 - tp/len(theta_p)
            if ii < errors.shape[2]:
                axs[ii].set_title("Error over Time, Varying Theta_p")
                axs[ii].set_xlabel('Time [sec]')
                axs[ii].set_ylabel(f'{prediction_dim_labs[ii]} Mean Error Over Theta_P')
                axs[ii].plot(time, errors[:, tp, ii], alpha=alpha, label=f"{_theta_p}")#, c='r')
            else:
                axs[ii].set_title("Error over Time")
                axs[ii].set_xlabel('Time [sec]')
                axs[ii].set_ylabel(f'2norm Error of Mean Over Theta_P')
                # axs[ii].plot(time, np.linalg.norm(np.mean(errors, axis=1), axis=1))
                axs[ii].plot(time, np.linalg.norm(errors[:, tp, :], axis=1), alpha=alpha, label=f"{_theta_p}")#, c='r')
        plt.legend(loc=1)

    if save:
        plt.savefig(f'{folder}/{label}error_over_time.jpg')
        print(f'Save figure to {folder}/{label}error_over_time.jpg')
    plt.show()


def plot_mean_time_error_vs_theta_p(
        theta, theta_p, errors, dt, prediction_dim_labs=('X', 'Y', 'Z'), save=False, label='', folder='Figures',
        figure=None, axs=None, show=False, legend_label=None, linestyle='-', title=None, all_constants=None,
        errors_gt=None):
    """
    Parameters
    ----------
    theta: float Optional (Default: max(theta_p))
        size of window we are predicting
    theta_p: float array
        the times into the future zhat predictions are in [sec]
    errors: float array
        the errors returns from utils.calc_shifted_error (steps, len(theta_p), m),
        where m is the number of output dims
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    time = np.linspace(0, errors.shape[0]*dt, errors.shape[0])

    # Plot avg error over theta_p, averaging over time
    if figure is None:
        figure = plt.figure(figsize=(12,10))
    if axs is None:
        axs = []
        gen_axs = True
    else:
        gen_axs = False

    if title is None:
        "Error over Theta_P"
    for ii in range(0, errors.shape[2]):
        if gen_axs:
            axs.append(plt.subplot(errors.shape[2], 1, ii+1))
        axs[ii].set_title(title)
        axs[ii].set_xlabel('Theta_P [sec]')
        axs[ii].set_ylabel(f'{prediction_dim_labs[ii]} Mean Error Over Time')
        axs[ii].plot(theta_p, np.mean(errors[:, :, ii], axis=0), label=legend_label, linestyle=linestyle)
        if errors_gt is not None:
            axs[ii].plot(
                theta_p,
                np.mean(errors_gt[:, :, ii], axis=0),
                label='GT_' + legend_label,
                linestyle='--'
            )
        axs[ii].legend(loc='center left', bbox_to_anchor=(1, 0.75), fontsize=8)
        if all_constants is not None:
            # def print_nested(d, indent=0):
            #     for key, value in d.items():
            #         if isinstance(value, dict):
            #             print('\t' * indent + str(key) + ': ')
            #             print_nested(value, indent+1)
            #         else:
            #             print('\t' * indent + str(key) + f': {value}')
            #
            # print_nested(all_constants)

            def dict_nested2str(d, indent=4, _recursive_call=False):
                str_dict = ''
                if _recursive_call:
                    internal_indent = indent
                else:
                    internal_indent = 0
                # print('internal: ', internal_indent)
                for key, value in d.items():
                    if isinstance(value, dict):
                        str_dict += '\n' + ' ' * internal_indent + str(key) + ': '
                        # str_dict += str(key) + ": "
                        # str_dict += '-woah-' + str(value)
                        str_dict += dict_nested2str(value, indent*2, _recursive_call=True)
                    else:
                        str_dict += '\n' + ' ' * internal_indent + str(key) + f': {value}'
                return str_dict

            axs[ii].text(
                1.1, 0.1,
                ('Constant Parameters\n'
                +'___________________\n'
                + dict_nested2str(all_constants)),
                fontsize=8
            )
            plt.subplots_adjust(right=0.6)

    if save:
        plt.tight_layout()
        plt.savefig(f'{folder}/{label}error_over_tp.jpg')
        print(f'Save figure to {folder}/{label}error_over_tp.jpg')

    if show:
        # for ax in axs:
        #     plt.grid(True)
        plt.tight_layout()
        print('showing fig')
        plt.show()
    return figure, axs

def plot_ldn_repr_error(error, theta, theta_p, z, dt, zhats, prediction_dim_labs=None, save_name=None, folder='data/figures/', max_rows=4, dim_labels=None):
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


    if save_name is None:
        save_name = 'testfig'

    for key in zhats:
        # output dimensionality in meat space
        mm = zhats[key].shape[2]
        break

    jj = min(mm, max_rows)
    kk = int(np.ceil(mm/jj))

    for key in error:
        # print(f"{key}: {error[key].shape}")
        for ii in range(0, len(theta_p)):
            plt.figure(figsize=(int(jj)*4, 12))
            for ll in range(0, mm):
                plt.subplot(jj, kk, ll+1)

                plt.title(f"LDN Representation: q={key} | theta_p={theta_p[ii]}")
                if dim_labels is not None:
                    plt.ylabel(dim_labels[ll])

                steps = z.shape[0]
                shifted_t = get_shifted_t(theta_p[ii], dt, steps, direction='forward')
                t = np.arange(0, z.shape[0]*dt, dt)

                # plt.plot(t, error[key][:, ii, mm-1], label=f'error_{theta_p[ii]}')
                plt.plot(t, z[:, ll-1], label='z')
                plt.plot(shifted_t, z[:, ll-1], label=f'z shift>>{theta_p[ii]}')
                plt.plot(t, zhats[key][:, ii, ll-1], label=f'zhat_{theta_p[ii]}', linestyle='--')
                plt.legend(loc=1)
                plt.xlabel('Time [sec]')
        plt.tight_layout()
        plt.savefig(f"{folder}{save_name}_q={key}-tp={theta_p[ii]}.jpg")
        plt.show()

        # if prediction_dim_labs is None:
        #     prediction_dim_labs = np.arange(0, z.shape[0])
        # plot_error(
        #     theta_p=theta_p, errors=error[key], dt=dt, prediction_dim_labs=prediction_dim_labs,
        #     theta=theta, save=True, label=f"q={key}_")

def plot_2d(time, dat_arr, labels=None, save_name=None, n_rows=4, title=None, show=True):
    j = min(dat_arr.shape[1], n_rows)
    k = int(np.ceil(dat_arr.shape[1]/j))
    if labels is None:
        labels = np.arange(0, dat_arr.shape[1])

    plt.figure(figsize=(int(n_rows)*4, 12))

    for ii in range(0, dat_arr.shape[1]):
        plt.subplot(j, k, ii+1)
        if title is not None:
            plt.title(title)
        plt.ylabel(labels[ii])
        plt.xlabel('Time [sec]')
        plt.plot(time, dat_arr[:, ii])

    plt.tight_layout()

    if save_name is not None:
        plt.savefig(save_name)
    if show:
        plt.show()


def plot_traj_error(time, state, targets, save_name=None):
    plt.figure(figsize=(10, 12))
    plt.subplot(411)
    plt.title('Trajectory Error with Path Planner (XYZ)')
    plt.ylabel('2norm error [m]')
    plt.xlabel('Time [sec]')
    error = []
    for ii in range(0, state.shape[0]):
        error.append(np.linalg.norm((targets[ii, :3]-state[ii, :3])))
    error = np.array(error)
    plt.plot(time, error, label=f"Avg Error={np.mean(error)}")
    plt.legend()
    plt.subplot(412)
    plt.title('X Trajectory')
    plt.ylabel('X Position [m]')
    plt.xlabel('Time [sec]')
    plt.plot(time, state[:, 0], c='k', label='state')
    plt.plot(time, targets[:, 0], linestyle='--', c='y', label='path')
    plt.legend()
    plt.subplot(413)
    plt.title('Y Trajectory')
    plt.ylabel('Y Position [m]')
    plt.xlabel('Time [sec]')
    plt.plot(time, state[:, 1], c='k', label='state')
    plt.plot(time, targets[:, 1], linestyle='--', c='y', label='path')
    plt.legend()
    plt.subplot(414)
    plt.title('Z Trajectory')
    plt.ylabel('Z Position [m]')
    plt.xlabel('Time [sec]')
    plt.plot(time, state[:, 2], c='k', label='state')
    plt.plot(time, targets[:, 2], linestyle='--', c='y', label='path')
    plt.legend()
    plt.tight_layout()
    plt.savefig('100_linear_targets_traj_error.png')
    plt.show()


def traj_3d_gif(
        dat_arr, time, save_name, sec_between_data_captures=1, tail_len=5,
        time_multiplier=10, flip_z=True, regen_figs=True, folder='Figures'):
    """
    Parameters
    ----------
    dat_arr: 2d array of floats
        shape(steps, dim)
    time: 1d array of floats
        shape(steps,)
        the cumulative time [sec]
    sec_between_data_captures: float
        The time resolution to capture data frames at
    fps: int
        frames per second of gif
    tail_len: float
        how many seconds of path to show at a given moment [sec]
    flip_z: bool, Optional (Default: True)
        flip sign of z. Helpful for plotting data in NED coordinates
    regen_figs: bool, Optional (Default:True)
        starts by clearing .cache then generates figures to create gif
        if False will read figures from file in the .cache
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists('.cache'):
        os.makedirs('.cache')
    if regen_figs:
        for file in os.listdir('.cache'):
            if file.endswith('.png'):
                os.remove(f".cache/{file}")

    if flip_z:
        dat_arr[:, 2] *= -1
        dat_arr[:, 5] *= -1
    print(dat_arr.shape)
    minx, maxx = [np.amin(dat_arr[:, 0]), np.amax(dat_arr[:, 0])]
    miny, maxy = [np.amin(dat_arr[:, 1]), np.amax(dat_arr[:, 1])]
    minz, maxz = [np.amin(dat_arr[:, 2]), np.amax(dat_arr[:, 2])]
    dt = np.mean(np.diff(time))
    tail_steps = int(tail_len/dt)

    last_cap = 0.0
    ii = 0
    fnames = []
    for step, runtime in enumerate(tqdm(time)):
        if (runtime - last_cap) >= sec_between_data_captures:
            last_cap = runtime

            fname = f".cache/{ii:04d}.png"
            fnames.append(fname)
            ii += 1

            if regen_figs:
                fig = plt.figure()
                ax = plt.subplot(111, projection='3d')
                ax.set_xlim(minx, maxx)
                ax.set_ylim(miny, maxy)
                ax.set_zlim(minz, maxz)
                plt.plot(
                    dat_arr[:step, 0],
                    dat_arr[:step, 1],
                    dat_arr[:step, 2],
                    alpha=0.25,
                    label='Past Trajectory'
                )
                plt.plot(
                    dat_arr[max(0, step-tail_steps):step, 0],
                    dat_arr[max(0, step-tail_steps):step, 1],
                    dat_arr[max(0, step-tail_steps):step, 2],
                    label=f"Last {tail_len} sec"
                )
                ax.scatter(
                    dat_arr[step-1, 0],
                    dat_arr[step-1, 1],
                    dat_arr[step-1, 2],
                    color='r',
                )
                plt.plot([maxx, maxx-1], [maxy, maxy], [maxz, maxz], label='1m in x', c='r')
                plt.plot([maxx, maxx], [maxy, maxy-1], [maxz, maxz], label='1m in y', c='g')
                plt.plot([maxx, maxx], [maxy, maxy], [maxz, maxz-1], label='1m in z', c='b')

                ax.text2D(0.75, 0.95, f"Time: {time[step]} sec", transform=ax.transAxes)
                plt.legend()

                plt.savefig(fname)
                fig.clear()
                plt.close()


    frames = []
    for fname in fnames:
        frames.append(imageio.imread(fname))

    # Save them as frames into a gif
    kargs = { 'duration': time[-1]/time_multiplier }
    imageio.mimsave(
        f"{folder}/{save_name}_{time_multiplier}x.gif",
        frames,
        # fps=120
        # duration=time[-1]/time_multiplier
    )#, time[-1]/time_multiplier)#'GIF', **kargs)

def plot_data_distribution(dat_arr, dim_labels=None, bins=None, n_rows=4, save_name=None, show=True, title=None):
    if bins is None:
        bins = 20

    j = min(dat_arr.shape[1], n_rows)
    k = int(np.ceil(dat_arr.shape[1]/j))
    plt.figure(figsize=(int(n_rows)*4, 12))
    for ii in range(0, dat_arr.shape[1]):
        plt.subplot(j, k, ii+1)
        if title is not None:
            plt.title(title)
        a = dat_arr[:, ii]
        _ = plt.hist(a, bins=bins)#'auto')  # arguments are passed to np.histogram
        # plt.title("Histogram with 'auto' bins")
        if dim_labels is not None:
            plt.ylabel(f"{dim_labels[ii]}")
        else:
            plt.ylabel(f"{ii}")
        # hist, bin_edges = np.histogram(a)#, bins=bins)
        # plt.hist(hist, bin_edges)
    plt.tight_layout()

    if save_name is not None:
        plt.savefig(save_name)
    if show:
        plt.show()

def plot_sliding_distance(positions, dt, theta):
    theta_p = np.linspace(dt, theta, 10)
    distances = []
    for tp in theta_p:
        tp_steps = int(tp/dt)
        # print('tp: ', tp)
        # print('tp steps: ', tp_steps)
        dists = []
        # print('pos shape: ', positions.shape)
        for pp in range(0, positions.shape[0]):
            if pp + tp_steps >= positions.shape[0]:
                continue
            dists.append(np.linalg.norm(positions[pp+tp_steps] - positions[pp]))
        distances.append(np.mean(dists))
        # print('dists: ', dists)
    plt.figure()
    plt.plot(theta_p, distances)
    plt.show()

if __name__ == '__main__':
    dat = DataHandler('llp_pd', 'data/databases')
    # data = dat.load(save_location='100_linear_targets', parameters=['state', 'target', 'time'])
    # plot_traj_error(data['time'], data['state'], data['target'])

    # plot_2d(
    #     data['time'],
    #     data['state'],
    #     labels=['x', 'y', 'z', 'dx', 'dy', 'dz', 'a', 'b', 'g', 'da', 'db', 'dg'],
    #     save_name=None,
    #     n_rows=4
    # )
    # dat = DataHandler('codebase_test_set', 'data/databases')
    # dat = DataHandler('llp_pd', 'data/databases')
    # # dat.load('100_linear_targts', ['state', 'ctrl'])
    # data = dat.load(
    #     save_location='100_linear_targets',
    #     parameters=['state', 'clean_u', 'time']
    # )

    # traj_3d_gif(data['state'], data['time'], save_name='100_linear_targets', time_multiplier=1, regen_figs=False)
    # plot_data_distribution(
    #     data['state'],
    #     dim_labels=['x', 'y', 'z', 'dx', 'dy', 'dz', 'a', 'b', 'g', 'da', 'db', 'dg']
    # )
