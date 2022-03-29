from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import imageio
import os
import numpy as np

def plot_pred(
        time, z, zhat, theta_p, size_out, gif_name, animate=True, window=None,
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
        axs[ii].plot(time, np.squeeze(zhat[:, :, ii]))

        plt.gca().set_prop_cycle(None)
        for pred in theta_p:
            axs[ii].plot(time-pred, z.T[ii], linestyle='--')
            axs[ii].plot(time, z.T[ii], linestyle='-')
            # axs[ii].plot(time, z.T[ii], linestyle='--')
            axs[ii].set_ylim(2*np.amin(z.T[ii]), 2*np.amax(z.T[ii]))

        axs[ii].legend(
            [f'{labels[ii]}hat at: ' + str(round(tp, 3)) for tp in theta_p]
            + [f'{labels[ii]} shifted: ' + str(round(tp, 3)) for tp in theta_p]
            + [f'{labels[ii]} actual: ' + str(round(tp, 3)) for tp in theta_p],
            loc=1)
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

def plot_error(theta_p, errors, dt, output_labs=('X', 'Y', 'Z'), theta=None, save=False, label='', folder='Figures'):
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
    if not os.path.exists(folder):
        os.makedirs(folder)

    time = np.linspace(0, errors.shape[0]*dt, errors.shape[0])

    if theta is None:
        theta = max(theta_p)

    if 1 <= len(theta_p) < 10:
        plt.figure(figsize=(8,8))
        for ii in range(0, len(theta_p)):
            plt.subplot(len(theta_p), 1, ii+1)
            plt.title(f"{theta_p[ii]} prediction 2norm error")
            plt.plot(errors[:, ii, :])
        plt.show()

        if save:
            plt.savefig(f'{folder}/{label}2norm_over_time.jpg')

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
            # axs[ii].zaxis.set_major_formatter('{x:.02f}')
            fig.colorbar(surf, shrink=0.5, aspect=5)
        # plt.show()

        if save:
            plt.savefig(f'{folder}/{label}3d_error_heat_map.jpg')


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
            plt.savefig(f'{folder}/{label}2d_error_heat_map.jpg')


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
            plt.savefig(f'{folder}/{label}error_over_time_avg_tp.jpg')

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
                axs[ii].set_ylabel(f'{output_labs[ii]} Mean Error Over Theta_P')
                axs[ii].plot(time, errors[:, tp, ii], alpha=alpha, label=f"{_theta_p}")#, c='r')
            else:
                axs[ii].set_title("Error over Time")
                axs[ii].set_xlabel('Time [sec]')
                axs[ii].set_ylabel(f'2norm Error of Mean Over Theta_P')
                # axs[ii].plot(time, np.linalg.norm(np.mean(errors, axis=1), axis=1))
                axs[ii].plot(time, np.linalg.norm(errors[:, tp, :], axis=1), alpha=alpha, label=f"{_theta_p}")#, c='r')
        plt.legend(loc=1)

    # plt.show()

    if save:
        plt.savefig(f'{folder}/{label}error_over_time.jpg')

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
            plt.savefig(f'{folder}/{label}error_over_tp.jpg')

    if not save:
        plt.show()

def plot_ldn_repr_error(error, theta, theta_p, z, dt, zhats, output_labs=None, label=None, folder='Figures', max_rows=4):
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

        if output_labs is None:
            output_labs = np.arange(0, z.shape[0])
        plot_error(
            theta_p=theta_p, errors=error[key], dt=dt, output_labs=output_labs,
            theta=theta, save=True, label=f"q={key}_")


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
        This is the 
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

def plot_data_distribution(dat_arr, dim_labels=None, bins=None, n_rows=4):
    if bins is None:
        bins = [0, 0.150]

    j = min(dat_arr.shape[1], n_rows)
    k = int(np.ceil(dat_arr.shape[1]/j))
    plt.figure()
    for ii in range(0, dat_arr.shape[1]):
        plt.subplot(j, k, ii+1)
        a = dat_arr[:, ii]
        _ = plt.hist(a, bins='auto')  # arguments are passed to np.histogram
        # plt.title("Histogram with 'auto' bins")
        if dim_labels is not None:
            plt.title(f"{dim_labels[ii]}")
        else:
            plt.title(f"{ii}")
        # hist, bin_edges = np.histogram(a)#, bins=bins)
        # plt.hist(hist, bin_edges)
    plt.show()



if __name__ == '__main__':
    from abr_analyze import DataHandler

    dat = DataHandler('llp_pd', 'data/databases')
    data = dat.load(save_location='100_linear_targets', parameters=['state', 'target', 'time'])
    plot_traj_error(data['time'], data['state'], data['target'])

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
