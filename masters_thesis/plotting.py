import matplotlib.pyplot as plt
import imageio
import os
import numpy as np

def plot_pred(time, z, zhat, theta_p, size_out, gif_name, animate=True, window=0.5, step=0.1):
    plt.figure()
    print(z.shape)
    print(zhat.shape)
    for tt, _theta_p in enumerate(theta_p):
        plt.subplot(len(theta_p), 1, tt+1)
        plt.title(f"{_theta_p} prediction")
        plt.xlabel("Time[sec]")
        plt.plot(time-_theta_p, z, label='z shifted by {_theta_p}')
        # plt.plot(time, z, label='z shifted by {_theta_p}')
        plt.plot(time, np.squeeze(zhat[:, tt, :]), label='zhat', linestyle='--')
        plt.legend()

    plt.figure()
    axs = []
    labels = ['x', 'y', 'z']
    for ii in range(0, size_out):
        axs.append(plt.subplot(size_out, 1, ii+1))
        axs[ii].plot(time, np.squeeze(zhat[:, :, ii]))

        plt.gca().set_prop_cycle(None)
        for pred in theta_p:
            axs[ii].plot(time-pred, z.T[ii], linestyle='--')

        axs[ii].legend(
            [f'{labels[ii]}hat at: ' + str(round(tp, 3)) for tp in theta_p]
            + [f'{labels[ii]} shifted: ' + str(round(tp, 3)) for tp in theta_p],
            loc=1)

    if animate:
        start = 0.0
        stop = window
        ss = 0
        filenames = []
        while stop <= time[-1]:
            for ax in axs:
                ax.set_xlim(start, stop)
            filename = f".cache/img_{ss:08d}.jpg"
            filenames.append(filename)
            plt.savefig(filename)
            start += step
            stop += step
            ss += 1

        with imageio.get_writer(gif_name, mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                os.remove(filename)
    plt.show()

def plot_error(theta_p, errors):
    plt.figure()
    for ii in range(0, len(theta_p)):
        plt.subplot(len(theta_p), 1, ii+1)
        plt.title(f"{theta_p[ii]} prediction 2norm error")
        plt.plot(errors[ii])
    plt.show()


