import matplotlib.pyplot as plt
import imageio
import os
import numpy as np

def plot_pred(data, theta_p, size_out, gif_name, animate=True, window=0.5, step=0.1):
    plt.figure()
    plt.title('Predictions over time')
    plt.plot(data['time'], data['z'])
    plt.plot(data['time'], data['zhat'])
    plt.legend(['z'] + [str(tp) for tp in theta_p])

    plt.figure()
    axs = []
    for ii in range(0, size_out):
        axs.append(plt.subplot(ii+1, 1, size_out))
        axs[ii].plot(data['time'], data['zhat'])

        plt.gca().set_prop_cycle(None)
        for pred in theta_p:
            axs[ii].plot(data['time']-pred, data['z'].T[ii], linestyle='--')

        axs[ii].legend(
            ['zhat at: ' + str(round(tp, 3)) for tp in theta_p]
            + ['z shifted: ' + str(round(tp, 3)) for tp in theta_p],
            loc=1)

    if animate:
        start = 0.0
        stop = window
        ss = 0
        filenames = []
        while stop <= data['time'][-1]:
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
