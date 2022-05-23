import nengo
import numpy as np
import matplotlib.pyplot as plt

def run(weights=None, dt=0.001):
    # input
    x = np.linspace(0, 1, 100)
    # desired output (scale by 2)
    y = np.linspace(0, 2, 100)

    model = nengo.Network()
    with model:
        ens = nengo.Ensemble(
            n_neurons=1000,
            dimensions=1,
            neuron_type=nengo.LIFRate(),
            seed=0
        )
        out = nengo.Node(size_in=1)

        # Connect first time with input/output points
        # and use NEF to solve for weights
        if weights is None:
            conn = nengo.Connection(
                ens,
                out,
                eval_points=x[:, np.newaxis],
                function=y[:, np.newaxis],
                synapse=None
            )
        # use weights from first pass to try and recreate results
        else:
            conn = nengo.Connection(
                ens.neurons,
                out,
                transform=weights,
                synapse=None
            )

            # pass each input point sequentially
            def in_func(t):
                return x[int((t-dt)/dt)]
            in_node = nengo.Node(in_func)

            nengo.Connection(in_node, ens, synapse=None)
            # probe output
            net_out = nengo.Probe(out, synapse=None)

    sim = nengo.Simulator(model)
    with sim:
        if weights is None:
            eval_pts, target_pts, decoded_pts = nengo.utils.connection.eval_point_decoding(
                conn, sim
            )
            print(len(eval_pts), 'steps')
        else:
            # NOTE does not seem to work for ens.neurons connections
            # eval_pts, target_pts, decoded_pts = nengo.utils.connection.eval_point_decoding(
            #     conn, sim,
            #     eval_points=x[:, np.newaxis]
            # )
            sim.run(x.shape[0]*dt)
            print(x.shape[0], ' steps')
            target_pts = y
            decoded_pts = sim.data[net_out]

        weights = sim.signals[sim.model.sig[conn]["weights"]]

    plt.figure()
    plt.plot(target_pts, label='target')
    plt.plot(decoded_pts, label='decoded')
    plt.legend()
    plt.show()

    return weights

weights = run()
_ = run(weights)
