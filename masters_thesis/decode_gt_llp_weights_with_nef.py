import nengo
import sys
import numpy as np
from abr_analyze import DataHandler
import json
from masters_thesis.utils.eval_utils import encode_ldn_data, load_data_from_json, RMSE
from masters_thesis.utils.plotting import plot_x_vs_xhat, plot_prediction_vs_gt

def run(json_params, weights=None):
    json_params, c_state, z_state, times = load_data_from_json(json_params)
    print('c state: ', c_state.shape)
    print('z state: ', z_state.shape)

    # split into separate dicts for easier referencing
    data_params = json_params['data']
    llp_params = json_params['llp']
    params = json_params['general']

    dat = DataHandler(data_params['db_name'], data_params['database_dir'])

    # Get the LDN representation of our state to predict.
    # This will be shifted back by theta and used as ground truth for the llp prediction.
    # Since the LDN represents a window of length theta with q coefficients,
    # if the LLP outputs q coefficients at time t that represent the signal theta
    # seconds into the future, this would be the same as the ldn representation
    # with the same q and theta, but in t+theta seconds
    GT_Z = encode_ldn_data(
        theta=llp_params['theta'],
        q=llp_params['q'],
        z=z_state,
        dt=params['dt']
    )

    # Shift GT back by theta
    theta_steps = int(llp_params['theta']/params['dt'])
    GT_Z = GT_Z[theta_steps:]
    # GT_z_decoded = z_state[theta_steps:]

    # stop theta_steps from the end since we won't have GT for those steps
    c_state = c_state[:-theta_steps]

    model = nengo.Network()
    with model:
        ens = nengo.Ensemble(
            n_neurons=llp_params['n_neurons'],
            dimensions=c_state.shape[1],
            neuron_type=llp_params['neuron_model'](),
            radius=1,#np.sqrt(c_state.shape[1]),
            seed=0
        )
        pred = nengo.Node(size_in=GT_Z.shape[1])
        if weights is None:
            conn = nengo.Connection(
                ens,
                pred,
                eval_points=c_state,
                function=GT_Z,
                synapse=None
            )
        else:
            print('Testing with trained weights')
            conn = nengo.Connection(
                ens.neurons,
                pred,
                transform=weights,
                synapse=None
            )

            def in_func(t):
                return c_state[int((t-params['dt'])/params['dt'])]
            in_node = nengo.Node(in_func)

            nengo.Connection(in_node, ens, synapse=None)
            # probe output
            net_out = nengo.Probe(pred, synapse=None)


    sim = nengo.Simulator(model, dt=params['dt'])
    with sim:
        if weights is None:
            eval_pt, tgt, decoded = nengo.utils.connection.eval_point_decoding(
                conn,
                sim
            )
        else:
            # NOTE eval function does not work on neur>post connection
            # eval_pt, tgt, decoded = nengo.utils.connection.eval_point_decoding(
            #     conn,
            #     sim,
            #     eval_points=c_state
            # )
            sim.run(c_state.shape[0]*params['dt'])
            tgt = GT_Z
            eval_pt = c_state
            decoded = sim.data[net_out]

        weights = sim.signals[sim.model.sig[conn]["weights"]]

    return(RMSE(tgt, decoded), eval_pt, tgt, decoded, weights)


if __name__ == '__main__':
    # load in all parameters
    # with open('parameter_sets/nni_nef_decode_params.json') as fp:
    with open(sys.argv[1]) as fp:
        json_params = json.load(fp)

    # NOTE manual varying of paramaters
    # json_params['llp']['n_neurons'] = 1000
    # json_params['data']['q_c'] = 0
    # json_params['data']['theta_c'] = 0.0
    # json_params['data']['q_u'] = 1
    # json_params['data']['theta_u'] = 3.59

    # NOTE first pass that saves weights to npz
    # rmse, eval_pts, target_pts, decoded_pts, weights = run(
    #     json_params, param_id, plot=True
    # )
    # # for ii in range(0, target_pts.shape[1]):
    # #     plot_x_vs_xhat(tgt[:, ii][:, np.newaxis], decoded[:, ii][:, np.newaxis])
    #
    # plot_prediction_vs_gt(target_pts, decoded_pts, json_params)
    # np.savez_compressed('weights.npz', weights=weights)

    # NOTE second pass loading in weights saved to npz
    rmse, eval_pts, target_pts, decoded_pts, weights = run(
        json_params,
        # weights=weights # np.load('weights.npz')['weights']
        weights=np.load('weights.npz')['weights']
    )
    plot_prediction_vs_gt(
        target_pts,
        decoded_pts,
        json_params['llp']['q'],
        json_params['llp']['theta'],
        json_params['general']['theta_p']
        )

    # NOTE example of looping through a parameter
    # plt.figure()
    # for neur in [1000, 2000, 3000, 5000]:
    #     json_params['llp']['n_neurons'] = neur
    #     rmse = run(json_params, param_id)
    #     plt.scatter(neur, rmse)
    # plt.show()


