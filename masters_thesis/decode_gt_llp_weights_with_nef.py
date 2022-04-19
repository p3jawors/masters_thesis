import nengo
import sys
import numpy as np
from abr_analyze import DataHandler
import json
from masters_thesis.utils.eval_utils import time_series_to_ldn_polynomials, load_data_from_json, decode_ldn_data
import matplotlib.pyplot as plt

# folder = 'data/figures/'
# if not os.path.exists(folder):
#     os.makedirs(folder)

def run(json_params, param_id, plot=False, weights=None):
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
    GT_Z = time_series_to_ldn_polynomials(
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

    # DEBUG PRINT
    # print('GT_Z: ', GT_Z.shape)
    # print('context: ', c_state.shape)
    # print('theta steps: ', theta_steps)
    # print(f"{llp_params['q']=}")
    # print(f"{llp_params['theta']=}")
    # print(f"{params['dt']=}")

    model = nengo.Network()
    with model:
        ens = nengo.Ensemble(
            n_neurons=llp_params['n_neurons'],
            dimensions=c_state.shape[1],
            neuron_type=llp_params['neuron_model'](),
            radius=np.sqrt(c_state.shape[1]),
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
            # eval_pt, tgt, decoded = nengo.utils.connection.eval_point_decoding(
            #     conn,
            #     sim,
            #     eval_points=c_state
            # )
            print('c state shape: ', c_state.shape)
            print('dt: ', params['dt'])
            print('steps: ', params['dt']*c_state.shape[0])
            sim.run(c_state.shape[0]*params['dt'])
            print(sim.trange())
            tgt = GT_Z
            eval_pt = c_state
            decoded = sim.data[net_out]

        print('tgt: ', tgt.shape)
        print('eval_pts: ', eval_pt.shape)
        print('decoded: ', decoded.shape)


        weights = sim.signals[sim.model.sig[conn]["weights"]]

    return(RMSE(tgt, decoded), eval_pt, tgt, decoded, weights)

def RMSE(x, xhat):
    err = 0
    for ii in range(0, x.shape[0]):
        for jj in range(0, x.shape[1]):
            err += (xhat[ii, jj] - x[ii, jj])**2
    err /= (ii+1)*(jj+1)
    err = np.sqrt(err)
    return err


def plot_x_vs_xhat(x, xhat):
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

def plot_prediction_vs_gt(tgt, decoded, json_params):
    plt.figure()
    for ii in range(0, tgt.shape[1]):
        plt.subplot(tgt.shape[1], 1, ii+1)
        plt.plot(tgt[:, ii])
        # plt.gca().set_prop_cycle(None)
        plt.plot(decoded[:, ii], linestyle='--')

    plt.figure()
    zhat_GT = decode_ldn_data(
        Z=tgt,
        q=json_params['llp']['q'],
        theta=json_params['llp']['theta'],
        theta_p=json_params['general']['theta_p']
    )
    zhat_pred = decode_ldn_data(
        Z=decoded,
        q=json_params['llp']['q'],
        theta=json_params['llp']['theta'],
        theta_p=json_params['general']['theta_p']
    )
    print(zhat_pred.shape)

    for ii in range(0, zhat_GT.shape[2]):
        for jj in range(0, zhat_GT.shape[1]):
            plt.subplot(zhat_GT.shape[2], zhat_GT.shape[1], ii*(zhat_GT.shape[1]) + jj+1)
            if ii == 0:
                plt.title(f"theta={json_params['llp']['theta']} | theta_p={json_params['general']['theta_p'][jj]}")
            if jj == 0:
                plt.ylabel(f"dim_{ii}")
            plt.plot(zhat_GT[:, jj, ii])
            # plt.gca().set_prop_cycle(None)
            plt.plot(zhat_pred[:, jj, ii], linestyle='--')

    plt.show()

if __name__ == '__main__':
    # load in all parameters
    # with open('parameter_sets/nni_nef_decode_params.json') as fp:
    with open(sys.argv[1]) as fp:
        json_params = json.load(fp)
    param_id = sys.argv[1].split('/')[-1].split('.')[0]
    # json_params['llp']['n_neurons'] = 1000
    # json_params['data']['q_c'] = 0
    # json_params['data']['theta_c'] = 0.0
    # json_params['data']['q_u'] = 1
    # json_params['data']['theta_u'] = 3.59

    rmse, eval_pts, target_pts, decoded_pts, weights = run(
        json_params, param_id, plot=True
    )
    # for ii in range(0, target_pts.shape[1]):
    #     plot_x_vs_xhat(tgt[:, ii][:, np.newaxis], decoded[:, ii][:, np.newaxis])

    plot_prediction_vs_gt(target_pts, decoded_pts, json_params)
    # np.savez_compressed('weights.npz', weights=weights)

    # plt.figure()
    # for neur in [1000, 2000, 3000, 5000]:
    #     json_params['llp']['n_neurons'] = neur
    #     rmse = run(json_params, param_id)
    #     plt.scatter(neur, rmse)
    # plt.show()

    rmse, eval_pts, target_pts, decoded_pts, weights = run(
        json_params, param_id, plot=True,
        weights=weights # np.load('weights.npz')['weights']
    )
    plot_prediction_vs_gt(target_pts, decoded_pts, json_params)
