import nengo
import sys
import numpy as np
from abr_analyze import DataHandler
import json
from masters_thesis.utils.eval_utils import encode_ldn_data, load_data_from_json, RMSE, flip_odd_ldn_coefficients, decode_ldn_data
from masters_thesis.utils.plotting import plot_x_vs_xhat, plot_prediction_vs_gt
import masters_thesis.utils.plotting as plotting

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
    # flip signs of odd coefficients per dim to create coefficients that align
    # with llp decoding where theta_p/theta=1 is theta in the future, not the past
    GT_Z = flip_odd_ldn_coefficients(Z=GT_Z, q=llp_params['q'])

    # Shift GT back by theta
    theta_steps = int(llp_params['theta']/params['dt'])
    GT_Z = GT_Z[theta_steps:]

    # stop theta_steps from the end since we won't have GT for those steps
    c_state = c_state[:-theta_steps]

    model = nengo.Network()
    # print(llp_params['neuron_model'])
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
            print("Solving for weights directly")
            eval_pt, tgt, decoded = nengo.utils.connection.eval_point_decoding(
                conn,
                sim
            )
        else:
            print("Running with pretrained weights")
            sim.run(c_state.shape[0]*params['dt'])
            tgt = GT_Z
            eval_pt = c_state
            decoded = sim.data[net_out]

        weights = sim.signals[sim.model.sig[conn]["weights"]]

    return(RMSE(tgt, decoded), eval_pt, tgt, decoded, weights, z_state)


def run_variation_comparison(json_fp, key_list, variation_list, labels, show_prediction=False, save=False, load=False):

    # for changing nested values
    def nested_set(dic, keys, value):
        for key in keys[:-1]:
            dic = dic.setdefault(key, {})
        dic[keys[-1]] = value

    figure = None
    axs = None

    with open(json_fp) as fp:
        json_params = json.load(fp)

    if save or load:
        dat = DataHandler(
            db_name=json_params['data']['db_name']+'_results',
            database_dir=json_params['data']['database_dir']
        )

        dat.save(
            save_location=json_fp.split('/')[-1] + '/params',
            data=json_params,
            overwrite=True
        )

    for vv, var in enumerate(variation_list):
        if save or load:
            save_name = ''
            save_name = '/'.join(key_list)
            save_name = json_fp.split('/')[-1] + '/results/' + save_name + '/' + str(var)

        if not load:
            nested_set(json_params, key_list, var)
            print(f"Updated json_params by changing {key_list} to {var}\n{json_params}")

            # NOTE first pass that saves weights to npz
            print(f"Getting decoded points from training set{json_fp}")
            json_params['data']['dataset_range'] = json_params['data']['train_range']
            rmse, eval_pts, target_pts, decoded_pts, weights, z_state = run(json_params)

            print('Getting test results')
            json_params['data']['dataset_range'] = json_params['data']['test_range']
            rmse, eval_pts, target_pts, decoded_pts, weights, z_state = run(json_params, weights=weights)

            theta_steps = int(json_params['llp']['theta']/json_params['general']['dt'])

            # account for steps removed from GT
            n_steps = np.diff(json_params['data']['dataset_range'])[0] - theta_steps

            print('Calculating RMSE for each theta_p')
            RMSEs = np.zeros((n_steps, int(len(json_params['general']['theta_p']))))#, m))
            for ii, tp in enumerate(json_params['general']['theta_p']):
                tp_steps = int(tp/json_params['general']['dt'])
                x = decode_ldn_data(
                    Z=target_pts,
                    q=json_params['llp']['q'],
                    theta=json_params['llp']['theta'],
                    theta_p=tp
                )
                xhat = decode_ldn_data(
                    Z=decoded_pts,
                    q=json_params['llp']['q'],
                    theta=json_params['llp']['theta'],
                )
                err = RMSE(x.T, xhat.T)
                RMSEs[:, ii] = err#[:, np.newaxis]
        else:
            print(f"Loading results from: {save_name}")
            data = dat.load(
                parameters=[
                    'RMSE',
                    'target_pts',
                    'decoded_pts',
                    'z_state',
                    'decoded_z',
                    'decoded_zhat'
                ],
                save_location=save_name
            )
            RMSEs = data['RMSE']
            target_pts = data['target_pts']
            decoded_pts = data['decoded_pts']
            z_state = data['z_state']
            x = data['decoded_z']
            xhat = data['decoded_zhat']


        if save:
            print(f"Saving results to: {save_name}")
            dat.save(
                save_location=save_name,
                data={
                    'RMSE': RMSEs,
                    'target_pts': target_pts,
                    'decoded_pts': decoded_pts,
                    'z_state': z_state,
                    'decoded_z': x,
                    'decoded_zhat': xhat
                },
                overwrite=True
            )

        if vv+1 == len(variation_list):
            save_fig = True
            print('saving image this time')
        else:
            save_fig = False
        show = save_fig
        # print('Plotting')
        figure, axs = plotting.plot_mean_time_error_vs_theta_p(
            theta_p=json_params['general']['theta_p'],
            errors=RMSEs[:, :, np.newaxis],
            dt=json_params['general']['dt'],
            theta=json_params['llp']['theta'],
            figure=figure,
            axs=axs,
            show=show,
            legend_label=labels[vv],
            save=save_fig,
            folder='data',
            label='nef_decode_vary_q_other_'
        )

        if show_prediction:
            plot_prediction_vs_gt(
                tgt=target_pts,
                decoded=decoded_pts,
                q=json_params['llp']['q'],
                theta=json_params['llp']['theta'],
                theta_p=json_params['general']['theta_p'],
                z_state=z_state,#[theta_steps:]
                xlim=[0, 1000],
                show=True,
                save=True,
                savename=f"data/pred_vs_gt_q_{json_params['llp']['q']}.jpg"
            )

def run_json_comparison(json_list, labels, show_prediction=False):
    figure = None
    axs = None
    for jj, json_fp in enumerate(json_list):
        with open(json_fp) as fp:
            json_params = json.load(fp)

        # NOTE first pass that saves weights to npz
        print(f"Parameter set: {json_fp}")
        print("Getting decoded points from training set")
        json_params['data']['dataset_range'] = [0, 80000]
        rmse, eval_pts, target_pts, decoded_pts, weights, z_state = run(json_params)

        print('Getting test results')
        json_params['data']['dataset_range'] = [80000, 100000]
        rmse, eval_pts, target_pts, decoded_pts, weights, z_state = run(json_params, weights=weights)

        theta_steps = int(json_params['llp']['theta']/json_params['general']['dt'])

        # Account for the steps removed from GT
        n_steps = np.diff(json_params['data']['dataset_range'])[0] - theta_steps

        # m = len(json_params['data']['z_dims'])
        # json_params['general']['theta_p'] = np.arange(
        #         json_params['general']['dt'], json_params['llp']['theta'], json_params['general']['dt']*10
        #     )
        #
        # json_params['general']['theta_p'] = np.linspace(
        #         json_params['general']['dt'], json_params['llp']['theta'], 10
        #     )

        print('Calculating RMSE for each theta_p')
        RMSEs = np.zeros((n_steps, int(len(json_params['general']['theta_p']))))#, m))
        for ii, tp in enumerate(json_params['general']['theta_p']):
            tp_steps = int(tp/json_params['general']['dt'])
            # decode GT coefficients
            x = decode_ldn_data(
                Z=target_pts,
                q=json_params['llp']['q'],
                theta=json_params['llp']['theta'],
                theta_p=tp
            )
            # decode prediction
            xhat = decode_ldn_data(
                Z=decoded_pts,
                q=json_params['llp']['q'],
                theta=json_params['llp']['theta'],
            )
            err = RMSE(x.T, xhat.T)
            RMSEs[:, ii] = err


        if jj+1 == int(len(json_list)):
            save = True
        else:
            save = False
        show = save

        # print('Plotting')
        figure, axs = plotting.plot_mean_time_error_vs_theta_p(
            theta_p=json_params['general']['theta_p'],
            errors=RMSEs[:, :, np.newaxis],
            dt=json_params['general']['dt'],
            theta=json_params['llp']['theta'],
            figure=figure,
            axs=axs,
            show=show,
            legend_label=labels[jj],
            save=save,
            folder='data',
            label='nef_decode_'
        )

        if show_prediction:
            plot_prediction_vs_gt(
                tgt=target_pts,
                decoded=decoded_pts,
                q=json_params['llp']['q'],
                theta=json_params['llp']['theta'],
                theta_p=json_params['general']['theta_p'],
                z_state=z_state,#[theta_steps:]
                xlim=[0, 1000],
                show=True,
                save=True,
                savename=f"data/pred_vs_gt_q_{json_params['llp']['q']}.jpg"
            )

if __name__ == '__main__':
    # labs = [
    #         'neurons=1000, theta=1', 'neurons=1000, theta=0.1',
    #         'neurons=10,000, theta=1', 'neurons=10,000, theta=0.1',
    #         'n_neurons=1000, theta=1, 5x data', 'n_neurons=1000, theta=0.1, 5x data'
    # ]
    #
    # json_list = [
    #     'parameter_sets/params_0015a.json',
    #     'parameter_sets/params_0015b.json',
    #     'parameter_sets/params_0015c.json',
    #     'parameter_sets/params_0015d.json',
    #     'parameter_sets/params_0015e.json',
    #     'parameter_sets/params_0015f.json',
    # ]
    # run_json_comparison(json_list, labs)

    run_variation_comparison(
        json_fp='parameter_sets/params_0016.json',
        key_list=['llp', 'n_neurons'],
        variation_list=[100, 200],
        labels=['100', '200'],
        show_prediction=False,
        save=False,
        load=True
    )
