import json
import numpy as np
import os
import nengo
import sys

from abr_analyze import DataHandler
from masters_thesis.utils.eval_utils import (
        decode_ldn_data,
        calc_shifted_error,
        load_data_from_json,
        RMSE,
        encode_ldn_data,
        decode_ldn_data,
        flip_odd_ldn_coefficients
)
from masters_thesis.utils.plotting import plot_pred, plot_prediction_vs_gt
import predict_llp

# TODO update eval so we can have separate db for train data and results

folder = 'data/figures/'
if not os.path.exists(folder):
    os.makedirs(folder)

def run(json_params, param_id, load_results=False, save=False, plot=True):
    json_params, c_state, z_state, times = load_data_from_json(json_params)
    # n_repeats = 10
    # c_state = np.tile(c_state, n_repeats)
    # z_state = np.tile(z_state, n_repeats)
    # times = np.arange(0, 

    # split into separate dicts for easier referencing
    data_params = json_params['data']
    llp_params = json_params['llp']
    params = json_params['general']

    dat = DataHandler(data_params['db_name'], data_params['database_dir'])

    if load_results:
        print('Loading previous results')
        results = dat.load(
            save_location=f"eval/{param_id}/results",
            parameters=['Z']
        )
    else:
        # NOTE scaling learning rate by dt in llp instantiation
        # llp class scales by 1/n_neurons
        results = predict_llp.run_model(
            c_state=c_state,
            z_state=z_state,
            # control=ctrl,
            dt=params['dt'],
            ens_args=json_params['ens_args'],
            **llp_params
        )

    # calculate absolute error between theta_p shifted zdims state
    # and decoded Z prediction at theta_p/theta
    zhat = decode_ldn_data(
        Z=results['Z'],
        q=llp_params['q'],
        theta=llp_params['theta'],
        theta_p=params['theta_p']
    )

    errors = calc_shifted_error(
        z=z_state,
        zhat=zhat,
        dt=params['dt'],
        theta_p=params['theta_p']
    )

    # encode state with ldn using same parameters as
    # the llp to see how well it is able to represent
    # the actual future state
    GT_Z = encode_ldn_data(
        theta=llp_params['theta'],
        q=llp_params['q'],
        z=z_state,
        dt=params['dt']
    )
    # flip coefficient signs to decode like llp (tp/t=1 is forward in time, not back)
    GT_Z = flip_odd_ldn_coefficients(Z=GT_Z, q=llp_params['q'])

    theta_steps = int(llp_params['theta']/params['dt'])
    # remove first theta steps so at time t GT is from theta in future
    GT_Z = GT_Z[theta_steps:]

    # =============== REPEATED CODE IN DECODE GT WITH NEF =================
    n_steps = GT_Z.shape[0] #- theta_steps
    # RMSE between decoded GT and decoded network output
    RMSEs = np.zeros((n_steps, int(len(params['theta_p']))))#, m))
    # RMSE beteween decoded GT and recorded state shifted in time
    RMSEs_gt = np.zeros(RMSEs.shape)
    # max theta steps
    t_steps = int(max(params['theta_p'])/json_params['general']['dt'])
    for ii, tp in enumerate(params['theta_p']):
        tp_steps = int(tp/json_params['general']['dt'])
        x = decode_ldn_data(
            Z=GT_Z,
            q=json_params['llp']['q'],
            theta=json_params['llp']['theta'],
            theta_p=tp
        )
        xhat = decode_ldn_data(
            Z=results['Z'],
            q=json_params['llp']['q'],
            theta=json_params['llp']['theta'],
        )

        # we cut the starting theta_steps out of GT_Z
        # to align it in time, to make the shapes match
        # up remove the ending theta steps from the prediction
        # that ran a sim with the untruncated data.
        err = RMSE(x.T, xhat[:-t_steps].T)
        RMSEs[:, ii] = err#[:, np.newaxis]

        if t_steps == tp_steps:
            err_gt = RMSE(z_state[tp_steps:, np.newaxis, :].T, x.T)
        else:
            err_gt = RMSE(z_state[tp_steps:-(t_steps-tp_steps), np.newaxis, :].T, x.T)
        RMSEs_gt[:, ii] = err_gt
    # =====================================================================

    # save params to the json name
    if save:
        dat.save(
            save_location=f"eval/{param_id}/results",
            data=results,
            overwrite=True
        )
        print(f"Saved data to eval/{param_id}/results of {data_params['db_name']} database located in folder {data_params['database_dir']}")

    if plot:
        plot_prediction_vs_gt(
            tgt=GT_Z,
            decoded=results['Z'][:-theta_steps],
            q=json_params['llp']['q'],
            theta=json_params['llp']['theta'],
            # theta_p=[0],#json_params['general']['theta_p'],
            theta_p=json_params['general']['theta_p'],
            z_state=z_state#[int(max(params['theta_p'])/params['dt']):, :]
        )

        # plot_error(theta_p=params['theta_p'], errors=errors, dt=params['dt'])

        plot_pred(
            time=times[:-theta_steps],
            z=z_state,
            zhat=zhat,#[:-theta_steps],
            theta_p=params['theta_p'],
            size_out=len(data_params['z_dims']),
            gif_name=f'{folder}{param_id}.gif',
            animate=False
        )

    return errors, results, times, z_state, zhat

if __name__ == '__main__':
    # load in all parameters
    with open(sys.argv[1]) as fp:
        json_params = json.load(fp)
    param_id = sys.argv[1].split('/')[-1].split('.')[0]

    load_results = False
    save = False
    if len(sys.argv) > 2:
        load_results = bool(sys.argv[2])
    errors, _, _, _, _, = run(json_params, param_id, load_results, save=save, plot=True)
    print('Error: ', errors)

