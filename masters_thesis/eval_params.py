import json
import numpy as np
import os
import nengo
import sys

from abr_analyze import DataHandler
from masters_thesis.utils.eval_utils import decode_ldn_data, calc_shifted_error, load_data_from_json
from masters_thesis.utils.plotting import plot_pred, plot_error
import predict_llp

# TODO update eval so we can have separate db for train data and results

folder = 'data/figures/'
if not os.path.exists(folder):
    os.makedirs(folder)

def run(json_params, param_id, load_results=False, save=False, plot=True):
    json_params, c_state, z_state, times = load_data_from_json(json_params)

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

        # save params to the json name
    if save:
        dat.save(
            save_location=f"eval/{param_id}/results",
            data=results,
            overwrite=True
        )
        print(f"Saved data to eval/{param_id}/results of {data_params['db_name']} database located in folder {data_params['database_dir']}")

    if plot:
        plot_error(theta_p=params['theta_p'], errors=errors, dt=params['dt'])

        plot_pred(
            time=times,
            z=z_state,
            zhat=zhat,
            theta_p=params['theta_p'],
            size_out=llp_params['size_z'],
            gif_name=f'{folder}{param_id}.gif',
            animate=False
        )

    return errors, results

if __name__ == '__main__':
    # load in all parameters
    with open(sys.argv[1]) as fp:
        json_params = json.load(fp)
    param_id = sys.argv[1].split('/')[-1].split('.')[0]

    load_results = False
    if len(sys.argv) > 2:
        load_results = bool(sys.argv[2])
    run(json_params, param_id, load_results, save=False)

