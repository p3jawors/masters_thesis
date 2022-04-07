import json
import numpy as np
import os
import nengo
import sys

from abr_analyze import DataHandler
from masters_thesis.utils.eval_utils import decode_ldn_data, calc_shifted_error
from masters_thesis.utils.plotting import plot_pred, plot_error
import predict_llp

# TODO update eval so we can have separate db for train data and results

folder = 'data/figures/'
if not os.path.exists(folder):
    os.makedirs(folder)

def run(json_params, param_id, load_results=False, save=False, plot=True):
    # split into separate dicts for easier use
    data_params = json_params['data']
    llp_params = json_params['llp']
    params = json_params['general']

    # get the neuron model object to match the string in the param file
    model_str = llp_params['neuron_model']
    if llp_params['neuron_model'] == 'nengo.LIFRate':
        llp_params['neuron_model'] = nengo.LIFRate
    elif llp_params['neuron_model'] == 'nengo.LifRectifiedLinear':
        llp_params['neuron_model'] = nengo.LIFRectifiedLinear
    elif llp_params['neuron_model'] == 'nengo.RectifiedLinear':
        llp_params['neuron_model'] = nengo.RectifiedLinear
    else:
        raise ValueError(f"{llp_params['neuron_model']} is not a valid neuron model")

    # Load training data
    if data_params['database_dir'] == '':
        data_params['database_dir'] = None

    dat = DataHandler(data_params['db_name'], data_params['database_dir'])
    data = dat.load(
        save_location=data_params['dataset'],
        parameters=[data_params['state_key'], data_params['ctrl_key'], 'time']
    )

    # extract our keys from the desired time range
    full_state = data[data_params['state_key']][
        data_params['dataset_range'][0]:data_params['dataset_range'][1]
    ]
    full_ctrl = data[data_params['ctrl_key']][
        data_params['dataset_range'][0]:data_params['dataset_range'][1]
    ]
    if full_ctrl.ndim == 1:
        full_ctrl = full_ctrl[:, np.newaxis]

    times = data['time'][data_params['dataset_range'][0]:data_params['dataset_range'][1]]

    # extract our desired dimensions to use as context, it is assumed all ctrl dims are used
    sub_state = np.take(full_state, indices=data_params['c_dims'], axis=1)
    sub_ctrl = np.take(full_ctrl, indices=data_params['u_dims'], axis=1)
    print(sub_state.shape)
    print(sub_ctrl.shape)
    c_state = np.hstack((sub_state, sub_ctrl))
    z_state = np.take(full_state, indices=data_params['z_dims'], axis=1)

    # clear some memory
    full_state = None
    del full_state
    sub_state = None
    del sub_state
    full_ctrl = None
    del full_ctrl

    # add a few missing llp params that we can calculate
    # llp_params['size_c'] = len(data_params['c_dims']) + ctrl.shape[1]
    llp_params['size_c'] = len(data_params['c_dims']) + len(data_params['u_dims'])
    llp_params['size_z'] = len(data_params['z_dims'])


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
    run(json_params, param_id, load_results, save=True)

