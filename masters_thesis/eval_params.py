import json
import numpy as np
import os
import nengo
import nni
import sys

from abr_analyze import DataHandler
from masters_thesis.utils.eval_utils import decode_ldn_data, calc_shifted_error, calc_nni_err
from masters_thesis.utils.plotting import plot_pred, plot_error
import predict_llp

folder = 'data/figures/'
if not os.path.exists(folder):
    os.makedirs(folder)

# load in all parameters
# TODO update this to work with new json parameters
# if params['run_nni']:
#     # Load nni params
#     experiment_id = nni.get_experiment_id()
#     params = nni.get_next_parameter()
#     cid = nni.get_sequence_id()
#     print(f'--Starting nni trial: {experiment_id} | {cid}--')
# else:
with open(sys.argv[1]) as fp:
    json_params = json.load(fp)

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
else:
    raise ValueError(f"{llp_params['neuron_model']} is not a valid neuron model")

# theta_p = np.linspace(params['dt'], params['theta'], int(params['theta']/params['dt']))

# Load training data
# NOTE remember nni needs a different database dir in its json
#  since nni is run from nni_scripts folder, go back a dir
dat = DataHandler(data_params['db_name'], data_params['database_dir'])
data = dat.load(
    save_location=data_params['dataset'],
    parameters=[data_params['state_key'], data_params['ctrl_key'], 'time']
)


# extract our keys from the desired time range
full_state = data[data_params['state_key']][
    data_params['dataset_range'][0]:data_params['dataset_range'][1]
]
ctrl = data[data_params['ctrl_key']][
    data_params['dataset_range'][0]:data_params['dataset_range'][1]
]
times = data['time'][data_params['dataset_range'][0]:data_params['dataset_range'][1]]

# extract our desired dimensions to use as context, it is assumed all ctrl dims are used
print('FULL: ', full_state.shape)
c_state = np.take(full_state, indices=data_params['c_dims'], axis=1)
z_state = np.take(full_state, indices=data_params['z_dims'], axis=1)

# add a few missing llp params that we can calculate
if ctrl.ndim == 1:
    # print(ctrl.shape)
    ctrl = ctrl[:, np.newaxis]
    # print(ctrl.shape)
print("CTRL SHAPE: ", ctrl.shape)
print("C: ", c_state.shape)
llp_params['size_in'] = len(data_params['c_dims']) + ctrl.shape[1]
llp_params['size_out'] = len(data_params['z_dims'])


# NOTE scaling learning rate by dt here, llp class scales by 1/n_neurons
llp_params['learning_rate'] *= params['dt']
results = predict_llp.run_model(
    c_state=c_state,
    z_state=z_state,
    control=ctrl,
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

if params['run_nni']:
    nni_error = calc_nni_err(errors)
    nni.report_final_result(error)
    print('final error: ', nni_error)
else:
    # save params to the json name
    param_id = sys.argv[1].split('/')[-1].split('.')[0]
    dat.save(
        save_location=f"eval/{param_id}/results",
        data=results,
        overwrite=True
    )
    json_params['llp']['neuron_model'] = model_str
    dat.save(
        save_location=f"eval/{param_id}/params",
        data=json_params,
        overwrite=True
    )

    plot_error(theta_p=params['theta_p'], errors=errors, dt=params['dt'])

    plot_pred(
        time=times,
        z=z_state,
        zhat=zhat,
        theta_p=params['theta_p'],
        size_out=llp_params['size_out'],
        gif_name=f'{folder}{param_id}.gif',
        animate=False
    )
