import nengo
import sys
import numpy as np
from abr_analyze import DataHandler
from itertools import product
import json
import hashlib
from masters_thesis.utils.eval_utils import encode_ldn_data, load_data_from_json, RMSE, flip_odd_ldn_coefficients, decode_ldn_data
from masters_thesis.utils.plotting import plot_x_vs_xhat, plot_prediction_vs_gt
import masters_thesis.utils.plotting as plotting
blue = "\033[94m"
endc = "\033[0m"
green = "\033[92m"
red = "\033[91m"

def print_nested(d, indent=0, return_val=False):
    if return_val:
        full_print = ''
    for key, value in d.items():
        if isinstance(value, dict):
            line = '\t' * indent + str(key) + ': '
            if return_val:
                full_print += line
            else:
                print(line)
            if return_val:
                nested_line = print_nested(value, indent+1, return_val=return_val)
                full_print += nested_line
            else:
                print_nested(value, indent+1)
        else:
            line = '\t' * indent + str(key) + f': {value}'
            if return_val:
                full_print += line
            else:
                print(line)

    if return_val:
        return full_print

def run(json_params, weights=None):
    loaded_json_params, c_state, z_state, times = load_data_from_json(json_params)
    print('c state: ', c_state.shape)
    print('z state: ', z_state.shape)

    # split into separate dicts for easier referencing
    data_params = loaded_json_params['data']
    llp_params = loaded_json_params['llp']
    params = loaded_json_params['general']

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

def gen_lookup_table(db_name, db_folder):
    dat = DataHandler(
        db_name=db_name,
        database_dir=db_folder
    )

    hashes = dat.load(
        save_location="params",
        parameters=dat.get_keys("params")
    )
    lookup = {}
    for hash_id in hashes:
        params = dat.load(
            save_location=f"params/{hash_id}",
            parameters=dat.get_keys(f"params/{hash_id}", recursive=True)
        )
        for key, val in params.items():
            # print(key)
            # print(val)
            # raise Exception
            if key not in lookup:
                lookup[key] = {str(val): [hash_id]}
            elif str(val) not in lookup[key].keys():
                lookup[key][str(val)] = [hash_id]
            elif str(val) in lookup[key].keys():
                lookup[key][str(val)].append(hash_id)
    # print('KEY: ', key)
    # print('len: ', len(lookup[key]['1']))
    # print('unique: ', len(set(lookup[key]['1'])))
    # print_nested(lookup[key])
    return lookup


# def add_parameters(missing_param_dict, db_name, db_folder=None):
#     dat = DataHandler(
#         db_name=db_name,
#         database_dir=db_folder
#     )
#     hash_ids = dat.load(save_location='params')
def compare_all_keys(dat, saved_exp_hashes, parameter_stems):
    """
    Input a DataHandler object and list of experiment hashes and will
    return a dictionary of constant parameters and a list of keys with
    that differ between any of the experiments. The differing values
    are to be used as legend keys to differentiate between experiments.
    The constant parameters can be printed alongside the figure.


    Saving format
    params  >hash_0 >parameter in json format
            >hash_1 >parameter in json format
            >hash_2 >parameter in json format

    results >script_0   >hash_0 >results_dict
                        >hash_2 >results_dict
            >script_1   >hash_1 >results_dict
    """
    if isinstance(parameter_stems, str):
        parameter_stems = [parameter_stems]

    final_legend = []
    final_constants = {}

    for group_name in parameter_stems:
        for ee, exp_hash in enumerate(saved_exp_hashes):
            # print(f"ee: {ee}")
            keys = dat.get_keys(f"params/{exp_hash}/{group_name}")
            # track any differing values keys'
            legend_keys = []
            if ee == 0:
                base_parameters = dat.load(save_location=f"params/{exp_hash}/{group_name}", parameters=keys)
            else:
                new_parameters = dat.load(save_location=f"params/{exp_hash}/{group_name}", parameters=keys)
                # temporary storage of differing keys to add to legend keys
                differing_keys = []
                for key in base_parameters:
                    if isinstance(base_parameters[key], (list, np.ndarray)):
                        try:
                            # if (base_parameters[key] != new_parameters[key]).any():
                            #     differing_keys.append(key)

                            if np.asarray(base_parameters[key]).shape != np.asarray(new_parameters[key]).shape:
                                differing_keys.append(key)
                            elif (base_parameters[key] != new_parameters[key]).any():
                                differing_keys.append(key)

                        except AttributeError as e:
                            print(f"Got AttributeError on {key} who's value is:\n{base_parameters[key]}")
                            print(f"Or possibly from const params:\n{new_parameters[key]}")
                            raise e


                    else:
                        if key not in new_parameters.keys():
                            new_parameters[key] = None
                        if base_parameters[key] != new_parameters[key]:
                            differing_keys.append(key)

                # add missing keys directly to legend keys
                for key in new_parameters:
                    if key not in base_parameters.keys():
                        legend_keys.append(f"{group_name}/{key}")

                # remove differing keys from base parameters, only leaving common ones
                for key in differing_keys:
                    base_parameters.pop(key)
                    legend_keys.append(f"{group_name}/{key}")

        final_constants[group_name] = base_parameters
        final_legend += legend_keys
    # print('FINAL LEGEND: ', final_legend)
    # print('''should match:
    #     variation_dict = {
    #         'llp/n_neurons': [1000, 2500, 5000],
    #         'llp/theta': [1, 0.1],
    #         'llp/q': [2, 4, 6, 8],
    #         'data/q_u': [1, 2],
    #     },
    #     ''')
    # raise Exception
    return final_constants, final_legend

def find_matches(dat, saved_exp_hashes, const_params):
    matches = []
    for exp_hash in saved_exp_hashes:
        data = dat.load(save_location=f"params/{exp_hash}", parameters=const_params.keys())
        # count the number of different key: value pairs, if zero save the hash
        # since looking for experiments with matching parameters
        num_diff = 0
        for param_key in const_params.keys():

            if isinstance(data[param_key], (list, np.ndarray)):
                try:
                    # print(param_key)
                    if np.asarray(data[param_key]).shape != np.asarray(const_params[param_key]).shape:
                        num_diff += 1
                    elif (data[param_key] != const_params[param_key]).any():
                        num_diff += 1
                except AttributeError as e:
                    print(f"Got AttributeError on {param_key} who's value is:\n{data[param_key]}")
                    print(f"Or possibly from const params:\n{const_params[param_key]}")
                    raise e

            else:
                if data[param_key] != const_params[param_key]:
                    num_diff += 1
            # print(data[param_key])
            # print(type(data[param_key]))
        if num_diff == 0:
            matches.append(exp_hash)
    print(f"{len(matches)} experiment hashes with matching parameters")
    return matches


def load_results(
        script_name,
        const_params,
        db_name,
        db_folder=None,
        ignore_keys=None,
        show_gt=False,
        show_prediction=False):

    dat = DataHandler(
        db_name=db_name,
        database_dir=db_folder
    )
    # Load the hashes of all experiments that have been run for this script
    saved_exp_hashes = dat.get_keys(f"results/{script_name}")
    print(f"{len(saved_exp_hashes)} experiments found with results from {script_name}")

    # if isinstance(const_params, list) or isinstance(const_params, np.ndarray):
    if const_params is not None:
        # Get all experiment id's that match a set of key value pairs
        print(f"Searching for results with matching parameters to: {const_params}")
        saved_exp_hashes = find_matches(dat, saved_exp_hashes, const_params)

    # Get a dictionary of common values and a list of keys for differing values
    # to use in the auto legend
    all_constants, legend_keys = compare_all_keys(dat, saved_exp_hashes, parameter_stems=['llp', 'data', 'general', 'ens_args'])
    if ignore_keys is not None:
        if isinstance(ignore_keys, str):
            ignore_keys = [ignore_keys]

        for key in ignore_keys:
            if key in legend_keys:
                legend_keys.remove(key)

    if const_params is None:
        # Use all hashes saved to this experiment script
        # Get a dictionary of common values and a list of keys for differing values
        # to use in the auto legend
        # all_constants, legend_keys = compare_all_keys(dat, saved_exp_hashes, parameter_stems=['llp', 'data', 'general', 'ens_args'])
        const_params = all_constants.keys

    figure = None
    axs = None
    for mm, match in enumerate(saved_exp_hashes):
        # load experiments parameters
        params = dat.load(
            save_location=f"params/{match}",
            parameters=[
                'llp/theta',
                'general/dt',
            ]
        )

        # load results from script_name
        data_parameters = [
                'RMSE',
                'target_pts',
                'decoded_pts',
                'z_state',
                'decoded_z',
                'decoded_zhat'
        ]
        if show_gt:
            data_parameters.append('RMSE_gt')

        data = dat.load(
            save_location=f'results/{script_name}/{match}',
            parameters=data_parameters
       )

        name_params = dat.load(
            save_location=f'params/{match}',
            parameters=legend_keys
        )
        name = ""
        for kk, key in enumerate(legend_keys):
            if kk > 0 and kk < len(legend_keys):
                name += " | "
            # name += f"{key.split('/')[-1]}={name_params[key]}"
            name += f"{key}={name_params[key]}"

        # print('======')
        # print(match)
        # print(name)
        # print_nested(dat.load(
        #     save_location=f"params/{match}",
        #     parameters=dat.get_keys(f"params/{match}")))


        # print('NAME PARAMS: ', name_params)
        # Plot while passing the figure and axes object through
        # iterations in the for loop to control when figure is shown
        RMSEs = data['RMSE']
        target_pts = data['target_pts']
        decoded_pts = data['decoded_pts']
        z_state = data['z_state']
        x = data['decoded_z']
        xhat = data['decoded_zhat']
        if show_gt:
            RMSE_gt = data['RMSE_gt'][:, :, np.newaxis]
        else:
            RMSE_gt = None


        save_fig = False
        constants_printout = None
        if mm+1 == len(saved_exp_hashes):
            save_fig = True
            constants_printout = all_constants
            print('saving image this time')
        show = save_fig

        if show_prediction:
            # print('showing prediction')
            pred_data = dat.load(
                save_location=f'params/{match}',
                parameters=['llp/q', 'llp/theta', 'general/dt']
            )

            plot_prediction_vs_gt(
                tgt=target_pts,
                decoded=decoded_pts,
                q=pred_data['llp/q'],
                theta=pred_data['llp/theta'],
                # theta_p = np.linspace(
                #         pred_data['general/dt'],
                #         pred_data['llp/theta'],
                #         10
                # ),
                theta_p=[1],
                z_state=z_state,#[theta_steps:]
                # xlim=[0, 1000],
                show=False,
                save=False,
                # savename=f"data/pred_vs_gt_q_{json_params['llp']['q']}.jpg"
            )


        # print('Plotting')
        # title = f"Error over Theta_p\n{const_params}"
        title = "Error over Theta_p"
        # if len(title) > 30:
        #     title = '\n'.join(x for x in title.split(','))
        figure, axs = plotting.plot_mean_time_error_vs_theta_p(
            theta_p = np.linspace(
                    params['general/dt'],
                    params['llp/theta'],
                    10
                ),
            errors=RMSEs[:, :, np.newaxis],
            dt=params['general/dt'],
            theta=params['llp/theta'],
            figure=figure,
            title=title,
            axs=axs,
            show=show,
            legend_label=name,
            save=save_fig,
            folder='data',
            label="",#f"{save_name.replace('.', '_').replace('/', '_')}_"
            all_constants=constants_printout,
            errors_gt=RMSE_gt,
        )




    # TODO update load to load all data if parameters is None
    # TODO add option to load recursively to get nested dicts
    # TODO add auto legend and text to show differences in legend, constants in sidebar


def run_variation_comparison(json_fps, variation_dict=None, show_error=False, show_prediction=False, save=False, load=False):

    # for changing nested values
    def nested_set(dic, keys, value):
        for key in keys[:-1]:
            dic = dic.setdefault(key, {})
        dic[keys[-1]] = value

    figure = None
    axs = None
    script_name = 'nef_decode_llp_weight_TEST3'

    for jj, json_fp in enumerate(json_fps):
        with open(json_fp) as fp:
            json_params = json.load(fp)

        if save or load:
            dat = DataHandler(
                db_name=json_params['data']['db_name']+'_results',
                database_dir=json_params['data']['database_dir']
            )

        if variation_dict is None:
            variations = [{}]
        else:
            variations = [dict(zip(variation_dict, v)) for v in product(*variation_dict.values())]

        print(f"\nRunning {len(variations)} variations of test\n")
        for vv, var in enumerate(variations):
            if len(var) > 0:
                name = ""
                for key in var:
                    nested_set(json_params, key.split('/'), var[key])
                    name += f"{key.split('/')[-1]}={var[key]} |"
            else:
                name = json_fp
            # nested_set(json_params, key_list, var)
            hash_name = hashlib.sha256(str(sorted(json_params)).replace(' ', '').encode('utf-8')).hexdigest()
            # save_name=f"{json_fp.split('/')[-1]}/{hash_name}/results"
            save_name=f"results/{script_name}/{hash_name}"


            # print(f"Updated json_params by changing {key_list} to {var}\n{json_params}")
            print(f"hash_name: {hash_name}")

            # account for steps removed from GT
            theta_steps = int(json_params['llp']['theta']/json_params['general']['dt'])
            m = len(json_params['data']['z_dims'])
            theta_p = np.linspace(
                    json_params['general']['dt'], json_params['llp']['theta'], 10
                )


            if not load:
                # NOTE first pass that saves weights to npz
                print(f"Getting decoded points from training set: {json_fp}")
                json_params['data']['dataset_range'] = json_params['data']['train_range']
                rmse, eval_pts, target_pts, decoded_pts, weights, z_state = run(json_params)
                np.savez_compressed('nef_weights.npz', weights=weights)

                print('Getting test results')
                json_params['data']['dataset_range'] = json_params['data']['test_range']
                rmse, eval_pts, target_pts, decoded_pts, weights, z_state = run(json_params, weights=weights)

                # save the altered parameters to the unique hash created from the altered json
                if save:
                    dat.save(
                        # save_location=f"{json_fp.split('/')[-1]}/{hash_name}/params",
                        save_location=f"params/{hash_name}",
                        data=json_params,
                        overwrite=True,
                        timestamp=False
                    )

                print('Calculating RMSE for each theta_p')
                # n_steps = np.diff(json_params['data']['dataset_range'])[0] - theta_steps
                n_steps = target_pts.shape[0] #- theta_steps
                # RMSE between decoded GT and decoded network output
                RMSEs = np.zeros((n_steps, int(len(theta_p))))#, m))
                # RMSE beteween decoded GT and recorded state shifted in time
                RMSEs_gt = np.zeros((n_steps, int(len(theta_p))))#, m))
                # print('PARAMS: ', json_params)
                # print('theta_p: ', theta_p)
                # print('theta: ', json_params['llp']['theta'])
                t_steps = int(max(theta_p)/json_params['general']['dt'])
                for ii, tp in enumerate(theta_p):
                    tp_steps = int(tp/json_params['general']['dt'])
                    # print('tp: ', tp)
                    # print('dt: ', json_params['general']['dt'])
                    # print('TP STEPS: ', tp_steps)
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

                    if t_steps == tp_steps:
                        err_gt = RMSE(z_state[tp_steps:, np.newaxis, :].T, x.T)
                    else:
                        err_gt = RMSE(z_state[tp_steps:-(t_steps-tp_steps), np.newaxis, :].T, x.T)
                    RMSEs_gt[:, ii] = err_gt
            else:
                print(f"Loading results from: {save_name}")
                json_params['data']['dataset_range'] = json_params['data']['test_range']
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
                RMSEs_gt = data['RMSE_gt']
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
                        'RMSE_gt': RMSEs_gt,
                        'target_pts': target_pts,
                        'decoded_pts': decoded_pts,
                        'z_state': z_state,
                        'decoded_z': x,
                        'decoded_zhat': xhat
                    },
                    overwrite=True,
                    timestamp=False
                )

            if show_error:
                save_fig = False
                # print('jj: ', jj)
                # print('vv: ', vv)
                # if vv+1 == len(variation_list):
                if vv+1 == len(variations):
                    if jj+1 == len(json_fps):
                        save_fig = True
                        print('saving image this time')
                else:
                    save_fig = False
                show = save_fig
                # print('Plotting')
                figure, axs = plotting.plot_mean_time_error_vs_theta_p(
                    theta_p=theta_p,
                    errors=RMSEs[:, :, np.newaxis],
                    errors_gt=RMSEs_gt[:, :, np.newaxis],
                    dt=json_params['general']['dt'],
                    theta=json_params['llp']['theta'],
                    figure=figure,
                    axs=axs,
                    show=show,
                    legend_label=name,#labels[vv],
                    save=save_fig,
                    folder='data',
                    label=f"{save_name.replace('.', '_').replace('/', '_')}_"
                )

            if show_prediction:
                plot_prediction_vs_gt(
                    tgt=target_pts,
                    decoded=decoded_pts,
                    q=json_params['llp']['q'],
                    theta=json_params['llp']['theta'],
                    # theta_p=theta_p,
                    theta_p=[json_params['llp']['theta']],
                    z_state=z_state,#[theta_steps:]
                    # xlim=[0, 1000],
                    show=True,
                    save=True,
                    savename=f"data/pred_vs_gt_q_{json_params['llp']['q']}.jpg"
                )

def get_script_RMSEs(script_name, db_name, db_folder=None, hash_ids=None):
    dat = DataHandler(db_name, db_folder)
    if hash_ids is None:
        hash_ids = dat.load(f"results/{script_name}")#, recursive=False)

    RMSEs = {}
    for hh, hash_id in enumerate(hash_ids):
        rmse = dat.load(
            save_location=f"results/{script_name}/{hash_id}",
            parameters=['RMSE']
        )['RMSE']
        RMSEs[hash_id] = np.mean(rmse)

    import operator
    sorted_d = sorted(RMSEs.items(), key=operator.itemgetter(1))
    return sorted_d

if __name__ == '__main__':

    # NOTE Current implementation for running experiment variations
    json_fps = [
        # 'parameter_sets/params_0016a.json',
        # 'parameter_sets/params_0016b.json',
        # 'parameter_sets/params_0018.json',
        # 'parameter_sets/params_0019.json',
        # 'parameter_sets/params_0021.json',
        # 'parameter_sets/params_0022.json',
        # 'parameter_sets/params_0023.json',
        'parameter_sets/params_0024.json',
    ]
    load = False
    lookup = False
    run_sim = False
    if len(sys.argv) > 1:
        if 'load' in sys.argv:
            load = True
        if 'lookup' in sys.argv:
            lookup = True
        if 'run' in sys.argv:
            run_sim = True
    if run_sim:
        # for json_fp in json_fps:
        run_variation_comparison(
            json_fps=json_fps,
            # variation_dict=None,
            variation_dict = {
                # 'llp/n_neurons': [1000, 5000, 10000],#, 50000],
                # 'llp/theta': [1, 0.1],
                # 'llp/q': [6],
                # 'data/z_dims': [[0, 1, 2]],
                # 'data/c_dims': [[0, 1, 2], [0, 1, 2, 8]],
                # 'data/q_c': [4, 6],
                # 'data/theta_c': [1, 3],
                #
                # 'data/q_u': [2, 4],
                # 'data/theta_u': [1, 3],
                #
                # 'data/path_dims': [[0, 1, 2], [0, 1, 2, 8]],
                # 'data/q_path': [4, 6],
                # 'data/theta_path': [1, 3],

            },
            # labels=['1000', '2000', '5000'],
            show_error=True,
            show_prediction=False,
            save=True,
        )

    # NOTE Current implementation for loading results and plotting

    const_params = {
        # == Data ==
        # "data/db_name": "llp_pd_d",
        # "data/database_dir": "data/databases",
        # "data/dataset": "9999_linear_targets_faster",
        # "data/dataset_range": [0, 100000],
        # "data/train_range": [0, 80000],
        # "data/test_range": [80000, 100000],

        # "data/state_key": "mean_shifted_normalized_ego_error",
        # "data/state_key": "state",
        "data/state_key": "mean_shift_abs_max_scale_state",
        "data/z_dims": [0, 1, 2],
        "data/c_dims": [0, 1, 2],
        "data/q_c": 4,
        "data/theta_c": 1,

        "data/ctrl_key": "clean_u",
        # "data/u_dims": [],
        # "data/u_dims": [0, 1, 2, 3],
        # "data/q_u": 0,
        # "data/theta_u": 0,

        "data/path_key": "target",
        "data/path_dims": [0, 1, 2],
        "data/q_path": 6,
        "data/theta_path": 1

        # == llp ==
        # "llp/model_type": "mine",
        # "llp/n_neurons": 1000,
        # "llp/theta": 1,
        # "llp/q": 6,
        # "llp/q_a": 5,
        # "llp/q_p": 2,
        # "llp/learning_rate": 8.656205265024589e-7,
        # "llp/neuron_model": "nengo.RectifiedLinear",

        # == ens args ==
        # "ens_args/seed": 0,
        # "ens_args/radius": 1,

        # == general ==
        # "general/run_nni": false,
        # "general/dt": 0.01,
        # "general/theta_p": [1]
    }
    # const_params = None
    script_name = 'nef_decode_llp_weight_TEST3'
    db_name = 'llp_pd_d_results'
    db_folder = '/home/pjaworsk/src/masters/masters_thesis/masters_thesis/data/databases'

    if load:
        load_results(
            script_name=script_name,
            const_params=const_params,
            db_name=db_name,
            db_folder=db_folder,
            ignore_keys=['general/theta_p', 'data/dataset_range'],
            show_gt=False,
            show_prediction=True
        )

    if lookup:
        lookup = gen_lookup_table(
            db_name=db_name,
            db_folder=db_folder,
        )
        print_nested(lookup)
        # from abr_analyze.utils import ascii_table
        # ascii_table.print_lookup(data=lookup, invert=False)

    # param = 'llp/q'
    # val = 2
    # hashes = lookup[param][str(val)]
    # dat = DataHandler(db_name, db_folder)
    # for hash_id in hashes:
    #     print('------')
    #     print(f"{hash_id} has {param} with a value of {str(val)}")
    #     print(dat.load(
    #         save_location=f"params/{hash_id}",
    #         parameters=[param]
    #         )[param]
    #     )
