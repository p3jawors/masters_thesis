import numpy as np
import json
import copy
import matplotlib.pyplot as plt

from abr_analyze import DataHandler

from masters_thesis.utils import eval_utils
from masters_thesis.utils import plotting


def _test_path_shift(data, params):
    """
    test if path from theta in future aligns with path at (t) if
    theta seconds are added to its time
    """
    plt.figure(figsize=(12, 12))
    plt.suptitle('Path and Future Path')
    for ii in range(0, 12):
        plt.subplot(4, 3, ii + 1)
        plt.plot(data["time"], data["target"][:, ii], label="path(t)", color="r")
        plt.plot(
            data["time"], data["target_future"][:, ii], label="path(t+theta)", color="y"
        )
        plt.plot(
            data["time"] + params["llp"]["theta"],
            data["target_future"][:, ii],
            label="path(t+theta) shifted forward",
            linestyle="--",
            color="k",
        )
        plt.legend()
    plt.tight_layout()
    plt.show()

def _plot_state_norm(data, params):
    plt.figure(figsize=(12, 12))
    plt.suptitle('State and State Norm')
    for ii in range(0, 12):
        plt.subplot(4, 3, ii+1)
        if ii not in params['data']['c_dims']:
            continue
        plt.title(
            (
                f"Mean: {params['data']['state_means'][ii]}"
                + f"\nVar: {params['data']['state_scales'][ii]}"
            )
        )
        plt.plot(data['state'][:, ii])
        plt.plot(data['norm_state'][:, ii], linestyle='--')
    plt.tight_layout()
    plt.show()

def _plot_path_norm(data, params):
    plt.figure(figsize=(12, 12))
    plt.suptitle('Path and Path norm')
    for ii in range(0, 12):
        plt.subplot(4, 3, ii+1)
        if ii not in params['data']['path_dims']:
            continue
        plt.title(
            (
                f"Mean: {params['data']['path_means'][ii]}"
                + f"\nVar: {params['data']['path_scales'][ii]}"
            )
        )
        plt.plot(data['target'][:, ii])
        plt.plot(data['norm_path'][:, ii], linestyle='--')
    plt.tight_layout()
    plt.show()

def _plot_ctrl_norm(data, params):
    plt.figure(figsize=(12, 12))
    plt.suptitle('Control and control norm')
    for ii in range(0, 4):
        plt.subplot(4, 1, ii+1)
        plt.title(
            (
                f"Mean: {params['data']['ctrl_means'][ii]}"
                + f"\nVar: {params['data']['ctrl_scales'][ii]}"
            )
        )
        plt.plot(data['ctrl_pd'][:, ii])
        plt.plot(data['norm_ctrl_0'][:, ii], linestyle='--')
    plt.tight_layout()
    plt.show()



def _compare_ldn_encoding_u(data, params):
    ctrl_ldn = eval_utils.encode_ldn_data(
        theta=params['data']['theta_u'],
        q=params['data']['q_u'],
        z=data['norm_ctrl_0'],
        dt=params['general']['dt']
    )
    plt.figure(figsize=(18,12))
    cnt = 0
    q_cnt = 0
    plt.suptitle('CTRL LDN Encoding')
    for ii in range(0, params['data']['q_u']*4):
        plt.subplot(4, params['data']['q_u'], ii+1)
        if ii%params['data']['q_u'] == 0:
            plt.ylabel(f"u_{cnt}")
            cnt += 1
        if cnt == 4:
            plt.xlabel(f"q_{q_cnt}")
            q_cnt += 1
        plt.plot(ctrl_ldn[:, ii], label='util encoded')
        plt.plot(data['ldn_ctrl_0'][:, ii], label='rt encoded', linestyle='--')
        if ii == 0:
            plt.legend()
    # plt.tight_layout()
    plt.show()

def _compare_ldn_encoding_c(data, params):
    c_ldn = eval_utils.encode_ldn_data(
        theta=params['data']['theta_c'],
        q=params['data']['q_c'],
        z=data['norm_state'],
        dt=params['general']['dt']
    )
    plt.figure(figsize=(18,12))
    cnt = 0
    q_cnt = 0
    plt.suptitle('State LDN Encoding')
    for ii in range(0, params['data']['q_c']*len(params['data']['c_dims'])):
        plt.subplot(len(params['data']['c_dims']), params['data']['q_c'], ii+1)
        if ii%params['data']['q_c'] == 0:
            plt.ylabel(f"state_{cnt}")
            cnt += 1
        if cnt == 4:
            plt.xlabel(f"q_{q_cnt}")
            q_cnt += 1
        plt.plot(c_ldn[:, ii], label='util encoded')
        plt.plot(data['ldn_state'][:, ii], label='rt encoded', linestyle='--')
        if ii == 0:
            plt.legend()
    # plt.tight_layout()
    plt.show()

def _compare_ldn_encoding_path(data, params):
    path_ldn = eval_utils.encode_ldn_data(
        theta=params['data']['theta_path'],
        q=params['data']['q_path'],
        z=data['norm_path'],
        dt=params['general']['dt']
    )
    plt.figure(figsize=(18,12))
    cnt = 0
    q_cnt = 0
    plt.suptitle('Path LDN Encoding')
    for ii in range(0, params['data']['q_path']*len(params['data']['path_dims'])):
        plt.subplot(len(params['data']['path_dims']), params['data']['q_path'], ii+1)
        if ii%params['data']['q_path'] == 0:
            plt.ylabel(f"path_{cnt}")
            cnt += 1
        if cnt == 4:
            plt.xlabel(f"q_{q_cnt}")
            q_cnt += 1
        plt.plot(path_ldn[:, ii], label='util encoded')
        plt.plot(data['ldn_path'][:, ii], label='rt encoded', linestyle='--')
        if ii == 0:
            plt.legend()
    # plt.tight_layout()
    plt.show()


def _compare_cz_state(data, params, json_fp):
    """
    NOTE**** path will not match because <norm_path> is from
    future path planner. load_data_from_json() performs this theta
    shift to the path, but it has already been done in the realtime
    encoding. To get a match just comment out the lines that remove
    the front theta_steps from path and back theta_steps from all else.
    As of writing this it is lines 491-491 in eval_utils.py
    """
    rt_params = copy.deepcopy(params)
    rt_params['data']['db_name'] = 'realtime_trained_predictor'
    rt_params['data']['database_dir'] = None
    rt_params['data']['dataset'] = f"{json_fp}"
    rt_params['data']['state_key'] = 'norm_state'
    rt_params['data']['ctrl_key'] = 'norm_ctrl_0'
    rt_params['data']['path_key'] = 'norm_path'

    loaded_json_params, c_state, z_state, times = eval_utils.load_data_from_json(rt_params)

    reset = True
    print('c state: ', c_state.shape)
    print('rt: ', data['c_0'].shape)
    for ii in range(0, c_state.shape[1]):
        if reset:
            plt.figure(figsize=(12,12))
            cnt = 0
            reset = False
        # plt.subplot(c_state.shape[1], 1, ii+1)
        plt.subplot(10, 1, cnt+1)
        plt.plot(c_state[:, ii], label='utils loaded')
        plt.plot(data['c_0'][:, ii], label='rt stacked', linestyle='--')
        if cnt == 0:
            plt.legend()
        cnt += 1
        if cnt == 10:
            cnt = 0
            reset = True
            plt.show()
    plt.show()

    plt.figure()
    for ii in range(0, len(params['data']['z_dims'])):
        plt.subplot(len(params['data']['z_dims']), 1, ii+1)
        plt.plot(z_state[:, ii], label='utils laded')
        plt.plot(data['z_0'][:, ii], label='rt stacked', linestyle='--')
        plt.legend()
    plt.show()

def _compare_zhat_to_decoded_Z(data, params):
    utils_zhat = eval_utils.decode_ldn_data(
        Z=data['Z_0'],
        q=params['llp']['q'],
        theta=params['llp']['theta'],
    )
    # second dim is len(theta_p), but just focussing on decoding at tp/t=1
    utils_zhat = np.squeeze(utils_zhat)
    plt.figure()
    plt.suptitle('Compare zhat to decoded Z')
    for ii in range(0, len(params['data']['z_dims'])):
        plt.subplot(len(params['data']['z_dims']), 1, ii+1)
        plt.plot(utils_zhat[:, ii], label='utils decoded')
        plt.plot(data['zhat_0'][:, ii], label='rt decoded', linestyle='--')
        plt.plot(data['z_0'][:, ii], label='z(t)', linestyle='--', color='k')
        plt.legend()
    plt.show()

def _plot_prediction(data, params):
    print(data['z_0'].shape)
    print(data['zhat_0'].shape)
    diff_errors = eval_utils.calc_shifted_error(
        z=data['z_0'],
        zhat=data['zhat_0'][:, np.newaxis, :],
        dt=params['general']['dt'],
        theta_p=params['llp']['theta']
    )
    abs_errors = abs(diff_errors)

    plotting.plot_mean_thetap_error_subplot_dims(
        theta=params['llp']['theta'],
        theta_p=params['llp']['theta'],
        errors=abs_errors,
        dt=params['general']['dt'],
        prediction_dim_labs=('X', 'Y', 'Z'),
        save=False,
        label='rt_test',
        folder='Figures',
        show=True
    )

    plotting.plot_pred(
        time=data['time'],
        z=data['z_0'],
        zhat=data['zhat_0'],
        theta_p=[params['llp']['theta']],
        size_out=len(params['data']['z_dims'])
    )

def _compare_to_offline_dataset_path(data, params):
    path_offline = DataHandler(
        params['data']['db_name'], params['data']['database_dir']
    ).load(
        save_location=params['data']['dataset'],
        parameters=['target']
    )['target']
    print(path_offline.shape)
    # # loaded_json_params, c_state, z_state, times = eval_utils.load_data_from_json(params)
    plt.figure(figsize=(12, 12))
    # plt.suptitle('State and State Norm')
    for ii in range(0, 12):
        plt.subplot(4, 3, ii+1)
        if ii not in params['data']['path_dims']:
            continue
        plt.title(
            (
                f"Mean: {params['data']['path_means'][ii]}"
                + f"\nVar: {params['data']['path_scales'][ii]}"
            )
        )
        plt.plot(data['target'][:, ii], label='rt targets')
        plt.plot(path_offline[:len(data['target']), ii], linestyle='--', label='offline targets')
        plt.legend()
    plt.tight_layout()
    plt.show()

def _compare_to_offline_dataset_state(data, params):
    state_offline = DataHandler(
        params['data']['db_name'], params['data']['database_dir']
    ).load(
        save_location=params['data']['dataset'],
        parameters=[params['data']['state_key']]
    )[params['data']['state_key']]
    # # loaded_json_params, c_state, z_state, times = eval_utils.load_data_from_json(params)
    plt.figure(figsize=(12, 12))
    # plt.suptitle('State and State Norm')
    for ii in range(0, 12):
        plt.subplot(4, 3, ii+1)
        if ii not in params['data']['c_dims']:
            continue
        plt.title(
            (
                f"Mean: {params['data']['state_means'][ii]}"
                + f"\nVar: {params['data']['state_scales'][ii]}"
            )
        )
        plt.plot(data['norm_state'][:, ii], label='rt state')
        plt.plot(state_offline[:len(data['norm_state']), ii], linestyle='--', label='offline state')
        plt.legend()
    plt.tight_layout()
    plt.show()

def _compare_future_rt_path_to_path_utils_decoded(data, params):
    """
    Realtime path from future path planner already shifted in time and ldn encoded.
    Path recorded from rt test not shifted in time or normalized. Normalize and
    do shift offline and see if the offline time shifted path matches the rt future path.
    """
    raise NotImplementedError
    plt.figure(figsize=(12, 12))
    plt.suptitle('RT future path vs offline path shifted in time')
    for ii in range(0, 12):
        plt.subplot(4, 3, ii+1)
        if ii not in params['data']['path_dims']:
            continue
        plt.title(
            (
                f"Mean: {params['data']['path_means'][ii]}"
                + f"\nVar: {params['data']['path_scales'][ii]}"
            )
        )
        #TODO plot offline encoded and scaled
        plt.plot(data['norm_path'][:, ii], linestyle='--')
    plt.tight_layout()
    plt.show()


dat = DataHandler("realtime_trained_predictor")
json_fp = "parameter_sets/rt_params_0000.json"
# json_fp = "parameter_sets/rt_params_0001.json"

print(dat.get_keys(json_fp))
with open(json_fp) as fp:
    params = json.load(fp)

data = dat.load(save_location=json_fp)

# _test_path_shift(data, params)
# _plot_state_norm(data, params)
# _plot_path_norm(data, params)
# _plot_ctrl_norm(data, params)
# _compare_ldn_encoding_u(data, params)
# _compare_ldn_encoding_c(data, params)
# _compare_ldn_encoding_path(data, params)
# _compare_cz_state(data, params, json_fp)
# _compare_zhat_to_decoded_Z(data, params)
# _compare_to_offline_dataset_path(data, params)
# _compare_to_offline_dataset_state(data, params)
# # _compare_future_rt_path_to_path_utils_decoded(data, params)
_plot_prediction(data, params)

