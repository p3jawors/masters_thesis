import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import json
from abr_analyze import DataHandler
from masters_thesis.utils import eval_utils, plotting

json_fp = "parameter_sets/thesis_rt_params.json"
with open(json_fp) as fp:
    params = json.load(fp)

db = DataHandler("thesis_realtime_trained_predictor")
data = db.load(f"{json_fp}_LLPSE_best")
# data = db.load(f"{json_fp}")
# print(data.keys())

data_type = 'zhat_'
bias_type = [
    ['up', 'down'],
    ['forward', 'backward'],
    ['left', 'right'],
    ['cw', 'ccw']
]
steps = int(params['llp']['theta']/params['general']['dt'])
print('theta: ', params['llp']['theta'])
print('dt: ', params['general']['dt'])
print(steps)
print(data['target'].shape)
yaw = data['target'][steps:, 8]

dt = 0.005
# plt.figure()
# plt.plot(
#     data['z_[0, 0, 0, 0]_no_bias_[0, 0, 0, 0]'],
# )
# plt.plot(
#     data['zhat_[0, 0, 0, 0]_no_bias_[0, 0, 0, 0]'], linestyle='--',
# )
# plt.show()
time=np.arange(
    0,
    data['z_[0, 0, 0, 0]_no_bias_[0, 0, 0, 0]'].shape[0]*dt,
    dt
)
# plotting.plot_pred(
#     time=time,
#     z=data['z_[0, 0, 0, 0]_no_bias_[0, 0, 0, 0]'],
#     zhat=data['zhat_[0, 0, 0, 0]_no_bias_[0, 0, 0, 0]'],
#     size_out=3,
#     theta_p=[1],
# )

for bias_group in bias_type:
    lns = ['-', '--']
    fig, axs = plt.subplots(3, 1, figsize=(12,8))
    # font = {'family' : 'normal',
    #         # 'weight' : 'bold',
    #         'size'   : 16}
    #
    # matplotlib.rc('font', **font)

    # fig.suptitle(bias_group)
    for bb, bias in enumerate(bias_group):
        print('_____________')
        print('BIAS TYPE: ', bias)
        bias_keys = []
        for key in data.keys():
            print(key)
            if f"_{bias}_" in key and data_type in key:
                print(f"- {key}")
                bias_keys.append(key)
            elif "no_bias" in key and data_type in key:
                no_bias = key

        body_no_bias_xyz = eval_utils.world_to_body_frame(
            positions=data[no_bias][:-steps, :],
            yaws=yaw,
        )

        for aa, ax in enumerate(axs):
            # plt.gca().set_prop_cycle(None)
            ax.set_prop_cycle(None)
            for key in bias_keys:
                body_xyz = eval_utils.world_to_body_frame(
                    positions=data[key][:-steps, :],
                    yaws=yaw,
                )
                # diff = body_no_bias_xyz[:, aa] - body_xyz[:, aa]
                diff = body_xyz[:, aa] - body_no_bias_xyz[:, aa]
                ax.plot(
                    diff,
                    label=f"{bias}_{np.sqrt(params['data']['ctrl_scales'][0] * abs(float(key.split('_')[-1].split(',')[1]))):.2f}rad/sec",
                    linestyle=lns[bb])

                # diff = data[no_bias][:, aa] - data[key][:, aa]
                # # ax.plot(data[key][:, aa], label=key.split('_')[-1])
                # ax.plot(diff, label=f"{bias}_{key.split('_')[-1]}", linestyle=lns[bb])

            # ax.hlines(0, len(diff), 0, color='k', linestyle='-.', label='_')
            # ax.plot(data[no_bias][:, aa], label='no_bias', color='k', linestyle='--')
            axs[0].set_ylabel('Difference to unbiased predictor')
            axs[1].set_ylabel('Difference to unbiased predictor')
            axs[2].set_ylabel('Difference to unbiased predictor')
            axs[0].set_xlabel('Steps')
            axs[1].set_xlabel('Steps')
            axs[2].set_xlabel('Steps')
            axs[0].set_title('X body frame')
            axs[1].set_title('Y body frame')
            axs[2].set_title('Z body frame')
            ax.legend(bbox_to_anchor=(1,1), ncol=2)

    plt.tight_layout()
    plt.show()

