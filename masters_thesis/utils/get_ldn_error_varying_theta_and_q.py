"""
Test LDN ability to represent activities and our state predictions
"""
from masters_thesis.utils.eval_utils import calc_ldn_err_vs_q
import matplotlib.pyplot as plt
import numpy as np
from masters_thesis.utils import plotting
from abr_analyze import DataHandler
import sys

# dat = DataHandler('codebase_test_set', 'data/databases')
# data = dat.load(save_location='sin5t', parameters=['time', 'state'])
# # z = data['state']
# z = data['state'][:, 0]
# z = z[:, np.newaxis]

n_steps = 10000
db_name = 'llp_pd'
dat = DataHandler(db_name, database_dir='data/databases')
data = dat.load(save_location='train/param_set_0003', parameters=['activities', 'Z', 'q'])
act = data['activities']
predicted_z = data['Z']
q = data['q']

data2 = dat.load(save_location='100_linear_targets', parameters=['state'])
state = data2['state']

dat2 = DataHandler('llp_pd_d', 'data/databases')
u = dat2.load(
    save_location='9999_linear_targets_faster',
    parameters=['clean_u_2000']
)['clean_u_2000']

to_show = 'activities'
if len(sys.argv) > 1:
    to_show = str(sys.argv[1])

dt = 0.01
theta_p = [1]
thetas = [1, 2, 4, 6, 8, 10]
qvals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# qvals = np.arange(1,201,1)

if to_show in ['Z', 'prediction']:
    z = predicted_z[:n_steps, :]
    ylabel = 'RMSE of Prediction'
    # prediction_dim_labs = []
    # for jj in ['x', 'y', 'z']:
    #     for ii in range(0, q):
    #         prediction_dim_labs.append(f"{jj}_{ii}")
    # label = 'LDN of Prediction'
    # qvals = [5]
    # error, zhats = calc_ldn_err_vs_q(
    #     z, qvals, theta, theta_p, dt=dt, return_zhat=True)
    # plotting.plot_ldn_repr_error(
    #     error, theta, theta_p, z, dt, zhats, prediction_dim_labs,
    #     max_rows=3,
    #     folder='data/presentation_figures/',
    #     save_name='LDN_prediction_Z')

elif to_show in ['act', 'activities', 'neurons', 'spikes']:
    n_neurons = 3
    z = act[:n_steps, :n_neurons]
    ylabel = 'RMSE of Activities'

    # prediction_dim_labs = ['neuron0', 'neuron1', 'neuron2']
    # label = 'Neural Activities'
    # qvals = [6]
    # plotting.plot_ldn_repr_error(
    #     error=error,
    #     theta=theta,
    #     theta_p=theta_p,
    #     z=z,
    #     dt=dt,
    #     zhats=zhats,
    #     prediction_dim_labs=prediction_dim_labs,
    #     max_rows=3,
    #     folder='data/presentation_figures/',
    #     save_name='LDN_prediction_activities'
    # )

elif to_show in ['state', 'context', 'c']:
    z = state[:n_steps, :3]
    # prediction_dim_labs = ['x', 'y', 'z']
    ylabel = 'RMSE of Position'
    # label = 'Input Context and GT'
    # plt.figure(figsize=(8,8))
    # plt.title('Error in LDN representation')
    # plt.ylabel('RMSE Error of Position')
    # plt.xlabel('LDN q value')
    #
    # for theta in thetas:
    #     error, zhats = calc_ldn_err_vs_q(
    #         # z, qvals, theta, theta_p, dt=dt, return_zhat=True)
    #         z, qvals, theta, theta, dt=dt, return_zhat=True)
    #
    #     error_means = []
    #     for key, val in error.items():
    #         # print('ERRR SHAPE: ', val.shape)
    #         error_means.append(np.sqrt(np.mean(val**2)))
    #     plt.plot(qvals, error_means, label=f'theta={theta}')
    # plt.legend()
    # plt.show()
    # print(zhats['4'].shape)
    # plotting.plot_ldn_repr_error(
    #     error, theta, theta_p, z, dt, zhats, prediction_dim_labs)#, label)
elif to_show in ['u', 'control', 'ctrl']:
    z = u[:n_steps, :]
    ylabel = 'RMSE of Control'



plt.figure(figsize=(8,8))
plt.title('Error in LDN representation')
plt.ylabel('RMSE Error of Activities')
plt.xlabel('LDN q value')
plt.ylabel(ylabel)

for theta in thetas:
    error, zhats = calc_ldn_err_vs_q(
        # z, qvals, theta, theta_p, dt=dt, return_zhat=True)
        z, qvals, theta, theta, dt=dt, return_zhat=True)

    error_means = []
    for key, val in error.items():
        # print('ERRR SHAPE: ', val.shape)
        error_means.append(np.sqrt(np.mean(val**2)))
    plt.plot(qvals, error_means, label=f'theta={theta}')
plt.legend()
plt.show()


