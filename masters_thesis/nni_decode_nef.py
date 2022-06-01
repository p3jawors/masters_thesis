import nni
import numpy as np
import json
from masters_thesis import decode_gt_llp_weights_with_nef as decode_nef
from masters_thesis.utils.eval_utils import RMSE, decode_ldn_data

# Get template params that won't change
# fname = '../parameter_sets/nni_nef_decode_params.json'
fname = '../parameter_sets/params_0024.json'
with open(fname) as fp:
    json_params = json.load(fp)

# Load nni params
experiment_id = nni.get_experiment_id()
nni_params = nni.get_next_parameter()
cid = nni.get_sequence_id()

json_params['data']['database_dir'] = f"../{json_params['data']['database_dir']}"
json_params["llp"]["n_neurons"] = nni_params["n_neurons"]
json_params["llp"]["q"] = nni_params["q"]
json_params["data"]["q_c"] = nni_params["q_c"]
json_params["data"]["theta_c"] = nni_params["theta_c"]
json_params["data"]["q_u"] = nni_params["q_u"]
json_params["data"]["theta_u"] = nni_params["theta_u"]
json_params["data"]["q_path"] = nni_params["q_path"]
json_params["data"]["theta_path"] = nni_params["theta_path"]
print("JSON PARAMS: ", json_params)

print(f'--Starting nni trial: {experiment_id} | {cid}--')
print('Decoding weights')
json_params['data']['dataset_range'] = json_params['data']['train_range']
rmse, eval_pts, target_pts, decoded_pts, weights, z_state = decode_nef.run(
    json_params=json_params, weights=None
)
print('Getting test results')
json_params['data']['dataset_range'] = json_params['data']['test_range']
rmse, eval_pts, target_pts, decoded_pts, weights, z_state = decode_nef.run(
    json_params, weights=weights
)


# =========== copied from decode nef
print('setting thetap to theta')
json_params['general']['theta_p'] = [json_params['llp']['theta']]
theta_p = [json_params['llp']['theta']]
print('Calculating RMSE for each theta_p')
# n_steps = np.diff(json_params['data']['dataset_range'])[0] - theta_steps
n_steps = target_pts.shape[0] #- theta_steps
# RMSE between decoded GT and decoded network output
RMSEs = np.zeros((n_steps, int(len(theta_p))))#, m))
# RMSE beteween decoded GT and recorded state shifted in time
# RMSEs_gt = np.zeros((n_steps, int(len(theta_p))))#, m))
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

# ===================================

# nni.report_final_result(np.sum(RMSEs))
err = np.mean(RMSEs)
nni.report_final_result(err)
print('final rmse: ', err)

