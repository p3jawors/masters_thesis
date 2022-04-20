import nni
import json
from masters_thesis import decode_gt_llp_weights_with_nef as decode_nef

# Get template params that won't change
fname = '../parameter_sets/nni_nef_decode_params.json'
with open(fname) as fp:
    json_params = json.load(fp)

# Load nni params
experiment_id = nni.get_experiment_id()
nni_params = nni.get_next_parameter()
cid = nni.get_sequence_id()

json_params["llp"]["n_neurons"] = nni_params["n_neurons"]
json_params["data"]["q_c"] = nni_params["q_c"]
json_params["data"]["theta_c"] = nni_params["theta_c"]
json_params["data"]["q_u"] = nni_params["q_u"]
json_params["data"]["theta_u"] = nni_params["theta_u"]

print(f'--Starting nni trial: {experiment_id} | {cid}--')
rmse, _, _, _, _ = decode_nef.run(
    json_params=json_params,
    param_id=cid
)
nni.report_final_result(rmse)
print('final rmse: ', rmse)

