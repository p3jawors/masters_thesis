import nni
import json
from masters_thesis import eval_params
from masters_thesis.utils.eval_utils import calc_nni_err

# Get template params that won't change
# fname = '../parameter_sets/nni_baseline_params.json'
fname = '../parameter_sets/nni_nef_decode_params.json'
with open(fname) as fp:
    json_params = json.load(fp)

# Load nni params
experiment_id = nni.get_experiment_id()
nni_params = nni.get_next_parameter()
cid = nni.get_sequence_id()

json_params["llp"]["q_a"] = nni_params["q_a"]
json_params["llp"]["q_p"] = nni_params["q_p"]
json_params["llp"]["learning_rate"] = nni_params["learning_rate"]

print(f'--Starting nni trial: {experiment_id} | {cid}--')
errors, _, _, _, _ = eval_params.run(
    json_params=json_params,
    param_id=cid,
    load_results=False,
    save=False,
    plot=False
)

# nni_error = calc_nni_err(errors)
# nni.report_final_result(nni_error)
# print('final error: ', nni_error)

nni.report_final_result(errors)
print('final error: ', errors)

