import json
from masters_thesis.utils.eval_utils import load_data_from_json

fname = 'test_context_history_gen.json'
with open(fname) as fp:
    json_params = json.load(fp)

model_str = json_params['llp']['neuron_model']

# no history of context or control
json_params['data']['c_dims'] = [0, 1]
json_params['data']['q_c'] = 4
json_params['data']['theta_c'] = 0

json_params['data']['z_dims'] = [1]

json_params['data']['u_dims'] = [0, 1]
json_params['data']['q_u'] = 3
json_params['data']['theta_u'] = 0

json_params, c_state, z_state, times = load_data_from_json(json_params)
print('---TEST 1: No history---')
print('should be (n, 4) and (n, 1)')
print(f"c: {c_state.shape}")
print(f"z: {z_state.shape}")

# history of context, but not control
json_params['llp']['neuron_model'] = model_str
json_params['data']['theta_c'] = 1
json_params, c_state, z_state, times = load_data_from_json(json_params)
print('---TEST 2: Context history---')
print('should be (n, 10) and (n, 1)')
print(f"c: {c_state.shape}")
print(f"z: {z_state.shape}")

# no history of context, but have control history
json_params['llp']['neuron_model'] = model_str
json_params['data']['theta_c'] = 0
json_params['data']['theta_u'] = 1
json_params, c_state, z_state, times = load_data_from_json(json_params)
print('---TEST 3: Control history---')
print('should be (n, 8) and (n, 1)')
print(f"c: {c_state.shape}")
print(f"z: {z_state.shape}")

# history of context and control
json_params['llp']['neuron_model'] = model_str
json_params['data']['theta_c'] = 1
json_params['data']['theta_u'] = 1
json_params, c_state, z_state, times = load_data_from_json(json_params)
print('---TEST 4: Context and Control history---')
print('should be (n, 14) and (n, 1)')
print(f"c: {c_state.shape}")
print(f"z: {z_state.shape}")
