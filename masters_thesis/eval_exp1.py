import nni
import json
import numpy as np
from masters_thesis import eval_params
from abr_analyze import DataHandler
import matplotlib.pyplot as plt

# Get template params that won't change
dat = DataHandler('exp_0001', 'data/databases')
fname = 'parameter_sets/exp1_params.json'
with open(fname) as fp:
    json_params = json.load(fp)

n_neurons = [1000, 2500, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
dat.save(save_location='param_file', data=json_params, overwrite=True)
dat.save(save_location='variation', data={'n_neurons': n_neurons}, overwrite=True)
errors = []

model_str = json_params['llp']['neuron_model']
for n_neur in n_neurons:
    try:
        json_params["llp"]["n_neurons"] = n_neur
        # we overwrite this to the nengo object, so it fails when we loop
        json_params['llp']['neuron_model'] = model_str

        # TODO update eval so we can have separate db for train data and results
        error, results = eval_params.run(
            json_params=json_params,
            param_id=fname,
            load_results=False,
            save=False,
            plot=False
        )
        dat.save(save_location=f'results/{n_neur}', data=results, overwrite=True)
        errors.append(np.linalg.norm(error))
    except Exception as e:
        print(f'Failed to run experiment with {n_neur} neurons')
        print(f"Error was raised:\n{e}")

dat.save(save_location='results', data={'2norm_error': errors}, overwrite=True)
plt.figure()
plt.plot(n_neurons[:len(errors)], errors)
plt.xlabel('n_neurons')
plt.ylabel('2norm error [m]')
plt.show()

