import nengo
import sys
import numpy as np
from abr_analyze import DataHandler
import json
from masters_thesis.utils.eval_utils import time_series_to_ldn_polynomials, load_data_from_json
import matplotlib.pyplot as plt

# folder = 'data/figures/'
# if not os.path.exists(folder):
#     os.makedirs(folder)

def run(json_params, param_id, load_results=False, save=False, plot=True):
    json_params, c_state, z_state, times = load_data_from_json(json_params)
    print('c state: ', c_state.shape)
    print('z state: ', z_state.shape)

    # split into separate dicts for easier referencing
    data_params = json_params['data']
    llp_params = json_params['llp']
    params = json_params['general']

    dat = DataHandler(data_params['db_name'], data_params['database_dir'])

    # Get the LDN representation of our state to predict.
    # This will be shifted back by theta and used as ground truth for the llp prediction.
    # Since the LDN represents a window of length theta with q coefficients,
    # if the LLP outputs q coefficients at time t that represent the signal theta
    # seconds into the future, this would be the same as the ldn representation
    # with the same q and theta, but in t+theta seconds
    GT_Z = time_series_to_ldn_polynomials(
        theta=llp_params['theta'],
        q=llp_params['q'],
        z=z_state,
        dt=params['dt']
    )

    # Shift GT back by theta
    theta_steps = int(llp_params['theta']/params['dt'])
    GT_Z = GT_Z[theta_steps:]
    # stop theta_steps from the end since we won't have GT for those steps
    c_state = c_state[:-theta_steps]

    # DEBUG PRINT
    # print('GT_Z: ', GT_Z.shape)
    # print('context: ', c_state.shape)
    # print('theta steps: ', theta_steps)
    # print(f"{llp_params['q']=}")
    # print(f"{llp_params['theta']=}")
    # print(f"{params['dt']=}")

    with nengo.Network() as model:
        ens = nengo.Ensemble(
            n_neurons=llp_params['n_neurons'],
            dimensions=c_state.shape[1],
            neuron_type=llp_params['neuron_model'](),
            radius=np.sqrt(c_state.shape[1])
        )
        pred = nengo.Node(size_in=GT_Z.shape[1])
        conn = nengo.Connection(
            ens,
            pred,
            eval_points=c_state,
            function=GT_Z,
            synapse=None
        )
    sim = nengo.Simulator(model)
    with sim:
        eval_pt, tgt, decoded = nengo.utils.connection.eval_point_decoding(
            conn, sim
        )

    def plot_decoded(x, xhat):
        plt.figure(figsize=(6, 8))
        plt.subplot(211)
        plt.title('Ideal and Decoded Value')
        plt.xlabel('Represented Value')
        plt.ylabel('Decoded Value')
        plt.plot(x, x, label='ideal')
        plt.plot(x, xhat, linestyle='--', label='decoded')
        plt.legend()

        rmse = RMSE(x=x, xhat=xhat)

        plt.subplot(212)
        plt.title('Error')
        plt.xlabel('Represented Value')
        plt.ylabel(r'$Error (x-\^x)$')
        plt.plot(x, x-xhat, label=f"RMSE: {rmse}")
        plt.legend()
        plt.tight_layout()
        plt.show()
        return rmse

    def RMSE(x, xhat):
        err = 0
        for ii in range(0, x.shape[0]):
            for jj in range(0, x.shape[1]):
                err += (xhat[ii, jj] - x[ii, jj])**2
        err /= (ii+1)*(jj+1)
        err = np.sqrt(err)
        return err

    # for ii in range(0, tgt.shape[1]):
    #     plot_decoded(tgt[:, ii][:, np.newaxis], decoded[:, ii][:, np.newaxis])
    # print('RMSE: ', RMSE(tgt, decoded))
    # print(f"{eval_pt.shape=}")
    # print(f"{tgt.shape=}")
    # print(f"{decoded.shape=}")
    # plt.figure()
    # for ii in range(0, tgt.shape[1]):
    #     plt.subplot(tgt.shape[1], 1, ii+1)
    #     plt.plot(tgt[:, ii])
    #     plt.gca().set_prop_cycle(None)
    #     plt.plot(decoded[:, ii], linestyle='--')
    # plt.show()
    return(RMSE(tgt, decoded))




if __name__ == '__main__':
    # load in all parameters
    with open(sys.argv[1]) as fp:
        json_params = json.load(fp)
    param_id = sys.argv[1].split('/')[-1].split('.')[0]

    load_results = False
    if len(sys.argv) > 2:
        load_results = bool(sys.argv[2])
    # run(json_params, param_id, load_results, save=True)
    plt.figure()
    for neur in [1000, 2000, 3000, 5000]:
        json_params['llp']['n_neurons'] = neur
        rmse = run(json_params, param_id, load_results, save=False)
        plt.scatter(neur, rmse)
    plt.show()
