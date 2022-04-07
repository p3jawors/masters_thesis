#TODO update multiplier to properly normalize errors based on a set max
# keeping vectors together (xyz scale together etc)
# you're too tired right now to get it going so multiplying error by 30
# for now since that gets your close
"""
calculate the difference between state and target and saves it to a new key
state_and_error, which is a horizontal stack of state and error
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from masters_thesis.utils import eval_utils
from abr_analyze import DataHandler
from runtime_utils import ego_state_error

view = False
if len(sys.argv) > 1:
    view = sys.argv[1]


db_name = 'llp_pd'
database_dir = None
train_data = '1000_linear_targets_faster'
# database_dir = 'data/databases'
# train_data = '100_linear_targets' #  90820 temporal data points

dat = DataHandler(db_name, database_dir=database_dir)
data = dat.load(
    save_location=train_data,
    parameters=['state', 'target', 'time']
)
# outlier_clipped_state = eval_utils.clipOutliers(data['state'])

state_and_targets = np.hstack((data['state'], data['target']))
ego_error = np.empty(data['state'].shape)
shift_norm_ego_error = np.empty(data['state'].shape)
for ii in range(0, data['state'].shape[0]):
    # ego = ego_state_error(state_and_targets[ii, :])
    # print(ego.shape)
    ego_error[ii, :] = ego_state_error(state_and_targets[ii, :])

for ii in range(0, ego_error.shape[1]):
    dim = ego_error[:, ii]
    shift_norm_ego_error[:, ii] = (dim - np.mean(dim))/np.amax(abs(dim-np.mean(dim)))

ego_error = np.array(ego_error)

labs = [
    'x', 'y', 'z', 'dx', 'dy' , 'dz', 'a', 'b' , 'g', 'da', 'db', 'dg'
]
if view:
    plt.figure()
    st = data['state']
    # st = outlier_clipped_state
    tg = data['target']
    df = tg-st
    ofst = 15
    for jj in range(0, st.shape[1]):
        plt.subplot(st.shape[1], 1, jj+1)
        plt.ylabel(f"{labs[jj]}")
        # plt.plot(st[ofst:, jj], label='state')
        # plt.plot(tg[ofst:, jj], label='target')
        plt.plot(df[ofst:, jj], label='diff')
        plt.plot(ego_error[ofst:, jj], label='ego error')
        plt.legend()
    plt.show()

    plt.figure()
    for jj in range(0, st.shape[1]):
        plt.subplot(st.shape[1], 1, jj+1)
        plt.ylabel(f"{labs[jj]}")
        if jj == 0:
            plt.title('ego error mean shifted and normalized')
        plt.plot(shift_norm_ego_error[ofst:, jj])
        plt.legend()
    plt.show()


if not view:
    dat.save(
        save_location=train_data,
        data={'ego_error': ego_error, 'mean_shifted_normalized_ego_error': shift_norm_ego_error},
        overwrite=True
    )
    print(f"New data saved to {train_data}")
