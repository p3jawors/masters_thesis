"""
subtracts gravity compensation and clips control signal, saving it to a new
key "clean_u"
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from abr_analyze import DataHandler

view = False
if len(sys.argv) > 1:
    view = sys.argv[1]

# db_name = 'llp_pd_e'
# database_dir = 'data/databases'
# train_data = '100_linear_targets_faster'

# db_name = 'llp_pd'
# database_dir = 'data/databases'
# train_data = '1000_linear_targets_faster'

db_name = 'llp_pd_d'
database_dir = 'data/databases'
train_data = '9999_linear_targets_faster'

dat = DataHandler(db_name, database_dir=database_dir)
data = dat.load(
    save_location=train_data,
    parameters=['ctrl', 'time']
)
print('NUM POINTS: ', len(data['time']))

data['gravity_rpm'] = 6800
data['rpm_lim'] = 2000
data[f"clean_u_{data['rpm_lim']}"] = np.clip(
    (data['ctrl'] - data['gravity_rpm']),
    -data['rpm_lim'],
    data['rpm_lim']
    )/data['rpm_lim']

# for key, val in data.items():
keys = [f"clean_u_{data['rpm_lim']}", 'ctrl']
for key in keys:
    val = data[key]
    n_subs = min(val.shape)
    plt.figure()
    for ii in range(0, n_subs):
        plt.subplot(n_subs, 1, ii+1)
        plt.title(f"{key}")
        plt.plot(data['time'], val.T[ii], label=f"{ii}")
        if key == 'ctrl':
            plt.ylim(6500, 7500)
plt.show()

if not view:
    print(f"Saving clean control signal to {train_data}")
    dat.save(
        save_location=train_data,
        data=data,
        overwrite=True
    )
