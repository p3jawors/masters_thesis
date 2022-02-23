"""
subtracts gravity compensation and clips control signal, saving it to a new
key "clean_u"
"""
import numpy as np
import matplotlib.pyplot as plt
from abr_analyze import DataHandler

db_name = 'llp_pd'
train_data = '100_linear_targets' #  90820 temporal data points
dat = DataHandler(db_name, database_dir='data/databases')
data = dat.load(
    save_location=train_data,
    parameters=['ctrl']
)

data['gravity_rpm'] = 6800
data['rpm_lim'] = 250
data['clean_u'] = np.clip(
    (data['ctrl'] - data['gravity_rpm']),
    -data['rpm_lim'],
    data['rpm_lim']
    )/data['rpm_lim']

# for key, val in data.items():
key = 'clean_u'
val = data[key]
n_subs = min(val.shape)
plt.figure()
for ii in range(0, n_subs):
    plt.subplot(n_subs, 1, ii+1)
    plt.title(f"{key}")
    plt.plot(val.T[ii], label=f"{ii}")
plt.show()

dat.save(
    save_location=train_data,
    data=data,
    overwrite=True
)
