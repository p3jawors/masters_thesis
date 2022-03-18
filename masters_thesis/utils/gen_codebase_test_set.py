import numpy as np
from abr_analyze import DataHandler
from eval_utils import gen_BLWN
import matplotlib.pyplot as plt

# xt, xW = gen_BLWN(T=10, dt=0.01, rms=1, limit=7, seed=37, sigma=0.5 , debug=True)
# t = np.arange(0, 10.01, 0.01)
# plt.figure()
# plt.plot(t, xt)
# plt.show()



def stim_func(t, freq=5):
    return [np.sin(t*2*np.pi*freq), np.cos(t*2*np.pi*freq)]

t = np.arange(0, 20, 0.01)
state = np.asarray(stim_func(t)).T
u = np.zeros(t.shape)
print(state.shape)

dat = DataHandler('codebase_test_set', 'data/databases')
dat.save(save_location='sin5t', data={'time': t, 'state': state, 'control': u}, overwrite=True)
