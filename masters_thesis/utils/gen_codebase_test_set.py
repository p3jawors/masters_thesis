import numpy as np
from abr_analyze import DataHandler
from eval_utils import gen_BLWN
import matplotlib.pyplot as plt

# xt, xW = gen_BLWN(T=10, dt=0.01, rms=1, limit=7, seed=37, sigma=0.5 , debug=True)
# t = np.arange(0, 10.01, 0.01)
# plt.figure()
# plt.plot(t, xt)
# plt.show()



dt = 0.001

def stim_func(t, freq=5):
    return [np.sin(t*2*np.pi*freq), np.cos(t*2*np.pi*freq)]

t = np.arange(0, 20, dt)
state = np.asarray(stim_func(t)).T
u = np.zeros(t.shape)
dat = DataHandler('codebase_test_set', 'data/databases')
dat.save(save_location='sin5t', data={'dt': dt, 'time': t, 'state': state, 'control': u}, overwrite=True)
print('state: ', state.shape)

def stim2_func(t):
    return [np.sin(t+np.sin(t)), (np.cos(t) + 1)*np.cos(t+np.sin(t))]
y = np.asarray(stim2_func(t)).T
# y = y[:, np.newaxis]
print('state2: ', y.shape)

dat = DataHandler('codebase_test_set', 'data/databases')
dat.save(save_location='sin(t+sin(t))', data={'dt': dt, 'time': t, 'state': y, 'control': u}, overwrite=True)

# plt.figure()
# plt.plot(t, y)
# plt.show()
def stim3_func(t):
    return [np.sin(t**2), 2*t*np.cos(t**2)]
y = np.asarray(stim3_func(t)).T
# y = y[:, np.newaxis]
print('state3: ', y.shape)

dat = DataHandler('codebase_test_set', 'data/databases')
dat.save(save_location='sin(t**2)', data={'dt': dt, 'time': t, 'state': y, 'control': u}, overwrite=True)


