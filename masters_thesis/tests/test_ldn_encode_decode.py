import matplotlib.pyplot as plt
from masters_thesis.utils.eval_utils import decode_ldn_data, time_series_to_ldn_polynomials
import numpy as np
def stim_func(t, freq=1):
    return [t*np.sin(t*2*np.pi*freq), np.cos(t*2*np.pi*freq)]

q = 10
theta = 1
theta_p = 0.5
dt = 0.01
theta_steps = int(theta_p/dt)
t = np.arange(0, 10, dt)
z = np.asarray(stim_func(t)).T
Z = time_series_to_ldn_polynomials(
    theta=theta,
    q=q,
    z=z,
    dt=dt)

print(f"z shape = {z.shape}")
print(f"Z shape = {Z.shape}")

zhat = decode_ldn_data(Z=Z, q=q, theta=theta, theta_p=theta_p)
plt.figure(figsize=(12,12))
for jj in range(0, z.shape[1]):
    plt.subplot(z.shape[1], 1, jj+1)
    plt.title(f'LDN Decoding: theta={theta}, theta_p={theta_p}')
    plt.ylabel(f"dim{jj}")
    plt.plot(t+theta_p, z[:, jj], label=f'z at t+{theta_p}')
    plt.plot(t, zhat[:, :, jj], linestyle='--', label=f'zhat_{theta_p} at t', c='r')
    plt.legend()
plt.show()
