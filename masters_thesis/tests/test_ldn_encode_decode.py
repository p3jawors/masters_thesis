import matplotlib.pyplot as plt
from masters_thesis.utils.eval_utils import decode_ldn_data, encode_ldn_data
import numpy as np
def stim_func(t, freq=0.7):
    return [t*np.sin(t*2*np.pi*freq), np.cos(t*2*np.pi*freq)]

# ldn parameters
q = 10
theta = 1
theta_p = 0.5
dt = 0.01
# number of discrete steps that make up theta
theta_steps = int(theta_p/dt)
t = np.arange(0, 10, dt)
# the value to represent
z = np.asarray(stim_func(t)).T
# the ldn encoding
Z = encode_ldn_data(
    theta=theta,
    q=q,
    z=z,
    dt=dt)

print(f"z shape = {z.shape}")
print(f"Z shape = {Z.shape}")

# decode the value to represent from the encoded ldn coefficients
zhat = decode_ldn_data(Z=Z, q=q, theta=theta, theta_p=theta_p)

# plot ground truth shifted by theta to align in time, and decoded value
plt.figure(figsize=(12,12))
for jj in range(0, z.shape[1]):
    plt.subplot(z.shape[1], 1, jj+1)
    plt.title(f'LDN Decoding: theta={theta}, theta_p={theta_p}')
    plt.ylabel(f"dim{jj}")
    plt.plot(t+theta_p, z[:, jj], label=f'z at t+{theta_p}')
    plt.plot(t, zhat[:, :, jj], linestyle='--', label=f'zhat_{theta_p} at t', c='r')
    plt.legend()
plt.show()
