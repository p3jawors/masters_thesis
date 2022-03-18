import numpy as np
import matplotlib.pyplot as plt
from masters_thesis.utils import eval_utils as utils

dt = 0.001
T = 1 # 1 sec sim
theta_p = 0.1
t = np.linspace(0, 2*np.pi, int(T/dt))

# get time windows for our ideal llp and ldn functions
# the ideal ldn would represent values from the past start-theta_p to t-theta_p
# the ideal llp would represent values from the past start+theta_p to t+theta_p
# ignoring times we don't have values to learn, this is just to test the error
# calculating function
t_base = t[int(2*theta_p/dt):-int(2*theta_p/dt)]
t_ldn = t[int(theta_p/dt):-int(3*theta_p/dt)]
t_llp = t[int(3*theta_p/dt):-int(theta_p/dt)]

# the output from an ideal ldn and llp representation of the base function
train_data = np.sin(t_base)
ldn_data = np.sin(t_ldn)
llp_data = np.sin(t_llp)

plt.figure()
ax = plt.subplot(211)
plt.title('Ideal llp and ldn output to test error function')
plt.xlabel('Time [sec]')
plt.plot(t_base, train_data, label='input data')
plt.plot(t_ldn, ldn_data, label='perfect ldn', linestyle='--')
plt.plot(t_llp, llp_data, label='perfect llp', linestyle='-.')
ax.axvspan(t_llp[0], t_base[0], alpha=0.1, color='red')
ax.axvspan(t_base[-1], t_ldn[-1], alpha=0.1, color='red')
plt.legend()

ax = plt.subplot(212)
plt.xlabel('Time steps')
plt.plot(train_data, label='input data', linestyle='--')
plt.plot(ldn_data, label='perfect ldn', linestyle='-.')
plt.plot(llp_data, label='perfect llp')
plt.legend()
plt.show()


print('Testing calc_shifted_error(ldn)')
# ldn
error_ldn = utils.calc_shifted_error(
    z=train_data[:, np.newaxis],
    zhat=ldn_data[:, np.newaxis, np.newaxis],
    dt=dt,
    theta_p=[theta_p],
    model='ldn'
)


print('Testing calc_shifted_error(llp)')
# llp
error_llp = utils.calc_shifted_error(
    z=train_data[:, np.newaxis],
    zhat=llp_data[:, np.newaxis, np.newaxis],
    dt=dt,
    theta_p=[theta_p],
    model='llp'
)

print('SUM LDN ERROR: ', sum(sum(error_ldn)))
print('SUM LLP ERROR: ', sum(sum(error_llp)))


print("testing calc_ldn_repr_error")
qvals = np.arange(1, 10)
#NOTE test repr error calc
results = utils.calc_ldn_repr_err(
    z=train_data[:, np.newaxis],
    qvals=qvals,
    theta=theta_p,
    theta_p=[theta_p],
    dt=dt
)
plt.figure()
for key, val in results.items():
    plt.plot(np.squeeze(val), label=key)
plt.legend()
plt.show()



