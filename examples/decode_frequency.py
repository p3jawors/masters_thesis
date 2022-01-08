"""
Example how to learn a non-linear function from and LDN

Trains on 1Hz and 2Hz signals and can tell whether an input
is the former or the latter
"""
from ldn import LDN
import nengo
import numpy as np
import matplotlib.pyplot as plt

blue = '\033[94m'
green = '\033[92m'
red = '\033[91m'
endc = '\033[0m'

print(f"{blue} So the system so far can be useful if we want to do some linear operation on our input, such as convolving it with any arbitrary pattern. But, it'd be good to do non-linear things as well.{endc}")
print(f"\n{blue} The easiest way to add a non-linearity is to just take the output from the LDN and feed it into a layer of neurons. We call this system an LMU. We can now use whatever methods we feel like to train the non-linear part, but importantly we don't have to train the linear part. This means we just have to do feed-forward training, which can be much more efficient than doing recurrent training.{endc}")
print(f"\n{blue} As an example, let's build a network that can detect whether the input is a 1Hz sine wave or a 2Hz sine wave. To do this, we start by generating some training data.{endc}")
theta = 0.5   # the size of the time window to remember
q = 6         # the number of basis functions to use to represent that window
dt = 0.001
t = np.arange(10000)*dt
stim1 = np.sin(t*2*np.pi).reshape(-1,1)
stim2 = np.sin(t*2*np.pi*2).reshape(-1,1)

plt.figure(figsize=(12,4))
plt.plot(t, stim1)
plt.plot(t, stim2)
plt.show()

print(f"{green}Now we feed those signals into the LDN. Since this is the data that would be fed into the non-linear neural network part, this is the actual training data for our system.{endc}")
print(f"{blue}Fortunately, nengo.Process objects have a nice helper function apply that does this for us.{endc}")

x1 = LDN(theta=theta, q=q).apply(stim1)
x2 = LDN(theta=theta, q=q).apply(stim2)


plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
plt.plot(x1)
plt.title('1Hz wave')
plt.subplot(1, 2, 2)
plt.plot(x2)
plt.title('2Hz wave')
plt.show()


print(f"{blue}Now we collect this data together so we can use it as training data in Nengo. This requires us to make an eval_points array (the inputs to the network) and a targets array (the corresponding desired outputs).{endc}")
eval_points = np.vstack([x1, x2])
targets = np.hstack([np.ones(len(x1)), -np.ones(len(x2))]).reshape(-1,1)

print(f"{blue}Now we build our model again and this time we add some neurons and tell Nengo to decode out our desired function using this training data. Note that this is the same as the normal nengo approach of specifying a function when we make a Connection, but here we're specifying the exact data to train with, rather than giving a function.{endc}")
model = nengo.Network()
with model:
    stim = nengo.Node(lambda t: np.sin(2*np.pi*t) if t<4 else np.sin(2*np.pi*t*2))

    ldn = nengo.Node(LDN(theta=theta, q=q))
    nengo.Connection(stim, ldn, synapse=None)

    neurons = nengo.Ensemble(n_neurons=200, dimensions=q, neuron_type=nengo.LIF())
    nengo.Connection(ldn, neurons)

    category = nengo.Node(None, size_in=1)
    nengo.Connection(neurons, category, eval_points=eval_points, function=targets)

    p_stim = nengo.Probe(stim)
    p_ldn = nengo.Probe(ldn)
    p_category = nengo.Probe(category, synapse=0.01)
sim = nengo.Simulator(model)
with sim:
    sim.run(8)
plt.figure(figsize=(12,4))
plt.plot(sim.trange(), sim.data[p_stim], label='stimulus')
plt.plot(sim.trange(), sim.data[p_category], label='output')
plt.legend()
plt.show()
print(f"{green}It works! For more complex situations, we'd probably want to generate a lot more training data to cover different inputs, but we'd use the same approach as shown here. We could also have multiple outputs, rather than just one.{endc}")
