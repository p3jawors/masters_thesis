"""
Example how to encode information into an LDN
"""
from ldn import LDN
import nengo
import numpy as np
import matplotlib.pyplot as plt

blue = '\033[94m'
green = '\033[92m'
red = '\033[91m'
endc = '\033[0m'

print(f"{blue}Let's build a simple model where we just feed a pulse into an LDN and see what happens...{endc}")
theta = 0.5   # the size of the time window to remember
q = 6         # the number of basis functions to use to represent that window

model = nengo.Network()
with model:
    stim = nengo.Node(lambda t: 1 if 0.2<t<0.25 else 0)  # our stimulus

    ldn = nengo.Node(LDN(theta=theta, q=q))
    nengo.Connection(stim, ldn, synapse=None)

    p_stim = nengo.Probe(stim)
    p_ldn = nengo.Probe(ldn)
sim = nengo.Simulator(model)
with sim:
    sim.run(1.5)


plt.figure(figsize=(14,5))
plt.subplot(2, 1, 1)
plt.plot(sim.trange(), sim.data[p_stim])
plt.ylabel('$u$')
plt.subplot(2, 1, 2)
plt.plot(sim.trange(), sim.data[p_ldn])
plt.ylabel('$x$')
plt.xlabel('time (s)')
plt.show()

print(f"{green}Okay, so this weird differential equation does something over time based on an input."
    + f" But how well is it representing that pulse?{endc} \n\n{blue}Let's try extracting some information out"
    + " of the LDN. In particular, let's extract what it thinks the value was 0.25 seconds ago."
    + " Since theta is 0.5s, 0.25s ago is half way through our window. We can ask the LDN for"
    + " the weights that will decode that information, which is just the value of the Legendre"
    + f" polynomials at that point.{endc}")

theta = 0.5
q = 20

model = nengo.Network()
with model:
    stim = nengo.Node(lambda t: 1 if 0.2<t<0.25 else 0)
    #stim = nengo.Node(nengo.processes.WhiteSignal(high=3, period=3))

    ldn = nengo.Node(LDN(theta=theta, q=q))
    nengo.Connection(stim, ldn, synapse=None)

    delayed_output = nengo.Node(None, size_in=1)
    nengo.Connection(ldn, delayed_output, transform=ldn.output.get_weights_for_delays([0.5]))

    p_stim = nengo.Probe(stim)
    p_ldn = nengo.Probe(ldn)
    p_delayed_output = nengo.Probe(delayed_output)
sim = nengo.Simulator(model)
with sim:
    sim.run(1.5)

plt.figure(figsize=(14,6))
plt.subplot(3, 1, 1)
plt.plot(sim.trange(), sim.data[p_stim])
plt.ylabel('$u$')
plt.xlim(0, 1.5)
plt.subplot(3, 1, 2)
plt.plot(sim.trange(), sim.data[p_ldn])
plt.ylabel('$x$')
plt.xlim(0, 1.5)
plt.subplot(3, 1, 3)
plt.plot(sim.trange(), sim.data[p_delayed_output], label='output')
plt.plot(sim.trange()+0.25, sim.data[p_stim], ls='--', label='ideal output')
plt.xlim(0, 1.5)
plt.legend()
plt.xlabel('time (s)')

t_plot = 0.3
w = np.linspace(0, 1, 1000)
decoder = LDN(theta=theta, q=q).get_weights_for_delays(w)
plt.plot(w*theta, decoder.dot(sim.data[p_ldn][int(t_plot/0.001)]))
plt.xlabel('amount of time into the past (s)')
plt.title(f'represented information at time T={t_plot}')
plt.show()

print(f"{green}Hmm, it sort of works, but it's not great. This is probably because we're asking"
        + f" it to encode a pulse, which is a very high frequency signal!{endc}"
        + f"\n\n{blue}Let's try increasing  so it is using more legendre polynomials,"
        + f" and thus is encoding higher frequencies.{endc}")


theta = 0.5
q = 20

model = nengo.Network()
with model:
    stim = nengo.Node(lambda t: 1 if 0.2<t<0.25 else 0)
    #stim = nengo.Node(nengo.processes.WhiteSignal(high=3, period=3))
    
    ldn = nengo.Node(LDN(theta=theta, q=q))
    nengo.Connection(stim, ldn, synapse=None)
    
    delayed_output = nengo.Node(None, size_in=1)
    nengo.Connection(ldn, delayed_output, transform=ldn.output.get_weights_for_delays([0.5]))
    
    p_stim = nengo.Probe(stim)
    p_ldn = nengo.Probe(ldn)
    p_delayed_output = nengo.Probe(delayed_output)
sim = nengo.Simulator(model)
with sim:
    sim.run(1.5)

plt.figure(figsize=(14,6))
plt.subplot(3, 1, 1)
plt.plot(sim.trange(), sim.data[p_stim])
plt.ylabel('$u$')
plt.xlim(0, 1.5)
plt.subplot(3, 1, 2)
plt.plot(sim.trange(), sim.data[p_ldn])
plt.ylabel('$x$')
plt.xlim(0, 1.5)
plt.subplot(3, 1, 3)
plt.plot(sim.trange(), sim.data[p_delayed_output], label='output')
plt.plot(sim.trange()+0.25, sim.data[p_stim], ls='--', label='ideal output')
plt.xlim(0, 1.5)
plt.legend()
plt.xlabel('time (s)')
plt.show()

print(f"{green}Sweet! If we want more accuracy, we could increase  further, or decrease the window size, but this is good for now.{endc}"
    + f"\n\n{blue}One other thing we could do is to take the  value at a particular point in time and plot what it thinks the value is over the whole window of time.{endc}")
t_plot = 0.3
w = np.linspace(0, 1, 1000)
decoder = LDN(theta=theta, q=q).get_weights_for_delays(w)
plt.plot(w*theta, decoder.dot(sim.data[p_ldn][int(t_plot/0.001)]))
plt.xlabel('amount of time into the past (s)')
plt.title(f'represented information at time T={t_plot}')
plt.show()

print(f"\n\n{blue}We could also try feeding in a band-limited white noise signal, to see how it handles a smoother input. In this case, we're also putting  back to 6 as that's probably fine for a slowly varying signal{endc}")

theta = 0.5
q = 6

model = nengo.Network()
with model:
    stim = nengo.Node(nengo.processes.WhiteSignal(high=3, period=3))
    
    ldn = nengo.Node(LDN(theta=theta, q=q))
    nengo.Connection(stim, ldn, synapse=None)
    
    delayed_output = nengo.Node(None, size_in=1)
    nengo.Connection(ldn, delayed_output, transform=ldn.output.get_weights_for_delays([0.5]))
    
    p_stim = nengo.Probe(stim)
    p_ldn = nengo.Probe(ldn)
    p_delayed_output = nengo.Probe(delayed_output)
sim = nengo.Simulator(model)
with sim:
    sim.run(1.5)

plt.figure(figsize=(14,6))
plt.subplot(3, 1, 1)
plt.plot(sim.trange(), sim.data[p_stim])
plt.ylabel('$u$')
plt.xlim(0, 1.5)
plt.subplot(3, 1, 2)
plt.plot(sim.trange(), sim.data[p_ldn])
plt.ylabel('$x$')
plt.xlim(0, 1.5)
plt.subplot(3, 1, 3)
plt.plot(sim.trange(), sim.data[p_delayed_output], label='output')
plt.plot(sim.trange()+0.25, sim.data[p_stim], ls='--', label='ideal output')
plt.xlim(0, 1.5)
plt.legend()
plt.xlabel('time (s)')
plt.show()
