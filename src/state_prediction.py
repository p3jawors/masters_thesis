import numpy as np
import nengo
from llp import LLP
from ldn import LDN

import math
import matplotlib.pyplot as plt

from nengo_interfaces.airsim import AirSim
from nengo_control.controllers.quadrotor import PD
from nengo_control.controllers.path_planners.path_planner_node import PathPlannerNode, ExitSim
from abr_control.controllers.path_planners import PathPlanner
from abr_control.controllers.path_planners.position_profiles import Linear
from abr_control.controllers.path_planners.velocity_profiles import Gaussian

# Test begins here
airsim_dt = 0.01
steps = 500

# Accepts 12D state as input and outputs a 4D control signal in radians/second
# in the rotor order: [front_right, rear_left, front_left, rear_right]
pd_ctrl = PD(
    gains=np.array(
        [
            8950.827941754635,
            5396.8148923228555,
            3797.2396183387336,
            2838.8455160747803,
            5817.333354627463,
            10763.75342891863,
            415.04893487790997,
            500.1385252571632,
        ]
    )
)
def Plinear(t):
    x = t
    y = t
    z = t
    return np.array([x, y, z])

def Pcurve(t):
    x = np.sin(t*np.pi/2)
    # y = np.sin(t*np.pi/2)
    # z = np.sin(t*np.pi/2)

    # x = t
    y = t
    z = t
    return np.array([x, y, z])


# # P = Plinear
# P = Pcurve


Pprof = Linear()
Vprof = Gaussian(dt=airsim_dt, acceleration=1)
path = PathPlanner(
        pos_profile=Pprof,
        vel_profile=Vprof,
        verbose=True
    )

interface = AirSim(
        dt=airsim_dt,
        show_display=True,
)
interface.connect(pause=True)
# get our starting state
# run twice as sometimes airsim doesn't return the updated state on connect
feedback = interface.get_feedback()
feedback = interface.get_feedback()
state = np.hstack(
            [
                feedback["position"],
                feedback["linear_velocity"],
                feedback["taitbryan"],
                feedback["angular_velocity"],
            ])

# targets = [np.array([2, 1, -3, 0, 0, 0, 0, 0, 1.57, 0, 0, 0])]
targets = [
        np.array([5, 3, -2, 0, 0, 0, 0, 0, 3.14, 0, 0, 0]),
        # np.array([-5, 3, -2, 0, 0, 0, 0, 0, 1.57, 0, 0, 0]),
        # np.array([3, -3, -5, 0, 0, 0, 0, 0, 3.14, 0, 0, 0]),
        # np.array([-1, -2, -2.5, 0, 0, 0, 0, 0, 3.14, 0, 0, 0])
    ]
interface.set_state("target", targets[0][:3], targets[0][6:9])

freq = 5
learning_rate = 5e-5
# t_delays = np.linspace(0, 1.0, 5)
t_delays = np.array([0.5, 1.0])
q = 6

model = nengo.Network()
try:
    with model:
        model.n_pred = len(t_delays)
        model.config[nengo.Connection].synapse = None

        # wrap our interface in a nengo Node
        interface_node = nengo.Node(
                interface,
                label="Airsim"
        )

        # path planner node
        path_node = PathPlannerNode(
                path_planner=path,
                targets=targets,
                dt=airsim_dt,
                buffer_reach_time=0,
                use_start_location=True,
                max_velocity=2,
                start_velocity=0,
                target_velocity=0
        )

        def sim_extras_func(t, x):
            # function for other things to do in the loop, like setting
            # locations of objects in sim, or other non-control related things
            interface.set_state("filtered_target", x[:3], x[6:9])

        # takes 12D target as input
        sim_extras = nengo.Node(sim_extras_func, size_in=12, size_out=0)

        # wrap our non-neural controller in a node that takes
        # in target and drone state, and outputs rotor velocities
        def ctrl_func(t, x):
            return pd_ctrl.generate(x[:12], x[12:]).flatten()

        ctrl_node = nengo.Node(ctrl_func, size_in=24, size_out=4)

        # connect our state info to our path planner to check error thresholds
        nengo.Connection(interface_node, path_node.input, synapse=None)

        # pass path info to sim extras to move a target object to visualize
        nengo.Connection(path_node.output, sim_extras, synapse=None)

        # get our next target from our path planner and pass it to our controller
        nengo.Connection(path_node.output, ctrl_node[12:], None)
        nengo.Connection(interface_node, ctrl_node[:12], None)

        # connect our controller output to our sim input
        nengo.Connection(ctrl_node, interface_node[:4], synapse=0)

        # def stim_func(t):
        #     return np.sin(t*2*np.pi*freq), np.cos(t*2*np.pi*freq)
        # c = nengo.Node(stim_func)

        c = nengo.Node(size_in=6, size_out=6)
        z = nengo.Node(size_in=3, size_out=3)
        nengo.Connection(interface_node[:6], c, synapse=None)
        nengo.Connection(c[:3], z, synapse=None)

        llp = []
        display = []
        def airsim_prediction_vis(t, x):
            xs = x[:model.n_pred]
            ys = x[model.n_pred: 2*model.n_pred]
            zs = x[2*model.n_pred:]
            pred = np.vstack((xs, np.vstack((ys, zs))))
            for rr, xyz in enumerate(pred.T):
                interface.set_state(
                    f"prediction_{rr}",
                    xyz=xyz,
                    orientation=[0, 0, 0])

        airsim_display = nengo.Node(airsim_prediction_vis, size_in=3*model.n_pred)
        for ll, label in enumerate(['x', 'y', 'z']):
            llp.append(LLP(
                    n_neurons=1000,
                    size_in=2,
                    size_out=1,
                    q_a=q,
                    q_p=q,
                    q=q,
                    theta=np.max(t_delays),
                    dt=airsim_dt,
                    learning=True,
                    K=learning_rate,
                    seed=0,
                    verbose=True)
                    # output_r=t_delays)
                    )

            # position
            nengo.Connection(c[ll], llp[ll].input[0], synapse=None)
            # velocity
            nengo.Connection(c[ll+3], llp[ll].input[1], synapse=None)
            nengo.Connection(z[ll], llp[ll].z, synapse=None)

            display.append(
                    nengo.Node(
                        size_in=1+model.n_pred,
                        size_out=1+model.n_pred
                    )
            )
            nengo.Connection(z[ll], display[ll][0])
            nengo.Connection(
                llp[ll].Z, display[ll][1:],
                transform=LDN(q=q, theta=np.max(t_delays)).get_weights_for_delays(t_delays/np.max(t_delays)))

            nengo.Connection(display[ll][1:], airsim_display[ll*model.n_pred:(ll+1)*model.n_pred], synapse=None)

        # add probes for plotting
        state_p = nengo.Probe(interface_node, synapse=0)
        target_p = nengo.Probe(path_node.output, synapse=0)
        ctrl_p = nengo.Probe(ctrl_node, synapse=0)
        predx_p = nengo.Probe(display[0], synapse=None)
        predy_p = nengo.Probe(display[1], synapse=None)
        predz_p = nengo.Probe(display[2], synapse=None)

    with nengo.Simulator(model, dt=airsim_dt) as sim:
        sim.run(steps * airsim_dt)

except ExitSim as e:
    print('Exiting Sim')

finally:
    interface.disconnect()
    # Plot results
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.title('Flight Path in NED Coordinates')
    ax.plot(
        sim.data[target_p].T[0],
        sim.data[target_p].T[1],
        sim.data[target_p].T[2],
        label='target'
        )
    ax.plot(
        sim.data[state_p].T[0],
        sim.data[state_p].T[1],
        sim.data[state_p].T[2],
        label='state',
        linestyle='--')
    ax.scatter(
        targets[0][0],
        targets[0][1],
        targets[0][2],
        label='final target'
        )
    ax.scatter(
        sim.data[state_p].T[0][0],
        sim.data[state_p].T[1][0],
        sim.data[state_p].T[2][0],
        label='start state',
        linestyle='--')

    plt.legend()

    plt.figure()
    plt.title('Control Commands')
    plt.ylabel('Rotor Velocities [rad/sec]')
    plt.xlabel('Time [sec]')
    plt.plot(sim.trange(), sim.data[ctrl_p])
    plt.legend(["front_right", "rear_left", "front_left", "rear_right"])

    plt.figure()
    labs = ['current']
    for delay in t_delays:
        labs.append(str(delay))

    plt.subplot(311)
    plt.title('LLP X Predictions')
    for ss in range(0, model.n_pred+1):
        plt.plot(sim.trange(), sim.data[predx_p].T[ss], label=labs[ss])
    plt.legend()
    plt.subplot(312)
    plt.title('LLP Y Predictions')
    for ss in range(0, model.n_pred+1):
        plt.plot(sim.trange(), sim.data[predy_p].T[ss], label=labs[ss])
    plt.legend()
    plt.subplot(313)
    plt.title('LLP Z Predictions')
    for ss in range(0, model.n_pred+1):
        plt.plot(sim.trange(), sim.data[predz_p].T[ss], label=labs[ss])
    plt.legend()



    plt.show()

