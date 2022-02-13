import numpy as np
import nengo
import sys

from nengo_interfaces.airsim import AirSim
from nengo_control.controllers.quadrotor import PD
from nengo_control.controllers.path_planners.path_planner_node import PathPlannerNode, ExitSim
from abr_control.controllers.path_planners import PathPlanner
from abr_control.controllers.path_planners.position_profiles import Linear
from abr_control.controllers.path_planners.velocity_profiles import Gaussian
from abr_analyze import DataHandler

if len(sys.argv)>1:
    n_targets = int(sys.argv[1])
else:
    n_targets = 100
print(f"Collecting data for {n_targets} target reach")

airsim_dt = 0.01
save_location = '100_linear_targets'

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

# Path planner motion profiles
Pprof = Linear()
Vprof = Gaussian(dt=airsim_dt, acceleration=1)
path = PathPlanner(
        pos_profile=Pprof,
        vel_profile=Vprof,
        verbose=True
    )

# Airsim API
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

targets = []
# in NED coordinates
for ii in range(0, n_targets):
    target = [
            np.random.uniform(low=-15, high=15, size=1)[0],
            np.random.uniform(low=-15, high=15, size=1)[0],
            np.random.uniform(low=-15, high=-1, size=1)[0],
            0, 0, 0,
            0, 0, np.random.uniform(low=-np.pi, high=np.pi, size=1)[0],
            0, 0, 0]

    targets.append(np.copy(target))
targets = np.asarray(targets)

# set our target object for visualzation purposes only
interface.set_state("target", targets[0][:3], targets[0][6:9])

model = nengo.Network()
try:
    with model:
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
                max_velocity=4,
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

        # add probes for plotting
        probes = {}
        probes['state'] = nengo.Probe(interface_node, synapse=0)
        probes['target'] = nengo.Probe(path_node.output, synapse=0)
        probes['ctrl'] = nengo.Probe(ctrl_node, synapse=0)

    with nengo.Simulator(model, dt=airsim_dt) as sim:
        # path planner will exit when we reach our final target
        sim.run(1e6 * airsim_dt)

except ExitSim as e:
    print('Exiting Sim')

finally:
    dat = DataHandler('llp_pd')
    data = {}
    for key, val in probes.items():
        data[key] = sim.data[val]
    data['time'] = sim.trange()
    dat.save(data=data, save_location=save_location, overwrite=True)
    interface.disconnect()
