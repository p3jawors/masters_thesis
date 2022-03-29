"""
Runs airsim sim of drone going to targets to generate training data
"""
# TODO
"""
- add adaptive controller
- add llp controller
"""

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
# force_params = {
#         'id': 'force_params',
#         'n_neurons': 1000,
#         'n_ensembles': 1,
#         'n_input': 4,
#         'n_output': 4,
#         'intercepts': None,
#         'intercept_bounds': [0.0, 0.0, 0.1],
#         'encoders': None,
#         'weights': None,
#         'learning_rate': None,
#         'tau_input': None,
#         'tau_output': 0.05,
#         'tau_training': 0.05,
#         'spherical': True,
#         'input_dims': [0, 1, 2],
#         'training_dims': np.arange(12),
#         'offchip_scale': None,
#         'training_error_limit': 300,
#     }
#
# angle_params = {
#         'id': 'angle_params',
#         'n_neurons': 1000,
#         'n_ensembles': 1,
#         'n_input': 1,
#         'n_output': 2,
#         'intercepts': None,
#         'intercept_bounds': [0.0, 0.0, 0.1],
#         'encoders': None,
#         'weights': None,
#         'learning_rate': None,
#         'tau_input': None,
#         'tau_output': 0.05,
#         'tau_training': 0.05,
#         'spherical': True,
#         'input_dims': [8],
#         'training_dims': np.arange(12),
#         'a_gains': [
#             2.8256250539071077,
#             4996.076580296213,
#             0.15854921256222543,
#             1073.0843029984721
#         ],
#         'pd_gains':[
#             9483.984536604656,
#             4157.641221757219,
#             2974.5495866882998,
#             11013.455417814257,
#             14788.192863046886,
#             9378.21231769378,
#             334.75328788213335,
#             6736.005631999111
#         ],
#         'offchip_scale': None,
#         'training_error_limit': 0.25,
#
#     }
#
# neural_params = {
#         'dt': dt,
#         'actm': False,
#         'ac': False,
#         'ac_allo': False,
#         'seed': neural_seed,
#         'force_params': force_params,
#         'angle_params': angle_params,
#         'means': np.array([
#             0.4, 0.0, 0.4,
#             0, 0, 0,
#             0, 0, 0,
#             0, 0, 0]),
#         'variances': np.array([
#             3.1, 2.8, 2.5,
#             1.4, 1.4, 1.4,
#             0.6, 0.523, 3.14,
#             1.6, 1.6, 1.6]),
#
#         'pes_wgt_exp': 4
# }


# if backend != 'pd':
#     print('Nengo CPU Control - learning')
#     neural_params['angle_params']['learning_rate'] = 1.968503937007874e-05
#     neural_params['force_params']['learning_rate'] = 0.11811023622047243
#     neural_params['angle_params']['offchip_scale'] = 1
#     neural_params['force_params']['offchip_scale'] = 1
#     neural_params['pes_wgt_exp'] = None
#
#
#     controller = neural.Neural(
#         seed=neural_params['seed'],
#         force_params=neural_params['force_params'],
#         angle_params=neural_params['angle_params'],
#         means=neural_params['means'],
#         variances=neural_params['variances'],
#         use_probes=debug,
#         backend=backend,
#         dt=neural_params['dt']
#     )
# else:
#     controller = pd.PD(
#         seed=neural_params['seed'],
#         use_probes=debug,
#         backend=backend,
#         dt=neural_params['dt'],
#         tau_output=neural_params['angle_params']['tau_output'],
#         gains=neural_params['angle_params']['pd_gains']
#     )


if len(sys.argv)>1:
    n_targets = int(sys.argv[1])
else:
    n_targets = 100

np.random.seed(n_targets)
print(f"Collecting data for {n_targets} target reach")

airsim_dt = 0.01
save_location = f'{n_targets}_linear_targets_faster'
notes = (
"""
-max_v=6, acc=3
"""
)

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
Vprof = Gaussian(dt=airsim_dt, acceleration=3)
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
                max_velocity=6,
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
    dat = DataHandler('llp_pd', 'data/databases')
    data = {}
    for key, val in probes.items():
        data[key] = sim.data[val]
    data['time'] = sim.trange()
    data['notes'] = notes
    dat.save(data=data, save_location=save_location, overwrite=True)
    interface.disconnect()
