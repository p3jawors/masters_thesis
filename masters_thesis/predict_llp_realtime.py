# NOTE will get error if not using LDN encoding, currently not accounting for that
# NOTE hardcoded to only get decoded predictions at tp/t==1
"""
Run the LLP Controller in airsim and visualize the predictions of future position
"""
import traceback
import numpy as np
import nengo
import airsim

# from llp import LLP
from masters_thesis.network.llp import LLP
from masters_thesis.network.ldn import LDN
from masters_thesis.decode_gt_llp_weights_with_nef import run as run_weight_solver
import masters_thesis.network.norm_context as norm_context

import matplotlib.pyplot as plt

from nengo_interfaces.airsim import AirSim
from nengo_control.controllers.quadrotor import PD

# from nengo_control.controllers.path_planners.path_planner_node import PathPlannerNode, ExitSim
from masters_thesis.path_planner_node import PathPlannerNode, ExitSim
from abr_control.controllers.path_planners import PathPlanner
from abr_control.controllers.path_planners.position_profiles import Linear
from abr_control.controllers.path_planners.velocity_profiles import Gaussian
from abr_analyze import DataHandler
import json

# json_fp = "parameter_sets/rt_params_0000.json"
# # json_fp = 'parameter_sets/rt_params_0001.json'
# model = nengo.Network()
# run_mpc = True
# gui = False
# =================
def run(json_fp, model=None, gui=False, run_mpc=False, use_probes=False, n_targets=1, run_pd_as_baseline=None, weights_npz=None, vis_predictions=True):
    with open(json_fp) as fp:
        params = json.load(fp)

    if run_pd_as_baseline is None:
        run_pd_as_baseline = not run_mpc

    # Accepts 12D state as input and outputs a 4D control signal in radians/second
    # in the rotor order: [front_right, rear_left, front_left, rear_right]
    pd_ctrl = PD(gains=params["control"]["gains"])

    # Path planner motion profiles
    Pprof = Linear()
    Vprof = Gaussian(
        dt=params["general"]["dt"], acceleration=params["path"]["path_acc"]
    )
    path = PathPlanner(pos_profile=Pprof, vel_profile=Vprof, verbose=True)

    Pprof_future = Linear()
    Vprof_future = Gaussian(
        dt=params["general"]["dt"], acceleration=params["path"]["path_acc"]
    )
    path_future = PathPlanner(
        pos_profile=Pprof_future, vel_profile=Vprof_future, verbose=True
    )

    # Airsim API
    interface = AirSim(
        dt=params["general"]["dt"],
        run_async=False,
        show_display=True,
    )
    interface.connect(pause=True)

    # Set starting position
    # start_state = [-11.00, -200.00, -12.00, 0, 0, 0]
    # interface.client.simSetVehiclePose(
    #     pose=airsim.Pose(
    #         airsim.Vector3r(start_state[0], start_state[1], start_state[2]),
    #         airsim.to_quaternion(start_state[3], start_state[4], start_state[5]),
    #     ),
    #     ignore_collision=True,
    # )
    # interface.connect(pause=True)

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
        ]
    )

    # targets = [np.array([2, 1, -3, 0, 0, 0, 0, 0, 1.57, 0, 0, 0])]
    # targets = [
    #         np.array([5, 3, -2, 0, 0, 0, 0, 0, 3.14, 0, 0, 0]),
    #         # np.array([-5, 3, -2, 0, 0, 0, 0, 0, 1.57, 0, 0, 0]),
    #         # np.array([3, -3, -5, 0, 0, 0, 0, 0, 3.14, 0, 0, 0]),
    #         # np.array([-1, -2, -2.5, 0, 0, 0, 0, 0, 3.14, 0, 0, 0])
    #     ]

    # Generate random targets
    # n_targets = 20
    # targets = []
    # np.random.seed(9999)
    # # in NED coordinates
    # targets_rng = np.random.RandomState(seed=params["ens_args"]["seed"])
    # for ii in range(0, n_targets):
    #     target = [
    #         targets_rng.uniform(low=-15, high=15, size=1)[0],  # + start_state[0],
    #         targets_rng.uniform(low=-15, high=15, size=1)[0],  # + start_state[1],
    #         targets_rng.uniform(low=-15, high=-1, size=1)[0],  # + start_state[2],
    #         0,
    #         0,
    #         0,
    #         0,  # + start_state[3],
    #         0,  # + start_state[4],
    #         targets_rng.uniform(low=-np.pi, high=np.pi, size=1)[0],  # + start_state[5],
    #         0,
    #         0,
    #         0,
    #     ]
    #
    #     targets.append(np.copy(target))
    # targets = np.asarray(targets)
    targets = np.array([
        [0, 0, -5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, -5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [5, 5, -5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [5, 5, -5, 0, 0, 0, 0, 0, 1.57, 0, 0, 0],
    ])

    # set our target object for visualzation purposes only
    interface.set_state("target", targets[0][:3], targets[0][6:9])

    if model is None:
        model = nengo.Network()
    try:
        with model:
            # model.config[nengo.Connection].synapse = None

            # wrap our interface in a nengo Node
            interface_node = nengo.Node(interface, label="Airsim")

            # TODO add version for testing with NODES for returning recorded state, ctrl, and path, before running in airsim
            # to confirm that encoding and predictions match

            # path planner node
            path_node_present = PathPlannerNode(
                path_planner=path,
                targets=targets,
                dt=params["general"]["dt"],
                buffer_reach_time=params["path"]["buffer_reach_time"],
                at_target_time=eval(params["path"]["at_target_time"]),
                start_state=np.zeros(12),
                debug=True,
                max_velocity=params["path"]["max_velocity"],
                start_velocity=params["path"]["start_velocity"],
                target_velocity=params["path"]["target_velocity"],
                name="path(t)",
            )
            path_node_future = PathPlannerNode(
                path_planner=path_future,
                targets=targets,
                dt=params["general"]["dt"],
                buffer_reach_time=params["path"]["buffer_reach_time"],
                at_target_time=eval(params["path"]["at_target_time"]),
                start_state=np.zeros(12),
                debug=True,
                max_velocity=params["path"]["max_velocity"],
                start_velocity=params["path"]["start_velocity"],
                target_velocity=params["path"]["target_velocity"],
                time_offset=params["llp"]["theta"],
                name="path(t+theta)",
            )

            def sim_extras_func(t, x):
                # function for other things to do in the loop, like setting
                # locations of objects in sim, or other non-control related things
                interface.set_state(
                    "target",
                    path_node_present.targets[path_node_present.target_index][:3],
                    path_node_present.targets[path_node_present.target_index][6:9],
                )
                interface.set_state("filtered_target", x[:3], x[6:9])
                # interface.set_state("predicted_state", x[-3:], [0, 0, 0])

            # takes 12D target as input
            sim_extras = nengo.Node(sim_extras_func, size_in=15, size_out=0)

            # wrap our non-neural controller in a node that takes
            # in target and drone state, and outputs rotor velocities
            def ctrl_func(t, x):
                return pd_ctrl.generate(x[:12], x[12:]).flatten()

            ctrl_node = nengo.Node(ctrl_func, size_in=24, size_out=4)

            # connect our state info to our path planner to check error thresholds
            nengo.Connection(interface_node, path_node_present.input, synapse=None)
            nengo.Connection(interface_node, path_node_future.input, synapse=None)

            # pass path info to sim extras to move a target object to visualize
            nengo.Connection(path_node_present.output, sim_extras[:12], synapse=None)

            # # get our next target from our path planner and pass it to our controller
            nengo.Connection(path_node_present.output, ctrl_node[12:], None)
            nengo.Connection(interface_node, ctrl_node[:12], None)

            # Context encoding
            encoded_state = norm_context.NormStates(params)
            nengo.Connection(
                interface_node,
                encoded_state.input,
                synapse=None,
                label="state>encoded_state.input",
            )

            encoded_path = norm_context.NormPath(params)
            nengo.Connection(
                path_node_future.output,
                encoded_path.input,
                synapse=None,
                label="future_path>encoded_path.input",
            )

            # size is summed length of state, path, and control dims, times their
            # respective number of legendre coefficients. If not LDN then use 1 instead
            # of 0, which is what is set when not using an LDN
            n_state_dims = len(params["data"]["c_dims"]) * max(1, params["data"]["q_c"])
            n_control_dims = len(params["data"]["u_dims"]) * max(
                1, params["data"]["q_u"]
            )
            n_path_dims = len(params["data"]["path_dims"]) * max(
                1, params["data"]["q_path"]
            )

            params["llp"]["size_c"] = n_state_dims + n_control_dims + n_path_dims
            params["llp"]["size_z"] = len(params["data"]["z_dims"])
            # add theta_p to llp params so that the network creates the decoded prediction node
            # params['llp']['theta_p'] = [params['llp']['theta']]

            def z_node_func(t, x):
                z = []
                for dim in params["data"]["z_dims"]:
                    # z.append(x[dim])
                    # 1. c dims are normalized in the order of c_dims in the json params
                    # 2. c dims are passed into the z node
                    # 3. z dims are a subset of c dims, so find the index of cdims
                    # where the value matches z dims
                    z.append(x[params["data"]["c_dims"].index(dim)])
                return z

            z_node = nengo.Node(
                z_node_func,
                # size_in=12,
                size_in=len(params["data"]["c_dims"]),
                size_out=len(params["data"]["z_dims"]),
                label="z_node",
            )

            nengo.Connection(encoded_state.state_norm, z_node, synapse=None)

            if use_probes:
                probes = {}
                probes["state"] = nengo.Probe(interface_node, synapse=None)
                probes["ctrl_pd"] = nengo.Probe(ctrl_node, synapse=None)
                probes["target"] = nengo.Probe(path_node_present.output, synapse=None)
                probes["target_future"] = nengo.Probe(path_node_future.output, synapse=None)
                probes["norm_state"] = nengo.Probe(encoded_state.state_norm, synapse=None)
                probes["ldn_state"] = nengo.Probe(encoded_state.state_ldn, synapse=None)
                probes["norm_path"] = nengo.Probe(encoded_path.path_norm, synapse=None)
                probes["ldn_path"] = nengo.Probe(encoded_path.path_ldn, synapse=None)

            # Initialize predictors
            # HACK FOR WEIGHTS
            # print("\nUSING HACKY WEIGHT LOADING AND SETTING LEARING TO 0\n")
            # with np.load('nef_rt_0001_weights.npz') as data:
            # with np.load("nef_rt_weights.npz") as data:
            if weights_npz is not None:
                print(f"Loading weights from {weights_npz}")
                with np.load(weights_npz) as data:
                # with np.load("nef_weights_rt_0000_750.npz") as data:
                    params["llp"]["decoders"] = np.reshape(
                        data["weights"].T,
                        (
                            params["llp"]["n_neurons"],
                            params["llp"]["q"],
                            len(params["data"]["z_dims"]),
                        ),
                    )
                    params["llp"]["learning_rate"] = 0
            else:
                print('Solving for weights with NEF solver...')
                RMSE, _, target_pts, decoded_pts, weights, z_state = run_weight_solver(params)
                from masters_thesis.utils import plotting
                theta_steps = int(params['llp']['theta']/params['general']['dt'])
                plotting.plot_prediction_vs_gt(
                    tgt=target_pts,
                    decoded=decoded_pts,
                    q=params['llp']['q'],
                    theta=params['llp']['theta'],
                    theta_p=[params['llp']['theta']],
                    # z_state=z_state,#[theta_steps:]
                    theta_steps=theta_steps,
                    # xlim=[0, 1000],
                    show=True,
                    save=True,
                )

                print(type(weights))
                print(weights)
                print(weights.shape)
                params["llp"]["decoders"] = np.reshape(
                    np.copy(weights).T,
                    (
                        params["llp"]["n_neurons"],
                        params["llp"]["q"],
                        len(params["data"]["z_dims"]),
                    ),
                )
                params["llp"]["learning_rate"] = 0

            if params["llp"]["neuron_model"] == "nengo.LIFRate":
                params["llp"]["neuron_model"] = nengo.LIFRate
            # elif params['llp']['neuron_model'] == 'nengo.LifRectifiedLinear':
            #     params['llp']['neuron_model'] = nengo.LIFRectifiedLinear
            elif params["llp"]["neuron_model"] == "nengo.RectifiedLinear":
                params["llp"]["neuron_model"] = nengo.RectifiedLinear
            elif params["llp"]["neuron_model"] == nengo.LIFRate or (
                # params['llp']['neuron_model'] == nengo.LifRectifiedLinear or (
                params["llp"]["neuron_model"]
                == nengo.RectifiedLinear
            ):
                pass
            else:
                raise ValueError(
                    f"{params['llp']['neuron_model']} is not a valid neuron model"
                )

            # if run_mpc:
            #     n_predictors = 1
            # else:
            #     n_predictors = 0

            # assert n_predictors <= 1 ("Currently not implemented for more predictors than one set of 8")
            predictors = []
            # ctrl_norms = []
            # ctrl_ldns = []
            encoded_ctrls = []
            denormed_predictions = []
            predicted_errors = []

            # selector_node = nengo.Node(lambda t, x: [list(x).index(min(x)), min(x)], size_in=2*n_predictors + 1, size_out=2)
            # Return index of min error, and min error
            if run_mpc:
                n_predictors = len(params['control']['bias_dist'])
            else:
                n_predictors = 0

            selector_node = nengo.Node(
                lambda t, x: [list(x).index(min(x)), min(x)],
                size_in=1 + n_predictors * 8,
                # size_in=1 + len(params['control']['bias_dist']) * 8,
                size_out=2,
            )

            # keep a local copy of the parameter choosing the llp implementation. Has to be deleted from dict
            # to avoid errors with kwargs on llp init, but need to know which implementation to use for
            # concurrent LLPs in predictive controller

            if params["llp"]["model_type"] == "mine":
                model_type = "mine"
            elif params["llp"]["model_type"] == "other":
                model_type = "other"
            del params["llp"]["model_type"]

            # NOTE this is the bias on the already normalized control (NOT in radians/sec)
            # bias = 0.0001
            def biases(index, bias):
                # [front_right_pwm, rear_left_pwm, front_left_pwm, rear_right_pwm]
                bias_list = [
                    # clockwise
                    [bias, bias, -bias, -bias],
                    # counterclockwise
                    [-bias, -bias, bias, bias],
                    # up
                    [bias, bias, bias, bias],
                    # down
                    [-bias, -bias, -bias, -bias],
                    # forward
                    [-bias, bias, -bias, bias],
                    # backward
                    [bias, -bias, bias, -bias],
                    # left
                    [bias, -bias, -bias, bias],
                    # right
                    [-bias, bias, bias, -bias],
                ]
                return bias_list[index]

            def init_llp_conns(llp_index, bias_step):
                bias_labels = ['cw', 'ccw', 'up', 'down', 'forward', 'backward', 'left', 'right']
                if llp_index > 0:
                    label = bias_labels[llp_index%8] + f'_{bias_step}'
                else:
                    label =f'no_bias_{bias_step}'
                print('LABEL: ', label)
                print(llp_index)
                # params['ens_args']['label'] = f'LLP_ens_{pp}'
                if model_type == "mine":
                    # scaling factor to better align with other model
                    # also used to account for different timesteps as
                    # this the LLP is implemented in a nengo node so it
                    # has to be accounted for manually
                    params["llp"]["learning_rate"] *= params["general"]["dt"]
                    predictors.append(
                        LLP(
                            ens_args=params["ens_args"],
                            **params["llp"],
                        )
                    )
                elif model_type == "other":
                    predictors.append(
                        LearnDynSys(
                            size_c=params["llp"]["size_c"],
                            size_z=params["llp"]["size_z"],
                            q=params["llp"]["q"],
                            theta=params["llp"]["theta"],
                            n_neurons=params["llp"]["n_neurons"],
                            learning_rate=params["llp"]["learning_rate"],
                            neuron_type=params["llp"]["neuron_model"](),
                            **params["ens_args"],
                        )
                    )

                # NOTE update here if want more predictions than tp/t==1
                def decode_ldn_data(t, x):  # Z, q, theta, theta_p=None):
                    transform = LDN(
                        theta=params["llp"]["theta"], q=params["llp"]["q"], size_in=1
                    ).get_weights_for_delays(np.asarray(params['general']['theta_p'])/params['llp']['theta'])
                    # zhat = []
                    # for _Z in x:
                    # _Z = np.asarray(_Z).reshape(
                    _Z = (
                        np.asarray(x)
                        .reshape((len(params["data"]["z_dims"]), params["llp"]["q"]))
                        .T
                    )
                    # zhat.append(np.dot(transform, _Z))
                    zhat = np.dot(transform, _Z).flatten()

                    return zhat

                # getting odd results using method in LLP, so decoding here
                predictors[-1].zhat = nengo.Node(
                    decode_ldn_data,
                    size_in=len(params["data"]["z_dims"]) * params["llp"]["q"],
                    size_out=len(params["data"]["z_dims"])*len(params['general']['theta_p']),
                    label=f"zhat_{bias_step}_{label}",
                )
                nengo.Connection(predictors[-1].Z, predictors[-1].zhat, synapse=None)

                # if pp == 0:
                #     bias = [0, 0, 0, 0]
                # else:
                #     bias = biases(index=pp - 1, bias=bias_step)

                encoded_ctrls.append(norm_context.NormControl(params, bias_step, run_mpc))

                # NOTE control added lower down after u_mpc is defined
                nengo.Connection(
                    # state_ldn,
                    encoded_state.state_ldn,
                    predictors[-1].c[:n_state_dims],
                    synapse=None,
                )
                nengo.Connection(
                    # ctrl_ldns[-1],
                    encoded_ctrls[-1].ctrl_ldn,
                    predictors[-1].c[n_state_dims : n_state_dims + n_control_dims],
                    synapse=None,
                )
                nengo.Connection(
                    # path_ldn,
                    encoded_path.path_ldn,
                    predictors[-1].c[n_state_dims + n_control_dims :],
                    synapse=None,
                )

                # nengo.Connection(c_node, predictors[-1].c, synapse=None)
                nengo.Connection(z_node, predictors[-1].z, synapse=None)


                # denormalize our prediction
                # TODO test denormalizing function
                def denormalize_state(t, x):
                    # norm_state = np.empty(len(params['data']['state_dims']))
                    denorm_state = []
                    # for dd, dim in enumerate(params['data']['state_dims']):
                    for dd, dim in enumerate(params["data"]["z_dims"]):
                        denorm_state.append(
                            (x[dd] * params["data"]["state_scales"][dim])
                            + params["data"]["state_means"][dim]
                        )

                    return denorm_state

                denormed_predictions.append(
                    nengo.Node(
                        denormalize_state,
                        size_in=len(params["data"]["z_dims"]),
                        size_out=len(params["data"]["z_dims"]),
                        label=f"denormalize_prediction_{bias_step}_{label}",
                    )
                )

                nengo.Connection(
                    predictors[-1].zhat[-len(params['data']['z_dims']):], denormed_predictions[-1], synapse=None
                )

                def calc_error(t, x):
                    full_path = np.asarray(x[:12])
                    # sub_path = np.take(full_path, indices=params['data']['path_dims']) #, axis=1)
                    sub_path = np.take(
                        full_path, indices=params["data"]["z_dims"]
                    )  # , axis=1)
                    pred_vals = np.asarray(x[12:])
                    err = np.linalg.norm(sub_path - pred_vals)
                    return err

                predicted_errors.append(
                    nengo.Node(
                        calc_error,
                        size_in=12 + len(params["data"]["z_dims"]),
                        size_out=1,
                        label=f"predicted_error_{bias_step}_{label}",
                    )
                )
                nengo.Connection(
                    path_node_future.output,
                    predicted_errors[-1][:12],
                    synapse=None,
                    label=f"future_path>error_calc_{bias_step}_{label}",
                )
                nengo.Connection(
                    denormed_predictions[-1],
                    predicted_errors[-1][12:],
                    synapse=None,
                    label=f"denormed_pred>error_calc_{bias_step}_{label}",
                )

                print('LLP INDEX: ', label)
                nengo.Connection(
                    predicted_errors[-1],
                    selector_node[llp_index],
                    # selector_node[bb*8 + (llp_index-1) + 1],
                    synapse=None,
                    label=f"predicted_error_{bias_step}_{label}>selector_node",
                )

                if use_probes:
                    # add probes for plotting
                    # probes['ctrl'] = nengo.Probe(ctrl_node, synapse=0)
                    probes[f"z_{bias_step}_{label}"] = nengo.Probe(predictors[-1].z, synapse=None)
                    probes[f"c_{bias_step}_{label}"] = nengo.Probe(predictors[-1].c, synapse=None)
                    probes[f"zhat_{bias_step}_{label}"] = nengo.Probe(predictors[-1].zhat, synapse=None)
                    probes[f"Z_{bias_step}_{label}"] = nengo.Probe(predictors[-1].Z, synapse=None)
                    if not run_mpc:
                        probes[f"norm_ctrl_{bias_step}_{label}"] = nengo.Probe(
                            encoded_ctrls[-1].ctrl_norm, synapse=None
                        )
                    probes[f"ldn_ctrl_{bias_step}_{label}"] = nengo.Probe(
                        encoded_ctrls[-1].ctrl_ldn, synapse=None
                    )

            # central llp
            init_llp_conns(0, [0, 0, 0, 0])

            # control variations
            if run_mpc:
                for bb, bias_step in enumerate(params['control']['bias_dist']):
                    # for pp in range(0, 8 * n_predictors + 1):
                    # for pp in range(1, 9):
                    for pp in range(0, 8):
                        init_llp_conns(
                            llp_index=1 + (bb*8) + pp,
                            bias_step=biases(index=pp-1, bias=bias_step)
                        )

            # Keep que of all control options and select one based on output of selector node
            def ctrl_que_func(t, x):
                if t > params['general']['dt']*150:
                    # print(f"{t}: {x}")
                    # receives all control options given the mpc bias variatons
                    index = x[-1]
                    # select the slice that has the lowest predicted error
                    u_mpc = x[int(index*4):int((index+1)*4)]
                    # print('MPC control')
                else:
                    u_mpc = [0.05, 0.05, 0.05, 0.05]
                    # u_mpc = [0.35, 0.35, 0.35, 0.35]
                return u_mpc

            # size in is 8 ctrl variations*n_predictors, + 1 for baseline, all
            # * 4 since 4 ctrl signals per variation, + 1 for index of selected ctrl
            ctrl_que = nengo.Node(
                ctrl_que_func,
                size_in=(8*n_predictors+1)*4 + 1,
                # size_in=(8*len(params['control']['bias_dist'])+1)*4 + 1,
                size_out=4,
                label='selected_mpc_ctrl'
            )

            # Add index of selection
            nengo.Connection(selector_node[0], ctrl_que[-1], synapse=None)

            # nengo.Connection(predictors[0].zhat, sim_extras[[12, 13, 14]], synapse=None)
            # show central controller prediction
            nengo.Connection(
                denormed_predictions[0], sim_extras[[12, 13, 14]], synapse=None
            )

            # TODO test denormalizing function
            def denormalize_ctrl(t, x):
                norm_ctrl = np.copy(x)
                denorm_ctrl = []
                # if t > 1:#/params['general']['dt']:
                # print('STARTING MPC')
                for dd, dim in enumerate(x):
                    denorm_ctrl.append(
                        dim
                        * params['data']['ctrl_scales'][dd]
                        + params['data']['ctrl_means'][dd]
                    )
                #     # print('MPC: ', denorm_ctrl)
                # else:
                #     # print('STARTUP SIGNAL')
                #     denorm_ctrl = [7000] * 4

                return [
                    denorm_ctrl[0], denorm_ctrl[1], denorm_ctrl[2], denorm_ctrl[3]#,
                    # norm_ctrl[0], norm_ctrl[1], norm_ctrl[2], norm_ctrl[3]
                ]


            u_mpc = nengo.Node(
                denormalize_ctrl,
                size_in=len(params["data"]["u_dims"]),
                size_out=len(params["data"]["u_dims"]),# *2,
                label=f"denormalize_u_mpc",
            )

            # probes[f"u_mpc_norm"] = nengo.Probe(selector_node, synapse=None)
            if use_probes:
                probes[f"u_mpc"] = nengo.Probe(u_mpc, synapse=None)

            nengo.Connection(ctrl_que, u_mpc, synapse=None)

            # ctrl_que outputs the lowest error prediction control input
            # u_mpc outputs the denormalized version (ready to send to airsim)
            if run_mpc:
                # connect mpc controller output to our sim input
                if run_pd_as_baseline:
                    # NOTE HACK TO GET PREDICTORS WITH PD BASE CONTROL
                    # Normalized version of pd control, as the ctrl_que may output
                    # a different u based on predictor errors
                    def normalize_ctrl(t, x):
                        # norm_ctrl = np.empty(len(params['data']['ctrl_dims']))
                        norm_ctrl = []
                        # for dd, dim in enumerate(params['data']['ctrl_dims']):
                        for dim in params['data']['u_dims']:
                            norm_ctrl.append(
                                (x[dim] - params['data']['ctrl_means'][dim])
                                /params['data']['ctrl_scales'][dim]
                            )
                        norm_ctrl = np.clip(norm_ctrl, -1, 1)

                        return list(norm_ctrl)

                    ctrl_norm = nengo.Node(
                        normalize_ctrl,
                        size_in=4,
                        size_out=len(params['data']['u_dims']),
                        label=f"ctrl_normalized"
                    )
                    nengo.Connection(ctrl_node, interface_node[:4], synapse=0)
                    nengo.Connection(ctrl_node, ctrl_norm, synapse=None)
                else:
                    #TODO check if this is being denormed
                    nengo.Connection(u_mpc[:4], interface_node[:4], synapse=0)

                for pp, enc_ctrl in enumerate(encoded_ctrls):
                    if run_pd_as_baseline:
                        # NOTE HACK TO GET PREDICTORS WITH PD BASE CONTROL
                        nengo.Connection(ctrl_norm, ctrl_que[pp*4:(pp+1)*4], synapse=0)
                        # Add bias
                        nengo.Connection(enc_ctrl.bias_node, ctrl_que[pp*4:(pp+1)*4], synapse=None)

                        nengo.Connection(
                            # u_mpc[4:],
                            ctrl_norm, # need the normalized control, not the one sent to drone
                            enc_ctrl.ctrl_ldn,
                            # enc_ctrl.ctrl_norm,
                            synapse=0,
                            label=f"ctrl_mpc>ctrl_ldn_{pp}",
                        )

                    else:
                        # Add base, which is the input control+bias of the predictor
                        # with the lowest error from the last step
                        # This is also the output from the ctrl_que, so setup feedback
                        # connection
                        # nengo.Connection(enc_ctrl.ctrl_norm, ctrl_que[pp*4:(pp+1)*4], synapse=None)
                        nengo.Connection(ctrl_que, ctrl_que[pp*4:(pp+1)*4], synapse=0)
                        # Add bias
                        # NOTE should synapse match connection of baseline?
                        nengo.Connection(enc_ctrl.bias_node, ctrl_que[pp*4:(pp+1)*4], synapse=None)

                        nengo.Connection(
                            # u_mpc[4:],
                            ctrl_que, # need the normalized control, not the one sent to drone
                            enc_ctrl.ctrl_ldn,
                            synapse=0,
                            label=f"ctrl_mpc>ctrl_ldn_{pp}",
                        )

            else:
                # connect pd controller output to our sim input
                nengo.Connection(ctrl_node, interface_node[:4], synapse=0)

                for pp, enc_ctrl in enumerate(encoded_ctrls):
                    nengo.Connection(
                        ctrl_node,
                        enc_ctrl.ctrl_norm,
                        synapse=0,
                        label=f"ctrl.output>ctrl_norms_{pp}.input",
                    )
            if vis_predictions:
                colors = []
                for cc in range(0, len(predictors)):
                    colors.append(np.random.uniform(0, 1, 3))
                    # col = np.random.uniform(0, 1, 3)
                    # for tp in range(0, len(params['general']['theta_p'])):
                    #     colors.append(
                    #         [col[0], col[1], col[2], 1 - tp/len(params['general']['theta_p'])*0.5]
                    #     )
                # print(colors)

                def prediction_dots(t, x):
                    duration = params['general']['dt']*10
                    if int(1000*t)%int(1000*duration) == 0:
                        n_tp = len(params['general']['theta_p'])
                        n_z = len(params['data']['z_dims'])
                        # dim per predictor
                        ndim = n_tp * n_z
                        # print(f"number of decoded values per predictor: {n_tp}")
                        # print(f"number of dimensions per decoded value: {n_z}")
                        # print(f"number of dimensions output per predictor: {ndim}")
                        # print(f"number of predictors: {len(predictors)}")

                        for npred in range(0, len(predictors)):
                            # print(f'decoding predictions for predictor_{npred}')
                            # all theta_p decoded predictions of z_dims for a single predictor
                            zhat_tp = x[npred*ndim:(npred+1)*ndim]
                            # print(f"zhat length: {len(zhat_tp)}")
                            # print(zhat_tp)
                            points = []
                            # loop through each decoded value
                            for ii in range(0, n_tp):
                                # print(f'looping through decoded values: {ii}')
                                # denormalze
                                # print(f'slicing: [{ii*n_z}, {ii*n_z+1}, {ii*n_z+2}]')
                                xyz_norm = [zhat_tp[ii*n_z], zhat_tp[(ii*n_z)+1], zhat_tp[(ii*n_z)+2]]
                                xyz_denorm = np.zeros(n_z)
                                for jj, xyz in enumerate(xyz_norm):
                                # for dd, dim in enumerate(params["data"]["z_dims"]):
                                    xyz_denorm[jj] = (
                                        (xyz * params["data"]["state_scales"][jj])
                                        + params["data"]["state_means"][jj]
                                    )

                                points.append(
                                    airsim.Vector3r(
                                        xyz_denorm[0], xyz_denorm[1], xyz_denorm[2]
                                    )
                                )
                            interface.client.simPlotPoints(
                                points=points,
                                # color_rgba=colors[npred*n_tp:(npred+1)*n_tp],
                                color_rgba=colors[npred].tolist(),
                                duration=duration
                            )

                nz = len(params['data']['z_dims'])
                ntp = len(params['general']['theta_p'])
                prediction_visualizer = nengo.Node(
                    prediction_dots,
                    size_in=len(predictors)*nz*ntp
                )

                for pp, predictor in enumerate(predictors):
                    nengo.Connection(
                        predictor.zhat,
                        prediction_visualizer[pp*nz*ntp:(pp+1)*nz*ntp],
                        synapse=None,
                    )



            if not gui:
                with nengo.Simulator(model, dt=params["general"]["dt"]) as sim:
                    # path planner will exit when we reach our final target
                    sim.run(50000 * params["general"]["dt"])

    except ExitSim as e:
        print("Exiting Sim")
        manual_stop = False

    except KeyboardInterrupt as e:
        print("TEST STOPPED MANUALLY:", e)
        print(traceback.format_exc())
        manual_stop = True

    except Exception as e:
        print("Exception Raised")
        print(type(e))
        print(traceback.format_exc())
        manual_stop = True

    finally:
        if use_probes:
            if manual_stop:
                txt = input("Do you want to save the partial test data? [y/n]")
                if txt == "y" or txt == "Y":
                    print("\n\nSaving partial test data...\n\n")
                    continue_with_save = True
                else:
                    print("\n\nClosing without saving data...\n\n")
                    continue_with_save = False

            else:
                continue_with_save = True

            if continue_with_save:
                dat = DataHandler("thesis_realtime_trained_predictor")
                data = {}
                for key, val in probes.items():
                    data[key] = sim.data[val]
                data["time"] = sim.trange()
                if run_mpc and run_pd_as_baseline:
                    dat.save(data=data, save_location=json_fp+"_LLPSE_best", overwrite=True)
                elif run_mpc:
                    dat.save(data=data, save_location=json_fp+"_LLPC_best", overwrite=True)
                else:
                    dat.save(data=data, save_location=json_fp, overwrite=True)

        print('Set use_probes=True to save probe data')
        interface.pause(False)
        interface.disconnect()


if __name__ == "__main__":
    # json_fp = "parameter_sets/rt_params_0000.json"
    # json_fp = 'parameter_sets/rt_params_0001.json'
    json_fp = "parameter_sets/thesis_rt_params.json"
    model = nengo.Network()
    # run(json_fp, model, gui=False, run_mpc=True, use_probes=True, n_targets=10, run_pd_as_baseline=False)
    # run(json_fp, model, gui=False, run_mpc=True, use_probes=True, n_targets=10, run_pd_as_baseline=True)
    run(json_fp, model, gui=False, run_mpc=False, use_probes=False, n_targets=10, run_pd_as_baseline=True)
