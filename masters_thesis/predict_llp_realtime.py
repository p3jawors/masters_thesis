# NOTE will get error if not using LDN encoding, currently not accounting for that
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
# from plotting import plot_pred

def run(json_fp):
    with open(json_fp) as fp:
        params = json.load(fp)

    # Accepts 12D state as input and outputs a 4D control signal in radians/second
    # in the rotor order: [front_right, rear_left, front_left, rear_right]
    pd_ctrl = PD(gains=params['control']['gains'])

    # Path planner motion profiles
    Pprof = Linear()
    Vprof = Gaussian(dt=params['general']['dt'], acceleration=params['path']['path_acc'])
    path = PathPlanner(
            pos_profile=Pprof,
            vel_profile=Vprof,
            verbose=True
    )

    Pprof_future = Linear()
    Vprof_future = Gaussian(dt=params['general']['dt'], acceleration=params['path']['path_acc'])
    path_future = PathPlanner(
            pos_profile=Pprof_future,
            vel_profile=Vprof_future,
            verbose=True
    )


    # Airsim API
    interface = AirSim(
            dt=params['general']['dt'],
            run_async=False,
            show_display=True,
    )
    interface.connect(pause=False)

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
                ])

    # targets = [np.array([2, 1, -3, 0, 0, 0, 0, 0, 1.57, 0, 0, 0])]
    # targets = [
    #         np.array([5, 3, -2, 0, 0, 0, 0, 0, 3.14, 0, 0, 0]),
    #         # np.array([-5, 3, -2, 0, 0, 0, 0, 0, 1.57, 0, 0, 0]),
    #         # np.array([3, -3, -5, 0, 0, 0, 0, 0, 3.14, 0, 0, 0]),
    #         # np.array([-1, -2, -2.5, 0, 0, 0, 0, 0, 3.14, 0, 0, 0])
    #     ]

    # Generate random targets
    n_targets = 3
    targets = []
    # in NED coordinates
    targets_rng = np.random.RandomState(seed=params['ens_args']['seed'])
    for ii in range(0, n_targets):
        target = [
            targets_rng.uniform(low=-15, high=15, size=1)[0], # + start_state[0],
            targets_rng.uniform(low=-15, high=15, size=1)[0], # + start_state[1],
            targets_rng.uniform(low=-31, high=-1, size=1)[0], # + start_state[2],
            0,
            0,
            0,
            0, # + start_state[3],
            0, # + start_state[4],
            targets_rng.uniform(low=-np.pi, high=np.pi, size=1)[0], # + start_state[5],
            0,
            0,
            0,
        ]

        targets.append(np.copy(target))
    targets = np.asarray(targets)

    # set our target object for visualzation purposes only
    interface.set_state("target", targets[0][:3], targets[0][6:9])

    model = nengo.Network()
    try:
        with model:
            # model.config[nengo.Connection].synapse = None

            # wrap our interface in a nengo Node
            interface_node = nengo.Node(
                    interface,
                    label="Airsim"
            )

            # TODO add version for testing with NODES for returning recorded state, ctrl, and path, before running in airsim
            # to confirm that encoding and predictions match

            # path planner node
            path_node_present = PathPlannerNode(
                path_planner=path,
                targets=targets,
                dt=params['general']['dt'],
                buffer_reach_time=params["path"]["buffer_reach_time"],
                at_target_time=eval(params["path"]["at_target_time"]),
                start_state=np.zeros(12),
                debug=True,
                max_velocity=params["path"]["max_velocity"],
                start_velocity=params["path"]["start_velocity"],
                target_velocity=params["path"]["target_velocity"],
                name='path(t)'
            )
            # TODO test shifting the path like this
            path_node_future = PathPlannerNode(
                path_planner=path_future,
                targets=targets,
                dt=params['general']['dt'],
                buffer_reach_time=params["path"]["buffer_reach_time"],
                at_target_time=eval(params["path"]["at_target_time"]),
                start_state=np.zeros(12),
                debug=True,
                max_velocity=params["path"]["max_velocity"],
                start_velocity=params["path"]["start_velocity"],
                target_velocity=params["path"]["target_velocity"],
                time_offset=params["llp"]["theta"],
                name='path(t+theta)'
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
            nengo.Connection(interface_node, path_node_present.input, synapse=None)
            nengo.Connection(interface_node, path_node_future.input, synapse=None)

            # pass path info to sim extras to move a target object to visualize
            nengo.Connection(path_node_present.output, sim_extras, synapse=None)

            # # get our next target from our path planner and pass it to our controller
            nengo.Connection(path_node_present.output, ctrl_node[12:], None)
            nengo.Connection(interface_node, ctrl_node[:12], None)

            # connect our controller output to our sim input
            nengo.Connection(ctrl_node, interface_node[:4], synapse=0)

            # Context encoding
            def normalize_state(t, x):
                # norm_state = np.empty(len(params['data']['state_dims']))
                norm_state = []
                # for dd, dim in enumerate(params['data']['state_dims']):
                for dim in params['data']['c_dims']:
                    norm_state.append(
                        (x[dim] - params['data']['state_means'][dim])
                        /params['data']['state_scales'][dim]
                    )

                return norm_state

            state_norm = nengo.Node(
                normalize_state,
                size_in=12,
                size_out=len(params['data']['c_dims'])
            )

            nengo.Connection(
                interface_node,
                state_norm,
                synapse=None,
                label='state>state_norm'
            )

            state_ldn = nengo.Node(
                LDN(
                    theta=params['data']['theta_c'],
                    q=params['data']['q_c'],
                    size_in=len(params['data']['c_dims']),
                ),
                label='ldn_state'
            )

            nengo.Connection(
                state_norm,
                state_ldn,
                synapse=None,
                label='state_norm>state_ldn'
            )


            def normalize_path(t, x):
                # norm_path = np.empty(len(params['data']['path_dims']))
                norm_path = []
                # for dd, dim in enumerate(params['data']['path_dims']):
                for dim in params['data']['path_dims']:
                    norm_path.append(
                        (x[dim] - params['data']['path_means'][dim])
                        /params['data']['path_scales'][dim]
                    )

                return list(norm_path)

            path_norm = nengo.Node(
                normalize_path,
                size_in=12,
                size_out=len(params['data']['c_dims'])
            )

            nengo.Connection(
                path_node_future.output,
                path_norm,
                synapse=None,
                label='future_path>path_norm'
            )

            path_ldn = nengo.Node(
                LDN(
                    theta=params['data']['theta_path'],
                    q=params['data']['q_path'],
                    size_in=len(params['data']['path_dims']),
                ),
                label='ldn_path'
            )

            nengo.Connection(
                path_norm,
                path_ldn,
                synapse=None,
                label='path_norm>path_ldn'
            )


            # size is summed length of state, path, and control dims, times their
            # respective number of legendre coefficients. If not LDN then use 1 instead
            # of 0, which is what is set when not using an LDN
            n_state_dims = len(params['data']['c_dims']) * max(1, params['data']['q_c'])
            n_control_dims =  len(params['data']['u_dims']) * max(1, params['data']['q_u'])
            n_path_dims = len(params['data']['path_dims']) * max(1, params['data']['q_path'])

            params['llp']['size_c'] = n_state_dims + n_control_dims + n_path_dims
            params['llp']['size_z'] = len(params['data']['z_dims'])
            # add theta_p to llp params so that the network creates the decoded prediction node
            params['llp']['theta_p'] = [params['llp']['theta']]

            # c_node = nengo.Node(
            #     size_out=n_state_dims + n_control_dims + n_path_dims,
            #     label='c_node'
            # )


            def z_node_func(t, x):
                z = []
                for dim in params['data']['z_dims']:
                    # z.append(x[dim])
                    # 1. c dims are normalized in the order of c_dims in the json params
                    # 2. c dims are passed into the z node
                    # 3. z dims are a subset of c dims, so find the index of cdims
                    # where the value matches z dims
                    z.append(x[params['data']['c_dims'].index(dim)])
                return z

            z_node = nengo.Node(
                z_node_func,
                # size_in=12,
                size_in=len(params['data']['c_dims']),
                size_out=len(params['data']['z_dims']),
                label='z_node'
            )

            nengo.Connection(state_norm, z_node, synapse=None)

            probes = {}
            probes['state'] = nengo.Probe(interface_node, synapse=0)
            probes['target'] = nengo.Probe(path_node_present.output, synapse=0)
            probes['target_future'] = nengo.Probe(path_node_future.output, synapse=0)

            # Initialize predictors
            # HACK FOR WEIGHTS
            print('\nUSING HACKY WEIGHT LOADING AND SETTING LEARING TO 0\n')
            with np.load('nef_rt_weights.npz') as data:
                params['llp']['decoders'] = np.reshape(
                    data['weights'].T,
                    (params['llp']['n_neurons'], params['llp']['q'], len(params['data']['z_dims']))
                )
                params['llp']['learning_rate'] = 0

            if params['llp']['neuron_model'] == 'nengo.LIFRate':
                params['llp']['neuron_model'] = nengo.LIFRate
            # elif params['llp']['neuron_model'] == 'nengo.LifRectifiedLinear':
            #     params['llp']['neuron_model'] = nengo.LIFRectifiedLinear
            elif params['llp']['neuron_model'] == 'nengo.RectifiedLinear':
                params['llp']['neuron_model'] = nengo.RectifiedLinear
            elif params['llp']['neuron_model'] == nengo.LIFRate or (
                    # params['llp']['neuron_model'] == nengo.LifRectifiedLinear or (
                        params['llp']['neuron_model'] == nengo.RectifiedLinear):
                pass
            else:
                raise ValueError(f"{params['llp']['neuron_model']} is not a valid neuron model")

            n_predictors = 0
            predictors = []
            ctrl_norms = []
            ctrl_ldns = []
            denormed_predictions = []
            predicted_errors = []

            # return index of lowest error and the error value
            # def select_lowest_error(t, x):
            #     lowest = 1e10
            #     for _x in x:

            selector_node = nengo.Node(lambda t, x: [list(x).index(min(x)), min(x)], size_in=2*n_predictors + 1, size_out=2)

            # keep a local copy of the parameter choosing the llp implementation. Has to be deleted from dict
            # to avoid errors with kwargs on llp init, but need to know which implementation to use for
            # concurrent LLPs in predictive controller

            if params['llp']['model_type'] == 'mine':
                model_type = 'mine'
            elif params['llp']['model_type'] == 'other':
                model_type = 'other'
            del params['llp']['model_type']

            for pp in range(0, 2*n_predictors+1):
                if model_type == 'mine':
                    # scaling factor to better align with other model
                    # also used to account for different timesteps as
                    # this the LLP is implemented in a nengo node so it
                    # has to be accounted for manually
                    params['llp']['learning_rate'] *= params['general']['dt']
                    predictors.append(
                        LLP(
                            ens_args=params['ens_args'],
                            **params['llp'],
                        )
                    )
                elif model_type == 'other':
                    predictors.append(
                        LearnDynSys(
                            size_c=params['llp']['size_c'],
                            size_z=params['llp']['size_z'],
                            q=params['llp']['q'],
                            theta=params['llp']['theta'],
                            n_neurons=params['llp']['n_neurons'],
                            learning_rate=params['llp']['learning_rate'],
                            neuron_type=params['llp']['neuron_model'](),
                            **params['ens_args'],
                        )
                    )

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

                ctrl_norms.append(
                    nengo.Node(
                        normalize_ctrl,
                        size_in=4,
                        size_out=len(params['data']['u_dims'])
                    )
                )

                # TODO connect to predictive controller output after scaling up
                nengo.Connection(
                    # ctrl_predictive.output,
                    ctrl_node,  # PD
                    ctrl_norms[-1],
                    synapse=None,
                    label=f'ctrl>ctrl_norms_{pp}'
                )
                # TODO add offset to control context and test
                # TODO sort out some better distribution of values
                if ii > 0:
                    bias = 0.1
                    if ii % 2 == 0:
                        scale = int(ii/2)
                    else:
                        scale = int((ii+1)/2)
                    bias_node = nengo.Node(
                        lambda t: [scale*bias * -1**pp] * len(params['data']['u_dims']),
                        size_out=len(params['data']['u_dims']),
                        label=f'bias_{pp}'
                    )
                    nengo.Connection(
                        bias_node,
                        ctrl_norms[-1],
                        synapse=None,
                        label=f'bias_node_{pp}>ctrl_norms_{pp}'
                    )

                ctrl_ldns.append(
                    nengo.Node(
                        LDN(
                            theta=params['data']['theta_u'],
                            q=params['data']['q_u'],
                            size_in=len(params['data']['u_dims'])
                        ),
                        label=f'ldn_ctrl_{pp}'
                    )
                )

                nengo.Connection(
                    ctrl_norms[-1],
                    ctrl_ldns[-1],
                    synapse=None,
                    label=f'ctrl_norms_{pp}>ctrl_ldns_{pp}'
                )

                nengo.Connection(state_ldn, predictors[-1].c[:n_state_dims], synapse=None)
                nengo.Connection(ctrl_ldns[-1], predictors[-1].c[n_state_dims:n_state_dims+n_control_dims], synapse=None)
                nengo.Connection(path_ldn, predictors[-1].c[n_state_dims+n_control_dims:], synapse=None)

                # nengo.Connection(c_node, predictors[-1].c, synapse=None)
                nengo.Connection(z_node, predictors[-1].z, synapse=None)

                # denormalize our prediction
                def denormalize_state(t, x):
                    # norm_state = np.empty(len(params['data']['state_dims']))
                    denorm_state = []
                    # for dd, dim in enumerate(params['data']['state_dims']):
                    for dd, dim in enumerate(params['data']['z_dims']):
                        denorm_state.append(
                            (x[dd] * params['data']['state_scales'][dim])
                            + params['data']['state_means'][dim]
                        )

                    return denorm_state

                denormed_predictions.append(
                    nengo.Node(
                        denormalize_state,
                        size_in=len(params['data']['z_dims']),
                        size_out=len(params['data']['z_dims'])
                    )
                )

                nengo.Connection(predictors[-1].zhat,  denormed_predictions[-1], synapse=None)

                def calc_error(t, x):
                    full_path = np.asarray(x[:12])
                    # sub_path = np.take(full_path, indices=params['data']['path_dims']) #, axis=1)
                    sub_path = np.take(full_path, indices=params['data']['z_dims']) #, axis=1)
                    pred_vals = np.asarray(x[12:])
                    err = np.linalg.norm(sub_path - pred_vals)
                    return err

                predicted_errors.append(
                    nengo.Node(
                        calc_error,
                        size_in=12 + len(params['data']['z_dims']),
                        size_out=1,
                        label=f"predicted_error_{pp}"
                    )
                )
                nengo.Connection(
                    path_node_future.output,
                    predicted_errors[-1][:12],
                    synapse=None,
                    label=f"future_path>error_calc_{pp}"
                )
                nengo.Connection(
                    denormed_predictions[-1],
                    predicted_errors[-1][12:],
                    synapse=None,
                    label=f"denormed_pred>error_calc_{pp}"
                )

                nengo.Connection(
                    predicted_errors[-1],
                    selector_node[pp],
                    synapse=None,
                    label=f"predicted_error_{pp}>selector_node"
                )

                # add probes for plotting
                # probes['ctrl'] = nengo.Probe(ctrl_node, synapse=0)
                probes[f'z_{pp}'] = nengo.Probe(predictors[-1].z, synapse=None)
                probes[f'zhat_{pp}'] = nengo.Probe(predictors[-1].zhat, synapse=None)
                probes[f'Z_{pp}'] = nengo.Probe(predictors[-1].Z, synapse=None)


            # TODO add scale up of input control when using predictive controller

            # TODO add visualization of prediction

        with nengo.Simulator(model, dt=params['general']['dt']) as sim:
            # path planner will exit when we reach our final target
            sim.run(50000 * params['general']['dt'])

    except ExitSim as e:
        print('Exiting Sim')
        manual_stop = False

    except KeyboardInterrupt as e:
        print("TEST STOPPED MANUALLY:", e)
        print(traceback.format_exc())
        manual_stop = True

    except Exception as e:
        print("Exception Raised")
        print(type(e))
        print(traceback.format_exc())
        manual_stop = False

    finally:
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
            dat = DataHandler('realtime_trained_predictor')
            data = {}
            for key, val in probes.items():
                data[key] = sim.data[val]
            data['time'] = sim.trange()
            dat.save(data=data, save_location=json_fp, overwrite=True)
        # data = dat.load(
        #         save_location="test_0000",
        #         parameters=['state', 'target', 'ctrl', 'z', 'zhat', 'Z', 'time']
        # )

        # plot_pred(
        #     data=data,
        #     theta_p=t_delays,
        #     size_out=llp_size_out,
        #     gif_name='llp_test.gif'
        # )

        interface.pause(False)
        interface.disconnect()
        # Plot results
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # plt.title('Flight Path in NED Coordinates')
        # ax.plot(
        #     sim.data[target_p].T[0],
        #     sim.data[target_p].T[1],
        #     sim.data[target_p].T[2],
        #     label='target'
        #     )
        # ax.plot(
        #     sim.data[state_p].T[0],
        #     sim.data[state_p].T[1],
        #     sim.data[state_p].T[2],
        #     label='state',
        #     linestyle='--')
        # ax.scatter(
        #     targets[0][0],
        #     targets[0][1],
        #     targets[0][2],
        #     label='final target'
        #     )
        # ax.scatter(
        #     sim.data[state_p].T[0][0],
        #     sim.data[state_p].T[1][0],
        #     sim.data[state_p].T[2][0],
        #     label='start state',
        #     linestyle='--')
        #
        # plt.legend()
        #
        # plt.figure()
        # plt.title('Control Commands')
        # plt.ylabel('Rotor Velocities [rad/sec]')
        # plt.xlabel('Time [sec]')
        # plt.plot(sim.trange(), sim.data[ctrl_p])
        # plt.legend(["front_right", "rear_left", "front_left", "rear_right"])
        #
        # plt.figure()
        # labs = ['current']
        # for delay in t_delays:
        #     labs.append(str(delay))
        #
        # plt.subplot(311)
        # plt.title('LLP X Predictions')
        # for ss in range(0, model.n_pred+1):
        #     plt.plot(sim.trange(), sim.data[predx_p].T[ss], label=labs[ss])
        # plt.legend()
        # plt.subplot(312)
        # plt.title('LLP Y Predictions')
        # for ss in range(0, model.n_pred+1):
        #     plt.plot(sim.trange(), sim.data[predy_p].T[ss], label=labs[ss])
        # plt.legend()
        # plt.subplot(313)
        # plt.title('LLP Z Predictions')
        # for ss in range(0, model.n_pred+1):
        #     plt.plot(sim.trange(), sim.data[predz_p].T[ss], label=labs[ss])
        # plt.legend()
        #
        #
        #
        # plt.show()
        #
if __name__ == '__main__':
    json_fp = 'parameter_sets/rt_params_0000.json'
    run(json_fp)
