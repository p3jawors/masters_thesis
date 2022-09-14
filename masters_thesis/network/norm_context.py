import numpy as np
import nengo
from masters_thesis.network.ldn import LDN
from nengo.network import Network
class NormStates(Network):
    def __init__(self, params):
        super().__init__(seed=params['ens_args']['seed'])

        self.input = nengo.Node(
            size_in=12,
            size_out=12,
        )

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

        self.state_norm = nengo.Node(
            normalize_state,
            size_in=12,
            size_out=len(params['data']['c_dims'])
        )

        nengo.Connection(
            # interface_node,
            self.input,
            self.state_norm,
            synapse=None,
            label='norm_state.input>state_norm'
        )

        self.state_ldn = nengo.Node(
            LDN(
                theta=params['data']['theta_c'],
                q=params['data']['q_c'],
                size_in=len(params['data']['c_dims']),
            ),
            label='ldn_state'
        )

        nengo.Connection(
            self.state_norm,
            self.state_ldn,
            synapse=None,
            label='state_norm>state_ldn'
        )


class NormPath(Network):
    def __init__(self, params):
        super().__init__(seed=params['ens_args']['seed'])

        self.input = nengo.Node(
            size_in=12,
            size_out=12,
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

        self.path_norm = nengo.Node(
            normalize_path,
            size_in=12,
            size_out=len(params['data']['path_dims'])
        )

        nengo.Connection(
            self.input,
            self.path_norm,
            synapse=None,
            label='norm_path.input>path_norm'
        )

        self.path_ldn = nengo.Node(
            LDN(
                theta=params['data']['theta_path'],
                q=params['data']['q_path'],
                size_in=len(params['data']['path_dims']),
            ),
            label='ldn_path'
        )

        nengo.Connection(
            self.path_norm,
            self.path_ldn,
            synapse=None,
            label='path_norm>path_ldn'
        )



class NormControl(Network):
    def __init__(self, params, biases, rt_control=True):

        super().__init__(seed=params['ens_args']['seed'])

        self.ctrl_ldn = nengo.Node(
            LDN(
                theta=params['data']['theta_u'],
                q=params['data']['q_u'],
                size_in=len(params['data']['u_dims'])
            ),
            label=f'ldn_ctrl'
        )

        # TODO add offset to control context and test
        # BIAS OF 0.05 looks like it would be good (check hist of du/dt)
        # if sum(biases) != 0:
        self.bias_node = nengo.Node(
            lambda t: biases,
            size_out=4,
            label=f'ctrl_bias'
        )

        nengo.Connection(
            self.bias_node,
            self.ctrl_ldn,
            synapse=None,
            label=f'bias_node>ctrl_ldn'
        )
        # If only using as a predictor, need some baseline control to
        # use as context. This will need to be normalized before being
        # summed with the biais going into the LDN
        if not rt_control:
           # TODO connect to predictive controller output after scaling up
            # Only connected if using as predictor. When used as controller
            # then connections are made externally
            self.input = nengo.Node(
                size_in=4,
                size_out=4,
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

            self.ctrl_norm = nengo.Node(
                normalize_ctrl,
                size_in=4,
                size_out=len(params['data']['u_dims']),
                label=f"ctrl_normalized"
            )

            nengo.Connection(
                self.ctrl_norm,
                self.ctrl_ldn,
                synapse=None,
                label=f'ctrl_norms>ctrl_ldns'
            )


            nengo.Connection(
                self.input,
                self.ctrl_norm,
                synapse=None,
                label=f'norm_ctrl.input>ctrl_norms'
            )


