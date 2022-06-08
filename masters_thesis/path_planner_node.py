import math
import timeit

import nengo
import numpy as np
from nengo.network import Network


class ExitSim(Exception):
    print("At end of target list, ending simulation")
    pass


class PathPlannerNode(Network):
    def __init__(
        self,
        path_planner,
        start_state,
        targets,
        dt,
        seed=0,
        buffer_reach_time=None,
        at_target_time=None,
        at_target_thresholds=None,
        probes=False,
        debug=False,
        time_offset=0,
        name='path_planner',
        **path_args,
    ):
        """
        Parameters
        ----------
        path_planner: instantiated path planner class, or list of them
            path planner class must have....
                - a next() function that returns the next 12D target state
                - a generate_path() function that will generate a 12D trajectry given a 12D state and target
                - a self.time_to_converge parameter that stores the time to target
                - a self.dt parameter
            see abr_control repo for various compatible path planners
        start_state: list
            12x1 array of starting state [x,y,z,dx,dy,dz,a,b,g,da,db,dg]
        targets: list
            n_targets x 12 array of target positions and orientations
        dt: float
            timestep size in sec
        seed: int, Optional (Default: 0)
            seed for rng functions
        buffer_reach_time: float, Optional (Default: None)
            time [sec] of buffer time to reach target past how
            long the path planner takes to converge before
            switching targets
        at_target_time: float, Optional (Default: None)
            the time [sec] to maintain a sub error threshold (consecutively)
            before switching targets.
            Note make sure to set buffer_reah_time larger than this time, otherwise
            we will reach our time threshold before we have had enough time to
            reach our at target time amount
            Default will be None, so we just switch targets after the path planners
            `time_to_converge + buffer_reach_time`
        at_target_thresholds:
            2d list of target error to maintain [meters, radians]
            before switching targets
        probes: bool, Optional (Default: False)
            whether or not to use probes for tracking network data
        debug: bool, Optional (Default: False)
            Set to true to get additional debug prints
        time_offset: Optional, (Default: 0)
            Offset to retrieve future path values. Set in seconds. This will also
            be added to the end of the target list so that the total number of steps
            remains constant.
        """
        self.name = name
        red = "\033[91m"
        yellow = "\u001b[33m"
        endc = "\033[0m"
        if buffer_reach_time is not None and at_target_time is not None:
            assert (
                buffer_reach_time > at_target_time,
                f"{red}at_target_time ({at_target_time}) > buffer_reach_time ({buffer_reach_time})"
                + "\nTo maintain a target error for some amount of time"
                + " we need to have that amount of threshold time past"
                + " path planner convergence time, otherwise we will always"
                + " switch targets before the controller has time to reach"
                + f" the target error count{endc}",
            )

        # make a list if there is only one item so we can use the
        # same looping later on
        if not isinstance(at_target_thresholds, list):
            if at_target_thresholds is None:
                at_target_thresholds = [None, None]

        if at_target_time is not None:
            assert (
                at_target_thresholds[0] is not None
                or at_target_thresholds[1] is not None
            ), (
                f"\n{red}If passing in an at_target_time, we must have at least one of either position"
                + f" or orientation at_target_thresholds be not None\nReceived {at_target_thresholds}{endc}"
            )

        if not isinstance(path_planner, (list, np.ndarray)):
            path_planner = [path_planner]

        if len(targets) > len(path_planner):
            # if only one path planner was passed in, then duplicate a reference
            # to that instance for each target we are going to.
            # At this point path_planner could be a list or array, this makes
            # sure we create a list with n_target references to path_planner
            path_planner = np.asarray(path_planner).tolist()
            path_planner *= len(targets)

        self.debug = debug
        self.dt = dt
        self.path_planner = path_planner
        self.targets = targets
        self.buffer_reach_time = buffer_reach_time
        self.at_target_time = at_target_time
        self.at_target_threshold = at_target_thresholds
        # place holder as we will add reaching time to this when we generate our path
        self.reach_time_limit = np.copy(buffer_reach_time)
        # self.state = None
        self.state = start_state
        # used for stopping the sim if one of the optional criteria are met for error or accuracy limits
        self.raise_error = False
        self.start_time = timeit.default_timer()
        self.test_accuracy = 0
        # offset for future path values
        self.time_offset = time_offset
        self.time_offset_steps = int(time_offset/self.dt)

        super().__init__(seed=seed)

        if probes:
            self.probes_dict = {}

        with self:
            # tracks the steps our path planners take to converge + buffer reach time
            # for paths up to and including the current one. This is used to compare
            # with the sim time to determine when to switch to the next target
            self.cumulative_path_time = dt
            # index of the target we are currently reaching
            self.target_index = -1
            # tracks the number of steps taken into the current path
            self.step_count = 0
            # the number of steps we have been within our target threshold
            self.at_target_count = 0
            # cumulative number of sim steps
            self.cumulative_steps = 0

            def input_func(t, x):
                self.state = x
                return x

            self.input = nengo.Node(
                input_func, size_in=12, size_out=12, label="input_node"
            )

            def generate_path_func(t, x):
                """
                input: int (1 or 0) whether to go to next target
                output: None

                Receives a 0 or 1 from the at_target_time node if the user sets a target time
                to maintain a sub position and orientation error. This gives another way to
                go to the next target. Otherwise we just wait for reach_time + buffer time to
                generate the next path
                """
                # t += self.time_offset
                # We need an update command from the thresholds being used
                if self.debug:
                    if t % 2.5 == 0:
                        print("----------DEBUG----------")
                        print("next target change: ", self.cumulative_path_time)
                        print(
                            "current run time: %.2f | real/sim: %.2f"
                            % (t, (timeit.default_timer() - self.start_time) / t)
                        )
                        print(f"reach time limit: {self.reach_time_limit}")

                # t > our path planner time to converge to target + buffer_reach_time
                if t >= self.cumulative_path_time or x == 1:
                    if x == 1:
                        # subtract the time we saved by reaching sub error thres early
                        # this value is fortunately also just the sim time, so set it to that
                        self.cumulative_path_time = t

                    # state should only be None on init, at which point we just wait
                    # for state feedback to initialize our path planner
                    if self.state is not None:
                        self.target_index += 1
                        # we have gone through all of our targets, end sim
                        if self.target_index >= len(self.targets):
                            print(f"{yellow}Total sim time: {t}{endc}")
                            raise ExitSim()
                            # else:
                            #     continue

                        else:
                            # if first target, see if the user wants to plan the path
                            # from a preset point, or the current drone's state
                            # subsequent targets will start the path from the last step
                            # of the previous path
                            if self.target_index == 0:
                                state = self.state
                                # self.target_index += 1
                            else:
                                # state = self.path_planner[self.target_index - 1].next()
                                state = self.path_planner[self.target_index-1].next_at_n(
                                    self.path_planner[self.target_index-1].n_timesteps-1
                                )

                            # generate our path from state to target
                            print(f"{self.name}")
                            self.path_planner[self.target_index].generate_path(
                                start_position=state[:3],
                                target_position=self.targets[self.target_index][:3],
                                start_orientation=state[6:9],
                                target_orientation=targets[self.target_index][6:9],
                                **path_args,
                            )
                            # update our time limit before the target changes
                            # this is the path planners time to converge + buffer_reach_time
                            self.reach_time_limit = (
                                self.buffer_reach_time
                                + self.path_planner[self.target_index].time_to_converge
                            )
                            # to offset future shifted path, need to cut first reach short
                            # and extend last so that it aligns with the non-shifted base sim
                            if self.target_index == 0:
                                self.reach_time_limit -= self.time_offset
                            elif self.target_index == len(self.targets)-1:
                                self.reach_time_limit += self.time_offset

                            # don't include takeoff as target
                            if self.target_index > 0:
                                print(
                                    "\n===== Target %i/%i : convergence_time: %f=====\n"
                                    % (
                                        self.target_index,
                                        len(self.targets) - 1,
                                        self.reach_time_limit,

                                    )
                                )
                            # reset our step counter for the next path
                            # if first target skip ahead by time offset steps
                            if self.target_index == 0:
                                self.step_count = self.time_offset_steps
                                print(f'0th TARGET, SETTING STEP COUNT TO OFFSET: {self.time_offset_steps}')
                            else:
                                self.step_count = 0
                            self.cumulative_path_time += self.reach_time_limit

            # run once to initialize path planner before nengo compiles network
            generate_path_func(t=dt, x=1)
            generate_path = nengo.Node(
                generate_path_func, size_in=1, size_out=0, label="generate_path"
            )

            if self.at_target_time:

                def error_thres_func(t, x):
                    """
                    input: 6d pos and orientation
                    output: 0 or 1 to generate next path if within error thres for at_target_time

                    Calculates the 2norm errors and checks if they are within the user defined thresholds
                    if they are then a counter is incremented. Once we reach ths desired at_target_time
                    return a 1 to tell the generate_path node to update to the next path early
                    """
                    pos = x[:3]
                    ori = x[6:9]
                    target_pos = self.targets[self.target_index][:3]
                    target_ori = self.targets[self.target_index][6:9]

                    # if a position threshold is set, check if we're meeting it
                    if self.at_target_threshold[0] is not None:
                        pos_error = np.linalg.norm(pos - target_pos)
                        if pos_error < self.at_target_threshold[0]:
                            at_pos = True
                        else:
                            at_pos = False
                    else:
                        at_pos = True

                    # if an orientation threshold is set, check if we're meeting it
                    if self.at_target_threshold[1] is not None:
                        # NOTE previously we just looked at yaw, this looks at abg
                        ori_error = np.linalg.norm(ori - target_ori)
                        if ori_error < self.at_target_threshold[1]:
                            at_ori = True
                        else:
                            at_ori = False
                    else:
                        at_ori = True

                    # if within target threshold, increase counter by 1
                    if at_ori and at_pos:
                        self.at_target_count += 1
                    # otherwise reset it to zero as we want consecutive steps at target
                    else:
                        self.at_target_count = 0

                    if self.at_target_count * self.dt >= self.at_target_time:
                        update_path = 1
                    else:
                        update_path = 0

                    return update_path

                # if we're not checking an error threshold, we don't need any input
                self.error_thres = nengo.Node(
                    error_thres_func, size_in=12, size_out=1, label="error_thres"
                )
                nengo.Connection(self.error_thres, generate_path, synapse=None)
                nengo.Connection(self.input, self.error_thres, synapse=None)

            def next_step_func(t):
                """
                updates our counters and returns the next point in our path
                """
                self.cumulative_steps += 1
                self.step_count += 1
                # return self.path_planner[self.target_index].next()
                return self.path_planner[self.target_index].next_at_n(self.step_count) # + int(self.time_offset/self.dt))

            self.output = nengo.Node(
                next_step_func, size_in=0, size_out=12, label="output"
            )

            if probes:
                # if self.buffer_reach_time:
                #     self.probes_dict['over_time_limit'] = nengo.Probe(check_time_limit, synapse=None)
                if self.at_target_time:
                    self.probes_dict["within_target_error"] = nengo.Probe(
                        self.error_thres, synapse=None
                    )

                def target_track(t):
                    return self.targets[self.target_index]

                path_track = nengo.Node(
                    target_track, size_in=0, size_out=12, label="return_target_at_index"
                )
                self.probes_dict["targets"] = nengo.Probe(path_track, synapse=None)
