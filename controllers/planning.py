import copy
from typing import List, Optional, Tuple, Literal
from enum import Enum

import numpy as np

from pydrake.all import (
    AbstractValue,
    Diagram,
    DiagramBuilder,
    MultibodyPlant,
    LeafSystem,
    RigidTransform,
    RotationMatrix,
    InverseKinematics,
    PiecewisePolynomial,
    PiecewisePose,
    Solve,
    RollPitchYaw,
    TrajectorySource,
)


def optimize_target_trajectory(keyframes: List[RigidTransform],
                               orientation_constraint_flags: List[bool],
                               duplicate_keyframe_flags: List[bool],
                               plant: MultibodyPlant,
                               q_nominal: np.array) -> Optional[np.array]:

    assert len(keyframes) == len(orientation_constraint_flags)
    assert len(keyframes) == len(duplicate_keyframe_flags)
    # q_nominal are the current positions
    print('||q_nominal|| =', q_nominal.shape, 'q_nominal', q_nominal)
    q_nominal_full = np.zeros(plant.num_positions(),)
    q_nominal_full[:7] = q_nominal

    print('dofs:', plant.num_actuated_dofs())
    print('num_model_instances:', plant.num_model_instances())
    print('num_joints:', plant.num_joints())

    joint_indices = [
        plant.GetJointByName(j).position_start()
        for j in map(lambda i: f'iiwa_joint_{i}', range(1, 8))
    ]
    print('joint_indices:', joint_indices, 'joints len=', len(joint_indices))

    q_keyframes = []
    is_first = True
    for keyframe, do_constrain_orientation, do_kf_duplicate in zip(
            keyframes, orientation_constraint_flags, duplicate_keyframe_flags):
        ik = InverseKinematics(plant)
        q_variables = ik.q()
        prog = ik.prog()

        if is_first:
            prog.SetInitialGuess(q_variables, q_nominal_full)
            is_first = False
        else:
            prog.SetInitialGuess(q_variables, q_keyframes[-1])
            
        prog.AddCost(np.square(np.dot(q_variables, q_nominal_full)))

        offset = np.array([0.01, 0.01, 0.01])
        offset_upper = keyframe.translation() + offset
        offset_lower = keyframe.translation() - offset

        pos = np.zeros((3,))

        ik.AddPositionConstraint(
            frameA=plant.world_frame(),
            frameB=plant.GetFrameByName("body"),
            p_BQ=pos,
            p_AQ_lower=offset_lower,
            p_AQ_upper=offset_upper)

        if do_constrain_orientation:
            ik.AddOrientationConstraint(
                    frameAbar=plant.GetFrameByName("body"),
                    R_AbarA=RotationMatrix.Identity(),
                    frameBbar=plant.world_frame(),
                    R_BbarB=keyframe.rotation(),
                    theta_bound=np.radians(0.5)
                )

        result = Solve(prog)
        if not result.is_success():
            print(f'no sol for i={kid}')
            print(result.GetInfeasibleConstraintNames(prog))
            return None
        else:
            q_keyframes.append(result.GetSolution(q_variables))
            if do_kf_duplicate:
                q_keyframes.append(q_keyframes[-1])

    return np.array(q_keyframes)


def make_cartesian_trajectory(keyframes: List[RigidTransform], timestamps: List[float]) -> PiecewisePose:
    if len(keyframes) != len(timestamps):
        raise Exception('bad input')
    return PiecewisePose.MakeLinear(timestamps, keyframes)


def make_wsg_trajectory(duplicate_flags: List[bool], timestamps: List[float], start: Literal['open', 'closed']):
    assert start in ('open', 'closed')

    def fix_duplicate_flags(flags):
        fl = []
        for flag in flags:
            if flag:
                fl.extend([False, True])
            else:
                fl.append(flag)
        return fl

    flags = fix_duplicate_flags(duplicate_flags)
    if len(flags) != len(timestamps):
        raise Exception('`duplicate flags` mess-up')

    opened = np.array([0.107])
    closed = np.array([0.0])
    tictac = [opened, closed]

    ind = 0 if start == 'open' else 1
    positions = []
    for duplicate_flag in flags:
        if duplicate_flag:
            ind = (ind + 1) % 2
        positions.append(tictac[ind % 2])
    positions = np.array(positions)
    return PiecewisePolynomial.FirstOrderHold(timestamps, positions.T)


def make_dummy_wsg_trajectory() -> PiecewisePolynomial:
    ts = [0, 1]
    wsg_keyframes = np.array([0.107, 0.107]).reshape(1, 2)
    return PiecewisePolynomial.FirstOrderHold(ts, wsg_keyframes)


def solve_for_grip_direction(X_WVstart: RigidTransform) -> RotationMatrix:
    possible_nut_rotations = []
    for ind, alpha in enumerate(np.radians(np.arange(30, 360, 60))):
        R_1 = RollPitchYaw([0, alpha, 0]).ToRotationMatrix() @ X_WVstart.rotation()
        vec = np.degrees(RollPitchYaw(R_1).vector()).tolist()
        vec_with_ind = [ind, np.round(np.degrees(alpha))] + vec
        #print("// ind={} angle {} RollPitchYaw: {:.1f} {:.1f} {:.1f}".format(*vec_with_ind))
        #AddMeshcatTriad(meshcat, f'nut-sides-{alpha:.3f}', X_PT=X_curr_rot, opacity=0.33)
        possible_nut_rotations.append(R_1)

    arg_min = 0
    least_ang_distance = float('inf')
    R_WNutInBetterOrientation = RotationMatrix(RollPitchYaw(np.radians([90., 60., 0.])))
    for ind, R_WNcand in enumerate(possible_nut_rotations):
        loss = np.linalg.norm(np.degrees(RollPitchYaw(R_WNcand.InvertAndCompose(R_WNutInBetterOrientation)).vector()))
        if loss < least_ang_distance:
            least_ang_distance = loss
            arg_min = ind

    return possible_nut_rotations[arg_min]

class TurnStage(Enum):
    UNSET = 'unset'
    APPROACH = 'approach'
    SCREW = 'screw'
    RETRACT = 'retract'
    FINISH = 'finish'

    def next(self):
        members = list(self.__class__)
        index = members.index(self)
        index = index + 1
        if index >= len(members):
            return None
        else:
            return members[index]


class MultiTurnPlanner(LeafSystem):
    def __init__(self,
                 mode: Literal['stiffness', 'hybrid'],
                 plant: MultibodyPlant):

        LeafSystem.__init__(self)
        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()
        self.stage: TurnStage = TurnStage.UNSET

        if mode not in ['stiffness', 'hybrid']:
            raise Exception('impossible mode', mode)
        else:
            self.mode = mode
            if mode == 'hybrid':
                raise Exception('didnt implement yet')

        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self.DeclareVectorInputPort("iiwa_state_measured", 14)

        self.X_WGinitial = None
        self.X_WVinitial = None
        self.gripper_index = self.plant.GetBodyByName('body').index()
        self.nut_index = self.plant.GetBodyByName('nut').index()

        self.DeclareAbstractOutputPort("stiffness_controller_switch", lambda: AbstractValue.Make(bool()), self.set_stiffness_switch)
        self.DeclareAbstractOutputPort("hybrid_controller_switch", lambda: AbstractValue.Make(bool()), self.set_hybrid_switch)

        self.iiwa_trajectory_index = int(self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())))
        self.taskspace_trajectory_index = int(self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePose())))
        self.wsg_trajectory_index = int(self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())))

        self.traj_published = { x: False for x in ['iiwa_trajectory', 'wsg_trajectory', 'taskspace_trajectory'] }
        self.DeclareAbstractOutputPort(
            "current_trajectory_for_stiffness", lambda: AbstractValue.Make(PiecewisePolynomial()), self.set_stiffness_trajectory)
        self.DeclareAbstractOutputPort(
            "current_trajectory_for_hybrid", lambda: AbstractValue.Make(PiecewisePose()), self.set_hybrid_trajectory)
        self.DeclareAbstractOutputPort(
            "current_trajectory_for_wsg", lambda: AbstractValue.Make(PiecewisePolynomial()), self.set_gripper_trajectory)
        self.DeclareAbstractOutputPort(
            "current_stage", lambda: AbstractValue.Make(TurnStage.UNSET), self.calc_current_stage)

        self.DeclarePeriodicUnrestrictedUpdateEvent(.1, .0, self.reinitialize_plan)


    def set_stiffness_trajectory(self, context, output):
        if not self.traj_published['iiwa_trajectory']:
            self.traj_published['iiwa_trajectory'] = True
            output.set_value(context.get_abstract_state(self.iiwa_trajectory_index).get_value())


    def set_hybrid_trajectory(self, context, output):
        if not self.traj_published['taskspace_trajectory']:
            self.traj_published['taskspace_trajectory'] = True
            output.set_value(context.get_abstract_state(self.taskspace_trajectory_index).get_value())


    def set_gripper_trajectory(self, context, output):
        if not self.traj_published['wsg_trajectory']:
            self.traj_published['wsg_trajectory'] = True
            wsg_traj = context.get_abstract_state(self.wsg_trajectory_index).get_value()
            output.set_value(wsg_traj)


    def get_stiffness_switch(self) -> bool:
        return self.mode == 'stiffness' or \
               self.stage in (TurnStage.APPROACH, TurnStage.RETRACT)


    def set_stiffness_switch(self, context, output):
        is_stiffness = self.get_stiffness_switch()
        output.set_value(is_stiffness)


    def set_hybrid_switch(self, context, output):
        is_force = not self.get_stiffness_switch()
        output.set_value(is_force)


    def calc_current_stage(self, context, output):
        output.set_value(self.stage)


    def solve_for_approach(self, keyframes, orientation_flags, duplicate_flags, next_stage, state, context):
        assert TurnStage.APPROACH == next_stage

        iiwa_state = self.GetInputPort("iiwa_state_measured").Eval(context)
        iiwa_positions = iiwa_state[:7]

        q_keyframes = optimize_target_trajectory(keyframes, orientation_flags, duplicate_flags,
                                                 self.plant, iiwa_positions)
        if q_keyframes is None:
            print('opt has failed')
            exit(1)
            return

        valid_timestamps = [0., 2., 4., 6]
                          # 0, start pose
                              # 1, reach pre grasp pose
                                  # 2, reach grasp pose
                                      # 3, reach closed grip

        if q_keyframes.shape[0] != len(valid_timestamps):
            raise Exception('logical mistake in traj building {}, {}'.format(q_keyframes.shape[0], len(valid_timestamps)))

        q_trajectory = PiecewisePolynomial.FirstOrderHold(valid_timestamps, q_keyframes[:, :7].T)
        start_gripper_position = 'open'
        wsg_trajectory = make_wsg_trajectory(duplicate_flags, valid_timestamps, start_gripper_position)

        state.get_mutable_abstract_state(self.iiwa_trajectory_index).set_value(q_trajectory)
        state.get_mutable_abstract_state(self.wsg_trajectory_index).set_value(wsg_trajectory)

        self.stage = next_stage
        for x in ['iiwa_trajectory', 'wsg_trajectory']:
            self.traj_published[x] = False


    def solve_for_turn(self, keyframes, orientation_flags, duplicate_flags, next_stage, state, context):
        assert TurnStage.SCREW == next_stage

        iiwa_state = self.GetInputPort("iiwa_state_measured").Eval(context)
        iiwa_positions = iiwa_state[:7]

        q_keyframes = optimize_target_trajectory(keyframes, orientation_flags, duplicate_flags,
                                                 self.plant, iiwa_positions)
        if q_keyframes is None:
            print('opt has failed')
            exit(1)
            return

        valid_timestamps = [6., 16.]
                          # 0, reach closed grip
                              # 1, completed turn
        if q_keyframes.shape[0] != len(valid_timestamps):
            raise Exception('logical mistake in traj building {}, {}'.format(q_keyframes.shape[0], len(valid_timestamps)))

        q_trajectory = PiecewisePolynomial.FirstOrderHold(valid_timestamps, q_keyframes[:, :7].T)
        start_gripper_position = 'closed'
        wsg_trajectory = make_wsg_trajectory(duplicate_flags, valid_timestamps, start_gripper_position)
        state.get_mutable_abstract_state(self.wsg_trajectory_index).set_value(wsg_trajectory)
        self.traj_published['wsg_trajectory'] = False

        if self.mode == 'stiffness':
            state.get_mutable_abstract_state(self.iiwa_trajectory_index).set_value(q_trajectory)
            self.traj_published['iiwa_trajectory'] = False

        elif self.mode == 'hybrid':
            cart_trajectory = make_cartesian_trajectory(keyframes, valid_timestamps)
            state.get_mutable_abstract_state(self.taskspace_trajectory_index).set_value(cart_trajectory)
            self.traj_published['taskspace_trajectory'] = False
        else:
            raise Exception('unreachable')

        self.stage = next_stage


    def solve_for_retract(self, keyframes, orientation_flags, duplicate_flags, next_stage, state, context):
        assert TurnStage.RETRACT == next_stage

        iiwa_state = self.GetInputPort("iiwa_state_measured").Eval(context)
        iiwa_positions = iiwa_state[:7]

        q_keyframes = optimize_target_trajectory(keyframes, orientation_flags, duplicate_flags,
                                                 self.plant, iiwa_positions)
        if q_keyframes is None:
            print('opt has failed')
            exit(1)
            return

        valid_timestamps = [16., 18., 20., 22.]
                          # 4, completed turn
                               # 5, reach open grip
                                    # 6, reach post grasp
                                         # 7, reach start pose
        if q_keyframes.shape[0] != len(valid_timestamps):
            raise Exception('logical mistake in traj building {}, {}'.format(q_keyframes.shape[0], len(valid_timestamps)))

        q_trajectory = PiecewisePolynomial.FirstOrderHold(valid_timestamps, q_keyframes[:, :7].T)
        start_gripper_position = 'closed'
        wsg_trajectory = make_wsg_trajectory(duplicate_flags, valid_timestamps, start_gripper_position)
        state.get_mutable_abstract_state(self.wsg_trajectory_index).set_value(wsg_trajectory)
        state.get_mutable_abstract_state(self.iiwa_trajectory_index).set_value(q_trajectory)
        self.stage = next_stage
        for x in ['iiwa_trajectory', 'wsg_trajectory']:
            self.traj_published[x] = False


    def get_latest_current_trajectory_time(self, context, state) -> Optional[float]:
        def get_opt_tj(context, state, state_index: int):
            traj = context.get_abstract_state(state_index).get_value()
            return None if 0 == traj.get_number_of_segments() else traj

        tjs = list(filter(lambda y: y is not None, map(lambda x: get_opt_tj(context, state, x),
                   [self.iiwa_trajectory_index, self.taskspace_trajectory_index])))
        if 0 == len(tjs):
            return None
        else:
            return max(map(lambda t: t.end_time(), tjs))


    def reinitialize_plan(self, context, state):
        now = context.get_time()
        have_plant_until_t = self.get_latest_current_trajectory_time(context, state)
        if have_plant_until_t is not None and now < have_plant_until_t:
            return

        next_stage = self.stage.next()
        if not next_stage:
            return

        if TurnStage.UNSET == next_stage:
            raise Exception('unreachable')
        elif TurnStage.FINISH == next_stage:
            self.stage = next_stage
            return

        current_poses = self.GetInputPort("body_poses").Eval(context)
        X_WG = current_poses[self.gripper_index]
        X_WV = current_poses[self.nut_index]
        if TurnStage.APPROACH == next_stage:
            if self.X_WGinitial is None:
                self.X_WGinitial = X_WG
            if self.X_WVinitial is None:
                self.X_WVinitial = X_WV
        elif next_stage in (TurnStage.SCREW, TurnStage.RETRACT):
            assert self.X_WGinitial is not None
            assert self.X_WVinitial is not None

        print('reinitialize_plan? at t=', context.get_time())

        R_GoalGripper = RotationMatrix(RollPitchYaw(np.radians([180, 0, 90]))).inverse()
        t_GoalGripper = [0., -0.085, -0.02]
        t_GoalGripperPreGrasp = [0., -0.225, -0.02]
        X_GoalGripper = RigidTransform(R_GoalGripper, np.zeros((3,)))
        X_gripper_offset = RigidTransform(RotationMatrix.Identity(), t_GoalGripper)
        X_pre_grasp_offset = RigidTransform(RotationMatrix.Identity(), t_GoalGripperPreGrasp)

        #AddMeshcatTriad(meshcat, 'pregrasp-at-initial-valve', X_PT=X_WGripperPreGraspAtTurnStart)
        #AddMeshcatTriad(meshcat, 'gripper-at-initial-valve', X_PT=X_WGripperAtTurnStart)
        #AddMeshcatTriad(meshcat, 'gripper-at-final-valve', X_PT=X_WGripperAtTurnEnd)
        #AddMeshcatTriad(meshcat, 'postgrasp-at-final-valve', X_PT=X_WGripperPostGraspAtTurnEnd)

        if TurnStage.APPROACH == next_stage:
            R_WVpreferred = solve_for_grip_direction(X_WV)
            X_WGripperAtTurnStart_ = RigidTransform(R_WVpreferred, X_WV.translation()) @ X_GoalGripper
            X_WGripperAtTurnStart = X_WGripperAtTurnStart_ @ X_gripper_offset
            X_WGripperPreGraspAtTurnStart = X_WGripperAtTurnStart_ @ X_pre_grasp_offset

            keyframes = [self.X_WGinitial, X_WGripperPreGraspAtTurnStart, X_WGripperAtTurnStart]
            orientation_flags = [False, True, True]
            duplicate_flags = [False, False, True]
            self.solve_for_approach(keyframes, orientation_flags, duplicate_flags, next_stage, state, context)

        elif TurnStage.SCREW == next_stage:
            X_WGripperAtTurnStart = X_WG

            X_WGripperAtTurnStart_ = X_WGripperAtTurnStart @ X_gripper_offset.inverse()
            R_WVinitial = (X_WGripperAtTurnStart_ @ X_GoalGripper.inverse()).rotation()

            R_WVend_ = RollPitchYaw(np.radians([0, 30, 0])).ToRotationMatrix() @ R_WVinitial
            X_WGripperAtTurnEnd_ = RigidTransform(R_WVend_, self.X_WVinitial.translation()) @ X_GoalGripper
            X_WGripperAtTurnEnd = X_WGripperAtTurnEnd_ @ X_gripper_offset

            keyframes = [X_WGripperAtTurnStart, X_WGripperAtTurnEnd]

            orientation_flags = [True, True]
            duplicate_flags = [False, False]
            self.solve_for_turn(keyframes, orientation_flags, duplicate_flags, next_stage, state, context)

        elif TurnStage.RETRACT == next_stage:
            X_WGripperAtTurnEnd = X_WG
            X_WGripperAtTurnEnd_ = X_WGripperAtTurnEnd @ X_gripper_offset.inverse()
            X_WGripperPostGraspAtTurnEnd = X_WGripperAtTurnEnd_ @ X_pre_grasp_offset

            keyframes = [X_WGripperAtTurnEnd, X_WGripperPostGraspAtTurnEnd, self.X_WGinitial]
            orientation_flags = [True, True, False]
            duplicate_flags = [True, False, False]
            self.solve_for_retract(keyframes, orientation_flags, duplicate_flags, next_stage, state, context)

        else:
            raise Exception('unreachable')

        # this is cheating. maybe try to manage without this:
        # self.plant.SetPositions(self.plant_context, self.plant.GetModelInstanceByName("iiwa"), trajectory.value(0))


def MakeWsgTrajectory() -> Diagram:

    class AbsToVectorProcessor(LeafSystem):
        def __init__(self, trajectory_source):
            LeafSystem.__init__(self)
            self.DeclareAbstractInputPort('trajectory', AbstractValue.Make(PiecewisePolynomial()))
            self.trajectory_source = trajectory_source
            self.DeclarePerStepDiscreteUpdateEvent(self.update_trajectory)
            self.old_traj_limits = None

        def update_trajectory(self, context, state):
            new_traj = self.GetInputPort('trajectory').Eval(context)
            new_limits = new_traj.start_time(), new_traj.end_time()
            if self.old_traj_limits is None or (self.old_traj_limits[0] != new_limits[0]) or (self.old_traj_limits[1] != new_limits[1]):
                print('resets wsg traj')
                self.trajectory_source.UpdateTrajectory(new_traj)
                self.old_traj_limits = new_limits


    builder = DiagramBuilder()
    trajectory_source = builder.AddSystem(TrajectorySource(make_dummy_wsg_trajectory()))
    processor = builder.AddSystem(AbsToVectorProcessor(trajectory_source))

    builder.ExportInput(processor.GetInputPort('trajectory'), 'abs_trajectory')
    builder.ExportOutput(trajectory_source.get_output_port(), 'positions_from_trajectory')

    diagram = builder.Build()
    diagram.set_name("wsg trajectory processor")
    return diagram
