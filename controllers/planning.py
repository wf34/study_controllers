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


def get_current_positions(plant, plant_context):
    q_ = np.zeros(plant.num_positions(),)
    q = plant.GetPositions(
        plant_context,
        plant.GetModelInstanceByName("iiwa"),
    )
    q_[:7] = q
    return q_


def optimize_target_trajectory(keyframes: List[RigidTransform], plant, plant_context) -> Tuple[Optional[PiecewisePolynomial], Optional[np.array]]:
    '''
    desc

    Args:
    Returns:
    '''
    iiwa_model_instance = plant.GetModelInstanceByName('iiwa')
    q_nominal = get_current_positions(plant, plant_context)
    print('q_nominal', q_nominal)

    num_q = plant.num_positions()
    num_iiwa_only = plant.num_positions(iiwa_model_instance)
    x = list(map(int, plant.GetJointIndices(model_instance=iiwa_model_instance)))
    print(f'GetJointIndices {x} of total {num_q}; of total iiwa-related: {num_iiwa_only}')
    print('dofs:', plant.num_actuated_dofs())
    print('num_model_instances:', plant.num_model_instances())
    print('num_joints:', plant.num_joints())

    joint_indices = [
        plant.GetJointByName(j).position_start()
        for j in map(lambda i: f'iiwa_joint_{i}', range(1, 8))
    ]
    print('joint_indices:', joint_indices)

    q_keyframes = []
    for kid, keyframe in enumerate(keyframes):
        ik = InverseKinematics(plant)
        q_variables = ik.q()
        prog = ik.prog()

        if 0 == kid:
            prog.SetInitialGuess(q_variables, q_nominal)
        else:
            prog.SetInitialGuess(q_variables, q_keyframes[-1])
            
        prog.AddCost(np.square(np.dot(q_variables, q_nominal)))

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

        if kid in (1, 2, 3, 4):
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
            break
        else:
            q_keyframes.append(result.GetSolution(q_variables))
            if kid in (2,3):
                q_keyframes.append(q_keyframes[-1])

    print('q_keyframes, keyframes:', len(q_keyframes), len(keyframes))
    if len(q_keyframes) - 2 == len(keyframes):
        q_keyframes = np.array(q_keyframes)
        valid_timestamps = [0., 2., 4., 6., 16., 18., 20., 22.]
                            # 0, start pose
                                # 1, reach pre grasp pose
                                    # 2, reach grasp pose
                                        # 3, reach closed grip
                                            # 4, completed turn
                                                 # 5, reach open grip
                                                      # 6, reach post grasp
                                                           # 7, reach start pose
        q_trajectory = PiecewisePolynomial.FirstOrderHold(valid_timestamps, q_keyframes[:, :7].T)
        return q_trajectory, valid_timestamps
    else:
        return None, None


def make_cartesian_trajectory(keyframes: List[RigidTransform], timestamps: List[float]) -> PiecewisePose:
    if len(keyframes) != len(timestamps):
        raise Exception('bad input')
    return PiecewisePose.MakeLinear(timestamps, keyframes)


def make_wsg_trajectory(timestamps: List[float]):
    assert 8 == len(timestamps)
    opened = np.array([0.107])
    closed = np.array([0.0])
    traj_wsg_command = PiecewisePolynomial.FirstOrderHold(
        timestamps[:2],
        np.hstack([[opened], [opened]]),
    )

    wsg_keyframes = [opened, opened, opened, closed, closed, opened, opened, opened]
    #                0       1       2       3       4       5       6       7
    for ts, wsg_position in zip(timestamps[2:], wsg_keyframes[2:]):
        traj_wsg_command.AppendFirstOrderSegment(ts, wsg_position)

    return traj_wsg_command


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

        self.DeclareAbstractOutputPort("stiffness_controller_switch", lambda: AbstractValue.Make(bool()), self.set_stiffness_switch)
        self.DeclareAbstractOutputPort("hybrid_controller_switch", lambda: AbstractValue.Make(bool()), self.set_hybrid_switch)

        self.iiwa_trajectory_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial()))
        self.taskspace_trajectory_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePose()))
        self.wsg_trajectory_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial()))
        self.needs_reinit = self.DeclareAbstractState(
            AbstractValue.Make(True))

        self.timings_index = self.DeclareAbstractState(
            AbstractValue.Make([]))

        self.inits = { x: False for x in ['iiwa_trajectory', 'wsg_trajectory', 'taskspace_trajectory', 'ts'] }
        self.DeclareAbstractOutputPort(
            "current_trajectory_for_stiffness", lambda: AbstractValue.Make(PiecewisePolynomial()), self.set_stiffness_trajectory)
        self.DeclareAbstractOutputPort(
            "current_trajectory_for_hybrid", lambda: AbstractValue.Make(PiecewisePose()), self.set_hybrid_trajectory)
        self.DeclareAbstractOutputPort(
            "current_trajectory_for_wsg", lambda: AbstractValue.Make(PiecewisePolynomial()), self.set_gripper_trajectory)
        self.DeclareAbstractOutputPort(
            "current_stage", lambda: AbstractValue.Make(TurnStage.UNSET), self.calc_current_stage)

        # self.DeclareInitializationUnrestrictedUpdateEvent(self.reinitialize_turn)
        self.DeclarePeriodicUnrestrictedUpdateEvent(.5, .0, self.reinitialize_turn)


    def set_stiffness_trajectory(self, context, output):
        if not self.inits['iiwa_trajectory']:
            self.inits['iiwa_trajectory'] = True
            output.set_value(context.get_abstract_state(int(self.iiwa_trajectory_index)).get_value())


    def set_hybrid_trajectory(self, context, output):
        if not self.inits['taskspace_trajectory']:
            self.inits['taskspace_trajectory'] = True
            output.set_value(context.get_abstract_state(int(self.taskspace_trajectory_index)).get_value())


    def set_gripper_trajectory(self, context, output):
        if not self.inits['wsg_trajectory']:
            print('sets wsg traj')
            self.inits['wsg_trajectory'] = True
            wsg_traj = context.get_abstract_state(int(self.wsg_trajectory_index)).get_value()
            output.set_value(wsg_traj)
            #self.wsg_traj_source.UpdateTrajectory(wsg_traj)


    def set_stiffness_switch(self, context, output):
        t = context.get_time()
        ts = context.get_abstract_state(int(self.timings_index)).get_value()
        if len(ts) == 0:
            return
        if self.mode == 'stiffness':
            output.set_value(ts[0] <= t and t < ts[-1])
        else:
            output.set_value((ts[0] <= t and t < ts[3]) or \
                             (ts[4] <= t and t < ts[-1]))


    def set_hybrid_switch(self, context, output):
        if self.mode == 'stiffness':
            output.set_value(False)
        else:
            ts = context.get_abstract_state(int(self.timings_index)).get_value()
            if 0 != len(ts):
                output.set_value( ts[3] <= t and t < ts[4] )


    def calc_current_stage(self, context, output):
        output.set_value(self.stage)


    def reinitialize_turn(self, context, state):
        print('reinitialize_turn? at t=', context.get_time())
        needs_reinit_ = state.get_mutable_abstract_state(int(self.needs_reinit)).get_value()
        if not needs_reinit_:
            print('doesn\'t need reinit')
            if context.get_time() > 10:
                next_stage = self.stage.next()
                if next_stage:
                    self.stage = next_stage
            return

        X_WG = self.plant.EvalBodyPoseInWorld(self.plant_context, self.plant.GetBodyByName('body'))
        X_WVstart = self.plant.EvalBodyPoseInWorld(self.plant_context, self.plant.GetBodyByName('nut'))

        R_GoalGripper = RotationMatrix(RollPitchYaw(np.radians([180, 0, 90]))).inverse()
        t_GoalGripper = [0., -0.085, -0.02]
        t_GoalGripperPreGrasp = [0., -0.225, -0.02]
        X_GoalGripper = RigidTransform(R_GoalGripper, np.zeros((3,)))
        X_gripper_offset = RigidTransform(RotationMatrix.Identity(), t_GoalGripper)
        X_pre_grasp_offset = RigidTransform(RotationMatrix.Identity(), t_GoalGripperPreGrasp)

        R_WVpreferred = solve_for_grip_direction(X_WVstart)

        X_WGripperAtTurnStart_ = RigidTransform(R_WVpreferred, X_WVstart.translation()) @ X_GoalGripper
        R_WVend_ = RollPitchYaw(np.radians([0, 30, 0])).ToRotationMatrix() @ R_WVpreferred
        X_WGripperAtTurnEnd_ = RigidTransform(R_WVend_, X_WVstart.translation()) @ X_GoalGripper

        X_WGripperAtTurnStart = X_WGripperAtTurnStart_ @ X_gripper_offset
        X_WGripperAtTurnEnd = X_WGripperAtTurnEnd_ @ X_gripper_offset

        X_WGripperPreGraspAtTurnStart = X_WGripperAtTurnStart_ @ X_pre_grasp_offset
        X_WGripperPostGraspAtTurnEnd = X_WGripperAtTurnEnd_ @ X_pre_grasp_offset

        #AddMeshcatTriad(meshcat, 'pregrasp-at-initial-valve', X_PT=X_WGripperPreGraspAtTurnStart)
        #AddMeshcatTriad(meshcat, 'gripper-at-initial-valve', X_PT=X_WGripperAtTurnStart)
        #AddMeshcatTriad(meshcat, 'gripper-at-final-valve', X_PT=X_WGripperAtTurnEnd)
        #AddMeshcatTriad(meshcat, 'postgrasp-at-final-valve', X_PT=X_WGripperPostGraspAtTurnEnd)

        trajectory, ts = optimize_target_trajectory(
            [X_WG, X_WGripperPreGraspAtTurnStart, X_WGripperAtTurnStart, X_WGripperAtTurnEnd, X_WGripperPostGraspAtTurnEnd, X_WG],
            self.plant, self.plant_context)

        if trajectory is None:
            print('opt didnt succeed')
            exit(0)

        print('timings ', ts)
        wsg_trajectory = make_wsg_trajectory(ts)
        cart_trajectory = make_cartesian_trajectory([X_WGripperAtTurnStart, X_WGripperAtTurnEnd], [ts[3], ts[4]])

        state.get_mutable_abstract_state(int(self.iiwa_trajectory_index)).set_value(trajectory)
        state.get_mutable_abstract_state(int(self.wsg_trajectory_index)).set_value(wsg_trajectory)
        state.get_mutable_abstract_state(int(self.taskspace_trajectory_index)).set_value(cart_trajectory)
        state.get_mutable_abstract_state(int(self.timings_index)).set_value(ts)


        self.plant.SetPositions(self.plant_context, self.plant.GetModelInstanceByName("iiwa"), trajectory.value(0))

        state.get_mutable_abstract_state(int(self.needs_reinit)).set_value(False)
        self.inits = { x: False for x in ['iiwa_trajectory', 'wsg_trajectory', 'taskspace_trajectory', 'ts'] }
        next_stage = self.stage.next()
        if next_stage:
            self.stage = next_stage


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
