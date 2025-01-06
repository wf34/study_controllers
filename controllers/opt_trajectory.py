from typing import List, Optional, Tuple

import numpy as np

from pydrake.all import (
    RigidTransform,
    RotationMatrix,
    InverseKinematics,
    PiecewisePolynomial,
    PiecewisePose,
    Solve,
    RollPitchYaw,
)


def get_current_positions(plant, plant_context):
    q_ = np.zeros(plant.num_positions(),)
    q = plant.GetPositions(
        plant_context,
        plant.GetModelInstanceByName("iiwa"),
    )
    q_[:3] = q
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
            for j in ("iiwa_joint_2", "iiwa_joint_4", "iiwa_joint_6")
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

        #offset = np.array([1e-2, 1e-2, 1e-2])
        offset = np.array([0.005, 0.05, 0.005])

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
        q_trajectory = PiecewisePolynomial.FirstOrderHold(valid_timestamps, q_keyframes[:, :3].T)
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

def make_dummy_wsg_trajectory():
    ts = [0, 1]
    wsg_keyframes = np.array([0.107, 0.107]).reshape(1, 2)
    return PiecewisePolynomial.FirstOrderHold(ts, wsg_keyframes)
