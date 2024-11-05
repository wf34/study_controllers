from typing import List, Optional

import numpy as np

from pydrake.all import (
    RigidTransform,
    RotationMatrix,
    InverseKinematics,
    PiecewisePolynomial,
    Solve,
    RollPitchYaw,
    PiecewisePolynomial,
)


def get_current_positions(plant, plant_context):
    q_ = np.zeros(plant.num_positions(),)
    q = plant.GetPositions(
        plant_context,
        plant.GetModelInstanceByName("iiwa"),
    )
    q_[:3] = q
    return q_


def optimize_target_trajectory(keyframes: List[RigidTransform], plant, plant_context) -> Optional[PiecewisePolynomial]:
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
    print(num_q)

    joint_indices = [
            plant.GetJointByName(j).position_start()
            for j in ("iiwa_joint_2", "iiwa_joint_4", "iiwa_joint_6")
        ]
    print('joint_indices:', joint_indices)

    q_keyframes = []
    for kid, keyframe in enumerate(keyframes):
        ik = InverseKinematics(plant)
        q_variables = ik.q()
        print(q_variables, q_nominal)
        prog = ik.prog()

        if 0 == kid:
            prog.SetInitialGuess(q_variables, q_nominal)
        else:
            prog.SetInitialGuess(q_variables, q_keyframes[-1])
            
        prog.AddCost(np.square(np.dot(q_variables, q_nominal)))

        offset = np.array([1e-2, 1e-2, 1e-2])

        if kid == 2:
            offset = np.array([0.025, 0.02, 0.025])

        offset_upper = keyframe.translation() + offset
        offset_lower = keyframe.translation() - offset

        pos = np.zeros(3,) if kid in (0, len(keyframes) - 1) else [0., 0.11, 0.]

        ik.AddPositionConstraint(
            frameA=plant.world_frame(),
            frameB=plant.GetFrameByName("body"),
            p_BQ=pos,
            p_AQ_lower=offset_lower,
            p_AQ_upper=offset_upper)

        if kid not in (0, len(keyframes) - 1):
            ik.AddOrientationConstraint(
                    frameAbar=plant.GetFrameByName("body"),
                    R_AbarA=RotationMatrix(),
                    frameBbar=plant.world_frame(),
                    R_BbarB=RotationMatrix(RollPitchYaw(-np.pi /2., -np.pi / 4, 0.)),
                    theta_bound=np.radians(5.)
                )

        result = Solve(prog)
        if not result.is_success():
            print(f'no sol for i={kid}')
            print(result.GetInfeasibleConstraintNames(prog))
            break
        else:
            q_keyframes.append(result.GetSolution(q_variables))

    if len(q_keyframes) == len(keyframes):
        q_keyframes = np.array(q_keyframes)
        valid_timestamps = np.arange(0, len(keyframes), step=1.) * 5.
        q_trajectory = PiecewisePolynomial.FirstOrderHold(valid_timestamps, q_keyframes[:, :3].T)
        return q_trajectory
