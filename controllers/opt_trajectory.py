from typing import List

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


def AddOrientationConstraint(plant, ik, target_frame, R_TG, bounds):
    """Add orientation constraint to the ik problem. Implements an inequality 
    constraint where the axis-angle difference between f_R(q) and R_WG must be
    within bounds. Can be translated to:
    ik.prog().AddBoundingBoxConstraint(angle_diff(f_R(q), R_WG), -bounds, bounds)
    """
    ik.AddOrientationConstraint(
        frameAbar=plant.GetFrameByName("body"), R_AbarA=R_TG, 
        frameBbar=target_frame, R_BbarB=RotationMatrix(),
        theta_bound=bounds
    )


def AddPositionConstraint(plant, ik, target_frame, p_BQ, p_NG_lower, p_NG_upper):
    """Add position constraint to the ik problem. Implements an inequality
    constraint where f_p(q) must lie between p_WG_lower and p_WG_upper. Can be
    translated to 
    ik.prog().AddBoundingBoxConstraint(f_p(q), p_WG_lower, p_WG_upper)
    """
    ik.AddPositionConstraint(
        frameA=plant.GetFrameByName("body"),
        frameB=target_frame,
        p_BQ=p_BQ,
        p_AQ_lower=p_NG_lower, p_AQ_upper=p_NG_upper)


def optimize_target_trajectory(keyframes: List[RigidTransform], plant, plant_context) -> None:
    '''
    desc

    Args:
    Returns:
    '''
    iiwa_model_instance = plant.GetModelInstanceByName('iiwa')
    q_nominal = get_current_positions(plant, plant_context)

    num_q = plant.num_positions()
    num_iiwa_only = plant.num_positions(iiwa_model_instance)
    x = list(map(int, plant.GetJointIndices(model_instance=iiwa_model_instance)))
    print(f'GetJointIndices {x} of total {num_q}; of total iiwa-related: {num_iiwa_only}')
    print('dofs:', plant.num_actuated_dofs())
    print('num_model_instances:', plant.num_model_instances())
    print('num_joints:', plant.num_joints())
    print(num_q)

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
        offset_upper = np.array([6e-2, 6e-2, 6e-2])
        offset_lower = -offset_upper

        ik.AddPositionConstraint(
            frameA=plant.GetFrameByName("body"),
            frameB=plant.world_frame(),
            p_BQ=keyframe.translation(),
            p_AQ_lower=offset_lower,
            p_AQ_upper=offset_upper)

        if kid not in (0, len(keyframes)-1):
            ik.AddOrientationConstraint(
                    frameAbar=plant.GetFrameByName("body"),
                    R_AbarA=RotationMatrix(RollPitchYaw(0., np.pi / 4, 0.)),
                    frameBbar=plant.world_frame(),
                    R_BbarB=RotationMatrix(),
                    theta_bound=np.pi / 2
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
        valid_timestamps = np.arange(0, len(keyframes), step=1.) * 2.
        print(valid_timestamps, valid_timestamps.shape)
        print(q_keyframes.shape)
        print(q_keyframes[:, :3])
        q_trajectory = PiecewisePolynomial.CubicShapePreserving(valid_timestamps, q_keyframes[:, :3].T)
        return q_trajectory
