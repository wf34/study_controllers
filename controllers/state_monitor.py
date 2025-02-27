import os
from typing import Union, List, Optional
import json
import numpy as np
from pydantic import BaseModel
from pydantic.json import pydantic_encoder

from pydrake.systems.framework import EventStatus
from pydrake.multibody.tree import MultibodyForces
from pydrake.multibody.math import SpatialForce

from pydrake.all import (
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    Quaternion,
)

import numpy as np

from catalog import have_matching_intervals
from planning import MultiTurnPlanner, TurnStage

def custom_encoder(**kwargs):
    def base_encoder(obj):
        if isinstance(obj, BaseModel):
            return obj.dict(**kwargs)
        else:
            return pydantic_encoder(obj)
    return base_encoder


class Vector3(BaseModel):
    x: float
    y: float
    z: float

    def instantiate_from_arr(arr: Union[np.array, List]):
        if isinstance(arr, np.ndarray):
            arr = arr.tolist()
        dct = {}
        for i, n in zip(range(3), ['x', 'y', 'z']):
            dct[n] = arr[i]

        return Vector3(**dct)

    def to_np(self):
        return np.array([self.x, self.y, self.z])


class Quat(BaseModel):
    w: float
    x: float
    y: float
    z: float

    def instantiate_from_rot_mat(R: RotationMatrix):
        vec = R.ToQuaternion()
        dct = {}
        for n in ['w', 'x', 'y', 'z']:
            dct[n] = getattr(vec, n)()

        return Quat(**dct)

    def to_drake_q(self) -> Quaternion:
        return Quaternion(self.w, self.x, self.y, self.z)


class DumpTransform(BaseModel):
    translation: Vector3
    rotation: Quat

    def instantiate_from_rt(rt: RigidTransform):
        return DumpTransform(translation=Vector3.instantiate_from_arr(rt.translation()),
                             rotation=Quat.instantiate_from_rot_mat(rt.rotation()))


class Datum(BaseModel):
    time: float
    Xee_desired_W: Optional[DumpTransform] = None
    Xee_observed_W: Optional[DumpTransform] = None
    force_sensed: Optional[Vector3] = None
    valve_pitch_angle: float
    turn: int
    stage: TurnStage


class StateMonitor:
    def __init__(self, path: Optional[str], diagram, outer_stage_obj: List[TurnStage]):
        self._diagram = diagram
        self._station = diagram.GetSubsystemByName('station')
        self._plant = self._station.GetSubsystemByName('plant')
        self.planner = diagram.GetSubsystemByName('MultiTurnPlanner')
        self.force_sensor = diagram.GetSubsystemByName('ForceSensor')

        self._iiwa = self._plant.GetModelInstanceByName('iiwa')
        self._gripper_body_instance = self._plant.GetBodyByName('body').index()
        self._valve_body_instance = self._plant.GetBodyByName('nut').index()

        self.torque_trajectory = None
        self.cart_trajectory = None
        self.stage_obj = outer_stage_obj
        self.path = path
        self.datums = []

        if self.path is not None and os.path.exists(path):
            if os.path.isdir(path):
                raise Exception('wrong input')
            if os.path.isfile(path):
                os.remove(path)


    def write_existing(self):
        if self.path is not None:
            with open(self.path, 'w') as the_file:
                the_file.write(json.dumps(self.datums, default=custom_encoder(by_alias=True)))
            self.path = None


    def get_turn(self, root_context) -> int:
        planner_context = self.planner.GetMyContextFromRoot(root_context)
        return self.planner.GetOutputPort('turn').Eval(planner_context)


    def get_stage(self, root_context) -> TurnStage:
        planner_context = self.planner.GetMyContextFromRoot(root_context)
        curr_stage = self.planner.GetOutputPort('current_stage').Eval(planner_context)
        self.stage_obj[0] = curr_stage
        return curr_stage


    def get_is_stiffness(self, root_context) -> bool:
        planner_context = self.planner.GetMyContextFromRoot(root_context)
        return self.planner.GetOutputPort('stiffness_controller_switch').Eval(planner_context)


    def keep_trajectories_up_to_date(self, root_context):
        planner_context = self.planner.GetMyContextFromRoot(root_context)
        def upd_trajectory(traj_name, topic):
            new_traj = self.planner.GetOutputPort(topic).Eval(planner_context)
            if 0 == new_traj.get_number_of_segments():
                return
            traj_ref = getattr(self, traj_name)
            if traj_ref is None or not have_matching_intervals(traj_ref, new_traj):
                setattr(self, traj_name, new_traj)

        for traj_name, topic in zip(['torque_trajectory', 'cart_trajectory'],
                                   ['current_trajectory_for_stiffness', 'current_trajectory_for_hybrid']):
            upd_trajectory(traj_name, topic)


    def get_goal_pose(self, root_context) -> RigidTransform:
        time = root_context.get_time()
        is_stiffness = self.get_is_stiffness(root_context)
        if is_stiffness:
            q_desired = self.torque_trajectory.value(time).T.ravel()
            plant_context = self._plant.GetMyContextFromRoot(root_context)
            saved_internal_coords = self._plant.GetPositions(plant_context, self._iiwa)
            self._plant.SetPositions(plant_context, self._iiwa, q_desired)
            X_WG = self._plant.EvalBodyPoseInWorld(plant_context, self._plant.GetBodyByName("body"))
            self._plant.SetPositions(plant_context, self._iiwa, saved_internal_coords)
            return X_WG

        else:
            return self.cart_trajectory.GetPose(time)


    def callback(self, root_context):
        self.keep_trajectories_up_to_date(root_context)
        stage = self.get_stage(root_context)
        if stage in (TurnStage.APPROACH, TurnStage.SCREW, TurnStage.RETRACT):
            station_context = self._station.GetMyContextFromRoot(root_context)
            ts = root_context.get_time()
            turn = self.get_turn(root_context)
            poses = self._station.GetOutputPort('body_poses').Eval(station_context)
            X_WV = poses[self._valve_body_instance]
            valve_pitch_angle = RollPitchYaw(X_WV.rotation()).pitch_angle()

            pose_desired = self.get_goal_pose(root_context)
            pose_measured = poses[self._gripper_body_instance]

            sensor_context = self.force_sensor.GetMyContextFromRoot(root_context)
            sensed_reaction_force = self.force_sensor.GetOutputPort('sensed_force_out').Eval(sensor_context)
            if stage == TurnStage.SCREW:
                self.datums.append(Datum(time=ts,
                                         Xee_desired_W=DumpTransform.instantiate_from_rt(pose_desired),
                                         Xee_observed_W= DumpTransform.instantiate_from_rt(pose_measured),
                                         force_sensed=Vector3.instantiate_from_arr(sensed_reaction_force),
                                         valve_pitch_angle=valve_pitch_angle,
                                         turn=turn,
                                         stage=stage))
            else:
                self.datums.append(Datum(time=ts,
                                         Xee_desired_W=DumpTransform.instantiate_from_rt(pose_desired),
                                         Xee_observed_W= DumpTransform.instantiate_from_rt(pose_measured),
                                         valve_pitch_angle=valve_pitch_angle,
                                         turn=turn,
                                         stage=stage))

            return EventStatus.DidNothing()

        elif TurnStage.FINISH == stage:
            print('/ FINISH /')
            self.write_existing()
            return EventStatus.ReachedTermination(self.planner, 'finished the task')

        else:
            return EventStatus.DidNothing()
