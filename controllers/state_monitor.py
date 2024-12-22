import os
from typing import Union, List, Optional

import numpy as np
from pydantic import BaseModel

from pydrake.systems.framework import EventStatus
from pydrake.multibody.tree import MultibodyForces
from pydrake.multibody.math import SpatialForce

from pydrake.all import (
    RigidTransform,
)

import numpy as np

from catalog import get_transl2d_from_transform, get_rot2d_from_transform

class Vector2(BaseModel):
    x: float
    z: float

    def instantiate_from_arr(arr: Union[np.array, List]):
        if isinstance(arr, np.ndarray):
            arr = arr.tolist()
        dct = {}
        for i, n in zip(range(2), ['x', 'z']):
            dct[n] = arr[i]

        return Vector2(**dct)


class Datum(BaseModel):
    time: float
    pe_s: Vector2
    f_s: Vector2


class StateMonitor:
    def __init__(self, path, plant, diagram, ttrajectory, ctrajectory, meshcat):
        self._plant = plant
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        gripper = plant.GetBodyByName("body")
        self._gripper_body_instance = gripper.index()
        self._shelf_body_instance = plant.GetBodyByName("shelf_body").index()
        self._diagram = diagram

        self._station = diagram.GetSubsystemByName("station")

        self.torque_trajectory = ttrajectory
        self.cart_trajectory = ctrajectory
        self._meshcat = meshcat
        self.X_Wshelf = None
        
        if os.path.exists(path):
            assert not os.path.isdir(path)
            if os.path.isfile(path):
                os.remove(path)

        self._file = open(path, 'a')
        self._file.write('[')

    def get_mode(self) -> str:
        if self.torque_trajectory is not None and self.cart_trajectory is None:
            return 'stiffness'
        elif self.torque_trajectory is None and self.cart_trajectory is not None:
            return 'hybrid'
        else:
            raise Exception('unreachable')

    def get_goal_pose(self, root_context) -> RigidTransform:
        time = root_context.get_time()
        if 'stiffness' == self.get_mode():
            q_desired = self.torque_trajectory.value(time).T.ravel()

            plant_context = self._plant.GetMyContextFromRoot(root_context)
            saved_internal_coords = self._plant.GetPositions(plant_context, self._iiwa)
            self._plant.SetPositions(plant_context, self._iiwa, q_desired)
            X_WG = self._plant.EvalBodyPoseInWorld(plant_context, self._plant.GetBodyByName("body"))
            self._plant.SetPositions(plant_context, self._iiwa, saved_internal_coords)
            return X_WG

        elif 'hybrid' == self.get_mode():
            return self.cart_trajectory.GetPose(time)


    def callback(self, root_context):
        current_time = root_context.get_time()
        if 2. <= current_time and current_time < 12.:
            station_context = self._station.GetMyContextFromRoot(root_context)
            poses = self._station.GetOutputPort('body_poses').Eval(station_context)
            if self.X_Wshelf is None:
                self.X_Wshelf = poses[self._shelf_body_instance]

            R_shelfW = get_rot2d_from_transform( self.X_Wshelf )  # because 3D->2D, XZ->XY

            pose_desired = self.get_goal_pose(root_context)
            plant_context = self._plant.GetMyContextFromRoot(root_context)

            pose_measured = poses[self._gripper_body_instance]

            pd_WG = get_transl2d_from_transform(pose_desired)
            pm_WG = get_transl2d_from_transform(pose_measured)

            pd_WS = R_shelfW @ pd_WG
            pm_WS = R_shelfW @ pm_WG

            pe_S = pd_WS - pm_WS

            force_sensor_system = self._diagram.GetSubsystemByName('ForceSensor')
            sensor_context = force_sensor_system.GetMyContextFromRoot(root_context)
            sensed_reaction_force = force_sensor_system.GetOutputPort('sensed_force_out').Eval(sensor_context)[1:]

            datum = Datum(time=root_context.get_time(),
                          pe_s=Vector2.instantiate_from_arr(pe_S),
                          f_s=Vector2.instantiate_from_arr(sensed_reaction_force))

            if self._file:
                self._file.write(datum.model_dump_json(exclude_none=True))
                self._file.write(',')
                self._file.flush()

        return EventStatus.DidNothing()
