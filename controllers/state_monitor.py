import os
from typing import Union, List

import numpy as np
from pydantic import BaseModel

from pydrake.systems.framework import EventStatus
from pydrake.multibody.tree import MultibodyForces
from pydrake.multibody.math import SpatialForce

import numpy as np

class Vector3(BaseModel):
    x: float
    y: float
    z: float

    def instantiate_from_arr(arr: Union[np.array, List]):
        if isinstance(arr, np.ndarray):
            arr = arr.tolist()
        if 3 != len(arr):
            raise Exception('size err')
        dct = {}
        for i, n in zip(range(3), ['x', 'y', 'z']):
            dct[n] = arr[i]

        return Vector3(**dct)


class Datum(BaseModel):
    time: float
    measured_iiwa_torque_coords: Vector3


class StateMonitor:
    def __init__(self, path, plant):
        self._plant = plant
        
        if os.path.exists(path):
            assert not os.path.isdir(path)
            if os.path.isfile(path):
                os.remove(path)

        self._file = open(path, 'a')
        self._counter = 0

    def callback(self, root_context):
        self._counter += 1
        if 0 != self._counter % 100:
            return

        #end_effector = self._plant.GetBodyByName("body")
        plant_context = self._plant.GetMyContextFromRoot(root_context)
        #X_WB = nut.EvalPoseInWorld(nut_context)
        #contact_results = self._plant.get_contact_results_output_port().Eval(nut_context)
        #multibody_forces = MultibodyForces(plant=self._plant)
        #forces = MultibodyForces(plant=self._plant)
        #self._plant.CalcForceElementsContribution(context=nut_context, forces=forces)
        #tau = self._plant.CalcGeneralizedForces(context=nut_context, forces=forces)
        #empty = np.array([0.]*6)
        #empty[-1] = np.max(tau) * 1.e+3

        iiwa_torque_coords = self._plant.GetPositions(
            plant_context,
            self._plant.GetModelInstanceByName("iiwa"),
        )
        datum = Datum(time=root_context.get_time(),
                      measured_iiwa_torque_coords=Vector3.instantiate_from_arr(iiwa_torque_coords))
        if self._file:
            self._file.write(datum.model_dump_json(exclude_none=True))
            self._file.flush()
        #print(self._plant.GetPositions(plant_context))
        #print(self._plant.GetPositions(plant_context, self._plant.GetModelInstanceByName("iiwa")))

        return EventStatus.DidNothing()
