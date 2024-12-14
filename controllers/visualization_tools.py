import typing
import numpy as np

from pydrake.all import (
    Cylinder,
    Sphere,
    Rgba,

    RotationMatrix,
    RigidTransform,
    Meshcat
)

def AddMeshcatTriad(meshcat, path, length=0.25, radius=0.01, opacity=1.0, X_PT=RigidTransform()):
    meshcat.SetTransform(path, X_PT)
    # x-axis
    X_TG = RigidTransform(
        RotationMatrix.MakeYRotation(np.pi / 2), [length / 2.0, 0, 0]
    )
    meshcat.SetTransform(path + '/x-axis', X_TG)
    meshcat.SetObject(
        path + '/x-axis', Cylinder(radius, length), Rgba(1, 0, 0, opacity)
    )

    # y-axis
    X_TG = RigidTransform(
        RotationMatrix.MakeXRotation(np.pi / 2), [0, length / 2.0, 0]
    )
    meshcat.SetTransform(path + '/y-axis', X_TG)
    meshcat.SetObject(
        path + '/y-axis', Cylinder(radius, length), Rgba(0, 1, 0, opacity)
    )

    # z-axis
    X_TG = RigidTransform([0, 0, length / 2.0])
    meshcat.SetTransform(path + '/z-axis', X_TG)
    meshcat.SetObject(
        path + '/z-axis', Cylinder(radius, length), Rgba(0, 0, 1, opacity)
    )
