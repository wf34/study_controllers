import numpy as np

from pydrake.all import (
    AbstractValue,
    PiecewisePolynomial,
    LeafSystem,
    RollPitchYaw,
    JacobianWrtVariable,
    JointStiffnessController,
    MultibodyForces,
)

class TorqueController(LeafSystem):
    """Wrapper System for Commanding Pure Torques to planar iiwa.
    @param plant MultibodyPlant of the simulated plant.
    @param ctrl_fun function object to implement torque control law.
    @param vx Velocity towards the linear direction.
    """
    def __init__(self, plant, ctrl_fun, vx):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()
        self._ctrl_fun = ctrl_fun
        self._vx = vx
        self._joint_indices = [
            plant.GetJointByName(j).position_start()
            for j in ("iiwa_joint_2", "iiwa_joint_4", "iiwa_joint_6")
        ]

        self.DeclareVectorInputPort("iiwa_position_measured", 3)
        self.DeclareVectorInputPort("iiwa_velocity_measured", 3)

        # If we want, we can add this in to do closed-loop force control on z.
        # self.DeclareVectorInputPort("iiwa_torque_external", 3)

        self.DeclareVectorOutputPort(
            "iiwa_position_command", 3, self.CalcPositionOutput
        )
        self.DeclareVectorOutputPort("iiwa_torque_cmd", 3, self.CalcTorqueOutput)
        # Compute foward kinematics so we can log the wsg position for grading.
        self.DeclareVectorOutputPort("wsg_position", 3, self.CalcWsgPositionOutput)

    def CalcPositionOutput(self, context, output):
        """Set q_d = q_now. This ensures the iiwa goes into pure torque mode in sim by setting the position control torques in InverseDynamicsController to zero.
        NOTE(terry-suh): Do not use this method on hardware or deploy this notebook on hardware.
        We can only simulate pure torque control mode for iiwa on sim.
        """
        q_now = self.get_input_port(0).Eval(context)
        output.SetFromVector(q_now)

    def CalcTorqueOutput(self, context, output):
        # Hard-coded position and force profiles. Can be connected from Trajectory class.
        if context.get_time() < 2.0:
            px_des = 0.65
        else:
            px_des = 0.65 + self._vx * (context.get_time() - 2.0)

        fz_des = 10

        # Read inputs
        q_now = self.get_input_port(0).Eval(context)
        v_now = self.get_input_port(1).Eval(context)
        # tau_now = self.get_input_port(2).Eval(context)

        self._plant.SetPositions(self._plant_context, self._iiwa, q_now)

        # 1. Convert joint space quantities to Cartesian quantities.
        X_now = self._plant.CalcRelativeTransform(self._plant_context, self._W, self._G)

        rpy_now = RollPitchYaw(X_now.rotation()).vector()
        p_xyz_now = X_now.translation()

        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kQDot,
            self._G,
            [0, 0, 0],
            self._W,
            self._W,
        )

        # Only select relevant terms. We end up with J_G of shape (3,3).
        # Rows correspond to (pitch, x, z).
        # Columns correspond to (q0, q1, q2).
        J_G = J_G[np.ix_([1, 3, 5], self._joint_indices)]
        v_pxz_now = J_G.dot(v_now)

        p_pxz_now = np.array([rpy_now[1], p_xyz_now[0], p_xyz_now[2]])

        # 2. Apply ctrl_fun
        F_pxz = self._ctrl_fun(p_pxz_now, v_pxz_now, px_des, fz_des)

        # 3. Convert back to joint coordinates
        tau_cmd = J_G.T.dot(F_pxz)
        output.SetFromVector(tau_cmd)

    def CalcWsgPositionOutput(self, context, output):
        """
        Compute Forward kinematics. Needed to log the position trajectory for grading.  TODO(russt): Could use MultibodyPlant's body_poses output port for this.
        """
        q_now = self.get_input_port(0).Eval(context)
        self._plant.SetPositions(self._plant_context, self._iiwa, q_now)
        X_now = self._plant.CalcRelativeTransform(self._plant_context, self._W, self._G)

        rpy_now = RollPitchYaw(X_now.rotation()).vector()
        p_xyz_now = X_now.translation()
        p_pxz_now = np.array([rpy_now[1], p_xyz_now[0], p_xyz_now[2]])

        output.SetFromVector(p_pxz_now)


class TrajFollowingJointStiffnessController(LeafSystem):
    def __init__(self, plant, kp, kd):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()

        self.kp_vec = np.zeros(3,) + kp
        self.kd_vec = np.zeros(3,) + kd

        #self.ctrl = JointStiffnessController()

        self.DeclareAbstractInputPort(
            "trajectory", AbstractValue.Make(PiecewisePolynomial()))
        self.trajectory = None
        self.qdot_trajectory = None

        self.DeclareVectorInputPort("iiwa_state_measured", 6)

        #self.DeclareVectorOutputPort(
        #    "iiwa_position_command", 3, self.CalcPositionOutput
        #)
        self.DeclareVectorOutputPort("iiwa_torque_cmd", 3, self.CalcTorqueOutput)


    def CalcPositionOutput(self, context, output):
        """Set q_d = q_now. This ensures the iiwa goes into pure torque mode in sim by setting the position control torques in InverseDynamicsController to zero.
        NOTE(terry-suh): Do not use this method on hardware or deploy this notebook on hardware.
        We can only simulate pure torque control mode for iiwa on sim.
        """
        q_now = self.GetInputPort("iiwa_position_measured").Eval(context)
        output.SetFromVector(q_now)


    def CalcTorqueOutput(self, context, output):
        if self.trajectory is None:
            self.trajectory = self.GetInputPort('trajectory').Eval(context)
            self.qdot_trajectory = self.trajectory.MakeDerivative()

        target_time = context.get_time()

        q_desired = self.trajectory.value(target_time).T.ravel()
        qdot_desired = self.qdot_trajectory.value(target_time).T.ravel()

        state_now = self.GetInputPort("iiwa_state_measured").Eval(context)
        q_now = state_now[:3]
        qdot_now = state_now[3:]

        self._plant.SetPositions(self._plant_context, self._iiwa, q_now)
        self._plant.SetVelocities(self._plant_context, self._iiwa, qdot_now)
        bias = self._plant.CalcBiasTerm(self._plant_context)

        forces = MultibodyForces(plant=self._plant)
        self._plant.CalcForceElementsContribution(self._plant_context, forces)
        tau = self._plant.CalcInverseDynamics(self._plant_context, np.zeros(5,), forces)
        #print('CalcTorqueOutput', self._plant.num_positions(), bias)

        tau -= bias
        tau = tau[:3]
        e = q_desired - q_now
        e_dot = qdot_desired - qdot_now

        tau += e * self.kp_vec
        tau += e_dot * self.kd_vec

        output.SetFromVector(tau)
