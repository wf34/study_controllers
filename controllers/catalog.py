import numpy as np
import numpy.linalg
from pydrake.all import (
    AbstractValue,
    BasicVector,
    PiecewisePolynomial,
    LeafSystem,
    RollPitchYaw,
    JacobianWrtVariable,
    JointStiffnessController,
    RigidTransform,
    MultibodyForces,
    SpatialForce,
)


def is_within_intervals(target_time: float, intervals: np.array) -> bool:
    for i in range(intervals.shape[0]):
        start, end = intervals[i]
        if intervals[i, 0] <= target_time and target_time < intervals[i, 1]:
            return True
    return False


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


def get_rot2d_from_transform(X: RigidTransform) -> np.array:
    pitch = RollPitchYaw(X.rotation()).pitch_angle()
    sin_pitch = np.sin(pitch)
    cos_pitch = np.cos(pitch)
    return np.array([cos_pitch, -sin_pitch, sin_pitch, cos_pitch]).reshape((2, 2))


class ForceSensor(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = self._plant.GetModelInstanceByName("iiwa")
        self.set_name('ForceSensor')

        self.DeclareVectorInputPort("iiwa_inner_forces_in", BasicVector(3))
        self.DeclareVectorInputPort("iiwa_state_measured", BasicVector(6))

        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self._shelf_body_instance = plant.GetBodyByName("shelf_body").index()

        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()

        self._joint_indices = [
            plant.GetJointByName(j).position_start()
            for j in ("iiwa_joint_2", "iiwa_joint_4", "iiwa_joint_6")
        ]

        #body_instance = plant.GetBodyByName("body")
        #self._ee = body_instance.body_frame()
        #self._ee_body_index = int(body_instance.index())

        #self._sensor_joint = self._plant.GetJointByName('iiwa_link_7_welds_to_body')
        #self._sensor_joint_index = self._sensor_joint.index()

        self.DeclareVectorOutputPort("sensed_force_out", 3, self.CalcForceOutput)
        self.X_Wshelf = None


    def CalcForceOutput(self, context, output):
        torques = self.GetInputPort("iiwa_inner_forces_in").Eval(context)

        if self.X_Wshelf is None:
            self.X_Wshelf = self.GetInputPort("body_poses").Eval(context)[self._shelf_body_instance]

        iiwa_state = self.GetInputPort("iiwa_state_measured").Eval(context)
        q_now = iiwa_state[:3]
        q_dot_now = iiwa_state[3:]

        self._plant.SetPositions(self._plant_context, self._iiwa, q_now)
        self._plant.SetVelocities(self._plant_context, self._iiwa, q_dot_now)
        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kQDot,
            self._G,
            [0, 0, 0],
            self._W,
            self._W,
        )
        J_G = J_G[np.ix_([1, 3, 5], self._joint_indices)]
        J_questionmark = np.linalg.pinv(J_G.T)

        #Jq_p_AoBi_E = self._plant.CalcJacobianPositionVector(
        #    context=self._plant_context,
        #    frame_B=self._G,
        #    p_BoBi_B=np.zeros(3),
        #    frame_A=self._W,
        #    frame_E=self._W
        #)
        #Jq_p_AoBi_E = Jq_p_AoBi_E[np.ix_([0, 2], self._joint_indices)]
        #J_questionmark_questionmark = np.linalg.pinv(Jq_p_AoBi_E.T)

        #print(J_G, '\n', J_questionmark, '\n---\nNew Jac:\n')
        #print(Jq_p_AoBi_E, '\n', J_questionmark_questionmark, '\n===\n')
        f = J_questionmark @ torques

        # Rows correspond to (pitch, x, z).
        R_Wshelf = get_rot2d_from_transform(self.X_Wshelf)
        f_shelf = R_Wshelf.T @ f[1:]

        tf = np.zeros((3),)
        tf[0] = f[0]
        tf[1:] = f_shelf
        output.SetFromVector(tf)


class TrajFollowingJointStiffnessController(LeafSystem):
    def __init__(self, plant, kp, kd):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()

        self._iiwa = plant.GetModelInstanceByName("iiwa")

        self.kp_vec = np.zeros(3,) + kp
        self.kd_vec = np.zeros(3,) + kd

        #self.ctrl = JointStiffnessController()

        self.DeclareAbstractInputPort(
            "trajectory", AbstractValue.Make(PiecewisePolynomial()))
        self.DeclareAbstractInputPort(
            "switched_on_intervals", AbstractValue.Make(np.array([])))

        self.trajectory = None
        self.qdot_trajectory = None
        self.switched_on_intervals = None

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

        if self.switched_on_intervals is None:
            self.switched_on_intervals = self.GetInputPort('switched_on_intervals').Eval(context)
            if 2 == len(self.switched_on_intervals.shape) and \
               2 != self.switched_on_intervals.shape[1]:
                raise Exception(f'each row must be an interval, but is mishaped: {self.switched_on_intervals.shape}')

        current_time = context.get_time()
        if not is_within_intervals(current_time, self.switched_on_intervals):
            output.SetFromVector(np.zeros((3),))
            return

        q_desired = self.trajectory.value(current_time).T.ravel()
        qdot_desired = self.qdot_trajectory.value(current_time).T.ravel()

        state_now = self.GetInputPort("iiwa_state_measured").Eval(context)
        q_now = state_now[:3]
        qdot_now = state_now[3:]

        self._plant.SetPositions(self._plant_context, self._iiwa, q_now)
        self._plant.SetVelocities(self._plant_context, self._iiwa, qdot_now)
        bias = self._plant.CalcBiasTerm(self._plant_context) # coreolis + gyscopic effects

        forces = MultibodyForces(plant=self._plant)
        self._plant.CalcForceElementsContribution(self._plant_context, forces) # gravity and forces aplied to model
        # 0 = dynamics(tau, state) ; a = (f, state) a = F / m; F = ma
        tau = self._plant.CalcInverseDynamics(self._plant_context, np.zeros(5,), forces)
        #print('CalcTorqueOutput', self._plant.num_positions(), bias)

        tau -= bias
        tau = tau[:3]

        e = q_desired - q_now
        e_dot = qdot_desired - qdot_now

        tau += e * self.kp_vec
        tau += e_dot * self.kd_vec
        output.SetFromVector(tau)


class HybridCartesianController(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self.X_Wshelf = None
        self.switched_on_intervals = None

        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()

        self._shelf_body_instance = plant.GetBodyByName("shelf_body").index()
        self._joint_indices = [
            plant.GetJointByName(j).position_start()
            for j in ("iiwa_joint_2", "iiwa_joint_4", "iiwa_joint_6")
        ]

        self.DeclareAbstractInputPort(
            "switched_on_intervals", AbstractValue.Make(np.array([])))

        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self.DeclareVectorInputPort("ee_force_measured", 3)
        self.DeclareVectorInputPort("iiwa_state_measured", 6)
        self.DeclareVectorOutputPort("iiwa_torque_cmd", 3, self.CalcTorqueOutput)


    def CalcTorqueOutput(self, context, output):
        if self.X_Wshelf is None:
            self.X_Wshelf = self.GetInputPort("body_poses").Eval(context)[self._shelf_body_instance]

        if self.switched_on_intervals is None:
            self.switched_on_intervals = self.GetInputPort('switched_on_intervals').Eval(context)
            if 2 == len(self.switched_on_intervals.shape) and \
               2 != self.switched_on_intervals.shape[1]:
                raise Exception(f'each row must be an interval, but is mishaped: {self.switched_on_intervals.shape}')

        current_time = context.get_time()
        if not is_within_intervals(current_time, self.switched_on_intervals):
            output.SetFromVector(np.zeros((3),))
            return

        f_measured = self.GetInputPort('ee_force_measured').Eval(context)
        state_now = self.GetInputPort("iiwa_state_measured").Eval(context)

        q_now = state_now[:3]
        qdot_now = state_now[3:]

        self._plant.SetPositions(self._plant_context, self._iiwa, q_now)
        self._plant.SetVelocities(self._plant_context, self._iiwa, qdot_now)

        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kQDot,
            self._G,
            [0, 0, 0],
            self._W,
            self._W,
        )
        J_G = J_G[np.ix_([1, 3, 5], self._joint_indices)]

        # force control
        R_Wshelf = get_rot2d_from_transform(self.X_Wshelf)
        f_goal = np.array([0., 10., 0.])
        e_forces_Shelf = f_goal - f_measured
        e_forces_W_ = R_Wshelf @ e_forces_Shelf[1:]
        e_forces_W = np.zeros((3,))
        e_forces_W[0] = e_forces_Shelf[0]
        e_forces_W[1:] = e_forces_W_[1:]
        e_tau = J_G.T @ e_forces_W
        tau = e_tau / 100

        output.SetFromVector(tau)
