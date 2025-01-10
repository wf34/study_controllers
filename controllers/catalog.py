import numpy as np
import numpy.linalg
from pydrake.all import (
    AbstractValue,
    BasicVector,
    PiecewisePolynomial,
    PiecewisePose,
    LeafSystem,
    RollPitchYaw,
    JacobianWrtVariable,
    JointStiffnessController,
    RigidTransform,
    MultibodyForces,
    SpatialForce,
    SpatialVelocity
)


def is_within_intervals(target_time: float, intervals: np.array) -> bool:
    for i in range(intervals.shape[0]):
        start, end = intervals[i]
        if intervals[i, 0] <= target_time and target_time < intervals[i, 1]:
            return True
    return False


def get_rot2d_from_transform(X_Wo: RigidTransform) -> np.array:
    pitch = RollPitchYaw(X_Wo.rotation()).pitch_angle()
    sin_pitch = np.sin(pitch)
    cos_pitch = np.cos(pitch)
    return np.array([cos_pitch, -sin_pitch, sin_pitch, cos_pitch]).reshape((2, 2))


def get_transl2d_from_transform(X_Wo: RigidTransform) -> np.array:
    t = X_Wo.translation()
    return np.array([t[0], t[2]])


def get_pitch_from_transform(X_Wo: RigidTransform) -> np.array:
    return RollPitchYaw(X_Wo.rotation()).pitch_angle()


def get_rotvel_from_spatial_velocity(v_o_W: np.array) -> float:
    if v_o_W.size == 6:
        return v_o_W.ravel()[1]
    else:
        raise Exception('unexpected vel notation')


def get_vel2d_from_spatial_velocity(v_o_W: np.array) -> np.array:
    if len(v_o_W.shape) == 2:
        x = v_o_W[3:, :].ravel()
    elif len(v_o_W.shape) == 1:
        x = v_o_W[3:]
    else:
        raise Exception('')

    assert x.size == 3
    return np.array([x[0], x[2]])


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
        self._shelf_body_instance = plant.GetBodyByName("nut").index()

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


    def CalcForceOutput(self, context, output):
        torques = self.GetInputPort("iiwa_inner_forces_in").Eval(context)

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
                         # ry, tx, tz
        J_G = J_G[np.ix_([1, 3, 5], self._joint_indices)]
        f_shelf_W = J_G @ torques
        # Rows correspond to (pitch, x, z).
        output.SetFromVector(f_shelf_W)


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
        tau = self._plant.CalcInverseDynamics(self._plant_context, np.zeros(6,), forces)
        #print('CalcTorqueOutput', self._plant.num_positions(), bias)

        tau -= bias
        tau = tau[:3]

        e = q_desired - q_now
        e_dot = qdot_desired - qdot_now

        tau += e * self.kp_vec
        tau += e_dot * self.kd_vec
        output.SetFromVector(tau)


class HybridCartesianController(LeafSystem):
    def __init__(self, plant, kp_tang: float, kd_tang: float, kf_norm: float, kfd_norm: float):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self.X_Wshelf = None
        self.switched_on_intervals = None

        self.cart_trajectory = None
        self.vel_trajectory = None

        self.kp_tang_vec = np.zeros(3,) + kp_tang
        self.kd_tang_vec = np.zeros(3,) + kd_tang
        self.kf_norm_vec = np.zeros(3,) + kf_norm
        self.kfd_norm_vec = np.zeros(3,) + kfd_norm

        self._iiwa = plant.GetModelInstanceByName("iiwa")
        end_effector = plant.GetBodyByName("body")
        self._G = end_effector.body_frame()
        self._W = plant.world_frame()

        self._shelf_body_instance = plant.GetBodyByName("nut").index()
        self._ee_body_instance = end_effector.index()

        self._joint_indices = [
            plant.GetJointByName(j).position_start()
            for j in ("iiwa_joint_2", "iiwa_joint_4", "iiwa_joint_6")
        ]

        self.DeclareAbstractInputPort(
            "switched_on_intervals", AbstractValue.Make(np.array([])))

        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self.DeclareAbstractInputPort(
            "body_spatial_velocities", AbstractValue.Make([SpatialVelocity()])
        )

        self.DeclareAbstractInputPort(
            "trajectory", AbstractValue.Make(PiecewisePose()))

        self.DeclareVectorInputPort("ee_force_measured", 3)
        self.DeclareVectorInputPort("iiwa_state_measured", 6)
        self.DeclareVectorOutputPort("iiwa_torque_cmd", 3, self.CalcTorqueOutput)


    def CalcTorqueOutput(self, context, output):
        if self.cart_trajectory is None:
            self.cart_trajectory = self.GetInputPort('trajectory').Eval(context)
            self.vel_trajectory = self.cart_trajectory.MakeDerivative()
            self.traj_intervals = np.array([self.cart_trajectory.start_time(), self.cart_trajectory.end_time()])[np.newaxis, :]

        if self.switched_on_intervals is None:
            self.switched_on_intervals = self.GetInputPort('switched_on_intervals').Eval(context)
            if 2 == len(self.switched_on_intervals.shape) and \
               2 != self.switched_on_intervals.shape[1]:
                raise Exception(f'each row must be an interval, but is mishaped: {self.switched_on_intervals.shape}')

        current_time = context.get_time()
        if not is_within_intervals(current_time, self.switched_on_intervals) or \
           not is_within_intervals(current_time, self.traj_intervals):
            output.SetFromVector(np.zeros((3),))
            return

        state_now = self.GetInputPort("iiwa_state_measured").Eval(context)

        q_now = state_now[:3]
        qdot_now = state_now[3:]

        self._plant.SetPositions(self._plant_context, self._iiwa, q_now)
        self._plant.SetVelocities(self._plant_context, self._iiwa, qdot_now)

        bias = self._plant.CalcBiasTerm(self._plant_context) # coreolis + gyscopic effects
        forces = MultibodyForces(plant=self._plant)
        inner_gravity = self._plant.CalcGravityGeneralizedForces(self._plant_context)
        np.copyto(forces.mutable_generalized_forces(), inner_gravity)

        tau = self._plant.CalcInverseDynamics(self._plant_context, np.zeros(6,), forces)
        tau -= bias
        tau = tau[:3]

        J_G = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kQDot,
            self._G,
            [0, 0, 0],
            self._W,
            self._W,
        )
                         # ry, tx, tz
        J_G = J_G[np.ix_([1, 3, 5], self._joint_indices)]

        # cartesian pos control
        pose_desired = self.cart_trajectory.GetPose(current_time)
        td_WG = get_transl2d_from_transform(pose_desired)

        velocity_desired = self.vel_trajectory.value(current_time)
        vd_WG = get_vel2d_from_spatial_velocity(velocity_desired)
        wd_WG = get_rotvel_from_spatial_velocity(velocity_desired)

        pose_measured = self.GetInputPort("body_poses").Eval(context)[self._ee_body_instance]
        tm_WG = get_transl2d_from_transform(pose_measured)
        rm_WG = get_pitch_from_transform(pose_measured)

        velocity_measured = self.GetInputPort("body_spatial_velocities").Eval(context)[self._ee_body_instance].get_coeffs()
        vm_WG = get_vel2d_from_spatial_velocity(velocity_measured)
        wm_WG = get_rotvel_from_spatial_velocity(velocity_measured)

        te_WG = td_WG - tm_WG
        ve_WG = vd_WG - vm_WG

        pitch_dot_err = wd_WG - wm_WG

        # print('t=', current_time, 've_WG, pitch_dot_err', ve_WG, pitch_dot_err)

        te_WG = np.pad(te_WG, (1, 0), mode='constant', constant_values=0.)
        ve_WG = np.pad(ve_WG, (1, 0), mode='constant', constant_values=0.)

        q_e_tang = J_G.T @ te_WG
        qdot_e_tang = J_G.T @ ve_WG

        tau += q_e_tang * self.kp_tang_vec
        tau += qdot_e_tang * self.kd_tang_vec

        # force control
        m_measured = self.GetInputPort('ee_force_measured').Eval(context)[0]
        m_goal = 2.
        me = m_goal - m_measured
        # print('me', me)

        # both directions arent used
        e_tau = J_G.T @ [me, 0, 0]
        ve_norm_tau = J_G.T @ [pitch_dot_err, 0, 0]

        tau += e_tau * self.kf_norm_vec
        tau += ve_norm_tau * self.kfd_norm_vec

        output.SetFromVector(tau)
