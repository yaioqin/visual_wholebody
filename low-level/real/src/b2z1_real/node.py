"""ROS1 orchestration node for B2 policy inference and Z1 IK control."""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import rospy
from geometry_msgs.msg import PointStamped, Twist
from std_msgs.msg import Bool, Float32MultiArray, Float64, String
from std_srvs.srv import Trigger, TriggerResponse

from b2z1_real.msg import B2LowCommand, B2LowState, Z1Command, Z1State

from .kinematics import Z1Kinematics
from .policy import ObservationBuilder, PolicyRunner, cartesian_to_sphere, sphere_to_cartesian
from .safety import ControlState, SafetyGate


class B2Z1PolicyNode:
    def __init__(self) -> None:
        self.config = rospy.get_param("~")
        policy_path = os.path.expanduser(str(self.config.get("policy_path", "")))
        if not policy_path:
            raise RuntimeError("~policy_path is empty; export a trained checkpoint before starting")
        if not os.path.isfile(policy_path):
            raise RuntimeError(f"policy file does not exist: {policy_path}")

        self.rate = float(self.config.get("control_rate", 50.0))
        self.dt = 1.0 / self.rate
        self.config["control_dt"] = self.dt
        self.builder = ObservationBuilder(self.config)
        self.policy = PolicyRunner(policy_path, str(self.config.get("policy_device", "cpu")))
        self.kinematics = Z1Kinematics()
        self.safety = SafetyGate(self.config["safety"])

        self.robot_cfg = self.config["robot"]
        self.command_cfg = self.config["command"]
        self.goal_cfg = self.config["goal"]
        self.deploy_cfg = self.config["deployment"]
        self.topics = self.config["topics"]

        self.default_leg = np.asarray(self.robot_cfg["default_leg_position"], dtype=np.float64)
        self.action_scale = np.asarray(self.robot_cfg["action_scale"], dtype=np.float64)
        self.leg_kp = np.asarray(self.robot_cfg["leg_kp"], dtype=np.float64)
        self.leg_kd = np.asarray(self.robot_cfg["leg_kd"], dtype=np.float64)
        self.goal_vector = np.asarray(self.goal_cfg["default_vector"], dtype=np.float64)
        self.base_height: Optional[float] = None

        margin = float(self.deploy_cfg["joint_limit_margin"])
        self.leg_lower = self.safety.leg_lower + margin
        self.leg_upper = self.safety.leg_upper - margin
        self.arm_lower = self.safety.arm_lower + margin
        self.arm_upper = self.safety.arm_upper - margin

        self.b2_state: Optional[B2LowState] = None
        self.z1_state: Optional[Z1State] = None
        self.b2_received_at: Optional[float] = None
        self.z1_received_at: Optional[float] = None
        self.velocity_received_at: Optional[float] = None
        self.velocity_command = np.zeros(3, dtype=np.float64)

        self.state = ControlState.DISARMED
        self.fault_reason = ""
        self.enable_requested = False
        self.estop_signal = False
        self.arming_started_at = 0.0
        self.arming_start_leg = self.default_leg.copy()
        self.previous_leg_target = self.default_leg.copy()
        self.last_leg_action = np.zeros(12, dtype=np.float64)
        delay = int(self.deploy_cfg.get("action_delay_steps", 1))
        if delay < 0:
            raise ValueError("action_delay_steps must be non-negative")
        self.action_queue = [np.zeros(12, dtype=np.float64) for _ in range(delay)]

        self.b2_command_pub = rospy.Publisher(self.topics["b2_command"], B2LowCommand, queue_size=1)
        self.z1_command_pub = rospy.Publisher(self.topics["z1_command"], Z1Command, queue_size=1)
        self.control_state_pub = rospy.Publisher("/b2z1/control_state", String, queue_size=1, latch=True)
        self.observation_pub = rospy.Publisher("/b2z1/debug/observation", Float32MultiArray, queue_size=1)
        self.action_pub = rospy.Publisher("/b2z1/debug/action", Float32MultiArray, queue_size=1)

        rospy.Subscriber(self.topics["b2_state"], B2LowState, self._b2_state_callback, queue_size=1)
        rospy.Subscriber(self.topics["z1_state"], Z1State, self._z1_state_callback, queue_size=1)
        rospy.Subscriber(self.topics["velocity_command"], Twist, self._velocity_callback, queue_size=1)
        rospy.Subscriber(self.topics["ee_goal"], PointStamped, self._goal_callback, queue_size=1)
        rospy.Subscriber(self.topics["base_height"], Float64, self._base_height_callback, queue_size=1)
        rospy.Subscriber(self.topics["enable"], Bool, self._enable_callback, queue_size=1)
        rospy.Subscriber(self.topics["emergency_stop"], Bool, self._estop_callback, queue_size=1)
        rospy.Service("/b2z1/clear_fault", Trigger, self._clear_fault)
        self.timer = rospy.Timer(rospy.Duration(self.dt), self._tick)
        self._publish_state()

    @staticmethod
    def _monotonic() -> float:
        return time.monotonic()

    def _b2_state_callback(self, message: B2LowState) -> None:
        self.b2_state = message
        self.b2_received_at = self._monotonic()

    def _z1_state_callback(self, message: Z1State) -> None:
        self.z1_state = message
        self.z1_received_at = self._monotonic()

    def _velocity_callback(self, message: Twist) -> None:
        linear = np.clip(message.linear.x, -float(self.command_cfg["max_linear_x"]),
                         float(self.command_cfg["max_linear_x"]))
        yaw = np.clip(message.angular.z, -float(self.command_cfg["max_yaw_rate"]),
                      float(self.command_cfg["max_yaw_rate"]))
        if (abs(linear) <= self.builder.walking_linear_threshold
                and abs(yaw) <= self.builder.walking_yaw_threshold):
            linear, yaw = 0.0, 0.0
        self.velocity_command[:] = [linear, 0.0, yaw]
        self.velocity_received_at = self._monotonic()

    def _goal_is_valid(self, candidate: np.ndarray) -> bool:
        candidate_sphere = cartesian_to_sphere(candidate)
        radius, pitch, yaw = candidate_sphere
        ranges = (
            (radius, self.goal_cfg["radius_range"], "radius"),
            (pitch, self.goal_cfg["pitch_range"], "pitch"),
            (yaw, self.goal_cfg["yaw_range"], "yaw"),
        )
        for value, limits, name in ranges:
            if not float(limits[0]) <= value <= float(limits[1]):
                rospy.logwarn("Rejected EE goal: %s %.3f outside [%.3f, %.3f]", name, value,
                              float(limits[0]), float(limits[1]))
                return False
        lower = np.asarray(self.goal_cfg["collision_lower"], dtype=np.float64)
        upper = np.asarray(self.goal_cfg["collision_upper"], dtype=np.float64)
        # Training interpolates trajectories in spherical coordinates before
        # running its static body/underground exclusion test.
        sphere_samples = np.linspace(
            cartesian_to_sphere(self.goal_vector),
            candidate_sphere,
            int(self.goal_cfg.get("collision_check_samples", 10)),
        )
        samples = np.stack([sphere_to_cartesian(sample) for sample in sphere_samples])
        collision = np.any(np.all((samples > lower) & (samples < upper), axis=1))
        underground = np.any(samples[:, 2] < float(self.goal_cfg["underground_limit"]))
        if collision or underground:
            rospy.logwarn("Rejected EE goal: path intersects the training collision exclusion")
            return False
        return True

    def _goal_callback(self, message: PointStamped) -> None:
        expected_frame = str(self.goal_cfg.get("frame_id", "b2_base_yaw"))
        if message.header.frame_id != expected_frame:
            rospy.logwarn("Rejected EE goal frame '%s'; expected '%s'",
                          message.header.frame_id, expected_frame)
            return
        candidate = np.asarray([message.point.x, message.point.y, message.point.z], dtype=np.float64)
        if np.all(np.isfinite(candidate)) and self._goal_is_valid(candidate):
            self.goal_vector = candidate

    def _base_height_callback(self, message: Float64) -> None:
        if np.isfinite(message.data) and 0.2 <= message.data <= 1.2:
            self.base_height = float(message.data)

    def _enable_callback(self, message: Bool) -> None:
        self.enable_requested = bool(message.data)
        if not self.enable_requested and self.state != ControlState.FAULT:
            self._set_state(ControlState.DISARMED)

    def _estop_callback(self, message: Bool) -> None:
        self.estop_signal = bool(message.data)
        if self.estop_signal:
            self._fault("emergency stop asserted")

    def _clear_fault(self, _request) -> TriggerResponse:
        if self.enable_requested:
            return TriggerResponse(False, "publish enable=false before clearing a fault")
        if self.estop_signal:
            return TriggerResponse(False, "emergency_stop topic is still true")
        self.fault_reason = ""
        self._set_state(ControlState.DISARMED)
        return TriggerResponse(True, "fault cleared; controller remains disarmed")

    def _set_state(self, state: ControlState) -> None:
        if state == self.state and state != ControlState.DISARMED:
            return
        self.state = state
        if state == ControlState.DISARMED:
            self.builder.reset()
            self.last_leg_action.fill(0.0)
            delay = int(self.deploy_cfg.get("action_delay_steps", 1))
            self.action_queue = [np.zeros(12, dtype=np.float64) for _ in range(delay)]
        self._publish_state()

    def _fault(self, reason: str) -> None:
        if self.state != ControlState.FAULT or reason != self.fault_reason:
            rospy.logerr("B2-Z1 controller FAULT: %s", reason)
        self.fault_reason = reason
        self.state = ControlState.FAULT
        self._publish_state()

    def _publish_state(self) -> None:
        suffix = f": {self.fault_reason}" if self.fault_reason else ""
        self.control_state_pub.publish(String(data=f"{self.state.value}{suffix}"))

    def _arm_arrays(self):
        if self.z1_state is None:
            return np.zeros(6), np.zeros(6)
        names = list(self.z1_state.joints.name)
        positions = np.asarray(self.z1_state.joints.position, dtype=np.float64)
        velocities = np.asarray(self.z1_state.joints.velocity, dtype=np.float64)
        expected = [f"joint{index}" for index in range(1, 7)]
        if names and all(name in names for name in expected):
            indices = [names.index(name) for name in expected]
            return positions[indices], velocities[indices]
        if len(positions) < 6 or len(velocities) < 6:
            raise ValueError("Z1 state must contain six joint positions and velocities")
        return positions[:6], velocities[:6]

    def _safety_reason(self, now: float) -> Optional[str]:
        if self.b2_state is None or self.z1_state is None:
            return "waiting for B2 and Z1 states"
        arm_position, _ = self._arm_arrays()
        return self.safety.check(
            now,
            self.b2_received_at,
            self.z1_received_at,
            self.b2_state.rpy,
            self.b2_state.q,
            arm_position,
            self.b2_state.motor_temperature,
            self.z1_state.temperature,
            bool(self.b2_state.valid and self.b2_state.crc_ok),
            bool(self.z1_state.valid),
        )

    def _publish_commands(self, leg_q: np.ndarray, arm_q: np.ndarray, arm_dq: np.ndarray, enabled: bool) -> None:
        now = rospy.Time.now()
        b2 = B2LowCommand()
        b2.header.stamp = now
        b2.q = leg_q.tolist()
        b2.dq = np.zeros(12).tolist()
        b2.kp = self.leg_kp.tolist()
        b2.kd = self.leg_kd.tolist()
        b2.tau = np.zeros(12).tolist()
        b2.enabled = enabled
        self.b2_command_pub.publish(b2)

        z1 = Z1Command()
        z1.header.stamp = now
        z1.q = arm_q.tolist()
        z1.dq = arm_dq.tolist()
        z1.enabled = enabled
        self.z1_command_pub.publish(z1)

    def _tick(self, _event) -> None:
        now = self._monotonic()
        try:
            arm_position, arm_velocity = self._arm_arrays()
            current_leg = (np.asarray(self.b2_state.q, dtype=np.float64)
                           if self.b2_state is not None else self.default_leg.copy())

            if self.state == ControlState.FAULT:
                self._publish_commands(current_leg, arm_position, np.zeros(6), False)
                return
            if not self.enable_requested:
                self._publish_commands(current_leg, arm_position, np.zeros(6), False)
                return

            reason = self._safety_reason(now)
            if reason:
                if self.state in (ControlState.ARMING, ControlState.ACTIVE):
                    self._fault(reason)
                return

            if self.state == ControlState.DISARMED:
                error = float(np.max(np.abs(current_leg - self.default_leg)))
                if error > float(self.deploy_cfg["initial_max_leg_error"]):
                    self._fault(
                        f"B2 is too far from deployment pose ({error:.3f} rad); suspend and position it first"
                    )
                    return
                self.arming_started_at = now
                self.arming_start_leg = current_leg.copy()
                self.previous_leg_target = current_leg.copy()
                self._set_state(ControlState.ARMING)

            if self.state == ControlState.ARMING:
                duration = max(float(self.deploy_cfg["arming_duration"]), self.dt)
                alpha = np.clip((now - self.arming_started_at) / duration, 0.0, 1.0)
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                leg_target = (1.0 - alpha) * self.arming_start_leg + alpha * self.default_leg
                self.previous_leg_target = leg_target.copy()
                self._publish_commands(leg_target, arm_position, np.zeros(6), True)
                if alpha >= 1.0:
                    self.builder.reset()
                    self._set_state(ControlState.ACTIVE)
                return

            if self.velocity_received_at is None or now - self.velocity_received_at > float(self.command_cfg["timeout"]):
                self.velocity_command.fill(0.0)

            orientation = self.b2_state.orientation
            current, transformed_goal = self.builder.build_current(
                [orientation.x, orientation.y, orientation.z, orientation.w],
                [self.b2_state.angular_velocity.x, self.b2_state.angular_velocity.y,
                 self.b2_state.angular_velocity.z],
                self.b2_state.q,
                self.b2_state.dq,
                arm_position,
                arm_velocity,
                self.last_leg_action,
                self.b2_state.foot_force,
                self.velocity_command,
                self.goal_vector,
                self.base_height,
            )
            policy_input = self.builder.policy_input(current)
            action = self.policy.infer(policy_input)
            action_clip = float(self.deploy_cfg["action_clip"])
            issued_leg_action = np.clip(action[:12], -action_clip, action_clip)
            self.action_queue.append(issued_leg_action.copy())
            applied_leg_action = self.action_queue.pop(0)
            self.last_leg_action = issued_leg_action

            desired_leg = self.default_leg + self.action_scale * applied_leg_action
            desired_leg = np.clip(desired_leg, self.leg_lower, self.leg_upper)
            max_leg_step = float(self.deploy_cfg["leg_target_velocity_limit"]) * self.dt
            desired_leg = self.previous_leg_target + np.clip(
                desired_leg - self.previous_leg_target, -max_leg_step, max_leg_step
            )
            self.previous_leg_target = desired_leg.copy()

            max_arm_step = float(self.deploy_cfg["arm_target_velocity_limit"]) * self.dt
            desired_arm, _ = self.kinematics.ik_step(
                arm_position,
                transformed_goal.arm_position,
                transformed_goal.arm_orientation,
                damping=float(self.deploy_cfg["ik_damping"]),
                max_delta=max_arm_step,
            )
            desired_arm = np.clip(desired_arm, self.arm_lower, self.arm_upper)
            desired_arm = arm_position + np.clip(
                desired_arm - arm_position, -max_arm_step, max_arm_step
            )
            desired_arm_velocity = np.clip(
                (desired_arm - arm_position) / self.dt,
                -float(self.deploy_cfg["arm_target_velocity_limit"]),
                float(self.deploy_cfg["arm_target_velocity_limit"]),
            )
            self._publish_commands(desired_leg, desired_arm, desired_arm_velocity, True)
            self.observation_pub.publish(Float32MultiArray(data=current.tolist()))
            self.action_pub.publish(Float32MultiArray(data=action.astype(np.float32).tolist()))
        except Exception as error:  # A control-loop exception must immediately disable both bridges.
            self._fault(f"control exception: {error}")
            arm = np.zeros(6) if self.z1_state is None else self._arm_arrays()[0]
            leg = self.default_leg if self.b2_state is None else np.asarray(self.b2_state.q)
            self._publish_commands(leg, arm, np.zeros(6), False)


def main() -> None:
    rospy.init_node("b2z1_policy")
    try:
        B2Z1PolicyNode()
        rospy.spin()
    except Exception as error:
        rospy.logfatal("Failed to start B2-Z1 policy node: %s", error)
        raise
