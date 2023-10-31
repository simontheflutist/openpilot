import math

from cereal import log
from openpilot.common.numpy_fast import clip, interp
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.controls.lib.pid import PIDController
from openpilot.selfdrive.controls.lib.vehicle_model import ACCELERATION_DUE_TO_GRAVITY

# At higher speeds (25+mph) we can assume:
# Lateral acceleration achieved by a specific car correlates to
# torque applied to the steering rack. It does not correlate to
# wheel slip, or to speed.

# This controller applies torque to achieve desired lateral
# accelerations. To compensate for the low speed effects we
# use a LOW_SPEED_FACTOR in the error. Additionally, there is
# friction in the steering wheel that needs to be overcome to
# move it at all, this is compensated for too.

LOW_SPEED_X = [0, 10, 20, 30]
LOW_SPEED_Y = [15, 13, 10, 5]


class LatControlTorque(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)
    self.torque_params = CP.lateralTuning.torque
    # don't clip the acceleration PID; clip the steer output.
    self.pid = PIDController(self.torque_params.kp, self.torque_params.ki,
                             k_f=self.torque_params.kf, pos_limit=1e308, neg_limit=-1e308)
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.use_steering_angle = self.torque_params.useSteeringAngle
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg
    self.relaxation_time = 3 # [s]

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction

  def get_torque(self, lateral_accel, vEgo, roll, actual_curvature, friction_compensation=False, diff=False):
    """
      Defines torque as a function of acceleration and other parameters.
    """
    low_speed_factor = interp(vEgo, LOW_SPEED_X, LOW_SPEED_Y)**2
    lateral_accel = lateral_accel + low_speed_factor * actual_curvature
    lateral_accel = lateral_accel - roll * ACCELERATION_DUE_TO_GRAVITY
    torque = self.torque_from_lateral_accel(lateral_accel, self.torque_params, 0,
                                            self.steering_angle_deadzone_deg,
                                            friction_compensation=friction_compensation,
                                            diff=diff)
    return torque
  
  def get_steer_from_desired_jerk(self, desired_jerk, relaxation_time, lateral_accel, vEgo, roll, actual_curvature):
    """Suppose torque-acceleration is
          t = f(a)
      and torque friction dynamics is
         Dt = -(t - s)/T
      where s is the steer command.
      Then the chain rule gives
         Da = [1/f'(a)] Dt = -[1 / (T * f'(a))] (f(a) - s)
      In order to track Da = u, we use the feedback law
          s = f(a) + (f'(a) * T) u.
    """
    # f(a)
    feedback_term = self.get_torque(lateral_accel, vEgo, roll, actual_curvature)

    # f'(a) * T
    gain = self.get_torque(lateral_accel, vEgo, roll, actual_curvature, diff=True) * relaxation_time
    return feedback_term + gain * desired_jerk    


  def update(self, active, CS, VM, params, last_actuators, steer_limited, desired_curvature, desired_curvature_rate, llk):
    """Feedback linearization lets us work with the model
        Da = j
      where j is a jerk input. Let us change variables to e = a - (desired a). We desire closed loop dynamics
        De = PI(e)
      which means
        Da = D(desired a) + PI(e).
      The command is the sum of a reference jerk and a PI control on acceleration tracking error.
    """
    pid_log = log.ControlsState.LateralTorqueState.new_message()

    if not active:
      output_torque = 0.0
      pid_log.active = False
    else:
      # Calculate actual curvature
      if self.use_steering_angle:
        actual_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
        # curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))
      else:
        actual_curvature_vm = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
        actual_curvature_llk = llk.angularVelocityCalibrated.value[2] / CS.vEgo
        actual_curvature = interp(CS.vEgo, [2.0, 5.0], [actual_curvature_vm, actual_curvature_llk])
        # curvature_deadzone = 0.0

      # Calculate actual acceleration
      actual_lateral_accel = actual_curvature * CS.vEgo ** 2
      desired_lateral_accel = desired_curvature * CS.vEgo ** 2

      # Calculate desired jerk from the reference trajectory Da = desired jerk
      feedforward_jerk = desired_curvature_rate * CS.vEgo ** 2

      # Calculate jerk feedback from deviation (a - desired a)
      pid_log.error = desired_lateral_accel - actual_lateral_accel
      freeze_integrator = steer_limited or CS.steeringPressed or CS.vEgo < 5
      feedback_jerk = self.pid.update(pid_log.error, speed=CS.vEgo, freeze_integrator=freeze_integrator)

      # Use feedback linearization to convert desired jerk to a steer command
      steer = self.get_steer_from_desired_jerk(feedforward_jerk + feedback_jerk, self.relaxation_time,
                                               actual_lateral_accel, CS.vEgo, params.roll, actual_curvature)
      
      # Clip like pid does
      output_torque = clip(steer, -self.steer_max, self.steer_max)

      # Finish composing log message
      pid_log.active = True
      pid_log.p = self.pid.p
      pid_log.i = self.pid.i
      pid_log.d = self.pid.d
      pid_log.f = self.pid.f
      pid_log.output = -output_torque
      pid_log.actualLateralAccel = actual_lateral_accel
      pid_log.desiredLateralAccel = desired_lateral_accel
      pid_log.saturated = self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited)

    # TODO left is positive in this convention
    return -output_torque, 0.0, pid_log
