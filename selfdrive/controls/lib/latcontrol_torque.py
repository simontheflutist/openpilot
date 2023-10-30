import math
import numpy as np

from cereal import log
from openpilot.common.numpy_fast import clip, interp
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.controls.lib.pid import PIDController
from openpilot.selfdrive.controls.lib.vehicle_model import ACCELERATION_DUE_TO_GRAVITY

# At higher speeds (25+mph) we can assume:
# Lateral acceleration achieved by a specific car correlates to
# torque applied to the steering rack. It does not correlate to
# wheel slip, or to speed.

class LatControlTorque(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)
    self.torque_params = CP.lateralTuning.torque
    # for integrating
    self.rate = 100 # [s^(-1)]
    # reference trajectory
    self.a_hat = 0 # [m/s^2]
    # adaptive control to track the true relaxation time of the system
    self.gamma = 1. # [dimensionless] dynamically scaled by self.get_time_constant
    self.adaptation_gain = 0.00005 # [s^3 / m^2]
    
    # for tracking acceleration to the reference model
    tracking_time_constant = 0.5 # [s]
    self.a_hat_rate = np.exp(-1 / (self.rate * tracking_time_constant))

    # TODO use in get_steer_command to enable PI control
    self.pid = PIDController(k_p=1,
                             k_i=0,
                             k_f=0., pos_limit=1e308, neg_limit=-1e308)
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.use_steering_angle = self.torque_params.useSteeringAngle
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg

    # use polynomials for velocity-dependent time constant
    p = np.array([ 1.16359733, -0.75694134,  0.5332686 , -0.21814991,  0.22976125,
                  -2.05787915,  0.45070959,  1.34722179,  0.25426358,  0.40487943,
                  -0.03346696,  0.23336259,  0.03495715])
    time_constant_cheby = np.polynomial.Chebyshev(p[0:5], domain=(0, 45))
    a_cheby = np.polynomial.Chebyshev(p[5:9], domain=(0, 45))
    b_cheby = np.polynomial.Chebyshev(p[9:13], domain=(0, 45))
    self.get_time_constant = lambda _, vEgo: time_constant_cheby(vEgo)
    self.get_a = lambda _, vEgo: a_cheby(vEgo)
    self.get_b = lambda _, vEgo: b_cheby(vEgo)
    

  def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction


  def update_a_hat(self, desired_acceleration):
    self.a_hat = self.a_hat_rate * self.a_hat + (1 - self.a_hat_rate) * desired_acceleration


  def update_gamma(self, desired_accel, actual_accel, freeze):
    if not freeze:
      self.gamma = self.adaptation_gain * (self.a_hat - actual_accel) * (desired_accel - actual_accel)
    return self.gamma
  

  def get_torque(self, lateral_accel, vEgo, friction_compensation=False, diff=False):
    """
      Defines torque as a function of acceleration and other parameters.
    """
    # velocity dependence
    a = self.get_a(vEgo)
    b = self.get_b(vEgo)
    
    if diff:
      return (1 + (lateral_accel / b)**2)**(-0.5) / b / a
    else:
      return np.arcsinh(lateral_accel / b) / a
  
  
  def get_steer_command(self, desired_accel, actual_accel, gamma, vEgo):
    # f(a)
    feedback_term = self.get_torque(actual_accel, vEgo)

    # f'(a) * T
    time_constant = self.get_time_constant(vEgo)
    gain = time_constant * gamma * self.get_torque(actual_accel, vEgo, diff=True)
    return feedback_term + gain * (desired_accel - actual_accel)    


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
      else:
        actual_curvature_vm = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
        actual_curvature_llk = llk.angularVelocityCalibrated.value[2] / CS.vEgo
        actual_curvature = interp(CS.vEgo, [2.0, 5.0], [actual_curvature_vm, actual_curvature_llk])

      # Calculate actual acceleration due to steering
      actual_lateral_accel = (actual_curvature * CS.vEgo ** 2 
                              - ACCELERATION_DUE_TO_GRAVITY * params.roll)
      desired_lateral_accel = (desired_curvature * CS.vEgo ** 2
                               - ACCELERATION_DUE_TO_GRAVITY * params.roll)

      # Update integrators
      freeze_integrator = steer_limited or CS.steeringPressed or CS.vEgo < 5
      self.update_a_hat(desired_lateral_accel)
      gamma = self.update_gamma(desired_lateral_accel, actual_lateral_accel, freeze=freeze_integrator)
      
      # Compute feedback
      steer = self.get_steer_command(desired_lateral_accel,
                                     actual_lateral_accel, gamma, CS.vEgo, params.roll)

      # Clip like pid does
      output_torque = clip(steer, -self.steer_max, self.steer_max)

      # Finish composing log message
      pid_log.active = True
      pid_log.error = 0
      pid_log.p = self.pid.p
      pid_log.i = self.pid.i
      # hijack for my uses lol
      pid_log.d = self.gamma
      pid_log.f = self.pid.f
      pid_log.output = -output_torque
      pid_log.actualLateralAccel = actual_lateral_accel
      pid_log.desiredLateralAccel = desired_lateral_accel
      pid_log.saturated = self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited)

    # model outputs correct sign
    return output_torque, 0.0, pid_log
