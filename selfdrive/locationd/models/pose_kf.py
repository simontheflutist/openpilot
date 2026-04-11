#!/usr/bin/env python3

import sys
import numpy as np

from openpilot.selfdrive.locationd.models.constants import ObservationKind

from rednose.helpers.kalmanfilter import KalmanFilter

if __name__=="__main__":
  import sympy as sp
  from rednose.helpers.ekf_sym import gen_code
  from rednose.helpers.sympy_helpers import euler_rotate, rot_to_euler
else:
  from rednose.helpers.ekf_sym_pyx import EKF_sym_pyx

EARTH_G = 9.81


class States:
  NED_ORIENTATION = slice(0, 3)  # roll, pitch, yaw in rad
  DEVICE_VELOCITY = slice(3, 6)  # device velocity in m/s
  ANGULAR_VELOCITY = slice(6, 9)  # roll, pitch and yaw rates in rad/s
  GYRO_BIAS = slice(9, 12)  # roll, pitch and yaw gyroscope biases in rad/s
  ACCELERATION = slice(12, 15)  # acceleration in device frame in m/s**2
  ACCEL_BIAS = slice(15, 18)  # Acceletometer bias in m/s**2


class PoseKalman(KalmanFilter):
  name = "pose"

  # state
  initial_x = np.array([0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0])
  # state covariance
  initial_P = np.diag([0.01**2, 0.01**2, 0.01**2,
                       10**2, 10**2, 10**2,
                       1**2, 1**2, 1**2,
                       1**2, 1**2, 1**2,
                       100**2, 100**2, 100**2,
                       0.01**2, 0.01**2, 0.01**2])

  # process noise (EM-tuned from driving data with IPLF smoother)
  Q = np.diag([0.001017**2, 0.001110**2, 0.0**2,   # orientation: roll, pitch, yaw
               0.02220**2, 0.005559**2, 0.003613**2,    # device velocity
               0.02588**2, 0.02595**2, 0.02583**2,      # angular velocity
               0.000997**2, 0.000833**2, 0.000949**2,   # gyro bias
               0.9000**2, 0.9000**2, 0.9000**2,         # acceleration
               0.001529**2, 0.001529**2, 0.001528**2])   # accel bias

  # Off-diagonal process noise correlations from EM (|r| >= 0.15)
  # Only correlations with clear physical/kinematic motivation are kept;
  # velocity cross-coupling, v↔gb, roll↔gb_yaw, and gb cross-axis terms
  # were removed as likely artifacts of the training data distribution.

  # pitch ↔ velocity: gravity projection — pitch tilts gravity between device axes
  Q[1, 3] = Q[3, 1] = +0.33 * np.sqrt(Q[1, 1] * Q[3, 3])      # pitch ↔ v_x
  Q[1, 5] = Q[5, 1] = +0.16 * np.sqrt(Q[1, 1] * Q[5, 5])      # pitch ↔ v_z

  # pitch ↔ gyro bias: orientation integrates ω which includes bias
  Q[1, 9] = Q[9, 1] = -0.44 * np.sqrt(Q[1, 1] * Q[9, 9])      # pitch ↔ gb_roll
  Q[1, 10] = Q[10, 1] = -0.62 * np.sqrt(Q[1, 1] * Q[10, 10])    # pitch ↔ gb_pitch
  Q[1, 11] = Q[11, 1] = -0.20 * np.sqrt(Q[1, 1] * Q[11, 11])    # pitch ↔ gb_yaw

  obs_noise = {
    ObservationKind.PHONE_GYRO: np.array([
      [1.281785411852e-04, -7.944940313955e-07, -4.546455422273e-08],
      [-7.944940313955e-07, 1.426107238360e-04, -4.724482745834e-07],
      [-4.546455422273e-08, -4.724482745834e-07, 9.722213391635e-05],
    ]),
    ObservationKind.PHONE_ACCEL: np.array([
      [1.856547720492e-01, -1.331549731447e-03, -2.052503168557e-02],
      [-1.331549731447e-03, 1.568372629413e-01, 4.216886642263e-03],
      [-2.052503168557e-02, 4.216886642263e-03, 2.446069553952e-01],
    ]),
    # CAMERA_ODO_TRANSLATION and CAMERA_ODO_ROTATION use per-observation R from model stds
    ObservationKind.CAMERA_ODO_TRANSLATION: np.diag([0.5**2, 0.5**2, 0.5**2]),
    ObservationKind.CAMERA_ODO_ROTATION: np.diag([0.05**2, 0.05**2, 0.05**2]),
  }

  @staticmethod
  def generate_code(generated_dir):
    name = PoseKalman.name
    dim_state = PoseKalman.initial_x.shape[0]
    dim_state_err = PoseKalman.initial_P.shape[0]

    state_sym = sp.MatrixSymbol('state', dim_state, 1)
    state = sp.Matrix(state_sym)
    roll, pitch, yaw = state[States.NED_ORIENTATION, :]
    velocity = state[States.DEVICE_VELOCITY, :]
    angular_velocity = state[States.ANGULAR_VELOCITY, :]
    vroll, vpitch, vyaw = angular_velocity
    gyro_bias = state[States.GYRO_BIAS, :]
    acceleration = state[States.ACCELERATION, :]
    acc_bias = state[States.ACCEL_BIAS, :]

    dt = sp.Symbol('dt')

    ned_from_device = euler_rotate(roll, pitch, yaw)
    device_from_ned = ned_from_device.T

    state_dot = sp.Matrix(np.zeros((dim_state, 1)))
    state_dot[States.DEVICE_VELOCITY, :] = acceleration

    f_sym = state + dt * state_dot
    device_from_device_t1 = euler_rotate(dt*vroll, dt*vpitch, dt*vyaw)
    ned_from_device_t1 = ned_from_device * device_from_device_t1
    f_sym[States.NED_ORIENTATION, :] = rot_to_euler(ned_from_device_t1)

    centripetal_acceleration = angular_velocity.cross(velocity)
    gravity = sp.Matrix([0, 0, -EARTH_G])
    h_gyro_sym = angular_velocity + gyro_bias
    h_acc_sym = device_from_ned * gravity + acceleration + centripetal_acceleration + acc_bias
    h_phone_rot_sym = angular_velocity
    h_relative_motion_sym = velocity
    obs_eqs = [
      [h_gyro_sym, ObservationKind.PHONE_GYRO, None],
      [h_acc_sym, ObservationKind.PHONE_ACCEL, None],
      [h_relative_motion_sym, ObservationKind.CAMERA_ODO_TRANSLATION, None],
      [h_phone_rot_sym, ObservationKind.CAMERA_ODO_ROTATION, None],
    ]
    gen_code(generated_dir, name, f_sym, dt, state_sym, obs_eqs, dim_state, dim_state_err)

  def __init__(self, generated_dir, max_rewind_age):
    dim_state, dim_state_err = PoseKalman.initial_x.shape[0], PoseKalman.initial_P.shape[0]
    self.filter = EKF_sym_pyx(generated_dir, self.name, PoseKalman.Q, PoseKalman.initial_x, PoseKalman.initial_P,
                              dim_state, dim_state_err, max_rewind_age=max_rewind_age)


if __name__ == "__main__":
  generated_dir = sys.argv[2]
  PoseKalman.generate_code(generated_dir)
