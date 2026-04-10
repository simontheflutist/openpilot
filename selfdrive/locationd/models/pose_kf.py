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
  Q = np.diag([0.000327**2, 0.000412**2, 0.0**2,  # orientation: roll, pitch, yaw
               0.04449**2, 0.02565**2, 0.02709**2,    # device velocity
               0.001926**2, 0.002299**2, 0.002151**2,  # angular velocity
               0.001057**2, 0.001404**2, 0.000732**2,  # gyro bias
               0.004243**2, 0.002564**2, 0.002709**2,  # acceleration
               0.000143**2, 0.000135**2, 0.000137**2]) # accel bias

  # Off-diagonal process noise correlations from EM (|r| >= 0.15)

  # roll ↔ pitch: coupled via IMU cross-axis sensitivity (r ≈ -0.22)
  Q[0, 1] = Q[1, 0] = -0.22 * np.sqrt(Q[0, 0] * Q[1, 1])

  # velocity ↔ acceleration: v_new = v + dt*a, so process noise is redundant (r ≈ -1.0)
  for i, j in [(3, 12), (4, 13), (5, 14)]:
    Q[i, j] = Q[j, i] = -1.0 * np.sqrt(Q[i, i] * Q[j, j])

  # pitch ↔ v_z / a_z: pitch change tilts gravity between device x and z axes
  Q[1, 5] = Q[5, 1] = -0.56 * np.sqrt(Q[1, 1] * Q[5, 5])    # r ≈ -0.56
  Q[1, 14] = Q[14, 1] = +0.56 * np.sqrt(Q[1, 1] * Q[14, 14])  # r ≈ +0.56

  # v_z ↔ gb_pitch: pitch-rate bias drives apparent v_z drift (r ≈ +0.31)
  Q[5, 10] = Q[10, 5] = +0.31 * np.sqrt(Q[5, 5] * Q[10, 10])

  # angular velocity ↔ gyro bias: separated only by camera odo rotation; noise is correlated
  Q[6, 9] = Q[9, 6] = +0.31 * np.sqrt(Q[6, 6] * Q[9, 9])      # ω_roll ↔ gb_roll
  Q[7, 10] = Q[10, 7] = -0.30 * np.sqrt(Q[7, 7] * Q[10, 10])    # ω_pitch ↔ gb_pitch
  Q[8, 11] = Q[11, 8] = -0.17 * np.sqrt(Q[8, 8] * Q[11, 11])    # ω_yaw ↔ gb_yaw
  # cross-axis ω ↔ gb coupling (IMU cross-talk between roll and pitch axes)
  Q[6, 10] = Q[10, 6] = +0.49 * np.sqrt(Q[6, 6] * Q[10, 10])    # ω_roll ↔ gb_pitch
  Q[7, 9] = Q[9, 7] = +0.51 * np.sqrt(Q[7, 7] * Q[9, 9])      # ω_pitch ↔ gb_roll

  # ω_roll ↔ ω_pitch: cross-axis angular velocity coupling (r ≈ -0.40)
  Q[6, 7] = Q[7, 6] = -0.40 * np.sqrt(Q[6, 6] * Q[7, 7])

  # orientation ↔ gyro bias: orientation integrates ω which includes bias
  Q[0, 9] = Q[9, 0] = -0.48 * np.sqrt(Q[0, 0] * Q[9, 9])      # roll ↔ gb_roll
  Q[0, 10] = Q[10, 0] = +0.28 * np.sqrt(Q[0, 0] * Q[10, 10])    # roll ↔ gb_pitch
  Q[0, 11] = Q[11, 0] = +0.32 * np.sqrt(Q[0, 0] * Q[11, 11])    # roll ↔ gb_yaw
  Q[1, 9] = Q[9, 1] = +0.30 * np.sqrt(Q[1, 1] * Q[9, 9])      # pitch ↔ gb_roll
  Q[1, 10] = Q[10, 1] = -0.57 * np.sqrt(Q[1, 1] * Q[10, 10])    # pitch ↔ gb_pitch
  Q[1, 11] = Q[11, 1] = +0.19 * np.sqrt(Q[1, 1] * Q[11, 11])    # pitch ↔ gb_yaw

  # orientation ↔ angular velocity: orientation rate ≈ ω
  Q[0, 7] = Q[7, 0] = -0.25 * np.sqrt(Q[0, 0] * Q[7, 7])      # roll ↔ ω_pitch
  Q[1, 6] = Q[6, 1] = -0.16 * np.sqrt(Q[1, 1] * Q[6, 6])      # pitch ↔ ω_roll
  Q[1, 7] = Q[7, 1] = +0.21 * np.sqrt(Q[1, 1] * Q[7, 7])      # pitch ↔ ω_pitch

  # orientation ↔ velocity / acceleration: gravity projection coupling
  Q[0, 4] = Q[4, 0] = -0.15 * np.sqrt(Q[0, 0] * Q[4, 4])      # roll ↔ v_y
  Q[0, 13] = Q[13, 0] = +0.15 * np.sqrt(Q[0, 0] * Q[13, 13])    # roll ↔ a_y

  # pitch ↔ accel bias: gravity tilt drives accel bias observability
  Q[1, 15] = Q[15, 1] = -0.29 * np.sqrt(Q[1, 1] * Q[15, 15])    # pitch ↔ ab_x
  Q[1, 17] = Q[17, 1] = +0.17 * np.sqrt(Q[1, 1] * Q[17, 17])    # pitch ↔ ab_z
  Q[5, 15] = Q[15, 5] = +0.17 * np.sqrt(Q[5, 5] * Q[15, 15])    # v_z ↔ ab_x

  # gyro bias cross-axis coupling
  Q[9, 10] = Q[10, 9] = -0.38 * np.sqrt(Q[9, 9] * Q[10, 10])    # gb_roll ↔ gb_pitch
  Q[10, 11] = Q[11, 10] = -0.16 * np.sqrt(Q[10, 10] * Q[11, 11])  # gb_pitch ↔ gb_yaw

  # gb_pitch ↔ a_z / ab_x: pitch-rate bias couples to vertical acceleration & accel bias
  Q[10, 14] = Q[14, 10] = -0.31 * np.sqrt(Q[10, 10] * Q[14, 14])  # gb_pitch ↔ a_z
  Q[10, 15] = Q[15, 10] = +0.18 * np.sqrt(Q[10, 10] * Q[15, 15])  # gb_pitch ↔ ab_x
  Q[14, 15] = Q[15, 14] = -0.17 * np.sqrt(Q[14, 14] * Q[15, 15])  # a_z ↔ ab_x

  obs_noise = {
    ObservationKind.PHONE_GYRO: np.array([
      [1.627868415126e-04, -1.855455818524e-05, -5.104060556272e-06],
      [-1.855455818524e-05, 3.282344983644e-04, 2.979683285392e-06],
      [-5.104060556272e-06, 2.979683285392e-06, 2.240175365704e-05],
    ]),
    ObservationKind.PHONE_ACCEL: np.array([
      [2.345583892000e-01, 6.951440175108e-03, 2.195350389558e-02],
      [6.951440175108e-03, 1.866283380261e-01, 1.272743620484e-02],
      [2.195350389558e-02, 1.272743620484e-02, 2.384804684119e-01],
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
