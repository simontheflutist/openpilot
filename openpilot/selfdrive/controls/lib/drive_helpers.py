from collections import deque

import numpy as np
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.realtime import DT_CTRL, DT_MDL

MIN_SPEED = 1.0
CONTROL_N = 17
CAR_ROTATION_RADIUS = 0.0
# This is a turn radius smaller than most cars can achieve
MAX_CURVATURE = 0.2
MIN_STABLE_DELAY = 0.3
HEADING_PREDICTOR_TAU = 1.0  # s, time constant for closing the predicted heading error
IN_FLIGHT_BUFFER_SECONDS = 1.0

# EU guidelines
MAX_LATERAL_JERK = 5.0  # m/s^3
MAX_LATERAL_ACCEL_NO_ROLL = 3.0  # m/s^2


def should_stop(v_ego: float, a_target: float) -> bool:
  return bool(v_ego < 0.3 and a_target < 0.1)

def clamp(val, min_val, max_val):
  clamped_val = float(np.clip(val, min_val, max_val))
  return clamped_val, clamped_val != val

def smooth_value(val, prev_val, tau, dt=DT_MDL):
  alpha = 1 - np.exp(-dt/tau) if tau > 0 else 1
  return alpha * val + (1 - alpha) * prev_val

def clip_curvature(v_ego, prev_curvature, new_curvature, roll) -> tuple[float, bool]:
  # This function respects ISO lateral jerk and acceleration limits + a max curvature
  v_ego = max(v_ego, MIN_SPEED)
  max_curvature_rate = MAX_LATERAL_JERK / (v_ego ** 2)  # inexact calculation, check https://github.com/commaai/openpilot/pull/24755
  new_curvature = np.clip(new_curvature,
                          prev_curvature - max_curvature_rate * DT_CTRL,
                          prev_curvature + max_curvature_rate * DT_CTRL)

  roll_compensation = roll * ACCELERATION_DUE_TO_GRAVITY
  max_lat_accel = MAX_LATERAL_ACCEL_NO_ROLL + roll_compensation
  min_lat_accel = -MAX_LATERAL_ACCEL_NO_ROLL + roll_compensation
  new_curvature, limited_accel = clamp(new_curvature, min_lat_accel / v_ego ** 2, max_lat_accel / v_ego ** 2)

  new_curvature, limited_max_curv = clamp(new_curvature, -MAX_CURVATURE, MAX_CURVATURE)
  return float(new_curvature), limited_accel or limited_max_curv


def get_accel_from_plan(speeds, accels, t_idxs, action_t=DT_MDL):
  if len(speeds) == len(t_idxs):
    v_now = speeds[0]
    a_now = accels[0]
    if action_t < MIN_STABLE_DELAY:
      v_target = v_now + (action_t / MIN_STABLE_DELAY) * (np.interp(MIN_STABLE_DELAY, t_idxs, speeds) - v_now)
    else:
      v_target = np.interp(action_t, t_idxs, speeds)
    a_target = 2 * (v_target - v_now) / (action_t) - a_now
  else:
    a_target = 0.0
  return a_target

def curv_from_psis(psi_target, psi_rate, vego, action_t):
  vego = np.clip(vego, MIN_SPEED, np.inf)
  curv_from_psi = psi_target / (vego * action_t)
  return 2*curv_from_psi - psi_rate / vego

def get_curvature_from_plan(yaws, yaw_rates, t_idxs, vego, action_t):
  if action_t < MIN_STABLE_DELAY:
    psi_target = (action_t / MIN_STABLE_DELAY) * np.interp(MIN_STABLE_DELAY, t_idxs, yaws)
  else:
    psi_target = np.interp(action_t, t_idxs, yaws)
  psi_rate = yaw_rates[0]
  return curv_from_psis(psi_target, psi_rate, vego, action_t)


class InFlightHeadingPredictor:
  """Predictor feedback on heading, assuming the steering responds to the curvature command as a unit-gain pure delay.

  The command issued now starts acting lat_delay later. The heading the car will have by then, relative to the pose
  the plan is expressed in, is set by the yaw rates already requested since that frame was captured. This predicts
  that heading from the request history and steers the predicted error against the plan to zero with time constant tau.
  """

  def __init__(self, dt=DT_CTRL, tau=HEADING_PREDICTOR_TAU):
    self.dt = dt
    self.tau = tau
    self.buffer_len = int(IN_FLIGHT_BUFFER_SECONDS / dt)
    self.yaw_rate_requests = deque([0.] * self.buffer_len, maxlen=self.buffer_len)

  def record(self, curvature, v_ego):
    self.yaw_rate_requests.append(v_ego * curvature)

  def predicted_heading(self, horizon):
    n = int(np.clip(round(horizon / self.dt), 0, self.buffer_len))
    if n == 0:
      return 0.
    return float(np.sum(np.array(self.yaw_rate_requests)[-n:])) * self.dt

  def get_curvature(self, yaws, yaw_rates, t_idxs, v_ego, plan_age, lat_delay):
    v_ego = max(v_ego, MIN_SPEED)
    t_act = plan_age + lat_delay  # plan time at which the command issued now starts acting
    psi_plan = np.interp(t_act, t_idxs, yaws)
    psi_pred = self.predicted_heading(t_act)
    curv_from_rate = np.interp(t_act, t_idxs, yaw_rates) / v_ego
    return float(curv_from_rate + (psi_plan - psi_pred) / (v_ego * self.tau))
