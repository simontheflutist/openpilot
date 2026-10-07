import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import InFlightHeadingPredictor
from openpilot.selfdrive.modeld.constants import ModelConstants

V_EGO = 20.0
PLAN_AGE = 0.05
LAT_DELAY = 0.3


def consistent_plan(curvature, v_ego=V_EGO):
  # heading and yaw rate the car follows when the curvature command is held constant
  t_idxs = np.array(ModelConstants.T_IDXS)
  yaw_rates = np.full_like(t_idxs, v_ego * curvature)
  yaws = yaw_rates * t_idxs
  return yaws, yaw_rates, t_idxs


class TestInFlightHeadingPredictor(OpenpilotTestCase):

  def test_consistent_history_reproduces_command(self):
    curvature = 0.01
    predictor = InFlightHeadingPredictor()
    for _ in range(predictor.buffer_len):
      predictor.record(curvature, V_EGO)
    yaws, yaw_rates, t_idxs = consistent_plan(curvature)
    out = predictor.get_curvature(yaws, yaw_rates, t_idxs, V_EGO, PLAN_AGE, LAT_DELAY)
    assert abs(out - curvature) < 1e-9

  def test_predicted_heading_uses_only_in_flight_window(self):
    predictor = InFlightHeadingPredictor()
    for _ in range(predictor.buffer_len):
      predictor.record(0.02, V_EGO)
    # the last PLAN_AGE + LAT_DELAY seconds of requests determine the heading at the moment the new command acts
    n = int(round((PLAN_AGE + LAT_DELAY) / DT_CTRL))
    for _ in range(n):
      predictor.record(0.0, V_EGO)
    assert predictor.predicted_heading(PLAN_AGE + LAT_DELAY) == 0.0
    assert abs(predictor.predicted_heading(1.0) - V_EGO * 0.02 * (1.0 - n * DT_CTRL)) < 1e-9

  def test_heading_error_feedback(self):
    curvature = 0.01
    predictor = InFlightHeadingPredictor()
    for _ in range(predictor.buffer_len):
      predictor.record(curvature, V_EGO)
    yaws, yaw_rates, t_idxs = consistent_plan(curvature)
    heading_error = 0.01
    out = predictor.get_curvature(yaws + heading_error, yaw_rates, t_idxs, V_EGO, PLAN_AGE, LAT_DELAY)
    assert abs(out - (curvature + heading_error / (V_EGO * predictor.tau))) < 1e-9

  def test_zero_history(self):
    predictor = InFlightHeadingPredictor()
    yaws, yaw_rates, t_idxs = consistent_plan(0.0)
    assert predictor.get_curvature(yaws, yaw_rates, t_idxs, V_EGO, PLAN_AGE, LAT_DELAY) == 0.0
