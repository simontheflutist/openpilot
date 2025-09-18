import numpy as np
from numbers import Number

class PIDController:
  def __init__(self, k_p, k_i, k_f=0., k_d=0., pos_limit=1e308, neg_limit=-1e308, rate=100):
    self._k_p = k_p
    self._k_i = k_i
    self._k_d = k_d
    self.k_f = k_f   # feedforward gain
    if isinstance(self._k_p, Number):
      self._k_p = [[0], [self._k_p]]
    if isinstance(self._k_i, Number):
      self._k_i = [[0], [self._k_i]]
    if isinstance(self._k_d, Number):
      self._k_d = [[0], [self._k_d]]

    self.set_limits(pos_limit, neg_limit)

    self.i_rate = 1.0 / rate
    self.speed = 0.0

    self.reset()

  @property
  def k_p(self):
    return np.interp(self.speed, self._k_p[0], self._k_p[1])

  @property
  def k_i(self):
    return np.interp(self.speed, self._k_i[0], self._k_i[1])

  @property
  def k_d(self):
    return np.interp(self.speed, self._k_d[0], self._k_d[1])

  def reset(self):
    self.p = 0.0
    self.i = 0.0
    self.d = 0.0
    self.f = 0.0
    self.control = 0

  def set_limits(self, pos_limit, neg_limit):
    self.pos_limit = pos_limit
    self.neg_limit = neg_limit

  def update(self, error, error_rate=0.0, speed=0.0, feedforward=0., freeze_integrator=False):
    self.speed = speed
    self.p = float(error) * self.k_p
    self.f = feedforward * self.k_f
    self.d = error_rate * self.k_d

    if not freeze_integrator:
      i = self.i + error * self.k_i * self.i_rate

      # Don't allow windup if already clipping
      test_control = self.p + i + self.d + self.f
      i_upperbound = self.i if test_control > self.pos_limit else self.pos_limit
      i_lowerbound = self.i if test_control < self.neg_limit else self.neg_limit
      self.i = np.clip(i, i_lowerbound, i_upperbound)

    control = self.p + self.i + self.d + self.f
    self.control = np.clip(control, self.neg_limit, self.pos_limit)
    return self.control


class MultiIntegralPIDController:
  """
  PID controller with multiple cascaded integral terms.

  Differences from PIDController:
  - k_i is a list where each element is either a number or [[speeds], [values]]
  - Maintains multiple integral states in self.error_integral[j]
  - self.i = k_i[0]*error_integral[0] + k_i[1]*error_integral[1] + ...
  - All integrators freeze when output is clipped
  """

  def __init__(self, k_p, k_i, k_f=0., k_d=0., pos_limit=1e308, neg_limit=-1e308, rate=100):
    # Store gains
    self._k_p = k_p if isinstance(k_p, list) else [[0], [float(k_p)]]
    self._k_d = k_d if isinstance(k_d, list) else [[0], [float(k_d)]]
    self.k_f = float(k_f)  # feedforward gain

    # Process k_i list
    if not isinstance(k_i, (list, tuple)) or not k_i:
      raise ValueError("k_i must be a non-empty list of coefficients/tables")

    self._k_i_list = []
    for ki in k_i:
      if isinstance(ki, Number):
        self._k_i_list.append([[0], [float(ki)]])
      else:
        # Assume already in [[speeds], [values]] format
        self._k_i_list.append(ki)

    self.n_integrators = len(self._k_i_list)
    self.set_limits(pos_limit, neg_limit)
    self.i_rate = 1.0 / rate
    self.speed = 0.0
    self.reset()

  @property
  def k_p(self):
    return float(np.interp(self.speed, self._k_p[0], self._k_p[1]))

  @property
  def k_d(self):
    return float(np.interp(self.speed, self._k_d[0], self._k_d[1]))

  @property
  def k_i(self):
    return [float(np.interp(self.speed, ki[0], ki[1])) for ki in self._k_i_list]

  def reset(self):
    self.p = 0.0
    self.d = 0.0
    self.f = 0.0
    self.i = 0.0
    self.control = 0.0
    # Initialize all integrator states to zero
    self.error_integral = [0.0] * self.n_integrators

  def set_limits(self, pos_limit, neg_limit):
    self.pos_limit = float(pos_limit)
    self.neg_limit = float(neg_limit)

  def update(self, error, error_rate=0.0, speed=0.0, feedforward=0., freeze_integrator=False):
    self.speed = speed

    # Update P, D, F terms
    self.p = float(error) * self.k_p
    self.d = error_rate * self.k_d
    self.f = feedforward * self.k_f

    # Get current integral gains
    k_i_values = self.k_i

    # Update integral terms if not frozen
    if not freeze_integrator and self.n_integrators > 0:
      # Update the cascade of integrators
      signal = float(error)  # Input to first integrator
      new_integrals = []

      for j in range(self.n_integrators):
        # Update this integrator: integral += input * dt
        new_val = self.error_integral[j] + signal * self.i_rate
        new_integrals.append(new_val)
        # Next integrator's input is this integrator's output
        signal = new_val

      # Check if we would saturate with the new integrator values
      new_i = sum(k * i for k, i in zip(k_i_values, new_integrals, strict=True))
      test_control = self.p + self.d + self.f + new_i

      # Only update integrators if we're not saturating
      if test_control <= self.pos_limit and test_control >= self.neg_limit:
        self.error_integral = new_integrals
      # else: all integrators freeze (don't update self.error_integral)

    # Calculate I-term as weighted sum of all integrals
    self.i = sum(k * i for k, i in zip(k_i_values, self.error_integral[:self.n_integrators], strict=True))

    # Calculate and clip final control output
    control = self.p + self.d + self.f + self.i
    self.control = float(np.clip(control, self.neg_limit, self.pos_limit))
    return self.control
