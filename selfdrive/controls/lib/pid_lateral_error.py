import numpy as np
from numbers import Number

from openpilot.common.numpy_fast import clip, interp


class LateralErrorPI():
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

    self.pos_limit = pos_limit
    self.neg_limit = neg_limit

    self.i_unwind_rate = 0.3 / rate
    self.i_rate = 1.0 / rate
    self.speed = 0.0

    self.reset()

  @property
  def k_p(self):
    return interp(self.speed, self._k_p[0], self._k_p[1])

  @property
  def old_k_i(self):
    return interp(self.speed, self._k_i[0], self._k_i[1])
  
  @property
  def k_i__and__k_ii(self):
    # unknown gain l
    #     d/dt e     = e_dot
    #     d/dt e_dot = -(psi_dot)^2 e - l * (k_ii * e + k_i * e_dot)
    # characteristic equation
    #     s^2 + (l * k_i) s + (l * k_ii * psi_dot^2) = 0
    # break frequency
    #     omega = sqrt(l * k_ii * psi_dot^2)
    #           = psi_dot * sqrt(l * k_ii)
    # damping
    #        xi = 0.5 * k_i/psi_dot * sqrt(l / k_ii)
    #
    # goal: use the "stock" k_i to get two gains, k_i for the first integral
    # of error and k_ii for the second
    # technically they should be uncorrelated, so by a scaling analysis, it makes
    # sense to use the "pythagorean theorem" since k_i has units of 1/s and
    # k_ii has units of 1/s^2
    #    old_k_i**2 = k_i**2 + k_ii
    # next let's define an anti-damping coefficient alpha so that
    #    k_ii = k_i**2 * alpha / 4
    alpha = 0.25
    k_i_squared = self.old_k_i**2 / (1 + alpha/4)
    k_i = k_i_squared ** 0.5
    k_ii = k_i_squared * alpha / 4
    return k_i, k_ii


  @property
  def k_d(self):
    return interp(self.speed, self._k_d[0], self._k_d[1])

  @property
  def error_integral(self):
    return self.e/self.k_ii

  def reset(self):
    self.p = 0.0
    self.e = 0.0
    self.e_dot = 0.0
    self.d = 0.0
    self.f = 0.0
    self.control = 0

  def update(self, error, yaw_rate=0.0,
             error_rate=0.0, speed=0.0, override=False, feedforward=0.,
             freeze_integrator=False, reset_integrator=False):
    self.speed = speed

    self.p = float(error) * self.k_p
    self.f = feedforward * self.k_f
    self.d = error_rate * self.k_d
    self.i = 0

    if override:
      self.e -= self.i_unwind_rate * float(np.sign(self.e))
      self.e_dot -= self.i_unwind_rate * float(np.sign(self.e_dot))
    else:
      # integrate
      #  d/dt e = e_dot
      #  d/dt e_dt = -(psi_dot)^2 * e + (acceleration-like error)
      # using semi-implicit Euler. bonus for symplectic!
      e = self.e + self.e_dot * self.i_rate 
      e_dot = self.e_dot + (-yaw_rate**2 * e + error) * self.i_rate

      k_i, k_ii = self.k_i__and__k_ii
      self.i = self.e * k_ii + self.e_dot * k_i
      control = self.p + self.i + self.d + self.f

      # Update when changing i will move the control away from the limits
      # or when i will move towards the sign of the error
      if reset_integrator:
        self.e = 0.
        self.e_dot = 0.
      elif (((error >= 0 and (control <= self.pos_limit or self.i < 0.0)) or
            (error <= 0 and (control >= self.neg_limit or self.i > 0.0))) and
            not freeze_integrator):
        self.e = e
        self.e_dot = e_dot
      else:
        # even when anti-windup activates, e still has to update due to inertial forces.
        # in this case, just do the rotation but skip the error
        e = self.e + self.e_dot * self.i_rate 
        e_dot = self.e_dot - yaw_rate**2 * e * self.i_rate
        self.e = e
        self.e_dot = e_dot

    control = self.p + self.i + self.d + self.f

    self.control = clip(control, self.neg_limit, self.pos_limit)
    return self.control
