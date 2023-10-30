# A new lateral controller

- $\sigma$ steering command (input)
- $\tau$ state variable that tracks $\sigma$ (with hysteresis/friction)
- $a$ acceleration (output)
- $a_\text{ref}$ reference (desired) acceleration
- $T$ time constant of real system
- $T_\text{ref}$ time constant of ideal tracking
- $\gamma$ live estimate of $T_\text{ref}$
- $\mu$ adaptation rate

Steering torque is _not_ statically related to lateral acceleration.
Rather than inverting friction with an ad-hoc gain boost, let's model the friction relationship as a spring-damper system between commanded torque and actual torque (which is statically related to lateral acceleration).

The friction dynamics are
$$
\dot \tau = \frac{\sigma - \tau}{T}
$$
and the output equation is given in inverse form as
$$
\tau = f(a).
$$
Using the chain rule, we compute the nonlinear dynamics of $a$ as
$$
\begin{align}
\dot a
&= \frac{\dot \tau}{f'(a)}
\\
&= \frac{\sigma - \tau}{T f'(a)}
\\
&= \frac{\sigma - f(a)}{T f'(a)}.
\end{align}
$$
Let $\sigma =  f(a) + \gamma f'(a) (a_\text{ref} - a) $, resulting in closed-loop dynamics
$$
\dot a  = \frac{\gamma}{T} (a_\text{ref} - a)
$$
Suppose we wish the track the reference dynamics
$$
\dot {\hat a} = \frac{a_\text{ref} - \hat a}{T_\text{ref}}.
$$
Choose as Lyapunov function
$$
V = \frac{1}{2} (\hat a - a)^2 + \frac{T}{2 \mu} \left(\frac{\gamma}{T} - 1\right)^2
$$
having time derivative is
$$
\begin{align}
\dot V &= (\hat a - a)
    \left(
        \frac{a_\text{ref} - \hat a}{T_\text{ref}}
        - \frac{\gamma}{T} (a_\text{ref} - a)
    \right)
    +
    \frac{1}{\mu} \left(\frac{\gamma}{T} - 1\right) \dot \gamma
    \\
    &= (\hat a - a)
    \left(
        \frac{a_\text{ref} - \hat a}{T_\text{ref}}
        - \frac{a_\text{ref} - a}{T_\text{ref}}
        + \frac{a_\text{ref} - a}{T_\text{ref}}
        - \frac{\gamma}{T} (a_\text{ref} - a)
    \right)
    +
    \frac{1}{\mu} \left(\frac{\gamma}{T} - 1\right) \dot \gamma
    \\
    &= -(\hat a - a)^2
    - \left(\frac{\gamma}{T} - 1\right) (\hat a - a)
         (a_\text{ref} - a)
    +
    \frac{1}{\mu} \left(\frac{\gamma}{T} - 1\right) \dot \gamma
\end{align}
$$
To make $\dot V$ negative semidefinite, we choose to adapt $\gamma$ as
$
\dot \gamma = \mu (\hat a - a) (a_\text{ref} - a)
$.
This way, $a$ tracks $\hat a$ without perfect knowledge of $T$.