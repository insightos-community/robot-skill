"""Retiming a fixed cubic joint path under velocity and acceleration limits.

The path parameter is independent of execution time. Backward propagation finds
controllable squared path speeds; forward propagation picks feasible local speeds.
No object geometry, preferred duration, or scene-specific multiplier is used.
"""
import math

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import linprog


def _extrema(path, squared_speed, acceleration):
    """Exact joint speed/acceleration extrema inside each cubic path cell."""
    vmax = np.zeros(path.c.shape[-1])
    amax = np.zeros_like(vmax)
    for i, length in enumerate(np.diff(path.x)):
        a, b, c, _ = path.c[:, i, :]
        u, x = acceleration[i], squared_speed[i]
        # q_ddot = q_ss * s_dot**2 + q_s * s_ddot.
        aa, ab, ac = 15*a*u, 6*a*x + 6*b*u, 2*b*x + c*u
        for j in range(len(vmax)):
            probes = [0., length]
            if abs(aa[j]) > 1e-15:
                vertex = -ab[j]/(2*aa[j])
                if 0 < vertex < length:
                    probes.append(vertex)
            amax[j] = max(amax[j], *(abs(aa[j]*z*z + ab[j]*z + ac[j]) for z in probes))
            probes = [0., length]
            for root in np.roots([aa[j], ab[j], ac[j]]):
                if abs(root.imag) < 1e-10 and 0 < root.real < length:
                    probes.append(float(root.real))
            vmax[j] = max(vmax[j], *(abs(3*a[j]*z*z + 2*b[j]*z + c[j])
                                    * math.sqrt(max(0., x + 2*u*z)) for z in probes))
    return vmax, amax


def retime_path(curve, velocity_limits, acceleration_limit, control_period):
    """Return uniform-period positions, with zero starting and terminal speed.

    Limits are inherited from the caller. The original geometric cubic curve is
    preserved, including its knots. A final analytic extremum calculation sets
    any necessary time correction, so feasibility does not depend on grid luck.
    """
    limits = np.asarray(velocity_limits, dtype=float)
    if (limits.ndim != 1 or not np.isfinite(limits).all() or np.any(limits <= 0)
            or not math.isfinite(acceleration_limit) or acceleration_limit <= 0
            or not math.isfinite(control_period) or control_period <= 0):
        raise ValueError('轨迹计时需要有限的正速度、加速度和控制周期')
    # At least two cells per original segment allow acceleration and deceleration
    # even when the segment is shorter than one control period.
    grid = np.concatenate([np.linspace(a, b, max(2, math.ceil((b-a)/control_period))+1)[:-1]
                           for a, b in zip(curve.x[:-1], curve.x[1:])] + [curve.x[-1:]])
    path = CubicHermiteSpline(grid, curve(grid), curve(grid, 1), axis=0)
    derivative = path(grid, 1)
    caps = np.min(np.divide(limits**2, derivative**2,
                           out=np.full_like(derivative, np.inf), where=abs(derivative) > 1e-12), axis=1)
    constraints = []
    for i, length in enumerate(np.diff(grid)):
        rows, bounds = [], []
        for fraction in (0., .5, 1.):
            first, second = path(grid[i]+fraction*length, 1), path(grid[i]+fraction*length, 2)
            matrix = np.column_stack([second*(1-fraction)-first/(2*length),
                                      second*fraction+first/(2*length)])
            rows.extend(matrix)
            rows.extend(-matrix)
            bounds.extend([acceleration_limit]*(2*len(limits)))
            cap = np.min(np.divide(limits**2, first**2,
                                  out=np.full_like(first, np.inf), where=abs(first) > 1e-12))
            if np.isfinite(cap):
                rows.append([1-fraction, fraction])
                bounds.append(cap)
        constraints.append((np.asarray(rows), np.asarray(bounds)))

    controllable = np.zeros(len(grid))
    for i in range(len(grid)-2, -1, -1):
        matrix, bounds = constraints[i]
        solution = linprog([-1., 0.], A_ub=matrix, b_ub=bounds,
                           bounds=[(0., caps[i] if np.isfinite(caps[i]) else None),
                                   (0., controllable[i+1])], method='highs')
        if not solution.success:
            raise ValueError(f'轨迹局部计时求解失败: {solution.message}')
        controllable[i] = max(0., solution.x[0])

    squared_speed = np.zeros(len(grid))
    for i, (matrix, bounds) in enumerate(constraints):
        residual = bounds-matrix[:, 0]*squared_speed[i]
        coefficient = matrix[:, 1]
        positive, negative = coefficient > 1e-12, coefficient < -1e-12
        high, low = controllable[i+1], 0.
        if positive.any():
            high = min(high, float(np.min(residual[positive]/coefficient[positive])))
        if negative.any():
            low = max(low, float(np.max(residual[negative]/coefficient[negative])))
        if high < low-1e-6:
            raise ValueError('轨迹局部计时没有连续可行速度')
        squared_speed[i+1] = max(0., high)

    denominator = np.sqrt(squared_speed[:-1])+np.sqrt(squared_speed[1:])
    if np.any(denominator <= 0):
        raise ValueError('轨迹包含无法通过的零速区段')
    durations = 2*np.diff(grid)/denominator
    acceleration = np.diff(squared_speed)/(2*np.diff(grid))
    vmax, amax = _extrema(path, squared_speed, acceleration)
    # Compute the correction from actual continuous extrema; no tuning factor.
    scale = max(1., float(np.max(vmax/limits)), math.sqrt(float(amax.max())/acceleration_limit))
    count = math.ceil(float(durations.sum())*scale/control_period)
    scale = count*control_period/float(durations.sum())
    squared_speed /= scale**2
    acceleration /= scale**2
    knots = np.r_[0., np.cumsum(durations*scale)]
    times = np.arange(count+1)*control_period
    indices = np.minimum(np.searchsorted(knots, times, side='right')-1, len(durations)-1)
    elapsed = times-knots[indices]
    parameter = grid[indices]+np.sqrt(squared_speed[indices])*elapsed+.5*acceleration[indices]*elapsed**2
    samples = path(np.clip(parameter, grid[0], grid[-1]))
    samples[0], samples[-1] = curve(curve.x[0]), curve(curve.x[-1])
    return samples
