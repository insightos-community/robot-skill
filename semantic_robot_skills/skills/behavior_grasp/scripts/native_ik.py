"""Recorded OmniGibson NumPy J-PARSE / orientation error for offline checks.

Source: existing native_ik_functions.py validation capture, 2026-09-18.
Runtime uses the corresponding Torch functions. No physics is simulated.
"""
import numpy as np
def _jparse_compute_numpy(
    jacobian,
    gamma=0.1,
    singular_direction_gain=1.0,
):
    """Batched J-PARSE pseudo-inverse (numpy implementation).

    Vectorized over the batch dimension for use with the IK controller.
    J-PARSE: Jacobian-based Projection Algorithm for Resolving Singularities
    Effectively. Clamps small singular values and adds smooth feedback in
    singular directions for singularity-robust IK control.
    Reference: https://github.com/armlabstanford/jparse
    """
    J = jacobian.astype(np.float64)
    N, m, _ = J.shape

    U, S, Vh = np.linalg.svd(J, full_matrices=False)

    sigma_max = S[:, 0]  # (N,)
    threshold = gamma * sigma_max  # (N,)

    nonsing_mask = S > threshold[:, None]  # (N, k)
    sing_mask = ~nonsing_mask

    # ---- J_safety: clamp singular values below threshold ----
    S_safety = np.clip(S, threshold[:, None], None)  # (N, k)
    J_safety = U @ (S_safety[:, :, None] * Vh)  # (N, m, n)

    # ---- J_proj: retain only non-singular directions ----
    S_proj = S * nonsing_mask  # (N, k)
    J_proj = U @ (S_proj[:, :, None] * Vh)

    # ---- Phi_singular: smooth feedback in singular directions ----
    n_sing = sing_mask.sum(axis=1)  # (N,)
    Phi_singular = np.zeros((N, m, m), dtype=np.float64)

    has_sing = n_sing > 0
    if has_sing.any():
        gains = np.full(m, singular_direction_gain, dtype=np.float64)
        Kp = np.diag(gains)  # (m, m)

        S_ratio = S[has_sing] / sigma_max[has_sing, None]  # (N_sing, k)
        phi_vals = (S_ratio / gamma) * sing_mask[has_sing]  # (N_sing, k)

        U_sing = U[has_sing]  # (N_sing, m, k)
        Phi_mat = np.apply_along_axis(np.diag, -1, phi_vals)  # (N_sing, k, k)
        Phi_singular[has_sing] = U_sing @ Phi_mat @ U_sing.transpose(0, 2, 1) @ Kp

    # ---- Combine ----
    J_safety_pinv = np.linalg.pinv(J_safety)
    J_proj_pinv = np.linalg.pinv(J_proj)

    J_parse = J_safety_pinv @ J_proj @ J_proj_pinv
    J_parse = J_parse + J_safety_pinv @ Phi_singular

    return J_parse.astype(np.float32)

def orientation_error(desired, current):
    """
    This function calculates a 3-dimensional orientation error vector for use in the
    impedance controller. It does this by computing the delta rotation between the
    inputs and converting that rotation to exponential coordinates (axis-angle
    representation, where the 3d vector is axis * angle).
    See https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation for more information.
    Optimized function to determine orientation error from matrices

    Args:
        desired (tensor): (..., 3, 3) where final two dims are 2d array representing target orientation matrix
        current (tensor): (..., 3, 3) where final two dims are 2d array representing current orientation matrix
    Returns:
        tensor: (..., 3) where final dim is (ax, ay, az) axis-angle representing orientation error
    """
    # convert input shapes
    input_shape = desired.shape[:-2]
    desired = desired.reshape(-1, 3, 3)
    current = current.reshape(-1, 3, 3)

    # grab relevant info
    rc1 = current[:, :, 0]
    rc2 = current[:, :, 1]
    rc3 = current[:, :, 2]
    rd1 = desired[:, :, 0]
    rd2 = desired[:, :, 1]
    rd3 = desired[:, :, 2]

    error = 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))

    # Reshape
    error = error.reshape(*input_shape, 3)

    return error


def _weighted_ik_step(q, jac, error, lower, upper, limited, velocity, dt, settings, solve):
    """Transform inputs / outputs around the supplied, unchanged J-PARSE solver."""
    scale, margin, max_weight, max_step = settings
    q, jac, error, lower, upper, velocity = [np.asarray(v, dtype=float) for v in (q, jac, error, lower, upper, velocity)]
    limited = np.asarray(limited, dtype=bool)
    if q.ndim != 1 or jac.shape != (6, q.size) or error.shape != (6,) or any((v.shape != q.shape for v in (lower, upper, limited, velocity))) or (not all((np.isfinite(v).all() for v in (q, jac, error, velocity)))) or np.any(velocity < 0) or (not 0 < dt < float('inf')):
        raise ValueError('Invalid differential IK state')
    if not np.isfinite(lower[limited]).all() or not np.isfinite(upper[limited]).all() or np.any(upper[limited] <= lower[limited]) or np.any(q[limited] < lower[limited] - 1e-06) or np.any(q[limited] > upper[limited] + 1e-06):
        raise ValueError('IK joint state outside valid limits')
    reference = q.copy()
    reference[limited] = np.clip(reference[limited], lower[limited], upper[limited])
    ratio = np.full(q.size, 0.5)
    ratio[limited] = np.minimum(reference[limited] - lower[limited], upper[limited] - reference[limited]) / (upper[limited] - lower[limited])
    weights = 1 + (max_weight - 1) * np.clip(1 - ratio / margin, 0, 1) ** 2
    d = 1 / np.sqrt(weights)
    weighted_jac = scale[:, None] * jac * d[None, :]
    weighted_error = scale * error
    raw = np.zeros_like(q) if not np.any(weighted_jac) else d * np.asarray(solve(weighted_jac, weighted_error))
    if raw.shape != q.shape or not np.isfinite(raw).all():
        raise ValueError('Nonfinite or invalid J-PARSE increment')
    caps = np.minimum(max_step, velocity * dt)
    if np.any(np.abs(reference - q) > caps + 1e-12):
        raise ValueError('IK limit roundoff recovery exceeds single-step allowance')
    caps = np.maximum(0, caps - np.abs(reference - q))
    moving = np.abs(raw) > 0
    alpha = min(1.0, float(np.min(caps[moving] / np.abs(raw[moving]))) if moving.any() else 1.0)
    for mask, room in ((limited & (raw > 0), upper - reference), (limited & (raw < 0), reference - lower)):
        if mask.any():
            alpha = min(alpha, float(np.min(room[mask] / np.abs(raw[mask]))))
    target = reference + alpha * raw
    diagnostic = dict(q=q.tolist(), weights=weights.tolist(), limit_margin_ratio=ratio.tolist(), delta_raw=raw.tolist(), delta_limited=(target - q).tolist(), step_scale=alpha)
    return (target, diagnostic)
