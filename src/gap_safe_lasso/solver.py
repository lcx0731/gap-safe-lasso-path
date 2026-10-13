"""Original implementation; equations and normalization are documented in README."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Certificate:
    primal: float
    dual: float
    gap: float
    gap_bound: float
    roundoff_padding: float
    radius: float
    kkt_violation: float
    theta: np.ndarray
    dual_correlations: np.ndarray


@dataclass(frozen=True)
class Result:
    coef: np.ndarray
    certificate: Certificate
    screened: np.ndarray
    converged: bool
    target_gap: float
    sweeps: int
    coordinate_visits: int
    certificate_checks: int
    trace: tuple[dict, ...]


class ConvergenceError(RuntimeError):
    """A run hit its iteration limit; the incomplete result remains inspectable."""

    def __init__(self, result):
        self.result = result
        super().__init__(
            f"Lasso did not converge in {result.sweeps} sweeps: "
            f"gap bound {result.certificate.gap_bound:.3g} > {result.target_gap:.3g}"
        )


def _problem(X, y, alpha):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or min(X.shape) == 0 or y.shape != (X.shape[0],):
        raise ValueError("X must be nonempty (n, p), and y must have shape (n,)")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must be finite")
    if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and strictly positive")
    return np.asfortranarray(X), y


def _coef(w, p):
    w = np.asarray(w, dtype=float)
    if w.shape != (p,) or not np.all(np.isfinite(w)):
        raise ValueError("coefficients must be a finite vector of length p")
    return w.copy()


def _certificate(X, y, w, alpha, residual):
    n, p = X.shape
    guard = 64 * np.finfo(float).eps * (n + p)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        correlations = X.T @ residual / n
        scale = min(1.0, alpha / max(float(np.max(np.abs(correlations))), alpha))
        # Move into the feasible region; recompute correlations rather than
        # assuming their floating-point scaling is exact.
        theta = residual * ((1 - guard) * scale / n)
        dual_correlations = X.T @ theta
        if np.max(np.abs(dual_correlations)) > alpha:
            raise FloatingPointError("dual feasibility could not be verified")
        primal = float(residual @ residual / (2 * n) + alpha * np.abs(w).sum())
        dual = float(y @ theta - (n / 2) * (theta @ theta))
        # Algebraically P-D, without subtracting two nearly equal objectives.
        mismatch = residual - n * theta
        gap = float(mismatch @ mismatch / (2 * n) + np.sum(
            np.abs(w) * (alpha - np.sign(w) * dual_correlations)
        ))
        padding = guard * max(1.0, primal, abs(dual))
        gap_bound = gap + padding
        kkt = np.where(
            w != 0,
            np.abs(correlations - alpha * np.sign(w)),
            np.maximum(np.abs(correlations) - alpha, 0.0),
        )
        scalars = [primal, dual, gap, gap_bound, float(np.max(kkt))]
        if not np.all(np.isfinite(scalars)) or gap < 0:
            raise FloatingPointError("non-finite or negative certificate")
    return Certificate(primal, dual, gap, gap_bound, padding,
                       float(np.sqrt(2 * gap_bound / n)), float(np.max(kkt)),
                       theta, dual_correlations)


def certificate(X, y, coef, alpha):
    """Evaluate the full problem, never just the retained coordinate subset."""
    X, y = _problem(X, y, alpha)
    w = _coef(coef, X.shape[1])
    with np.errstate(over="raise", invalid="raise"):
        return _certificate(X, y, w, alpha, y - X @ w)


def solve_lasso(X, y, alpha, *, initial=None, screening=True,
                rtol=1e-8, atol=1e-10, max_sweeps=20000):
    """Solve ||y-Xw||²/(2n)+alpha*||w||1, without an implicit intercept.

    Fresh screening masks are created on EVERY call, including warm starts.
    The stopping target is atol + rtol*||y||²/(2n). Unconverged runs raise.
    """
    X, y = _problem(X, y, alpha)
    n, p = X.shape
    if not isinstance(screening, (bool, np.bool_)):
        raise ValueError("screening must be boolean")
    if (not np.isfinite(rtol) or not np.isfinite(atol)
            or rtol < 0 or atol < 0 or rtol + atol <= 0):
        raise ValueError("tolerances must be finite, nonnegative, and not both zero")
    if isinstance(max_sweeps, bool) or not isinstance(max_sweeps, (int, np.integer)) or max_sweeps < 1:
        raise ValueError("max_sweeps must be a positive integer")
    w = np.zeros(p) if initial is None else _coef(initial, p)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        norms = np.sum(X * X, axis=0) / n
        target = float(atol + rtol * (y @ y) / (2 * n))
    if not np.all(np.isfinite(norms)) or not np.isfinite(target):
        raise FloatingPointError("problem scale exceeds float64 range")
    if np.any((norms == 0) & np.any(X != 0, axis=0)):
        raise FloatingPointError("a nonzero column norm underflowed; rescale the problem")
    active = norms > 0
    w[~active] = 0
    visits = 0
    trace = []
    for sweep in range(max_sweeps + 1):
        # Refresh incremental residuals and verify every feature at each sweep.
        with np.errstate(over="raise", invalid="raise"):
            residual = y - X @ w
        cert = _certificate(X, y, w, alpha, residual)
        if screening:
            upper = np.abs(cert.dual_correlations) + np.sqrt(n * norms) * cert.radius
            margin = 64 * np.finfo(float).eps * (n + p) * max(1.0, alpha)
            # Retaining a nonzero iterate is conservative and avoids a
            # discontinuous reset; it may be screened once a CD update zeros it.
            active &= ~((upper < alpha - margin) & (w == 0))
        trace.append({"sweep": sweep, "gap_bound": cert.gap_bound,
                      "primal": cert.primal, "retained": int(active.sum()),
                      "coordinate_visits": visits, "kkt_violation": cert.kkt_violation})
        if cert.gap_bound <= target or sweep == max_sweeps:
            result = Result(w.copy(), cert, ~active, cert.gap_bound <= target,
                            target, sweep, visits, len(trace), tuple(trace))
            if not result.converged:
                raise ConvergenceError(result)
            return result
        for j in np.flatnonzero(active):
            old = w[j]
            rho = float(X[:, j] @ residual / n + norms[j] * old)
            w[j] = np.sign(rho) * max(abs(rho) - alpha, 0.0) / norms[j]
            residual -= X[:, j] * (w[j] - old)
            visits += 1
    raise AssertionError("unreachable")


def lasso_path(X, y, alphas, **kwargs):
    """Warm-start a strictly decreasing positive alpha path; do not reuse masks."""
    alphas = np.asarray(alphas, dtype=float)
    if (alphas.ndim != 1 or alphas.size == 0 or not np.all(np.isfinite(alphas))
            or np.any(alphas <= 0) or np.any(np.diff(alphas) >= 0)):
        raise ValueError("alphas must be finite, positive, and strictly decreasing")
    if "initial" in kwargs:
        raise ValueError("lasso_path manages its own warm starts")
    results = []
    w = None
    for alpha in alphas:
        result = solve_lasso(X, y, float(alpha), initial=w, **kwargs)
        results.append(result)
        w = result.coef
    return results
