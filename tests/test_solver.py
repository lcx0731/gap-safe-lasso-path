import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.linear_model import Lasso

from gap_safe_lasso import ConvergenceError, certificate, lasso_path, solve_lasso
from gap_safe_lasso.preprocessing import TrainingScale
from gap_safe_lasso.experiment import synthetic


def problem(seed=4, rho=0.8, n=60, p=90):
    rng = np.random.default_rng(seed)
    common = rng.normal(size=(n, 1))
    X = np.sqrt(rho) * common + np.sqrt(1 - rho) * rng.normal(size=(n, p))
    X -= X.mean(axis=0)
    X /= np.sqrt(np.mean(X ** 2, axis=0))
    y = 2 * X[:, 0] - X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y - y.mean()


@pytest.mark.parametrize("screening", [True, False])
def test_orthogonal_soft_threshold_oracle(screening):
    Q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(12, 6)))
    X = np.sqrt(12) * Q
    beta = np.array([2, -1, 0.3, -0.2, 0, 0.6])
    result = solve_lasso(X, X @ beta, 0.6, screening=screening)
    expected = np.sign(beta) * np.maximum(np.abs(beta) - 0.6, 0)
    assert_allclose(result.coef, expected, atol=1e-14)
    assert result.certificate.gap_bound <= result.target_gap


@pytest.mark.parametrize("rho", [0.0, 0.8, 0.98])
@pytest.mark.parametrize("fraction", [0.7, 0.1, 0.025])
def test_against_sklearn_and_screening_off(rho, fraction):
    X, y = problem(rho=rho)
    alpha = fraction * np.max(np.abs(X.T @ y)) / len(y)
    safe = solve_lasso(X, y, alpha, rtol=1e-9)
    plain = solve_lasso(X, y, alpha, screening=False, rtol=1e-9)
    ref = Lasso(alpha=alpha, fit_intercept=False, tol=1e-13,
                max_iter=100000).fit(X, y)
    ref_cert = certificate(X, y, ref.coef_, alpha)
    assert abs(safe.certificate.primal - ref_cert.primal) <= safe.target_gap
    assert abs(safe.certificate.primal - plain.certificate.primal) <= 2 * safe.target_gap
    assert_allclose(X @ safe.coef, X @ ref.coef_, atol=3e-4)
    assert_allclose(ref.coef_[safe.screened], 0, atol=1e-8)
    assert safe.coordinate_visits <= plain.coordinate_visits
    assert safe.certificate.kkt_violation < 1e-6


@pytest.mark.parametrize("seed", [1, 7, 51])
def test_dual_feasible_gap_identity_and_ball(seed):
    X, y = problem(seed=seed, n=25, p=12)
    w = np.random.default_rng(seed).normal(size=12)
    alpha = 0.2
    c = certificate(X, y, w, alpha)
    assert np.max(np.abs(X.T @ c.theta)) <= alpha
    assert_allclose(c.gap, c.primal - c.dual, atol=1e-12, rtol=1e-12)
    assert c.gap_bound >= c.gap >= 0
    ref = Lasso(alpha=alpha, fit_intercept=False, tol=1e-13,
                max_iter=100000).fit(X, y)
    optimum = certificate(X, y, ref.coef_, alpha)
    assert c.primal - optimum.primal <= c.gap_bound
    theta_star = (y - X @ ref.coef_) / len(y)
    assert np.linalg.norm(theta_star - c.theta) <= c.radius + 1e-10


def test_alpha_max_returns_zero_without_sweeps():
    X, y = problem()
    alpha = np.max(np.abs(X.T @ y)) / len(y)
    result = solve_lasso(X, y, alpha)
    assert_array_equal(result.coef, 0)
    assert result.sweeps == result.coordinate_visits == 0
    assert result.certificate_checks == 1


def test_zero_response_and_zero_design():
    a = solve_lasso(np.ones((5, 3)), np.zeros(5), 0.1)
    b = solve_lasso(np.zeros((5, 3)), np.arange(5), 0.1, initial=np.ones(3))
    assert_array_equal(a.coef, 0)
    assert_array_equal(b.coef, 0)
    assert a.screened.all() and b.screened.all()


def test_duplicate_columns_compare_predictions_not_unique_coefficients():
    z = np.arange(-4, 5, dtype=float)
    X = np.column_stack([z, z, -z, np.zeros_like(z)])
    y = 2 * z
    alpha = 0.3
    result = solve_lasso(X, y, alpha)
    reference = Lasso(alpha=alpha, fit_intercept=False, tol=1e-13).fit(X, y)
    assert_allclose(X @ result.coef, X @ reference.coef_, atol=1e-10)
    assert result.screened[-1]
    # Duplicates on the dual boundary cannot all be safely discarded.
    assert not result.screened[:3].any()


def test_path_resets_screened_mask_when_penalty_decreases():
    X = 2 * np.eye(4)
    y = X @ np.array([1, 0.4, -0.2, 0])
    high, low = lasso_path(X, y, [0.8, 0.1])
    assert high.screened[1] and high.screened[2]
    assert low.coef[1] != 0 and low.coef[2] != 0
    assert_allclose(low.coef, [0.9, 0.3, -0.1, 0], atol=1e-12)


def test_early_limit_raises_with_full_problem_certificate():
    X, y = problem(rho=0.98)
    with pytest.raises(ConvergenceError) as caught:
        solve_lasso(X, y, 0.01, max_sweeps=1, rtol=1e-12)
    result = caught.value.result
    assert not result.converged and result.sweeps == 1
    assert result.certificate.gap_bound > result.target_gap
    assert_allclose(certificate(X, y, result.coef, 0.01).gap, result.certificate.gap)


def test_trace_monotonicity_and_work_accounting():
    X, y = problem()
    for screening in [False, True]:
        result = solve_lasso(X, y, 0.2, screening=screening)
        assert np.all(np.diff([row["primal"] for row in result.trace]) <= 1e-12)
        assert np.all(np.diff([row["retained"] for row in result.trace]) <= 0)
        assert result.certificate_checks == result.sweeps + 1
        assert result.coordinate_visits == sum(row["retained"] for row in result.trace[:-1])
        if not screening:
            assert result.coordinate_visits == X.shape[1] * result.sweeps


def test_warm_start_does_not_modify_inputs():
    X, y = problem()
    initial = np.arange(X.shape[1], dtype=float) / 100
    copies = [a.copy() for a in [X, y, initial]]
    solve_lasso(X, y, 0.3, initial=initial)
    for a, b in zip([X, y, initial], copies):
        assert_array_equal(a, b)


def test_column_permutation_and_sign_invariance():
    X, y = problem(n=80, p=10)
    base = solve_lasso(X, y, 0.15, rtol=1e-10)
    perm = np.arange(10)[::-1]
    signs = (-1.0) ** np.arange(10)
    changed = solve_lasso(X[:, perm] * signs, y, 0.15, rtol=1e-10)
    assert_allclose(X @ base.coef, (X[:, perm] * signs) @ changed.coef, atol=1e-6)


@pytest.mark.parametrize("alpha", [0, -1, np.nan, np.inf])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        solve_lasso(np.eye(2), np.ones(2), alpha)


@pytest.mark.parametrize("X,y", [([], []), (np.empty((2, 0)), np.ones(2)),
                                    (np.eye(2), [1]), ([[np.nan]], [1]), ([[1]], [np.inf]),
                                    (np.eye(2), [[1], [2]])])
def test_invalid_problem(X, y):
    with pytest.raises(ValueError):
        solve_lasso(X, y, 0.1)


@pytest.mark.parametrize("kwargs", [{"rtol": -1}, {"atol": np.nan},
                                      {"rtol": 0, "atol": 0}, {"max_sweeps": 0},
                                      {"max_sweeps": 1.5}, {"max_sweeps": True},
                                      {"screening": "yes"}, {"initial": [1]},
                                      {"initial": [np.inf, 1]}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        solve_lasso(np.eye(2), np.ones(2), 0.1, **kwargs)


@pytest.mark.parametrize("alphas", [[], [0.1, 0.2], [0.1, 0.1], [0.1, 0], [np.nan], [[1]]])
def test_invalid_paths(alphas):
    with pytest.raises(ValueError):
        lasso_path(np.eye(2), np.ones(2), alphas)


def test_training_scaling_and_restored_intercept():
    X = np.array([[1, 3, 5], [2, 3, 10], [3, 3, 15]], dtype=float)
    y = np.array([2, 3, 7], dtype=float)
    scale = TrainingScale.fit(X, y)
    A = scale.transform(X)
    assert_allclose(A.mean(axis=0), 0, atol=1e-15)
    assert_allclose(np.mean(A ** 2, axis=0), [1, 0, 1])
    V = X + [100, 0, 200]
    assert np.max(np.abs(scale.transform(V).mean(axis=0))) > 10
    # Validation features cannot change the stored training transformation.
    assert_array_equal(scale.mean, [2, 3, 10])
    coef = np.array([0.5, 0, -0.7])
    raw, intercept = scale.raw_parameters(coef)
    assert_allclose(V @ raw + intercept, scale.target_mean + scale.transform(V) @ coef)


@pytest.mark.parametrize("scale", [1e200, 1e-200])
def test_unrepresentable_column_norm_fails_explicitly(scale):
    with pytest.raises(FloatingPointError):
        solve_lasso(np.eye(2) * scale, np.ones(2), 0.1)


@pytest.mark.parametrize("method", ["fit", "transform", "raw_parameters"])
def test_preprocessing_rejects_invalid_shape_or_nonfinite(method):
    scale = TrainingScale.fit(np.eye(2), np.ones(2))
    with pytest.raises(ValueError):
        if method == "fit":
            TrainingScale.fit([[np.nan]], [1])
        elif method == "transform":
            scale.transform([[1, 2, 3]])
        else:
            scale.raw_parameters([np.inf, 0])


@pytest.mark.parametrize("rho", [0.2, 0.9])
def test_synthetic_reproducibility_and_nonoverlapping_split_sizes(rho):
    first, second = synthetic(rho), synthetic(rho)
    for a, b in zip(first, second):
        assert_array_equal(a, b)
    assert [len(a) for a in first] == [240, 240, 160, 160, 400, 400]
