"""Seeded synthetic, held-out Lasso experiment; no data downloads."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso

from .preprocessing import TrainingScale
from .solver import certificate, lasso_path


def synthetic(rho, seed=20261013):
    """Thirty blocks of ten features, six nonzeros, and three separate splits."""
    rng = np.random.default_rng(seed)
    n, p = 800, 300
    block = rng.normal(size=(n, 30))
    independent = rng.normal(size=(n, p))
    Z = np.sqrt(rho) * np.repeat(block, 10, axis=1) + np.sqrt(1 - rho) * independent
    scales = np.geomspace(0.5, 3.0, p)
    means = np.linspace(-2.0, 2.0, p)
    X = Z * scales + means
    beta = np.zeros(p)
    beta[np.arange(6) * 10] = [2.0, -1.8, 1.5, -1.3, 1.0, -0.8]
    y = 4.0 + Z @ beta + 0.7 * rng.normal(size=n)
    return X[:240], y[:240], X[240:400], y[240:400], X[400:], y[400:]


def run_experiment():
    report = {"seed": 20261013, "data": "deterministic synthetic Gaussian block design",
              "train_n": 240, "validation_n": 160, "test_n": 400, "p": 300,
              "noise_variance": 0.49, "scenarios": []}
    traces = []
    for rho in [0.2, 0.9]:
        X, y, V, v, T, t = synthetic(rho)
        scaling = TrainingScale.fit(X, y)
        A, b = scaling.transform(X), y - scaling.target_mean
        AV, AT = scaling.transform(V), scaling.transform(T)
        alpha_max = float(np.max(np.abs(A.T @ b)) / len(b))
        alphas = alpha_max * np.geomspace(1.0, 0.025, 12)
        safe = lasso_path(A, b, alphas, screening=True, rtol=1e-9)
        plain = lasso_path(A, b, alphas, screening=False, rtol=1e-9)
        rows = []
        for alpha, s, u in zip(alphas, safe, plain):
            ref = Lasso(alpha=alpha, fit_intercept=False, tol=1e-12,
                        max_iter=100000, selection="cyclic").fit(A, b)
            ref_cert = certificate(A, b, ref.coef_, alpha)
            if ref_cert.gap_bound > max(1e-10, 0.1 * s.target_gap):
                raise AssertionError("reference fit did not reach a tighter gap tolerance")
            if abs(s.certificate.primal - ref_cert.primal) > 2 * s.target_gap:
                raise AssertionError("reference objective disagreement exceeds tolerance")
            if np.max(np.abs(ref.coef_[s.screened]), initial=0.0) > 1e-7:
                raise AssertionError("screened feature is nonzero in reference fit")
            if abs(s.certificate.primal - u.certificate.primal) > 2 * s.target_gap:
                raise AssertionError("screened and unscreened objectives disagree")
            rows.append({
                "alpha": float(alpha), "alpha_fraction": float(alpha / alpha_max),
                "primal": s.certificate.primal, "dual": s.certificate.dual,
                "gap_bound": s.certificate.gap_bound, "target_gap": s.target_gap,
                "kkt_violation": s.certificate.kkt_violation,
                "reference_objective_abs_difference": abs(s.certificate.primal - ref_cert.primal),
                "reference_train_prediction_max_abs_difference": float(np.max(np.abs(A @ (s.coef - ref.coef_)))),
                "screened": int(s.screened.sum()), "nonzero": int(np.count_nonzero(s.coef)),
                "safe_coordinate_visits": s.coordinate_visits,
                "plain_coordinate_visits": u.coordinate_visits,
                "safe_certificate_matvecs": 3 * s.certificate_checks,
                "plain_certificate_matvecs": 3 * u.certificate_checks,
                "validation_mse": float(np.mean((v - scaling.target_mean - AV @ s.coef) ** 2)),
            })
        chosen = int(np.argmin([row["validation_mse"] for row in rows]))
        selected = safe[chosen]
        raw, intercept = scaling.raw_parameters(selected.coef)
        np.testing.assert_allclose(T @ raw + intercept, AT @ selected.coef + scaling.target_mean,
                                   rtol=1e-12, atol=1e-12)
        report["scenarios"].append({
            "rho": rho, "alpha_max": alpha_max, "path": rows, "selected_index": chosen,
            "selected_alpha_fraction": rows[chosen]["alpha_fraction"],
            "selected_test_mse": float(np.mean((t - scaling.target_mean - AT @ selected.coef) ** 2)),
            "total_safe_coordinate_visits": sum(s.coordinate_visits for s in safe),
            "total_plain_coordinate_visits": sum(u.coordinate_visits for u in plain),
            "total_safe_certificate_matvecs": 3 * sum(s.certificate_checks for s in safe),
            "total_plain_certificate_matvecs": 3 * sum(u.certificate_checks for u in plain),
        })
        traces.append({"rho": rho, "alpha_fraction": 0.025, "target_gap": safe[-1].target_gap,
                       "safe": list(safe[-1].trace), "plain": list(plain[-1].trace)})
    return report, traces


def plot_results(report, traces, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for col, scenario in enumerate(report["scenarios"]):
        rows = scenario["path"]
        fractions = [row["alpha_fraction"] for row in rows]
        for key, label, color in [("safe_coordinate_visits", "Gap Safe", "#007F86"),
                                   ("plain_coordinate_visits", "Unscreened", "#B55D32")]:
            axes[0, col].plot(fractions, [row[key] for row in rows], "o-", label=label, color=color)
        axes[0, col].set(title=f"Within-block correlation = {scenario['rho']}", ylabel="Coordinate visits")
        axes[0, col].legend()
        axes[1, col].plot(fractions, [row["screened"] for row in rows], "o-", color="#007F86", label="Screened")
        axes[1, col].plot(fractions, [row["nonzero"] for row in rows], "s--", color="#B55D32", label="Nonzero coefficients")
        axes[1, col].set(ylabel="Feature count (p = 300)", ylim=(0, 310))
        axes[1, col].legend()
        for ax in axes[:, col]:
            ax.set_xscale("log")
            ax.invert_xaxis()
            ax.set_xlabel("alpha / alpha_max (decreasing warm-start path)")
            ax.grid(alpha=0.2)
    fig.suptitle("Screening removes coordinate work, not full certificate checks", fontsize=14)
    fig.savefig(output / "path-screening.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, scenario in zip(axes, traces):
        for key, label, color in [("safe", "Gap Safe", "#007F86"), ("plain", "Unscreened", "#B55D32")]:
            ax.semilogy([r["coordinate_visits"] for r in scenario[key]],
                        [r["gap_bound"] for r in scenario[key]], label=label, color=color)
        ax.axhline(scenario["target_gap"], linestyle=":", color="black", label="Stopping target")
        ax.set(title=f"Correlation = {scenario['rho']}; alpha / alpha_max = 0.025",
               xlabel="Cumulative coordinate visits (certificate cost excluded)", ylabel="Padded duality gap")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Optimization error check at the weakest penalty", fontsize=14)
    fig.savefig(output / "convergence.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    report, traces = run_experiment()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, payload in [("summary.json", report), ("traces.json", traces)]:
        (args.output / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    plot_results(report, traces, args.output)
    for scenario in report["scenarios"]:
        ratio = scenario["total_safe_coordinate_visits"] / scenario["total_plain_coordinate_visits"]
        print(f"rho={scenario['rho']}: coordinate-visit ratio={ratio:.3f}; "
              f"selected test MSE={scenario['selected_test_mse']:.4f}")


if __name__ == "__main__":
    main()
