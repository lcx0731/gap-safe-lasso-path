"""Dense, educational Lasso optimization with inspectable certificates."""

from .solver import ConvergenceError, certificate, lasso_path, solve_lasso

__all__ = ["ConvergenceError", "certificate", "lasso_path", "solve_lasso"]
