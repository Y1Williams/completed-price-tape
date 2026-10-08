"""Numerical invariants and optimizer failure handling, with no new experiments."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from tape_example.bounds import pulse_information
from tape_example.model import H, MODEL, gauss, matrices, mixture_quadrature
from tape_example.numerics import feasibility, optimize, starts, validate_metadata


class NumericalInvariants(unittest.TestCase):
    def test_density_has_unit_mass_and_is_positive(self):
        beta, weights = gauss(0.5, 2, 64)
        for theta in (-0.5, 0, 0.5):
            density = (2 / 3) * (1 + theta * (5 - 4 * beta) / 3)
            self.assertAlmostEqual(float(weights @ density), 1, places=14)
            endpoints = (2 / 3) * (1 + theta * (5 - 4 * np.array([0.5, 2])) / 3)
            self.assertGreaterEqual(float(endpoints.min()), 1 / 3)

    def test_changed_model_or_pulse_range_is_rejected(self):
        correct = {"model": MODEL.copy(), "tape_global_pulse_proved_for": "U>=2"}
        self.assertTrue(all(item["passed"] for item in validate_metadata(MODEL, correct)))
        with self.assertRaisesRegex(ValueError, "Numerical results"):
            validate_metadata({**MODEL, "tau": 1}, correct)
        with self.assertRaisesRegex(ValueError, "Continuum results"):
            validate_metadata(MODEL, {**correct, "model": {**MODEL, "R": 4}})
        with self.assertRaisesRegex(ValueError, "optimality range"):
            validate_metadata(MODEL, {**correct, "tape_global_pulse_proved_for": "U>=0"})

    def test_zero_mass_tangent_and_origin(self):
        _, weights = mixture_quadrature()
        self.assertLess(abs(float(weights.sum())), 1e-14)
        self.assertEqual(float(H(0)), 0.0)

    def test_uniform_control_brownian_variance(self):
        _, variance, tape = matrices(16, 2)
        weights = np.full(16, 1 / 16)
        self.assertAlmostEqual(float(weights @ variance @ weights), 1 / 3, places=14)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(tape).min()), -1e-14)

    def test_independent_pulse_formula_matches_matrices(self):
        for U in (0, 2, 6):
            with self.subTest(U=U):
                _, _, tape = matrices(16, U)
                pulse = np.r_[np.full(8, 1 / 8), np.zeros(8)]
                closed_form, _ = pulse_information(U)
                self.assertLess(abs(float(pulse @ tape @ pulse) - closed_form), 1e-11)

    def test_all_initial_schedules_are_admissible(self):
        for n in (16, 32, 64):
            self.assertTrue(all(feasibility(weights)["feasible"] for weights in starts(n)))

    def test_failed_optimizer_is_never_selected(self):
        weights = np.full(4, 0.25)
        failed = SimpleNamespace(x=weights, success=False, status=9,
                                 message="Iteration limit reached", nit=700, nfev=701)
        diagnostics = []
        with patch("tape_example.numerics.minimize", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "No successful feasible"):
                optimize(4, np.eye(4), np.eye(4), np.eye(4), "tape", [weights],
                         diagnostics=diagnostics)
        self.assertFalse(diagnostics[0]["starts"][0]["accepted"])
        self.assertTrue(diagnostics[0]["starts"][0]["feasible"])

    def test_successful_but_infeasible_optimizer_is_rejected(self):
        initial = np.full(4, 0.25)
        infeasible = SimpleNamespace(x=np.array([0.6, 0.4, 0, 0]), success=True, status=0,
                                     message="success", nit=1, nfev=2)
        with patch("tape_example.numerics.minimize", return_value=infeasible):
            with self.assertRaisesRegex(RuntimeError, "No successful feasible"):
                optimize(4, np.eye(4), np.eye(4), np.eye(4), "tape", [initial])


if __name__ == "__main__":
    unittest.main()
