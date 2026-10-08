"""Deterministic multistart optimization and independent reference verification."""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.optimize import minimize

from .bounds import PAD, cost_envelope, pulse_information, sign_margins
from .model import DEFAULT_QUADRATURE, REFINED_QUADRATURE, MODEL, information, matrices

SUM_TOLERANCE = 1e-9
BOX_TOLERANCE = 1e-10
VALUE_ATOL = 2e-11
VALUE_RTOL = 1e-9
RESULT_FILES = ("numerical_results.json", "continuum_results.json")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def runtime_metadata():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "timestamp_utc": datetime.now(timezone.utc).isoformat()}


def feasibility(weights):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) < 2 or not np.all(np.isfinite(w)):
        return {"feasible": False, "sum_error": None, "box_violation": None}
    sum_error = abs(float(w.sum()) - 1)
    box_violation = max(0.0, float(-w.min()), float(w.max() - 2 / len(w)))
    return {"feasible": sum_error <= SUM_TOLERANCE and box_violation <= BOX_TOLERANCE,
            "sum_error": sum_error, "box_violation": box_violation}


def starts(n):
    """Fixed uniform, interval and cosine ordered initial schedules."""
    if n < 2 or n % 2:
        raise ValueError("Initial schedules require positive even n.")
    result = [np.ones(n) / n]
    k = n // 2
    for shift in range(0, n - k + 1, max(1, n // 8)):
        w = np.zeros(n)
        w[shift:shift + k] = 2 / n
        result.append(w)
    for j in range(1, 6):
        order = np.argsort(np.cos(j * np.pi * (np.arange(n) + 0.5) / n))
        w = np.zeros(n)
        w[order[:k]] = 2 / n
        result.extend([w, 2 / n - w])
    return result


def optimize(n, A, V, P, kind, initials=None, *, diagnostics=None, label=None):
    """Select the best successful, feasible SLSQP run from fixed starts.

    This is local optimization and does not certify global optimality.
    Every attempted start is saved to the supplied diagnostics list.
    """
    if kind not in ("cost", "tape"):
        raise ValueError("kind must be 'cost' or 'tape'.")
    scale = max(float(np.max(np.abs(P))), 1e-15)

    def value_grad(w):
        if kind == "tape":
            return float(w @ P @ w), 2 * P @ w
        Aw, Vw = A @ w, V @ w
        d, variance = float(w @ Aw), float(w @ Vw)
        return d * d / variance, 4 * d / variance * Aw - 2 * d * d / variance ** 2 * Vw

    def fun(w):
        value, gradient = value_grad(w)
        return -value / scale, -gradient / scale

    records = []
    best = None
    for index, initial in enumerate(starts(n) if initials is None else initials):
        initial = np.asarray(initial, dtype=float)
        if len(initial) != n or not feasibility(initial)["feasible"]:
            raise ValueError(f"Invalid initial control for {label or kind}, start {index}.")
        result = minimize(
            fun, initial, jac=True, method="SLSQP", bounds=[(0, 2 / n)] * n,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                         "jac": lambda w: np.ones(n)},
            options={"ftol": 1e-12, "maxiter": 700},
        )
        check = feasibility(result.x)
        value = float(value_grad(result.x)[0]) if np.all(np.isfinite(result.x)) else float("nan")
        accepted = bool(result.success and check["feasible"] and np.isfinite(value))
        record = {
            "start_index": index, "initial_value": float(value_grad(initial)[0]),
            "success": bool(result.success), "accepted": accepted,
            "status": int(result.status), "message": str(result.message),
            "iterations": int(result.nit), "function_evaluations": int(result.nfev),
            "value": value if np.isfinite(value) else None, **check,
            "weights": result.x.tolist() if np.all(np.isfinite(result.x)) else None,
        }
        records.append(record)
        if accepted and (best is None or value > best["value"]):
            best = {"value": value, "weights": result.x.tolist(), "success": True,
                    "message": str(result.message), "selected_start": index}
    if diagnostics is not None:
        diagnostics.append({"label": label or kind, "n": n, "kind": kind,
                            "selected_start": None if best is None else best["selected_start"],
                            "accepted_starts": sum(r["accepted"] for r in records),
                            "attempted_starts": len(records), "starts": records})
    if best is None:
        raise RuntimeError(f"No successful feasible optimizer result for {label or kind} (n={n}).")
    return best


def evaluate_continuum(previous, weights):
    """Evaluate the continuum formulas for one supplied control."""
    w = np.asarray(weights, dtype=float)
    n = len(w)
    A, V, P = matrices(n, 2)
    bounds = cost_envelope(w, A, V)
    check = cost_envelope(w, A, V, m=32768, beta_order=64)
    us = np.linspace(0.05, 6, 240)
    upper, lower, pulse = [], [], []
    for U in us:
        j, infinite = pulse_information(float(U))
        upper.append(float(min(1, bounds["upper"] / (j - PAD))))
        lower.append(float(bounds["lower"] / ((j if U >= 2 else infinite) + PAD)))
        pulse.append(j)
    j2, infinite = pulse_information(2)
    j6, _ = pulse_information(6)
    values = information(w, A, V, P)
    pulse_w = np.r_[np.full(n // 2, 2 / n), np.zeros(n // 2)]
    cross_error = abs(j2 - float(pulse_w @ P @ pulse_w))
    margins = sign_margins()
    if cross_error >= 1e-11:
        raise RuntimeError("Pulse formula and block-matrix quadrature disagree.")
    if not bounds["lower"] < bounds["upper"] < 0.003:
        raise RuntimeError("The expected continuous cost envelope failed.")
    if not bounds["upper"] < check["upper"]:
        raise RuntimeError("The fine cost envelope did not improve the coarse envelope.")
    if min(margins.values()) <= 0:
        raise RuntimeError("An analytical sign margin is nonpositive.")
    return {
        "model": MODEL.copy(), "cost_control_weights": w.tolist(), "cost_control_blocks": n,
        "cost_continuum_bounds": bounds, "coarser_envelope_check": check,
        "tape_infinite_optimum": infinite, "tape_exact_at_U2": j2, "tape_exact_at_U6": j6,
        "tape_global_pulse_proved_for": "U>=2", "pulse_quadrature_crosscheck_error": cross_error,
        **margins,
        "same_cost_design": {"J_cost": values["J_cost"], "J_tape_at_U2": values["J_tape"],
                             "cost_share": values["J_cost"] / values["J_tape"],
                             "additional_share": 1 - values["J_cost"] / values["J_tape"]},
        "ratio_at_U2": [bounds["lower"] / (j2 + PAD), bounds["upper"] / (j2 - PAD)],
        "ratio_at_U6": [bounds["lower"] / (j6 + PAD), bounds["upper"] / (j6 - PAD)],
        "U": us.tolist(), "ratio_lower": lower, "ratio_upper": upper,
        "pulse_information": pulse, "numerical_grid_screen": previous["grids"],
    }


def compute(output_dir):
    """Rerun the existing three grid screens and n=128 cost/envelope evaluation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    started = time.perf_counter()
    status = "failed"
    try:
        allrows = []
        us = np.r_[0, np.linspace(0.1, 2, 20), 3, 4, 6]
        for n in (16, 32, 64):
            A, V, P = matrices(n, 2)
            cost = optimize(n, A, V, P, "cost", diagnostics=diagnostics, label=f"n{n}/cost")
            tape = optimize(n, A, V, P, "tape", diagnostics=diagnostics, label=f"n{n}/tape_U2")
            print(f"n={n}: cost={cost['value']:.3f}, tape(U=2)={tape['value']:.3f}, "
                  f"ratio={cost['value'] / tape['value']:.3f}", flush=True)
            rows = []
            for U in us:
                _, _, PP = matrices(n, float(U))
                initials = starts(n) if n == 16 else [
                    np.array(tape["weights"]), np.array(cost["weights"]), *starts(n)[:4]
                ]
                result = optimize(n, A, V, PP, "tape", initials, diagnostics=diagnostics,
                                  label=f"n{n}/tape_U{float(U):.17g}")
                rows.append({"U": float(U), "tape": result, "ratio": cost["value"] / result["value"]})
            allrows.append({"n": n, "cost": cost, "tape_at_2": tape, "rows": rows})
        numerical = {**MODEL, "method": "deterministic quadrature and multistart constrained optimization",
                     "global_optimality": "not certified by local optimization", "grids": allrows}
        A, V, P = matrices(128, 2)
        initial = np.repeat(allrows[-1]["cost"]["weights"], 2) / 2
        cost128 = optimize(128, A, V, P, "cost", [initial], diagnostics=diagnostics, label="n128/cost")
        continuum = evaluate_continuum(numerical, cost128["weights"])
        write_json(output_dir / RESULT_FILES[0], numerical)
        write_json(output_dir / RESULT_FILES[1], continuum)
        status = "passed"
    finally:
        write_json(output_dir / "optimizer_diagnostics.json", {
            **runtime_metadata(), "status": status, "elapsed_seconds": time.perf_counter() - started,
            "quadrature": DEFAULT_QUADRATURE.as_dict(), "sum_tolerance": SUM_TOLERANCE,
            "box_tolerance": BOX_TOLERANCE, "require_optimizer_success": True,
            "method": "SLSQP", "ftol": 1e-12, "maxiter": 700,
            "global_optimality": "not certified by local optimization", "runs": diagnostics,
        })
    print(f"n=128: ratio(U=2)=[{continuum['ratio_at_U2'][0]:.3f}, "
          f"{continuum['ratio_at_U2'][1]:.3f}], "
          f"same-design cost share={continuum['same_cost_design']['cost_share']:.3f}", flush=True)
    return numerical, continuum


def _comparison(label, actual, expected, atol=VALUE_ATOL, rtol=VALUE_RTOL):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if actual.shape != expected.shape:
        return {"label": label, "passed": False, "reason": "shape mismatch"}
    error = np.abs(actual - expected)
    return {"label": label, "passed": bool(np.all(np.isfinite(actual)) and
                                             np.all(error <= atol + rtol * np.abs(expected))),
            "max_absolute_error": float(np.max(error)), "atol": atol, "rtol": rtol}


def _continuum_comparisons(actual, expected, *, atol=VALUE_ATOL, rtol=VALUE_RTOL):
    result = []
    for key in ("cost_continuum_bounds", "coarser_envelope_check", "same_cost_design"):
        for field in expected[key]:
            result.append(_comparison(f"continuum/{key}/{field}", actual[key][field],
                                      expected[key][field], atol, rtol))
    for key in ("tape_infinite_optimum", "tape_exact_at_U2", "tape_exact_at_U6",
                "autocorrelation_derivative_margin", "H_second_derivative_at_2",
                "ratio_at_U2", "ratio_at_U6", "U", "ratio_lower", "ratio_upper", "pulse_information"):
        result.append(_comparison(f"continuum/{key}", actual[key], expected[key], atol, rtol))
    return result


def validate_metadata(numerical, continuum):
    """Require saved metadata to describe the fixed model evaluated by this code."""
    if {key: numerical.get(key) for key in MODEL} != MODEL:
        raise ValueError("Numerical results contain different model metadata.")
    if continuum.get("model") != MODEL:
        raise ValueError("Continuum results contain different model metadata.")
    if continuum.get("tape_global_pulse_proved_for") != "U>=2":
        raise ValueError("The pulse optimality range must be U>=2.")
    return [{"label": "model/numerical_metadata", "passed": True},
            {"label": "model/continuum_metadata", "passed": True},
            {"label": "model/pulse_optimality_range", "passed": True}]


def verify(data_dir, output_dir, *, quadrature=DEFAULT_QUADRATURE, refined=REFINED_QUADRATURE,
           report_name="verification.json"):
    """Reevaluate saved controls without calling an optimizer.

    Check objective values, feasibility, ratios, all continuum curves, a finer
    quadrature and the independent closed pulse formula. This is numerical QA,
    not interval-arithmetic certification of the continuum enclosure.
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    numerical, continuum = [json.loads((data_dir / name).read_text(encoding="utf-8")) for name in RESULT_FILES]
    checks = validate_metadata(numerical, continuum)
    controls, quadrature_checks = [], []

    @lru_cache(maxsize=None)
    def matrix_pair(n, U):
        base = matrices(n, U, quadrature=quadrature)
        fine = matrices(n, U, quadrature=refined)
        matrix_checks = [_comparison(f"quadrature/n{n}/U{U}/{key}", x, y)
                         for key, x, y in zip(("A", "V", "P"), base, fine)]
        quadrature_checks.extend(matrix_checks)
        return base, fine

    def control_check(label, weights, n, U, kind, stored_value):
        if len(weights) != n:
            raise ValueError(f"Control length mismatch for {label}.")
        feasible = feasibility(weights)
        controls.append({"label": label, "n": n, "U": U, **feasible})
        base, fine = matrix_pair(n, U)
        values = information(weights, *base)
        refined_values = information(weights, *fine)
        checks.append(_comparison(label + "/stored_value", values[kind], stored_value))
        checks.append(_comparison(label + "/refined_value", values[kind], refined_values[kind]))
        return values[kind]

    for grid in numerical["grids"]:
        n = grid["n"]
        cost = control_check(f"n{n}/cost", grid["cost"]["weights"], n, 2,
                             "J_cost", grid["cost"]["value"])
        control_check(f"n{n}/tape_U2", grid["tape_at_2"]["weights"], n, 2,
                      "J_tape", grid["tape_at_2"]["value"])
        for row in grid["rows"]:
            value = control_check(f"n{n}/tape_U{row['U']:.17g}", row["tape"]["weights"], n,
                                  row["U"], "J_tape", row["tape"]["value"])
            checks.append(_comparison(f"n{n}/ratio_U{row['U']:.17g}", cost / value, row["ratio"]))
    n = continuum["cost_control_blocks"]
    w = continuum["cost_control_weights"]
    control_check("continuum/cost_control", w, n, 2, "J_cost", continuum["same_cost_design"]["J_cost"])
    recomputed = evaluate_continuum(numerical, w)
    checks.extend(_continuum_comparisons(recomputed, continuum))
    checks.append({"label": "continuum/numerical_grid_screen",
                   "passed": continuum["numerical_grid_screen"] == numerical["grids"]})
    A, V, P = matrix_pair(n, 2)[0]
    fine_A, fine_V, _ = matrix_pair(n, 2)[1]
    fine_envelope = cost_envelope(w, fine_A, fine_V, beta_order=max(128, refined.beta_order))
    base_envelope = cost_envelope(w, A, V)
    for field in ("lower", "upper", "d", "variance", "support_slack"):
        checks.append(_comparison(f"continuum/envelope_beta_refinement/{field}",
                                  base_envelope[field], fine_envelope[field]))
    pulse_w = np.r_[np.full(n // 2, 2 / n), np.zeros(n // 2)]
    for U in (0, 2, 6):
        pulse, infinite = pulse_information(U)
        fine_pulse, fine_infinite = pulse_information(U, order=max(128, refined.beta_order))
        pulse_P = matrix_pair(n, U)[0][2]
        checks.append(_comparison(f"pulse/U{U}/matrix", float(pulse_w @ pulse_P @ pulse_w), pulse))
        checks.append(_comparison(f"pulse/U{U}/beta_refinement", [pulse, infinite], [fine_pulse, fine_infinite]))
    passed = all(item["passed"] for item in checks + quadrature_checks) and all(c["feasible"] for c in controls)
    report = {
        **runtime_metadata(), "status": "passed" if passed else "failed", "optimizer_called": False,
        "source_files": list(RESULT_FILES),
        "sum_tolerance": SUM_TOLERANCE, "box_tolerance": BOX_TOLERANCE,
        "value_atol": VALUE_ATOL, "value_rtol": VALUE_RTOL,
        "base_quadrature": quadrature.as_dict(), "refined_quadrature": refined.as_dict(),
        "numerical_allowance": PAD,
        "numerical_interpretation": "PAD is a numerical allowance. Total floating point and quadrature errors require separate bounds.",
        "control_count": len(controls), "controls": controls,
        "checks": checks, "quadrature_checks": quadrature_checks,
    }
    write_json(output_dir / report_name, report)
    if not passed:
        failed = [item["label"] for item in checks + quadrature_checks if not item["passed"]]
        failed.extend(item["label"] for item in controls if not item["feasible"])
        raise RuntimeError("Verification failed: " + ", ".join(failed))
    print(f"Verified {len(controls)} saved controls and {len(checks) + len(quadrature_checks)} checks. "
          "No optimization was run.", flush=True)
    return report


def compare_rerun(reference_dir, output_dir):
    """Compare objective values and displayed quantities, not optimizer weights."""
    reference_dir, output_dir = Path(reference_dir), Path(output_dir)
    ref_num, ref_cont = [json.loads((reference_dir / name).read_text(encoding="utf-8")) for name in RESULT_FILES]
    new_num, new_cont = [json.loads((output_dir / name).read_text(encoding="utf-8")) for name in RESULT_FILES]
    checks = []
    # Cross-platform local solvers can converge to slightly different controls.
    # Compare their scientific values using explicit rerun tolerances.
    atol, rtol = 5e-9, 5e-6
    checks.append(_comparison("grid_sizes", [grid["n"] for grid in new_num["grids"]],
                              [grid["n"] for grid in ref_num["grids"]], 0, 0))
    for old, new in zip(ref_num["grids"], new_num["grids"]):
        checks.append(_comparison(f"n{old['n']}/cost", new["cost"]["value"], old["cost"]["value"], atol, rtol))
        checks.append(_comparison(f"n{old['n']}/tape_U2", new["tape_at_2"]["value"], old["tape_at_2"]["value"], atol, rtol))
        checks.append(_comparison(f"n{old['n']}/U", [r["U"] for r in new["rows"]], [r["U"] for r in old["rows"]], 0, 0))
        checks.append(_comparison(f"n{old['n']}/tape_curve", [r["tape"]["value"] for r in new["rows"]],
                                  [r["tape"]["value"] for r in old["rows"]], atol, rtol))
        checks.append(_comparison(f"n{old['n']}/ratio_curve", [r["ratio"] for r in new["rows"]],
                                  [r["ratio"] for r in old["rows"]], 5e-6, 5e-5))
    # Supporting-function gradients and a near-optimal control can vary more
    # than objective values. The comparison tolerances are saved per quantity.
    for key in ("cost_continuum_bounds", "coarser_envelope_check"):
        for field in ("lower", "upper"):
            checks.append(_comparison(f"continuum/{key}/{field}", new_cont[key][field], ref_cont[key][field], atol, rtol))
    for key in ("ratio_at_U2", "ratio_at_U6", "ratio_lower", "ratio_upper", "pulse_information"):
        checks.append(_comparison(f"continuum/{key}", new_cont[key], ref_cont[key], 5e-6, 5e-5))
    for key in ("J_cost", "J_tape_at_U2", "cost_share", "additional_share"):
        checks.append(_comparison(f"continuum/same_cost_design/{key}", new_cont["same_cost_design"][key],
                                  ref_cont["same_cost_design"][key], 5e-6, 5e-5))
    passed = all(item["passed"] for item in checks)
    report = {**runtime_metadata(), "status": "passed" if passed else "failed", "checks": checks,
              "note": "Local optimizer weights need not be identical. Tolerances apply to scientific values and are not global-optimality certificates."}
    write_json(output_dir / "rerun_comparison.json", report)
    if not passed:
        raise RuntimeError("Rerun comparison failed. Inspect rerun_comparison.json.")
    print(f"Fresh results agree with the reference in {len(checks)} tolerance checks.", flush=True)
    return report
