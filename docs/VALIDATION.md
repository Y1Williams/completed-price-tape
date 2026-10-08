# Numerical verification

Run the tests and verify the saved calculation from the repository root:

```text
python -m unittest discover -s tests -v
python reproduce.py verify
```

Verification reevaluates the saved schedules and information values using the formulas in [METHODS](METHODS.md). The output is written to `output/verification.json`.

## Checks and tolerances

- The recovery density has unit mass and remains positive over the parameter interval. The perturbation integrates to zero.
- Each schedule satisfies the quantity constraint within $10^{-9}$ and the rate bound within $10^{-10}$.
- Saved information values are compared with direct evaluation using an absolute tolerance of $2\times10^{-11}$ and a relative tolerance of $10^{-9}$.
- Quadrature orders increase from 64, 12 and 80 to 96, 24 and 120 for recovery rates, execution intervals and recovery time. Additional envelope and pulse checks use at least 128 recovery rate nodes.
- The tests cover admissible controls, model consistency, Brownian cost variance, the pulse formula, optimizer acceptance and the data used by the demonstration.

Saved numerical reports are available for the [reference calculation](validation/verification.json), a [repeated calculation](validation/computed_verification.json) and their [comparison](validation/rerun_comparison.json). The [solver diagnostics](validation/optimizer_diagnostics.json) record the attempted initial controls, solver outcomes and selected schedules.

The verification workflow runs tests and saved result checks on Windows and Linux with Python 3.12. The pinned dependencies are listed in `requirements.txt`.

## Demonstration data

The English demonstration uses the 64 interval calculations at 23 positive recovery windows. At each saved node, it displays the original numerical information values. Between nodes, it linearly interpolates tape information while keeping cost information fixed. The displayed ratio and multiplier are calculated from those two information amounts. The page identifies calculated nodes and interpolation endpoints.

At $U/T=2$, the cost to tape information ratio is approximately 0.316 and its reciprocal is approximately 3.165. Displayed values use at most three decimal places. JSON data retain full precision.

## Interpretation

The blue curve evaluates an analytical bound over continuous schedules. Circles show the ratio obtained by separately optimizing the two feedback models on a finite grid. Numerical precision and the distinction between analytical inequalities and their floating point evaluation are explained in [METHODS](METHODS.md). The paper states the regularity conditions that connect information to calibration episode complexity.
