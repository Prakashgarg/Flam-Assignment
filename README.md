
# FLAM R&D / AI Assignment Solution

## Problem Statement
We are tasked with estimating unknown parameters (θ, M, X) in the given parametric curve equations:

\[
x = (t \cos(\theta) - e^{M|t|} \sin(0.3t) \sin(\theta) + X) \\
y = (42 + t \sin(\theta) + e^{M|t|} \sin(0.3t) \cos(\theta))
\]

## Approach Overview
1. Loaded `xy_data.csv` containing points (x, y) sampled for t ∈ [6, 60].
2. Assumed data corresponds to uniformly spaced t values.
3. Implemented model equations in Python.
4. Defined **L1 loss** between observed and predicted points.
5. Used `scipy.optimize.differential_evolution` for global optimization and `minimize` for fine-tuning.

## Parameter Bounds
- θ ∈ (0°, 50°)
- M ∈ (-0.05, 0.05)
- X ∈ (0, 100)

## Results
| Parameter | Symbol | Estimated Value |
|------------|---------|----------------:|
| Theta (degrees) | θ | 28.12350 |
| M | M | 0.02142 |
| X | X | 54.89879 |

### Final Curve Equation
\[
\left(t * \cos(28.12350) - e^{0.02142|t|} \cdot \sin(0.3t) \sin(28.12350) + 54.89879,\ 
42 + t * \sin(28.12350) + e^{0.02142|t|} \cdot \sin(0.3t) \cos(28.12350) \right)
\]

This expression can be directly used in Desmos for verification.

## Files Included
- `fit_flam.py` — Python script to estimate parameters
- `xy_data.csv` — Provided data
- `fit_plot.png` — Visualization of fitted vs observed data
- `README.md` — Explanation of approach and results

## Author
Prepared by **Prakash Garg**
