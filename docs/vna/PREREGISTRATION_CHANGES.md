# Stage B measurement amendments (2026-10-15)

Round 2 changes the N1 gate statistic from rho = V / ||J^T e||² to
nu = V / ||e||². Permuting residuals preserves ||e||² but can reduce
||J^T e||², inflating null rho. We retain rho and lambda p-values as diagnostics;
real and null rows now also report V, e_norm2, nu and nu p-values for N1/N2.

This decision follows inspection of regression B5 data and precedes all CIFAR
measurement data. Regression B5 will be rerun for all six arms and reported
under the new gate; the old regression gate is superseded.

The listed 55, 60, …, 100 grid supplies twenty second-half points. Task 51
is added (first and mid) to satisfy the simultaneous requirement of 22.
The second-half B5 schedule now provides at least twenty points (22 for the
100-task regression). Fewer than ten points in the gate window is INVALID.
B3 measures the middle of all ten tasks and the end of tasks 2, 5 and 10;
end-of-task points remain reference-only.

Solver selection is auto: exact output space for n_out <= 4096, CG otherwise.
This is a computational choice with the same damping and mathematical target.
