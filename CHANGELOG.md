# Release Notes

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For earlier releases, see the [GitHub Releases](https://github.com/JuliaQuantumControl/GRAPELinesearchAnalysis.jl/releases).


## [Unreleased]


## [v0.8.0] — 2026-09-17

* Changed: `GRAPELinesearchAnalysis` now requires GRAPE 1.2, which requires Optim 2 (instead of Optim 1) for an Optim.jl `optimizer`. Upgrade GRAPE and Optim together with `GRAPELinesearchAnalysis`. Since the package relies on functions of GRAPE that are only stable within bugfix releases, each release of `GRAPELinesearchAnalysis` supports a single minor version of GRAPE [[#3]]
* Changed: The minimum supported Julia version is now 1.10 (LTS)
* Changed: The package no longer depends on `Optim`, `LBFGSB`, `LineSearches`, `QuantumControl`, or `QuantumPropagators`
* Changed: The plots show the full optimization functional `J` that the optimizer minimizes, including running costs, instead of only the final-time functional `J_T`. This also supports a `J_T` that does not take a `tau` keyword argument
* Changed: The layout of the plots. The third panel compares the normalized negative gradient and search direction. The fourth panel shows the functional along the search direction as a function of the step width α. The bottom panel compares the functional along the negative gradient and along the search direction as a function of the step length (the norm of the pulse update), instead of using the same values of α for both directions
* Changed: The environment variable for using the active Plots backend is now spelled `GRAPE_LINESEARCH_ANALYSIS_USE_CURRENT_BACKEND`. Replace `GRAPE_LINESEARCH_ANALYSIS_USE_CURRENT_BACKED=1` with `GRAPE_LINESEARCH_ANALYSIS_USE_CURRENT_BACKEND=1`
* Changed: The ratio and the step width α in `print_ls_table` use exponential notation
* Added: Keyword argument `rtol` for `plot_linesearch`, the relative tolerance for checking whether the pulse update is along the search direction
* Fixed: The callback no longer changes the `tau_vals` of the optimization result. It restores the GRAPE workspace for the optimized pulses, as GRAPE 1.2 requires for an Optim.jl `optimizer`
* Fixed: For an optimizer that does not update the pulses along the search direction (e.g., `Optim.ConjugateGradient`), the callback no longer aborts the optimization with an assertion error. It warns once and plots the line search along the pulse update instead, with α = 1 for the actual update
* Fixed: Keyword arguments of `plot_linesearch` for the plotting functions were ignored
* Fixed: The callback threw a `MethodError` when switching back from the GR backend to a different active backend
* Fixed: The upper limit of the y-axis of the line search plots was fixed to 1, and the logarithmic scale did not check for non-positive values of the functional


## [v0.7.2] — 2024-09-04

* Changed: Depend directly on `QuantumControl` instead of `QuantumControlBase`


[Unreleased]: https://github.com/JuliaQuantumControl/GRAPELinesearchAnalysis.jl/compare/v0.8.0..HEAD
[v0.8.0]: https://github.com/JuliaQuantumControl/GRAPELinesearchAnalysis.jl/releases/tag/v0.8.0
[v0.7.2]: https://github.com/JuliaQuantumControl/GRAPELinesearchAnalysis.jl/releases/tag/v0.7.2
[#3]: https://github.com/JuliaQuantumControl/GRAPELinesearchAnalysis.jl/pull/3
