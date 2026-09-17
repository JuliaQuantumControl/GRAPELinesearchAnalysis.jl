# Release Notes

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For earlier releases, see the [tags](https://github.com/JuliaQuantumControl/GRAPELinesearchAnalysis.jl/tags) of the repository.


## [Unreleased]

* Changed: `GRAPELinesearchAnalysis` now requires GRAPE 1.2, which supports Optim 2 instead of Optim 1. Since the package relies on functions of GRAPE that are only stable within bugfix releases, each release supports a single minor version of GRAPE
* Changed: The minimum supported Julia version is now 1.10 (LTS)
* Changed: The package no longer depends on `Optim`, `LBFGSB`, `LineSearches`, `QuantumControl`, or `QuantumPropagators`
* Changed: The plots show the full optimization functional `J`, including running costs, instead of only the final-time functional `J_T`, evaluated with `GRAPE.evaluate_functional`. This also supports a `J_T` that does not take a `tau` keyword argument
* Changed: The plot along the negative gradient uses the step length (the norm of the pulse update) instead of the step width α of the search direction. The fourth panel now shows the functional along the search direction as a function of α
* Changed: The environment variable `GRAPE_LINESEARCH_ANALYSIS_USE_CURRENT_BACKED` is replaced by the correctly spelled `GRAPE_LINESEARCH_ANALYSIS_USE_CURRENT_BACKEND`
* Changed: The ratio and the step width α in `print_ls_table` use exponential notation
* Added: Keyword argument `rtol` for `plot_linesearch`
* Added: Tests and continuous integration
* Fixed: The callback no longer changes the `tau_vals` of the optimization result, and restores the GRAPE workspace for the optimized pulses, as required with GRAPE 1.2 for an Optim.jl optimizer
* Fixed: For an optimizer that does not update the pulses along the search direction (e.g., `Optim.ConjugateGradient`), the callback no longer aborts the optimization with an assertion error. It warns once and plots the line search along the pulse update instead
* Fixed: Keyword arguments of `plot_linesearch` for the plotting functions were ignored
* Fixed: The callback threw a `MethodError` when switching back from the GR backend to a different active backend
* Fixed: The upper limit of the y-axis of the line search plots was fixed to 1, and the logarithmic scale did not check for non-positive values of the functional

## [v0.7.2] — 2024-09-04

* Changed: Depend directly on `QuantumControl` instead of `QuantumControlBase`


[Unreleased]: https://github.com/JuliaQuantumControl/GRAPELinesearchAnalysis.jl/compare/v0.7.2..HEAD
[v0.7.2]: https://github.com/JuliaQuantumControl/GRAPELinesearchAnalysis.jl/releases/tag/v0.7.2
