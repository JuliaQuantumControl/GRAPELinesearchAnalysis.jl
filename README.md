# GRAPELinesearchAnalysis.jl

A package to analyze and debug [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)/[LBFGS](https://julianlsolvers.github.io/Optim.jl/stable/#algo/lbfgs/) linesearch behavior inside [GRAPE.jl](https://github.com/JuliaQuantumControl/GRAPE.jl).

The package allows to explore how the value of the functional varies both in the direction of the gradient and in the search direction chosen by LBFGS for different step widths, around the step width α chosen by Optim.jl.

## Installation

The `GRAPELinesearchAnalysis` package is not registered. Thus, you can only install it from Github

## Usage

The package is designed to be used as an `info_hook` for a [GRAPE.jl](https://github.com/JuliaQuantumControl/GRAPE.jl) optimization.
