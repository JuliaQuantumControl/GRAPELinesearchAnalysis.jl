using Test
using IOCapture: IOCapture
using QuantumControl: optimize
using QuantumControl.Functionals: J_a_fluence
using GRAPE
using GRAPELinesearchAnalysis: plot_linesearch, print_ls_table
import Optim
import Plots

include("problem.jl")

ENV["GKSwstype"] = "100"  # headless GR

const PNG_FILES = ["linesearch-001.png", "linesearch-002.png", "linesearch-003.png"]


function test_unchanged_result(res, res_ref)
    @test res.iter == res_ref.iter
    @test res.J_T == res_ref.J_T
    @test res.J_a == res_ref.J_a
    @test res.tau_vals == res_ref.tau_vals
    @test res.optimized_controls == res_ref.optimized_controls
    @test res.f_calls == res_ref.f_calls
    @test res.fg_calls == res_ref.fg_calls
end


@testset "$name$(isempty(kwargs) ? "" : " with running cost")" for (name, optkwargs) in [
        ("LBFGSB", (;)),
        ("Optim.LBFGS", (; optimizer = Optim.LBFGS())),
    ],
    kwargs in [(;), (; J_a = J_a_fluence, lambda_a = 0.1)]

    problem = tls_problem()
    res_ref = optimize(problem; method = GRAPE, optkwargs..., kwargs...)
    outdir = mktempdir()
    callback = plot_linesearch(outdir; samples = 3)
    res = optimize(problem; method = GRAPE, callback, optkwargs..., kwargs...)
    test_unchanged_result(res, res_ref)
    @test sort(readdir(outdir)) == PNG_FILES
    @test length(res.records) == 3
    @test [r[1] for r in res.records] == [1, 2, 3]
    @test all(r[end] > 0 for r in res.records)  # step width α

    captured = IOCapture.capture() do
        print_ls_table(res)
    end
    lines = filter(!isempty, split(captured.output, "\n"))
    @test length(lines) == 4
    @test contains(lines[1], "step α")

end


@testset "print_iters" begin

    # `print_iters` runs after the callback and reads the workspace
    problem = tls_problem(print_iters = true)
    captured_ref = IOCapture.capture() do
        optimize(problem; method = GRAPE, optimizer = Optim.LBFGS())
    end
    outdir = mktempdir()
    captured = IOCapture.capture() do
        optimize(
            problem;
            method = GRAPE,
            optimizer = Optim.LBFGS(),
            callback = plot_linesearch(outdir; samples = 3)
        )
    end
    strip_secs(output) = [rsplit(line, limit = 2)[1] for line in split(output, "\n")[1:5]]
    @test strip_secs(captured.output) == strip_secs(captured_ref.output)
    @test sort(readdir(outdir)) == PNG_FILES

end


@testset "pulse update not along search direction" begin

    problem = tls_problem()
    res_ref = optimize(problem; method = GRAPE, optimizer = Optim.ConjugateGradient())
    outdir = mktempdir()
    callback = plot_linesearch(outdir; samples = 3)
    captured = IOCapture.capture() do
        optimize(problem; method = GRAPE, callback, optimizer = Optim.ConjugateGradient())
    end
    res = captured.value
    test_unchanged_result(res, res_ref)
    @test sort(readdir(outdir)) == PNG_FILES
    msg = "The pulse update is not along the search direction"
    @test count(msg, captured.output) == 1
    @test res.records[2][end] == 1.0  # α

end


@testset "non-GR backend" begin

    Plots.unicodeplots()
    try
        problem = tls_problem()
        outdir = mktempdir()
        captured = IOCapture.capture() do
            optimize(
                problem;
                method = GRAPE,
                callback = plot_linesearch(outdir; samples = 3, verbose = true)
            )
        end
        @test Plots.backend() isa Plots.UnicodePlotsBackend
        @test sort(readdir(outdir)) == PNG_FILES
        @test count("Written GRAPELinesearchAnalysis plot", captured.output) == 3
    finally
        Plots.gr()
    end

end
