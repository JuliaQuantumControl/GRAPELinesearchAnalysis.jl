using Test
using QuantumControl: optimize
using QuantumControl.Functionals: J_a_fluence
using GRAPE
using GRAPELinesearchAnalysis: sample_functional, _get_step
import Optim

include("problem.jl")


# A callback that samples the functional at α = 0 and at the step width chosen
# by the optimizer, and records whether these samples match the functional for
# the guess and optimized pulses, and whether the workspace is unchanged
function make_check_sampling()
    J_prev = NaN
    records = []
    function check_sampling(wrk, iter)
        J = sum(wrk.J_parts)
        if iter > 0
            pulsevals = copy(wrk.pulsevals)
            J_parts = copy(wrk.J_parts)
            tau_vals = copy(wrk.result.tau_vals)
            fg_count = copy(wrk.fg_count)
            f_calls = wrk.result.f_calls
            α, d, is_search_direction = _get_step(wrk; rtol = 1e-8)
            (J_vals,) = sample_functional(copy(wrk.pulsevals_guess), wrk, (d, [0.0, α]))
            workspace_ok = (
                (wrk.pulsevals == pulsevals) &&
                (wrk.J_parts == J_parts) &&
                (wrk.result.tau_vals == tau_vals) &&
                (wrk.fg_count == fg_count) &&
                (wrk.result.f_calls == f_calls)
            )
            push!(
                records,
                (;
                    iter,
                    is_search_direction,
                    guess_ok = (J_vals[1] ≈ J_prev),
                    opt_ok = (J_vals[2] ≈ J),
                    workspace_ok
                )
            )
        end
        J_prev = J
        return nothing
    end
    return check_sampling, records
end


@testset "$name$(isempty(kwargs) ? "" : " with running cost")" for (name, optkwargs) in [
        ("LBFGSB", (;)),
        ("Optim.LBFGS", (; optimizer = Optim.LBFGS())),
    ],
    kwargs in [(;), (; J_a = J_a_fluence, lambda_a = 0.1)]

    problem = tls_problem()
    callback, records = make_check_sampling()
    res = optimize(problem; method = GRAPE, callback, optkwargs..., kwargs...)
    @test res.iter == 3
    @test length(records) == 3
    @test all(r.is_search_direction for r in records)
    @test all(r.guess_ok for r in records)
    @test all(r.opt_ok for r in records)
    @test all(r.workspace_ok for r in records)

end


@testset "J_T without tau" begin

    problem = tls_problem(J_T = J_T_without_tau, chi = chi_without_tau)
    callback, records = make_check_sampling()
    res = optimize(problem; method = GRAPE, callback)
    @test res.iter == 3
    @test length(records) == 3
    @test all(r.guess_ok for r in records)
    @test all(r.opt_ok for r in records)
    @test all(r.workspace_ok for r in records)

end


@testset "pulse update not along search direction" begin

    problem = tls_problem()
    callback, records = make_check_sampling()
    res = optimize(problem; method = GRAPE, callback, optimizer = Optim.ConjugateGradient())
    @test res.iter == 3
    # The first iteration of the conjugate gradient method is along the
    # negative gradient
    @test records[1].is_search_direction
    @test !records[2].is_search_direction
    @test all(r.guess_ok for r in records)
    @test all(r.opt_ok for r in records)
    @test all(r.workspace_ok for r in records)

end
