module GRAPELinesearchAnalysis

using Printf
using QuantumControlBase.ConditionalThreads: @threadsif
using QuantumPropagators
using GRAPE
using Optim
using LBFGSB
using LinearAlgebra
using PyPlot: matplotlib


# Given a search direction (e.g., the negative gradient) and a vector of step
# widths α, return a vector of J_T for going downhill in the direction of the
# gradient by a distance α, from `pulsevals0`. Plotting the resul is useful for
# getting a feeling for the gradient and the linesearch.
function explore_linesearch(search_direction, α_vals, pulsevals0, wrk)
    J_T_vals = zeros(length(α_vals))
    J_T_func = wrk.kwargs[:J_T]
    N_T = length(wrk.result.tlist) - 1
    N = length(wrk.objectives)
    Ψ = wrk.fw_states
    τ = wrk.result.tau_vals
    for (i, α) in enumerate(α_vals)
        pulsevals = pulsevals0 + α * search_direction
        @threadsif wrk.use_threads for k = 1:N
            copyto!(Ψ[k], wrk.objectives[k].initial_state)
            for n = 1:N_T  # `n` is the index for the time interval
                local (G, dt) = GRAPE._fw_gen(pulsevals, k, n, wrk)
                propstep!(Ψ[k], G, dt, wrk.prop_wrk[k])
            end
            τ[k] = dot(wrk.objectives[k].target_state, Ψ[k])
        end
        J_T_vals[i] = J_T_func(Ψ, wrk.objectives; τ=τ)
    end
    return J_T_vals
end


function _get_linesearch_data(
        wrk,
        optimization_state::Optim.OptimizationState,
        optimizer_state::Optim.AbstractOptimizerState,
)
    # TODO: use wrk only (and compare with current implementation)
    α = optimizer_state.alpha # the stepwidth chosen by the linesearch
    lbfgs_direction = optimizer_state.s # the quasi-Newton update direction
    gradient_direction = -1 * optimizer_state.g_previous
    pulsevals_guess = optimizer_state.x_previous
    pulsevals_opt = optimizer_state.x

    return α, lbfgs_direction, gradient_direction, pulsevals_guess, pulsevals_opt

end


function _get_linesearch_data(
        wrk,
        iter_res::GRAPE.LBFGSB_Result,
        optimizer::LBFGSB.L_BFGS_B,
)
    # TODO: use wrk only (and compare with current implementation)
    lbfgs_direction = wrk.searchdirection
    gradient_direction =  -1 * wrk.gradient
    if norm(lbfgs_direction) ≈ 0.0
        # LBFGSB hasn't yet built up Hessian (first iteration)
        # TODO: this should be fixed in GRAPE
        lbfgs_direction = gradient_direction
    end
    pulsevals_opt = iter_res.x
    pulsevals_guess = iter_res.x_previous
    Δϵ = pulsevals_opt - pulsevals_guess
    norm_Δϵ = norm(Δϵ)
    norm_ls = norm(lbfgs_direction)
    proj_ls = Δϵ ⋅ lbfgs_direction
    cosθ = proj_ls / (norm_Δϵ * norm_ls)
    @assert cosθ ≈ 1.0  # search direction and Δϵ are parallel
    α = proj_ls / norm_ls^2
    @assert α ≈ optimizer.dsave[14]
    @assert norm(Δϵ - α * lbfgs_direction) < 1e-12

    return α, lbfgs_direction, gradient_direction, pulsevals_guess, pulsevals_opt

end


function plot_linesearch(outdir)

    function _plot_linesearch(
            wrk,
            optimization_state,
            optimizer_state,
            iteration, args...)

        (iteration == 0) && (return nothing)

        # TODO: linesearch data sources should be entirely in wrk, so we don't
        # need different methods.
        α, lbfgs_direction, gradient_direction, pulsevals_guess, pulsevals_opt = _get_linesearch_data(wrk, optimization_state, optimizer_state)
        α_vals = collect(range(0, 2α, length=100))
        J_α_gradient = explore_linesearch(gradient_direction, α_vals, pulsevals_guess, wrk)
        J_α_lbfgs = explore_linesearch(lbfgs_direction, α_vals, pulsevals_guess, wrk)
        J_T = wrk.result.J_T
        J_T_guess = wrk.result.J_T_prev

        fig, axs = matplotlib.pyplot.subplots(figsize = (7, 15), nrows=5)
        fig.suptitle(
            "Iteration $iteration; \$J_T\$ = $(@sprintf("%.2e", J_T_guess)) → $(@sprintf("%.2e", J_T))"
        )

        axs[1].plot(pulsevals_guess, label="guess")
        axs[1].plot(pulsevals_opt, label="opt")
        axs[1].set_xlabel("time step")
        axs[1].set_ylabel("pulse amplitude")
        axs[1].legend()

        axs[2].plot(gradient_direction, label="gradient")
        axs[2].set_xlabel("time step")
        axs[2].set_ylabel("gradient direction")
        axs[2].legend()

        axs[3].plot(gradient_direction, "--", label="gradient")
        axs[3].plot(lbfgs_direction, label="LBFGS direction")
        axs[3].set_xlabel("time step")
        axs[3].set_ylabel("LBFGS direction")
        axs[3].legend()

        axs[4].plot(α_vals, J_α_gradient, label="gradient")
        axs[4].scatter([0, ], [J_T_guess, ])
        axs[4].axhline(y=J_T_guess, color="black", linewidth=0.5)
        axs[4].axvline(x=α, color="black", linewidth=0.5)
        axs[4].set_xlabel("step width α")
        axs[4].set_ylabel("\$J_T\$")
        axs[4].set_title("Gradient Linesearch")
        axs[4].legend()

        axs[5].plot(α_vals, J_α_gradient, "--", label="gradient")
        axs[5].plot(α_vals, J_α_lbfgs, label="LBFGS")
        axs[5].scatter([0, α], [J_T_guess, J_T])
        axs[5].axhline(y=J_T_guess, color="black", linewidth=0.5)
        axs[5].axhline(y=J_T, color="black", linewidth=0.5)
        axs[5].axvline(x=α, color="black", linewidth=0.5)
        axs[5].set_xlabel("step width α")
        axs[5].set_ylabel("\$J_T\$")
        if J_T < 0.1
            axs[5].set_yscale("log")
            ylim = axs[5].get_ylim()
            axs[5].set_ylim((ylim[1], 1))
        else
            ylim = axs[5].get_ylim()
            axs[5].set_ylim((ylim[1], 1))
        end
        axs[5].set_title("LBFGS Linesearch α = $α")
        axs[5].legend()

        fig.tight_layout()

        fig.savefig(
            joinpath(outdir, @sprintf("linesearch-%03i.png", iteration))
        )
        return nothing
    end

    return _plot_linesearch
end

end # module
