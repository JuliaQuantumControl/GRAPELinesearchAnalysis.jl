module GRAPELinesearchAnalysis

using Printf
using QuantumControlBase.ConditionalThreads: @threadsif
using QuantumPropagators
using GRAPE
using Optim
using LBFGSB
using LinearAlgebra
using Plots
using Plots.PlotMeasures: px


# Given a search direction (e.g., the negative gradient) and a vector of step
# widths α, return a vector of J_T for going downhill in the direction of the
# gradient by a distance α, from `pulsevals0`. Plotting the resul is useful for
# getting a feeling for the gradient and the linesearch.
function explore_linesearch(search_direction, α_vals, pulsevals0, wrk)
    J_T_vals = zeros(length(α_vals))
    J_T_func = wrk.kwargs[:J_T]
    N_T = length(wrk.result.tlist) - 1
    N = length(wrk.objectives)
    τ = wrk.result.tau_vals
    pulsevals_orig = copy(wrk.pulsevals)
    for (i, α) in enumerate(α_vals)
        wrk.pulsevals .= pulsevals0 .+ α .* search_direction
        @threadsif wrk.use_threads for k = 1:N
            reinit_prop!(wrk.fw_propagators[k], wrk.objectives[k].initial_state)
            for n = 1:N_T  # `n` is the index for the time interval
                prop_step!(wrk.fw_propagators[k])
            end
            τ[k] = dot(wrk.objectives[k].target_state, wrk.fw_propagators[k].state)
        end
        Ψ = [propagator.state for propagator in wrk.fw_propagators]
        J_T_vals[i] = J_T_func(Ψ, wrk.objectives; τ=τ)
    end
    wrk.pulsevals .= pulsevals_orig
    return J_T_vals
end


function _get_linesearch_data(
    wrk,
    iteration,
    args...
)

    searchdirection = wrk.searchdirection
    gradient_direction = -1 * wrk.gradient
    if norm(searchdirection) ≈ 0.0
        # LBFGSB hasn't yet built up Hessian (first iteration)
        # TODO: this should be fixed in GRAPE
        searchdirection = gradient_direction
    end
    pulsevals_opt = wrk.pulsevals
    pulsevals_guess = wrk.pulsevals_guess
    Δϵ = pulsevals_opt - pulsevals_guess
    norm_Δϵ = norm(Δϵ)
    norm_ls = norm(searchdirection)
    proj_ls = Δϵ ⋅ searchdirection
    # check that search direction and Δϵ are parallel
    cosθ = proj_ls / (norm_Δϵ * norm_ls)
    i = iteration
    @assert abs(cosθ - 1.0) < 1e-12  "iter $i: cos(θ) = $cosθ ≠ 1"
    α = proj_ls / norm_ls^2
    @assert abs(α - wrk.alpha) < 1e-12 "iter $i: α = $α ≠ wrk.alpha = $(wrk.alpha)"
    @assert norm(Δϵ - α * searchdirection) < 1e-12

    return α, searchdirection, gradient_direction, pulsevals_guess, pulsevals_opt

end


function plot_linesearch(
    outdir;
    samples=100,
    verbose=(get(ENV, "GRAPE_LINESEARCH_ANALYSIS_VERBOSE", "0") == "1"),
    use_current_backend=(
        get(ENV, "GRAPE_LINESEARCH_ANALYSIS_USE_CURRENT_BACKED", "0") == "1"
    ),
    kwargs...
)

    defaults = Dict(
        :linewidth => 3,
        :marker_size => 3,
        :foreground_color_legend => nothing,
        :background_color_legend => RGBA(1, 1, 1, 0.8)
    )
    merge!(default, kwargs)

    mkpath(outdir)

    function _plot_linesearch(wrk, iteration, args...)

        current_backend = backend()
        if !use_current_backend && !(current_backend isa Plots.GRBackend)
            gr()
        end

        (iteration == 0) && (return nothing)

        # TODO: linesearch data sources should be entirely in wrk, so we don't
        # need different methods.
        α, ls_direction, gradient_direction, pulsevals_guess, pulsevals_opt =
            _get_linesearch_data(wrk, iteration, args...)
        α_vals = collect(range(0, 2α, length=samples))
        J_α_gradient = explore_linesearch(gradient_direction, α_vals, pulsevals_guess, wrk)
        J_α_ls = explore_linesearch(ls_direction, α_vals, pulsevals_guess, wrk)
        J_T = wrk.result.J_T
        J_T_guess = wrk.result.J_T_prev

        ax1 = plot(pulsevals_guess; label="guess", defaults...)
        plot!(
            ax1,
            pulsevals_opt;
            label="opt",
            xlabel="time step (control parameter index)",
            ylabel="control amplitude",
            defaults...
        )
        plot!(
            ax1,
            title="Iteration $iteration; J = $(@sprintf("%.2e", J_T_guess)) → $(@sprintf("%.2e", J_T))"
        )

        ax2 = plot(
            gradient_direction;
            label="gradient",
            xlabel="time step (control parameter index)",
            ylabel="gradient directon",
            defaults...
        )

        ax3 = plot(gradient_direction; linestyle=:dash, label="gradient", defaults...)
        plot!(
            ax3,
            ls_direction;
            label="search direction",
            xlabel="time step (control parameter index)",
            ylabel="search direction",
            defaults...
        )

        ax4 = plot(
            α_vals,
            J_α_gradient;
            label="gradient",
            xlabel="step width α",
            ylabel="Functional",
            title="Gradient Linesarch",
            defaults...
        )
        scatter!(ax4, [0,], [J_T_guess,], label="")
        hline!(ax4, [J_T_guess], color="black", linewidth=0.5, label="")
        vline!(ax4, [α], color="black", linewidth=0.5, label="")

        ax5 = plot(
            α_vals,
            J_α_gradient;
            linestyle=:dash,
            label="gradient",
            title="Linesearch α = $α",
            defaults...
        )
        plot!(
            ax5,
            α_vals,
            J_α_ls;
            label="search direction",
            xlabel="step width α",
            ylabel="Functional",
            defaults...
        )
        scatter!(ax5, [0, α], [J_T_guess, J_T], label="")
        hline!(ax5, [J_T_guess], color="black", linewidth=0.5, label="")
        hline!(ax5, [J_T], color="black", linewidth=0.5, label="")
        vline!(ax5, [α], color="black", linewidth=0.5, label="")
        if J_T < 0.1
            plot!(ax5; yaxis=:log)
            ylim = ylims(ax5)
            ylims!(ax5, (ylim[1], 1))
        else
            ylim = ylims(ax5)
            ylims!(ax5, (ylim[1], 1))
        end

        fig = plot(ax1, ax2, ax3, ax4, ax5, layout=(5, 1),)
        plot!(fig, size=(700, 1500), left_margin=50px,)

        outfile = joinpath(outdir, @sprintf("linesearch-%03i.png", iteration))
        savefig(fig, outfile)
        if verbose
            @info "Written GRAPELinesearchAnalysis plot to $outfile"
        end

        if !use_current_backend && !(current_backend isa Plots.GRBackend)
            current_backend() # switch back
        end

        return nothing

    end

    return _plot_linesearch
end

end # module
