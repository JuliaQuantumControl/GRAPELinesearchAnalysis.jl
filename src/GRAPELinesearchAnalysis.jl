module GRAPELinesearchAnalysis

import GRAPE
using LinearAlgebra: dot, norm
using Printf: @printf, @sprintf
using Plots:
    Plots, RGBA, backend, gr, hline!, plot, plot!, savefig, scatter!, vline!, ylims!
using Plots.PlotMeasures: px


# Evaluate the full functional `J` for the pulse values `pulsevals0 + α d` for
# the samples `(d, α_vals)`, each consisting of a direction `d` and a vector of
# step widths `α_vals`. Returns a vector of `J` values for each sample. On exit,
# the workspace is restored to the state for the optimized pulse values: GRAPE's
# Optim.jl backend requires `wrk.pulsevals` to be unchanged after a callback,
# LBFGSB aliases `wrk.pulsevals`, and callbacks that run later (e.g.,
# `print_iters`) read `wrk.J_parts`.
function sample_functional(pulsevals0, wrk, samples...)
    pulsevals_opt = copy(wrk.pulsevals)
    pulsevals = similar(pulsevals0)
    try
        return map(samples) do (d, α_vals)
            map(α_vals) do α
                pulsevals .= pulsevals0 .+ α .* d
                GRAPE.evaluate_functional(pulsevals, wrk; count_call = false)
            end
        end
    finally
        GRAPE.evaluate_functional(pulsevals_opt, wrk; count_call = false)
    end
end


# Return the scalar `α` and the direction `d` so that the pulse update of the
# current iteration is `Δu = α d`. For an optimizer that updates the pulses along
# `GRAPE.search_direction`, `d` is that search direction and `α` is the step
# width. Otherwise, `d` is `Δu` and `α = 1`. The third return value indicates
# whether `d` is the search direction.
function _get_step(wrk; rtol)
    Δu = GRAPE.pulse_update(wrk)
    s = GRAPE.search_direction(wrk)
    s_norm_sq = dot(s, s)
    if s_norm_sq > 0
        # `GRAPE.step_width` warns (for some optimizers) if `Δu` is not
        # parallel to `s`, so we calculate the step width directly
        α = dot(Δu, s) / s_norm_sq
        if norm(Δu - α * s) ≤ rtol * norm(Δu)
            return α, s, true
        end
    end
    return 1.0, Δu, false
end


# Limits for the y-axis of a plot of `J` values, restricting the range above the
# value `J_guess` at α = 0 to be the same as the range below it (on a linear or
# log scale). Values larger than that are not interesting for a line search.
function _ylims(J_vals, J_guess; log_scale)
    J_min = minimum(J_vals)
    J_max = maximum(J_vals)
    if log_scale
        upper = min(J_max, J_guess * (J_guess / J_min))
        pad = (upper / J_min)^0.05
        return (J_min / pad, upper * pad)
    else
        upper = min(J_max, J_guess + (J_guess - J_min))
        pad = 0.05 * (upper - J_min)
        return (J_min - pad, upper + pad)
    end
end


"""Create a callback for plotting the line search in each iteration.

```julia
callback = GRAPELinesearchAnalysis.plot_linesearch(outdir; kwargs...)
```

creates a function `callback` that can be passed as `callback` to `optimize`
with `method = GRAPE`. For every iteration, it writes an image
`linesearch-NNN.png` (with the iteration number `NNN`) to `outdir`.

Each image shows the guess and optimized pulses of the iteration, the negative
gradient and the search direction, and the value of the optimization functional
``J`` (including any running costs) for different step widths ``α`` along the
search direction and along the negative gradient. Every sampled value requires
a full propagation of all trajectories. Any propagation `callback` defined for
the trajectories is also called for these propagations.

With an optimizer that does not update the pulses along
`GRAPE.search_direction` (e.g., `Optim.ConjugateGradient`), the images show the
pulse update ``Δu`` of the iteration as the search direction, with ``α = 1``
for the actual update. A warning is shown the first time this happens.

The `callback` returns a tuple `(iter, ǁgǁ, ǁsǁ, ratio, angle, α)` with the norm
of the gradient `g`, the norm of the search direction `s`, the ratio `ǁsǁ/ǁgǁ`,
the angle between `-g` and `s` in degrees, and the step width `α`, for each
iteration. These are stored in the `records` of the optimization result, see
[`print_ls_table`](@ref).

# Keyword arguments

* `samples=100`: The number of values of ``α`` for which to evaluate the
  functional, along each of the two directions.
* `verbose=false`: If `true`, show a message for every image that is written.
  The default can be set with the environment variable
  `GRAPE_LINESEARCH_ANALYSIS_VERBOSE=1`.
* `use_current_backend=false`: If `true`, create the plots with the active
  Plots backend. Otherwise, switch to the GR backend while creating the plots.
  The default can be set with the environment variable
  `GRAPE_LINESEARCH_ANALYSIS_USE_CURRENT_BACKEND=1`.
* `rtol=1e-8`: The relative tolerance for checking whether the pulse update is
  along the search direction.

All other keyword arguments are passed to the `plot` functions of each panel,
e.g., `linewidth` (default: 3).
"""
function plot_linesearch(
    outdir;
    samples = 100,
    verbose = (get(ENV, "GRAPE_LINESEARCH_ANALYSIS_VERBOSE", "0") == "1"),
    use_current_backend = (
        get(ENV, "GRAPE_LINESEARCH_ANALYSIS_USE_CURRENT_BACKEND", "0") == "1"
    ),
    rtol = 1e-8,
    kwargs...
)

    defaults = Dict{Symbol,Any}(
        :linewidth => 3,
        :foreground_color_legend => nothing,
        :background_color_legend => RGBA(1, 1, 1, 0.8)
    )
    merge!(defaults, kwargs)

    mkpath(outdir)
    warned = false

    function _plot_linesearch(wrk, iteration, args...)

        (iteration == 0) && (return nothing)

        α, direction, is_search_direction = _get_step(wrk; rtol)
        if !is_search_direction && !warned
            @warn "The pulse update is not along the search direction. Plotting the line search along the pulse update instead." iteration
            warned = true
        end
        dirlabel = is_search_direction ? "search direction" : "pulse update"
        αlabel =
            is_search_direction ? "step width α" : "step width α (fraction of pulse update)"
        gradient_direction = -1 * GRAPE.gradient(wrk)
        g_norm = norm(gradient_direction)
        d_norm = norm(direction)
        if (g_norm == 0) || (d_norm == 0)
            @warn "Cannot plot the line search for a vanishing gradient or pulse update" iteration
            return nothing
        end
        # With the same step length for both directions, the gradient samples
        # are comparable to the samples along the search direction
        unit_gradient_direction = gradient_direction / g_norm
        unit_direction = direction / d_norm
        pulsevals_guess = copy(wrk.pulsevals_guess)
        pulsevals_opt = copy(wrk.pulsevals)
        J_opt = sum(wrk.J_parts)

        α_vals = collect(range(0, 2α, length = samples))
        step_lengths = abs.(α_vals) .* d_norm
        J_vals, J_vals_gradient = sample_functional(
            pulsevals_guess,
            wrk,
            (direction, α_vals),
            (unit_gradient_direction, step_lengths),
        )
        J_guess = J_vals[1]  # α = 0

        all_J_vals = [J_vals; J_vals_gradient; J_opt]
        log_scale = (minimum(all_J_vals) > 0) && (J_guess > 10 * minimum(all_J_vals))
        angle = GRAPE.vec_angle(gradient_direction, direction; unit = :degree)

        current_backend = backend()
        switch_backend = !use_current_backend && !(current_backend isa Plots.GRBackend)
        switch_backend && gr()
        try

            fmt(x) = @sprintf("%.2e", x)

            ax1 = plot(pulsevals_guess; label = "guess", defaults...)
            plot!(
                ax1,
                pulsevals_opt;
                label = "optimized",
                xlabel = "control parameter index",
                ylabel = "control amplitude",
                title = "Iteration $iteration: J = $(fmt(J_guess)) → $(fmt(J_opt))",
                defaults...
            )

            ax2 = plot(
                gradient_direction;
                label = "negative gradient",
                xlabel = "control parameter index",
                ylabel = "-∇J",
                defaults...
            )

            ax3 = plot(
                unit_gradient_direction;
                linestyle = :dash,
                label = "negative gradient",
                defaults...
            )
            plot!(
                ax3,
                unit_direction;
                label = dirlabel,
                xlabel = "control parameter index",
                ylabel = "normalized direction",
                title = "norm ratio = $(@sprintf("%.3g", d_norm / g_norm)), angle = $(@sprintf("%.1f", angle))°",
                defaults...
            )

            ax4 = plot(
                α_vals,
                J_vals;
                label = dirlabel,
                xlabel = αlabel,
                ylabel = "functional J",
                title = "Line search along $dirlabel: α = $(fmt(α))",
                defaults...
            )
            scatter!(ax4, [0, α], [J_guess, J_opt]; label = "")
            hline!(ax4, [J_guess, J_opt]; color = "black", linewidth = 0.5, label = "")
            vline!(ax4, [α]; color = "black", linewidth = 0.5, label = "")

            ax5 = plot(
                step_lengths,
                J_vals_gradient;
                linestyle = :dash,
                label = "negative gradient",
                defaults...
            )
            plot!(
                ax5,
                step_lengths,
                J_vals;
                label = dirlabel,
                xlabel = "step length ǁΔuǁ",
                ylabel = "functional J",
                title = "Line search along gradient and $dirlabel",
                defaults...
            )
            Δu_norm = abs(α) * d_norm
            scatter!(ax5, [0, Δu_norm], [J_guess, J_opt]; label = "")
            hline!(ax5, [J_guess, J_opt]; color = "black", linewidth = 0.5, label = "")
            vline!(ax5, [Δu_norm]; color = "black", linewidth = 0.5, label = "")

            ylim_ax4 = _ylims([J_vals; J_opt], J_guess; log_scale)
            ylim_ax5 = _ylims(all_J_vals, J_guess; log_scale)
            for (ax, ylim) in ((ax4, ylim_ax4), (ax5, ylim_ax5))
                log_scale && plot!(ax; yaxis = :log)
                (ylim[2] > ylim[1]) && ylims!(ax, ylim)
            end

            fig = plot(ax1, ax2, ax3, ax4, ax5; layout = (5, 1))
            plot!(fig; size = (700, 1500), left_margin = 50px)

            outfile = joinpath(outdir, @sprintf("linesearch-%03i.png", iteration))
            savefig(fig, outfile)
            if verbose
                @info "Written GRAPELinesearchAnalysis plot to $outfile"
            end

        finally
            switch_backend && backend(current_backend)
        end

        return (iteration, g_norm, d_norm, d_norm / g_norm, angle, α)

    end

    return _plot_linesearch
end


"""Print information about the search direction in each iteration.

```julia
print_ls_table(res)
```

prints a table of the records in the optimization result `res` obtained from
`optimize` with `method = GRAPE` and
`callback = GRAPELinesearchAnalysis.plot_linesearch(…)`. For each iteration, the
table shows the norm of the gradient, the norm of the search direction, the
ratio of these norms, the angle between the negative gradient and the search
direction, and the step width `α`.
"""
function print_ls_table(res)
    println("")
    @printf("%6s", "iter")
    @printf("%10s", "|grad|")
    @printf("%10s", "|search|")
    @printf("%10s", "ratio")
    @printf("%10s", "angle(°)")
    @printf("%10s", "step α")
    println("")
    for (iter, g_norm, s_norm, ratio, angle, α) in res.records
        @printf("%6d", iter)
        @printf("%10.2e", g_norm)
        @printf("%10.2e", s_norm)
        @printf("%10.2e", ratio)
        @printf("%10.2f", angle)
        @printf("%10.2e", α)
        println("")
    end
    println("")
end

end # module
