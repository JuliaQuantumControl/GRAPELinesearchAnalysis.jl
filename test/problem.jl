using QuantumControl: ControlProblem, Trajectory, hamiltonian
using QuantumControl.Shapes: flattop
using QuantumControl.Functionals: J_T_sm, chi_sm
using QuantumPropagators: ExpProp


# A state-to-state transfer in a two-level system that reaches a low `J_T`
# within a few iterations
function tls_problem(; kwargs...)
    ϵ(t) = 0.2 * flattop(t, T = 5, t_rise = 0.3, func = :blackman)
    H = hamiltonian(ComplexF64[-0.5 0; 0 0.5], (ComplexF64[0 1; 1 0], ϵ))
    tlist = collect(range(0, 5, length = 201))
    trajectories = [Trajectory(ComplexF64[1, 0], H, target_state = ComplexF64[0, 1])]
    return ControlProblem(
        trajectories,
        tlist;
        iter_stop = 3,
        prop_method = ExpProp,
        J_T = J_T_sm,
        print_iters = false,
        rethrow_exceptions = true,
        kwargs...
    )
end


# Final-time functional and boundary condition that do not take a `tau` keyword
# argument
J_T_without_tau(Ψ, trajectories) = J_T_sm(Ψ, trajectories)
chi_without_tau(Ψ, trajectories) = chi_sm(Ψ, trajectories)
