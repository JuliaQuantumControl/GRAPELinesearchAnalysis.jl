using Test
using SafeTestsets

# Note: comment outer @testset to stop after first @safetestset failure
@time @testset verbose = true "GRAPELinesearchAnalysis.jl Package" begin

    println("\n* Sampling the Functional (test_sample_functional.jl)")
    @time @safetestset "Sampling the Functional" begin
        include("test_sample_functional.jl")
    end

    println("\n* Plotting the Line Search (test_plot_linesearch.jl)")
    @time @safetestset "Plotting the Line Search" begin
        include("test_plot_linesearch.jl")
    end

end
nothing
