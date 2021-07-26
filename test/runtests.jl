using Pythia, Test

# @time @testset "Basic Methods Tests" begin include("basicMethods_test.jl") end
@time @testset "Exponential Smoothing Tests" begin include("ETS_test.jl") end
