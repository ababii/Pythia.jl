using Pythia, Test

tic()
@time @testset "Basic Methods Tests" begin include("basicMethods_test.jl") end
toc()
