module Pythia

using Statistics, Distributions

export sglasso, glasso, lasso
export MeanForecast, fit

include("algorithms/sglasso.jl")
include("algorithms/basicMethods.jl")

end
