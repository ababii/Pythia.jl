module Pythia

using Statistics, Distributions

export sglasso, glasso, lasso
export MeanForecast, NaiveForecast
export fit

include("algorithms/sglasso.jl")
include("algorithms/basicMethods.jl")

end
