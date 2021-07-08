module Pythia

using Base: Integer, Float64, sign_mask
using Statistics, Distributions

export sglasso, glasso, lasso
export MeanForecast, NaiveForecast, SES, Holt
export fit, predict

include("algorithms/sglasso.jl")
include("algorithms/basicMethods.jl")
include("algorithms/ETS.jl")

end
