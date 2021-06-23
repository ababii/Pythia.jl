abstract type MeanForecastModel end
abstract type ResultsObj end

### Mean Forecast Model
mutable struct MeanForecast <: MeanForecastModel
    y::Vector{AbstractFloat}
    h::Integer
    level::Union{Nothing, Vector{Integer}}
    function MeanForecast(y = []; h = 5, level = (80, 95)) # Constructor
        if length(y) <= 0
            error("The input array is empty")
        end
        if h <= 0
            error("the number of prediction steps must be positive")
        end

        # cleanPredInterval_(level) # TODO: implement this
        return new(y, h, level)
    end
end

struct MeanForecastResults <: ResultsObj # Stores results from fit(MeanForecast)
    model::MeanForecastModel
    fittedvalues::Vector{Float64} # Vector of fitted values
    lower::Matrix{Float64}
    upper::Matrix{Float64}
    function MeanForecastResults(model, fittedvalues, lower, upper)
        return new(model, fittedvalues, lower, upper)
    end
end

function fit(model::MeanForecastModel)
    y = model.y
    h = model.h
    level = model.level
    data_size = length(y)

    # Compute mean, concatenate it onto the original data
    data_mean = mean(y)
    mean_vector = ones(h) .* data_mean
    fittedvalues = vcat(y, mean_vector)

    # Compute Prediction Intervals
    lower = zeros(length(level), h)
    upper = lower
    for i in 1:length(level)
        dist = Normal(0.0, 1.0)
        qq = quantile(dist, 0.5 * (1 + level[i]/100))
        lower[i, :] = mean_vector - qq * mean_vector
        upper[i, :] = mean_vector + qq * mean_vector
    end

    results = MeanForecastResults(model, fittedvalues, lower, upper)
    return results
end
