abstract type MeanForecastModel end
abstract type NaiveForecastModel end
abstract type SeasonalNaiveForecastModel end
abstract type ModelResults end

### Mean Forecast Model
mutable struct MeanForecast <: MeanForecastModel
    y::Vector{AbstractFloat}
    h::Integer
    level::Union{Nothing, Vector{AbstractFloat}}
    function MeanForecast(y = []; h = 5, level = [80, 95]) # Constructor
        if length(y) <= 0
            error("The input array is empty")
        end
        if h <= 0
            error("The h-value must be positive")
        end

        cleanPredInterval_(level)
        return new(y, h, level)
    end
end

function predict(model::MeanForecastModel)
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

    results = BasicMethodResults(model, fittedvalues, lower, upper)
    return results
end

function fit(model::MeanForecastModel)
    return model
end

### Naive Forecast Model
mutable struct NaiveForecast <: NaiveForecastModel
    y::Vector{AbstractFloat}
    h::Integer
    level::Union{Nothing, Vector{AbstractFloat}}
    function NaiveForecast(y = []; h = 5, level = [80, 95]) # Constructor
        if length(y) <= 0
            error("The input array is empty")
        end
        if h <= 0
            error("The h-value must be positive")
        end
        cleanPredInterval_(level)
        return new(y, h, level)
    end
end

function predict(model::NaiveForecastModel)
    y = model.y
    h = model.h
    level = model.level
    data_size = length(y)

    # Forecasts are equal to last value of observations
    lastValue = last(y)
    fitted_chunk = fill(lastValue, (h))
    fittedvalues = vcat(y, fitted_chunk)

    # Compute Prediction Intervals
    lower = zeros(length(level), h)
    upper = lower
    for i in 1:length(level)
        dist = Normal(0.0, 1.0)
        qq = quantile(dist, 0.5 * (1 + level[i]/100))
        lower[i, :] = fitted_chunk - qq * fitted_chunk
        upper[i, :] = fitted_chunk + qq * fitted_chunk
    end

    results = BasicMethodResults(model, fittedvalues, lower, upper)
    return results
end

function fit(model::NaiveForecastModel)
    return model
end

### Seasonal Naive Model
mutable struct SeasonalNaiveForecast <: SeasonalNaiveForecastModel
    y::Vector{AbstractFloat}
    h::Integer
    level::Union{Nothing, Vector{AbstractFloat}}
    function SeasonalNaiveForecast(y = []; h = 5, level = [80, 95]) # Constructor
        if length(y) <= 0
            error("The input array is empty")
        end
        if h <= 0
            error("The h-value must be positive")
        end
        cleanPredInterval_(level)
        return new(y, h, level)
    end
end

function fit(model::SeasonalNaiveForecastModel)
    y = model.y
    h = model.h
    level = model.level
    data_size = length(y)

    # Forecasts are equal to last value of observations
    lastValue = last(y)
    fitted_chunk = fill(lastValue, (h))
    fittedvalues = vcat(y, fitted_chunk)

    # Compute Prediction Intervals
    lower = zeros(length(level), h)
    upper = lower
    for i in 1:length(level)
        dist = Normal(0.0, 1.0)
        qq = quantile(dist, 0.5 * (1 + level[i]/100))
        lower[i, :] = fitted_chunk - qq * fitted_chunk
        upper[i, :] = fitted_chunk + qq * fitted_chunk
    end

    results = BasicMethodResults(model, fittedvalues, lower, upper)
    return results
end


# Ensures Prediction Intervals are numerical and on the interval (0, 100)
function cleanPredInterval_(level)
    for i in level
        if !isa(i, Number)
            print(isa(i, Number))
            error("Ensure that all level inputs are numeric")
        elseif i < 0 || i > 100
            error("Ensure that all confidence interval inputs are in the interval [0, 100]")
        end
    end
end

# Results Object - Common across all "Basic Method" models
mutable struct BasicMethodResults <: ModelResults # Stores results from fit(MeanForecast)
    model::MeanForecastModel
    fittedvalues::Vector{Float64} # Vector of fitted values
    lower::Matrix{Float64} # Lower bound of prediction intervals
    upper::Matrix{Float64} # Upper bound of prediction intervals
    function BasicMethodResults(model, fittedvalues, lower, upper)
        return new(model, fittedvalues, lower, upper)
    end
end