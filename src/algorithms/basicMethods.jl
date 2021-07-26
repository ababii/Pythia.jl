abstract type BasicModel end
abstract type ModelResults end

### Mean Forecast Model
mutable struct MeanForecast <: BasicModel
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

function predict(model::MeanForecast)
    y = model.y
    h = model.h
    level = model.level
    data_size = length(y)

    # Compute mean, concatenate it onto the original data
    data_mean = mean(y)
    mean_vector = ones(h + data_size) .* data_mean
    fittedvalues = mean_vector
    residuals = y - fittedvalues[1:data_size] # compute residuals

    ### Coverage Probability Estimate, will calculate for 80 and 95 for now.
    c_80 = 1.28
    c_95 = 1.96

    K = 0 # number of parameters estimated in the forecasting method

    lower = zeros(length(level), h)
    upper = lower

    # Compute Prediction Intervals  
    if h != 0
        # Calculate one-step prediction intervals
        one_step = sqrt(1 / (data_size - K) * (transpose(residuals) * residuals))
        σ = one_step
    end
    if h > 1
        # Calculate multi-step prediction intervals
        multi_step = fill(one_step * sqrt(1 + 1/data_size), h)
        multi_step[1] = one_step
        σ = multi_step
    end
    
    # Get the last h values
    yhat = fittedvalues[data_size + 1 : data_size + h]

    # Calculate prediction intervals, store in upper / lower matrices
    lower[1, :] = yhat - c_80 * σ
    lower[2, :] = yhat - c_95 * σ

    upper[1, :] = yhat + c_80 * σ
    upper[2, :] = yhat + c_95 * σ

    results = BasicMethodResults(model, fittedvalues, lower, upper)
    return results
end

function fit(model::MeanForecast)
    return model
end

### Naive Forecast Model
mutable struct NaiveForecast <: BasicModel
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

function predict(model::NaiveForecast)
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

function fit(model::NaiveForecast)
    return model
end

### Seasonal Naive Model
mutable struct SeasonalNaiveForecast <: BasicModel
    y::Vector{AbstractFloat}
    h::Integer
    m::Integer
    level::Union{Nothing, Vector{AbstractFloat}}
    function SeasonalNaiveForecast(y = []; h = 5, m = 12, level = [80, 95]) # Constructor
        if length(y) <= 0
            error("The input array is empty")
        end
        if h <= 0
            error("The h-value must be positive")
        end
        if m <= 0
            error("The m-value must be positive")
        end
        cleanPredInterval_(level)
        return new(y, h, m, level)
    end
end

function fit(model::SeasonalNaiveForecast)
    y = model.y
    h = model.h
    m = model.m
    level = model.level
    data_size = length(y)

    # Forecasts are equal to last value of observations
    k = trunc(Integer, (h-1)/m)
    fitted_chunk = y[data_size + 1 - m * (k+1) : data_size + h - m * (k+1)]
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
    model::BasicModel
    fittedvalues::Vector{Float64} # Vector of fitted values
    lower::Matrix{Float64} # Lower bound of prediction intervals
    upper::Matrix{Float64} # Upper bound of prediction intervals
    function BasicMethodResults(model, fittedvalues, lower, upper)
        return new(model, fittedvalues, lower, upper)
    end
end