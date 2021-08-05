abstract type ARIMAModel end
using PyCall

py_arima = PyNULL()
np = PyNULL()

function __init__() # import python libraries and if they aren't installed, install them.
    copy!(tsa, pyimport_conda("statsmodels.tsa.arima.model.ARIMA", "statsmodels"))
    copy!(np, pyimport_conda("numpy", "numpy"))
end

mutable struct ARIMA <: ARIMAModel
    y::Vector{AbstractFloat}
    order::Union{Nothing, Tuple}
    seasonal_order::Union{Nothing, Tuple}
    trend::Union{Nothing, String}
    enforce_stationarity::Union{Nothing, Bool}
    enforce_invertibility::Union{Nothing, Bool}
    concentrate_scale::Union{Nothing, Bool}
    trend_offset::Union{Nothing, Integer}
    missing_val::String
    function ARIMA(y = []; order = (0, 0, 0), seasonal_order = (0, 0, 0, 0), trend = nothing, enforce_stationarity = true,
        enforce_invertibility = true, concentrate_scale = false, trend_offset = 1, missing_val = "none") # Constuctor 
        
        pyMdl = py_arima(endog = y, order = order, seasonal_order = seasonal_order, trend = trend, enforce_stationarity = enforce_stationarity, enforce_invertibility = enforce_invertibility, concentrate_scale = concentrate_scale, trend_offset = trend_offset)

        return pyMdl
    end 
end

