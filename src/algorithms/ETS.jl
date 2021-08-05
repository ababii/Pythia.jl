abstract type ETSModel end
import Optim: optimize
"""
SES
Description:
    - Contains parameters and data to apply Simple Exponential Smoothing (SES)
    - Subtype of ETSModel, for future proofing
Constuctor:
    - Vector{AbstractFloat} y: contains observations for forecasting
    - Integer h: Stores the number of steps after time T to forecast (T = latest time in the observed data)
        - Default = 10, must be > 0
    - Union{Nothing, Float64} alpha: smoothing parameter [1]
        - 0 < alpha < 1
        - If no value is specified, an optimal value will be chosen using LBFGS.
    - Union{Nothing, Float64} init_level: initial level value [1]
        -Inf < init_level < Inf
        - If no value is specified, an optimal value will be chosen uisng LBFGS.

EXAMPLE USAGE:
    observations = [1.0, 2.0, 3.0]
    # Inputted values will be used if they are specified. Otherwise, they will be computed.
    mdl = SES(observations, h = 5) 
    mdl = SES(observations, alpha = 0.4) 
    mdl = SES(observations, alpha = 0.25, init_level = 500.0) 
    mdl = SES(observations, h = 15, alpha = 0.3, init_level = 750.0)

"""
mutable struct SES <: ETSModel
    y::Vector{AbstractFloat}
    h::Integer
    alpha::Union{Nothing, Float64}
    init_level::Union{Nothing, Float64}
    function SES(y = []; h = 5, alpha = nothing, init_level = nothing) # Constuctor 
        # Input Cleaning
        if length(y) <= 0
            error("The input array is empty.")
        end
        if h <= 0
            error("The number of prediction steps must be positive")
        end

        cleanParams_(alpha, "alpha")

        return new(y, h, alpha, init_level)
    end 
end

"""
makeForecast_()
Description:
    - Helper function
    - computes a forecast with Simple Exponential Smoothing (SES) [1]
Parameters
    - model: SES struct (defined above)
    - Note: when makeForecast_ is called, all parameters have been calibrated.
Returns:
    A tuple containing the following:
    - Vector{Float64} forecast: contains fitted values
    - SSE::Float64 SSE: the Sum of the Squares (error term)
        - used for parameter optimzation
References: 
    [1] Hyndman, R.J., & Athanasopoulos, G. (2019) *Forecasting:
        principles and practice*, 3rd edition, OTexts: Melbourne,
        Australia. OTexts.com/fpp3. Accessed on April 19th 2020.
"""
function makeForecast_(model::SES) # Return vector of fitted values of length h + SSE
    h = model.h
    alpha = model.alpha
    init_level = model.init_level
    y = model.y
    data_size = length(y)

    lvls = zeros(data_size + 1) # Stores smoothed values
    forecast = zeros(data_size + h) # Stores fitted values
    lvls[1] = init_level # Set l_0
    forecast[1] = init_level
    SSE = 0.0 # Compute Sum of Squared Errors

    for i in 2:(length(lvls)) # Compute l_t's
        lvls[i] = alpha * y[i-1] + (1 - alpha) * lvls[i-1];
        forecast[i] = lvls[i]
        SSE += (forecast[i-1] - y[i-1])^2
    end

    for i in 1:h # Set forecasted values
        forecast[data_size + i] = lvls[end]
    end

    return forecast, SSE
end

"""
SSE_() 
Description:
    - Returns the Sum of Squared Errors for a given alpha and init_level.
Parameters:
    - model: SES struct (defined above)
    - alpha: value of alpha used for computation
    - init_level: value of init_level used for computation
    - verbose: if verbose > 0, then the parameters will be printed as the are optimized.
Returns:
    - Float64 res: the Sum of Squared Errors for the given alpha and init_level values.
"""
function SSE_(model::SES; alpha = nothing, init_level = nothing, verbose = 0) # Helper function for optimization
    model.alpha = alpha
    model.init_level = init_level

    res = makeForecast_(model)[2]
    if verbose > 0
        println("Error: $res, Alpha: $(model.alpha), L0: $(model.init_level)")
    end

    return res
end
"""
cleanParams_()
Description:
    Throws an error if a given parameters is out of the specified bounds.
Parameters:
    - param: value of parameter to be checked
    - name: name of parameter to be checked
    - lb: lower bound of parameter, default = 0.0
    - ub: upper bound of parameter, defualt = 1.0
Note: if param is not of type Nothing, then param must be in (0.0, 1.0)
Returns:
    - nothing
"""
function cleanParams_(param::Union{Nothing, Float64}, name::String, lb::Float64 = 0.0, ub::Float64 = 1.0)
    if !isnothing(param) && (param < lb || param > ub)
        error(name * " needs to be in the range ($lb, $ub)")
    end
end
"""
fit!(model::ETSModel)
Description:
    fits an ETSModel object
    - computes initial values of parameters: 'alpha' and 'init_value'
    - modifies the 'model' directly.
    - after fit! is called, predict(model) is ready to be called
Parameters:
    - model: SES struct
    - v: verbosity, default = 0. if v > 0, verbose will be used.
Computation of model parameters:
'alpha' and 'init_value' are computed using LBFGS
Returns:
    - nothing
"""
function fit(model::SES; v=0) # Set alpha + init_level
    # Creating a copy of the user inputted params for the model as model.alpha and model.init_level will be updated later
    base_alpha = model.alpha
    base_init_level = model.init_level

    # Model to return to ensure that input model is not modified
    retModel = deepcopy(model)

    if (isnothing(retModel.alpha))
        @warn "Since no value was entered for 'alpha', it will be chosen"
    end
    if (isnothing(retModel.init_level))
        # model.init_level = compute_init_heuristic_(model.y)
        @warn "Since no value was entered for 'init_level', it will be chosen"
    end 

    # Optimize predict(model) - directly updates model
    f(x) = SSE_(retModel, 
                alpha=(isnothing(base_alpha) ? x[1] : base_alpha),
                init_level=(isnothing(base_init_level) ? x[2] : base_init_level),
                verbose=v)

    if isnothing(retModel.alpha) || isnothing(retModel.init_level) # if atleast one value needs to be optimized
        # lower and upper limits for alpha and init_level
        lower = [0.0, -Inf] 
        upper = [1.0, Inf]
        # Initial values for alpha and init_level
        init_inputs = [0.0, model.y[1]]
        res = optimize(f, lower, upper, init_inputs)
        if v > 0
            println(res)
        end
    end

    return retModel
end
"""
predict()
Description:
    - Exposed function
    - computes a forecast with Simple Exponential Smoothing (SES) [1]
    - calls makeForecast_ a helper function, which returns a vector of fitted values and SSE error term.
Parameters
    - model: SES struct (defined above)
    - Note: when predict is called, all parameters have been calibrated.
Returns:
    A tuple containing the following:
    - Vector{Float64} forecast: contains fitted values
References: 
    [1] Hyndman, R.J., & Athanasopoulos, G. (2019) *Forecasting:
        principles and practice*, 3rd edition, OTexts: Melbourne,
        Australia. OTexts.com/fpp3. Accessed on April 19th 2020.
"""
function predict(model::ETSModel)
    fitted_vals = makeForecast_(model)[1]
    h = model.h
    return fitted_vals[end-(h-1):end]
end

### Holt's Method - implementing Holt's linear trend method
mutable struct Holt <: ETSModel
    y::Vector{AbstractFloat}
    h::Integer
    alpha::Union{Nothing, Float64}
    beta::Union{Nothing, Float64}
    phi::Union{Nothing, Float64}
    init_level::Union{Nothing, Float64}
    init_trend::Union{Nothing, Float64}
    damped::Bool
    function Holt(y = []; h = 5, alpha = nothing, beta = nothing, phi = nothing, init_level = nothing, init_trend = nothing, damped = false) # Constuctor 
        # Input Cleaning
        if length(y) <= 0
            error("The input array is empty.")
        end
        if h <= 0
            error("The number of prediction steps must be positive")
        end

        cleanParams_(alpha, "alpha")
        cleanParams_(beta, "beta")
        cleanParams_(phi, "phi", 0.80, 0.98)

        return new(y, h, alpha, beta, phi, init_level, init_trend)
    end 
end

function makeForecast_(model::Holt) # Return vector of fitted values of length h + SSE
    h = model.h
    alpha = model.alpha
    beta = model.beta
    phi = model.phi
    init_level = model.init_level
    init_trend = model.init_trend
    y = model.y
    data_size = length(y)

    lvls = zeros(data_size + 1) # Stores level values, indices (0, T)
    trends = zeros(data_size + 1) # Stores trend values, indices (0, T)
    forecast = zeros(data_size + h) # Stores forecasted values, yhat, indices (1, T+h)

    lvls[1] = init_level # Set l_0
    trends[1] = init_trend # Set b_0

    SSE = 0.0 # Compute Sum of Squared Errors 

    for i in 2:(length(lvls)) # Compute l_t's
        lvls[i] = alpha * y[i-1] + (1 - alpha) * (lvls[i-1] + phi * trends[i-1]) # level equation
        trends[i] = beta * (lvls[i] - lvls[i-1]) + (1 - beta) * phi * trends[i-1] # trned equation 
        forecast[i-1] = lvls[i-1] + phi * trends[i-1] # one-step ahead forecast
        SSE += (forecast[i-1] - y[i-1])^2
    end
    
    for i in 1:h # Set forecasted values
        if i == 1
            trend_coefficient = phi
        elseif phi != 1
            trend_coefficient = phi * (1 - phi ^ h) / (1 - phi)
        else
            trend_coefficient = i
        end
        forecast[data_size + i] = lvls[end] + trend_coefficient * trends[end]
    end

    return forecast, SSE
end

function SSE_(model::Holt; alpha = nothing, beta = nothing, phi = nothing, init_level = nothing, init_trend = nothing, verbose = 0) # Helper function for optimization
    model.alpha = alpha
    model.beta = beta
    model.phi = phi
    model.init_level = init_level
    model.init_trend = init_trend

    res = makeForecast_(model)[2]
    if verbose > 0
        println("Error: $res, Alpha: $(model.alpha), Beta*: $(model.beta), Phi: $(model.phi), L0: $(model.init_level), BO:$(model.init_trend)")
    end

    return res
end

function fit(model::Holt; v=0) # Set alpha + init_level
    # Creating a copy of the user inputted params for the model as model.alpha and model.init_level will be updated later
    base_alpha = model.alpha
    base_beta = model.beta
    base_phi = model.phi
    base_init_level = model.init_level
    base_init_trend = model.init_trend

    # copy of model to return to ensure that input model is not modified
    retModel = deepcopy(model)

    if (isnothing(retModel.alpha))
        @warn "Since no value was entered for 'alpha', it will be chosen"
    end
    if (isnothing(retModel.beta))
        @warn "Since no value was entered for 'beta', it will be chosen"
    end
    if (isnothing(retModel.init_level))
        @warn "Since no value was entered for 'init_level', it will be chosen"
    end
    if (isnothing(retModel.init_trend))
        @warn "Since no value was entered for 'init_trend', it will be chosen"
    end

    if retModel.damped
        print("The Damped Trend Method will be used.\n")
        if (isnothing(retModel.phi))
            @warn "Since no value was entered for `phi`, it will be chosen"
        end
    else
        if (!isnothing(retModel.phi))
            @warn "Since a value was entered for `phi`, damping will occur despite 'damped' being set to false"
        else
            base_phi = 1 # No damping is equivalent to setting the damping parameter to 1
        end
    end

    # Optimize makeForecast_(model) - directly updates model
    f(x) = SSE_(retModel, 
                alpha=(isnothing(base_alpha) ? x[1] : base_alpha),
                beta=(isnothing(base_beta) ? x[2] : base_beta),
                phi=(isnothing(base_phi) ? x[3] : base_phi),
                init_level=(isnothing(base_init_level) ? x[4] : base_init_level),
                init_trend=(isnothing(base_init_trend) ? x[5] : base_init_trend),
                verbose=v)

    
    if isnothing(retModel.alpha) || isnothing(retModel.init_level) || isnothing(retModel.init_trend) || isnothing(retModel.beta) || (damped && isnothing(retModel.phi))# if at least one value needs to be optimized
        # lower and upper limits for alpha, beta, phi, init_level, init_trend
        lower = [0.0, 0.0, 0.8, -Inf, -Inf] 
        upper = [1.0, 1.0, 0.995, Inf, Inf]

        # Initial values for alpha, beta, init_level, init_trend (optimization starting points)
        first_l0 = model.y[1]
        first_b0 = model.y[1]
        first_alpha = 0.1
        first_beta = 0.1
        first_phi = 0.9

        # Optimize the SSE of the model through the function "f"
        init_inputs = [first_alpha, first_beta, first_phi, first_l0, first_b0]
        res = optimize(f, lower, upper, init_inputs)
        
        if v > 0
            println(res)
        end
    end

    return retModel
end

### HoltWinters's Method - implementing the HoltWinters Additive method
mutable struct HoltWinters <: ETSModel
    y::Vector{AbstractFloat}
    h::Integer
    alpha::Union{Nothing, Float64}
    beta::Union{Nothing, Float64}
    gamma::Union{Nothing, Float64}
    m::Integer
    init_level::Union{Nothing, Float64}
    init_trend::Union{Nothing, Float64}
    init_season::Vector{AbstractFloat}
    function HoltWinters(y = []; h = 5, alpha = nothing, beta = nothing, gamma = nothing, m, init_level = nothing, init_trend = nothing, init_season) # Constuctor 
        # Input Cleaning
        if length(y) <= 0
            error("The input array is empty.")
        end
        if h <= 0
            error("The number of prediction steps must be positive")
        end
        if isnothing(m)
            error("You must provide a seasonal period 'm'")
        end
        if m <= 1
            error("'m' must be an integer greater than 1")
        end
        if length(init_season) != m
            error("The length of 'init_season' must be equal to `m`")
        end

        cleanParams_(alpha, "alpha")
        cleanParams_(beta, "beta")
        cleanParams_(gamma, "gamma")

        return new(y, h, alpha, beta, gamma, m, init_level, init_trend, init_season)
    end 
end

function makeForecast_(model::HoltWinters) # Return vector of fitted values of length h + SSE
    h = model.h
    alpha = model.alpha
    beta = model.beta
    gamma = model.gamma
    m = model.m
    init_level = model.init_level
    init_trend = model.init_trend
    init_season = model.init_season
    y = model.y
    data_size = length(y)

    lvls = zeros(data_size + 1) # Stores level values, indices (0, T)
    trends = zeros(data_size + 1) # Stores trend values, indices (0, T)
    seasons = zeros(data_size + m) # Stores seasonal values
    forecast = zeros(data_size + h) # Stores forecasted values, yhat, indices (1, T+h)

    lvls[1] = init_level # Set l_0
    trends[1] = init_trend # Set b_0
    seasons[1:m] = init_season # Set s_0 ... s_m-1

    SSE = 0.0 # Compute Sum of Squared Errors

    k = trunc(Integer, (h-1)/m)

    for i in 2:(length(lvls)) # Compute l_t's
        season_idx = i + m - 1 # to start updating values after index m

        lvls[i] = alpha * (y[i-1] - seasons[season_idx - m]) + (1 - alpha) * (lvls[i-1] + trends[i-1]) # level equation
        trends[i] = beta * (lvls[i] - lvls[i-1]) + (1 - beta) * trends[i-1] # trend equation 
        seasons[season_idx] = gamma * (y[i-1] - lvls[i-1] - trends[i-1]) + (1 - gamma) * seasons[season_idx - m] # season equation
        forecast[i-1] = lvls[i-1] + 1 * trends[i-1] + seasons[season_idx + 1 - m] # one-step ahead forecast
        SSE += (forecast[i-1] - y[i-1])^2
    end

    for i in 1:h # Set forecasted values
        forecast[data_size + i] = lvls[end] + i * trends[end + i - m * (k+1)]
    end

    return forecast, SSE
end

function SSE_(model::HoltWinters; alpha = nothing, beta = nothing, gamma = nothing, init_level = nothing, init_trend = nothing, verbose = 0) # Helper function for optimization
    model.alpha = alpha
    model.beta = beta
    model.gamma = gamma
    model.init_level = init_level
    model.init_trend = init_trend

    res = makeForecast_(model)[2]
    if verbose > 0
        println("Error: $res, Alpha: $(model.alpha), Beta*: $(model.beta), Gamma: $(model.gamma), L0: $(model.init_level), BO:$(model.init_trend), S0:$(model.init_season)")
    end

    return res
end

function fit(model::HoltWinters; v=0) # Set alpha + init_level
    # Creating a copy of the user inputted params for the model as model.alpha and model.init_level will be updated later
    base_alpha = model.alpha
    base_beta = model.beta
    base_gamma = model.gamma
    base_init_level = model.init_level
    base_init_trend = model.init_trend

    # copy of model to return to ensure that input model is not modified
    retModel = deepcopy(model)

    if (isnothing(retModel.alpha))
        @warn "Since no value was entered for 'alpha', it will be chosen"
    end
    if (isnothing(retModel.beta))
        @warn "Since no value was entered for 'beta', it will be chosen"
    end
    if (isnothing(retModel.gamma))
        @warn "Since no value was entered for 'gamma', it will be chosen"
    end
    if (isnothing(retModel.init_level))
        @warn "Since no value was entered for 'init_level', it will be chosen"
    end
    if (isnothing(retModel.init_trend))
        @warn "Since no value was entered for 'init_trend', it will be chosen"
    end

   
    # Optimize makeForecast_(model) - directly updates model
    f(x) = SSE_(retModel, 
                alpha=(isnothing(base_alpha) ? x[1] : base_alpha),
                beta=(isnothing(base_beta) ? x[2] : base_beta),
                gamma=(isnothing(base_gamma) ? x[3] : base_gamma),
                init_level=(isnothing(base_init_level) ? x[4] : base_init_level),
                init_trend=(isnothing(base_init_trend) ? x[5] : base_init_trend),
                verbose=v)

    
    if isnothing(retModel.alpha) || isnothing(retModel.beta) || (isnothing(retModel.gamma)) || isnothing(retModel.init_level) || isnothing(retModel.init_trend) # if at least one value needs to be optimized
        # lower and upper limits for alpha, beta, gamma, init_level, init_trend
        lower = [0.0, 0.0, 0.0, -Inf, -Inf] 
        upper = [1.0, 1.0, 1.0, Inf, Inf]

        # Initial values for alpha, beta, init_level, init_trend (optimization starting points)
        first_l0 = model.y[1]
        first_b0 = model.y[1]
        first_alpha = 0.0
        first_beta = 0.0
        first_gamma = 0.0

        # Optimize the SSE of the model through the function "f"
        init_inputs = [first_alpha, first_beta, first_gamma, first_l0, first_b0]
        res = optimize(f, lower, upper, init_inputs)
        
        if v > 0
            println(res)
        end
    end

    return retModel
end

