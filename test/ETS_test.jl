using Pythia
using Test
using CSV
using DataFrames

# Using Examples from Forecasting: Principles and Practice (2nd ed.) by Rob J Hyndman and George Athanasopoulos

### Exponential Smoothing Tests ###

## Simple Exponential Smoothing Tests

# Import Data
oildata = CSV.read("fpp2_datasets/oil.csv", DataFrame)
y = oildata[!, 3] 
y = y[32:49]

# Initialize Model and Make Prediction
mdl = SES(y, h = 5)
fitted_mdl = fit(mdl, v = 1)
yhat = predict(fitted_mdl)
y_actual = 

print(yhat) # Not sure what to compare this to but I know its correct though.