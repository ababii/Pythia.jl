using Pythia
using Test
using CSV
using DataFrames

# Using Examples from Forecasting: Principles and Practice (2nd ed.) by Rob J Hyndman and George Athanasopoulos
# Ground Truth Taken from statsmodels

### Exponential Smoothing Tests ###

## Simple Exponential Smoothing ##

# Import Data
oildata = CSV.read("fpp2_datasets/oil.csv", DataFrame)
y = oildata[!, 3] 
y = y[32:49]

# Initialize Model and Make Prediction
mdl = SES(y, h = 5)
fitted_mdl = fit(mdl, v = 0)
yhat = predict(fitted_mdl)
yactual = [542.6803683882322, 542.6803683882322, 542.6803683882322, 542.6803683882322, 542.6803683882322]

@test yhat ≈ yactual rtol=0.05
# Compare yhat with ground truth

## Holt's Linear Trend Method ##

# Import Data
ausair = CSV.read("fpp2_datasets/ausair.csv", DataFrame)
y = ausair[!, 3]
y = y[21:47]

# Initialize Model and Make Prediction
mdl = Holt(y, h = 5)
fitted_mdl = fit(mdl, v = 0)
yhat = predict(fitted_mdl)
yactual = [74.59331490568717, 76.69123025401483, 78.7891456023425, 80.88706095067016, 82.98497629899784]
@test yhat ≈ yactual rtol=0.05

## Holt's Linear Method - Damped ##
mdl = Holt(y, h = 5, damped = true)
fitted_mdl = fit(mdl, v = 0)
yhat = predict(fitted_mdl)
yactual = [74.4245006855701, 76.3634111773908, 78.2926270423973, 80.21219675409546, 82.12216854362171]
@test yhat ≈ yactual rtol=0.05

## HoltWinter's Method ##

# Import Data
austourists = CSV.read("fpp2_datasets/austourists.csv", DataFrame)
y = austourists[!, 3]
y = y[25:68]

# Initialize Model and Make Prediction
mdl = HoltWinters(y, h = 5, m = 4, init_season = [9.70, -9.31, -1.69, 1.31])
fitted_mdl = fit(mdl, v = 0)
yhat = predict(fitted_mdl)
# @test yhat ≈ yactual rtol=0.05