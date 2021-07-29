using Pythia
using Test
 
### Mean Forecast Tests ###
y = [1.0, 2.0, 3.0, 4.0, 5.0]
mdl = MeanForecast(y, h = 5, level = [80, 95])
fittedMdl = fit(mdl)
results = predict(fittedMdl)

@test results.fittedvalues == [1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 3.0, 3.0, 3.0, 3.0]
### Will add tests for prediction intervals in later commits.

### Naive Forecast Tests ###
y = [1.0, 2.0, 3.0, 4.0, 5.0]
mdl = NaiveForecast(y, h = 5, level = [80, 95])
fittedMdl = fit(mdl)
results = predict(fittedMdl)

@test results.fittedvalues == [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
### Will add tests for prediction intervals in later commits.

### Seasonal Naive Model ###