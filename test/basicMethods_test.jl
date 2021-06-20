using Pythia
using Test

y = [1.0, 2.0, 3.0, 4.0, 5.0]
mdl = MeanForecast(y, h = 5, level = [80, 95])
mdlResult = fit(mdl)

@test mdlResult.fittedvalues = [1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 3.0, 3.0, 3.0, 3.0]
