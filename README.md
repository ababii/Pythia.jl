# Pythia.jl
Machine learning time series regressions in Julia.

Currently the package supports LASSO, group LASSO, and sparse group LASSO solved via proximal gradient descent.

The sparse group LASSO will be used for mixed frequency data regressions after applying suitable transformations with Legendre polynomials.
