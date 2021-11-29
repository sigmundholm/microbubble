# Heat Equation

This program solves the Heat Equation,
```math
  ∂_t u - νΔu  = f   in Ω
             u = g   on ∂Ω,
```
using CutFEM.

The implemented time step methods for a moving domain are
 - BDF-1 (Implicit Euler)
 - Crank-Nicholson
 - BDF-2 (see below)
 - BDF-3 (see below)

For a stationary domain, only BDF-methods with interpolated initial steps are supported.

### Convergence problems
Note that when using BDF-2, the implementation does not attain full convergence in the L^2-norm, when the first step is computed using BDF-1. When the first step is computed using Crank-Nicholson, the expected 2. order convergence is achieved. 

The same is observed for BDF-3, when BDF-1 and BDF-2 are used for initial steps. The lower than expected convergence seems to come from the error in the beginning from the initial BDF-methods. When this initial error is cut away from the error computations, the correct order of convergence is attained. This can be seen by running the file `tau_comparison.py`. However, the correct convergence is attained when the initial steps are interpolated. 
