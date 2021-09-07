# Heat Equation

This program solves the Heat Equation,
```math
  ∂_t u - νΔu  = f   in Ω
             u = g   on ∂Ω,
```
using CutFEM.

The implemented time step methods are
 - BDF-1 (Implicit Euler)
 - Crank-Nicholson
 - BDF-2
 - BDF-3 (?)

Note that when using BDF-2, the implementation does not attain full convergence when the first step is computed using BDF-1. When the first step is computed using Crank-Nicholson, the expected 2. order convergence is achieved.
