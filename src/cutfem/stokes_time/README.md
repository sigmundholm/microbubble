# Time Dependent Stokes

This solves the Time Dependent Stokes Equation:

```math
  ∂_t u - νΔu + ∇·p = f   in Ω
                ∇·u = 0   in Ω
                  u = g   on ∂Ω,
```
on an unfitted mesh, using cutFEM. The time derivative is discretized using Implicit Euler, so in each time step the Generalized Stokes equation is solved.