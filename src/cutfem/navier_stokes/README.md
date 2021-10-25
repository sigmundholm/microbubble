# Navier-Stokes Equations

This program solves the Navier-Stokes Equations:

```math
  ∂_t u + (u·∇)u - νΔu + ∇·p = f   in Ω
                         ∇·u = 0   in Ω
                           u = g   on ∂Ω,
```
on an unfitted mesh, using cutFEM.