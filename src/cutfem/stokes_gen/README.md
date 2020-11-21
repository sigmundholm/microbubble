# Generalized Stokes

This solves the Generalized Stokes Equation:

```math
  δu - τνΔu + τ∇·p = f   in Ω
               ∇·u = 0   in Ω
                 u = g   on ∂Ω,
```

on an unfitted mesh, using cutFEM.