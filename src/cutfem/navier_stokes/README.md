# Navier-Stokes Equations

This program solves the Navier-Stokes Equations:

```math
  ∂_t u + (u·∇)u - νΔu + ∇·p = f   in Ω
                         ∇·u = 0   in Ω
                           u = g   on ∂Ω,
```
on an unfitted mesh, using cutFEM.

The convection term can either be assembled explicitly, by including it on the right hand side, or by using a 
semi-implicit method. Then the convection term is linearised by extrapolating the solution from the earlier time
steps. The desired assembly method is chosen by setting the flag `semi_implicit` in the constructor. If `semi_implcit`
is set to `false`, the convection term is assembled explicitly.

## Solvers
 - Stationary Navier-Stokes
    - Set the flag `stationary` to `true` in the constructor.
    - Explicit convection term
       - Use the method `run_step` or `run_step_non_linear`.
    - Semi-implicit convection term
       - Use the method `run_step_non_linear`.
 - Time dependent Navier-Stokes
    - Set the flag `stationary` to `false` in the constructor.
    - Moving domain
       - Either explicit or semi-implicit convection term.
       - Use the method `run_time_moving_domain`.
    - Stationary domain
       - Either explicit or semi-implicit convection term.
       - Use the method `run_time`. 
