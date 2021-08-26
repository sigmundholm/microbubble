# Generalised Poisson -> Heat Equation

The goal of this program is to solve the Heat Equation. The time discretisation method will be BDF-1 and BDF-2. The Heat equation is given by
```math
  ∂_t u - νΔu  = f   in Ω
             u = g   on ∂Ω.
```
As a first step the generalised Poisson equation will be solved, given by
```math
  u - τνΔu  = τf   in Ω
          u = g   on ∂Ω.
```
