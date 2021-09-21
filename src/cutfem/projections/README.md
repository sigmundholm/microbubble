# L2 Projections with CutFEM

This program performs L^2 projections by solving systems on the form

```math
  (u, v) = (g, v) for all v ∈ V,
```
over some domain Ω, on an unfitted mesh, using CutFEM.

Implemented:
 - `projections_mixed`: For mixed finite element problems. Made for projecting a vector velocity 
   function and a pressure scalar field given as Functions into a finite element space.
