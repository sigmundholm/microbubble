# Microbubble

This repository contains the code written for my master thesis.

The goal of my thesis is to simulate the movement of a microbubble in a capillary blood vein, using the Cut Finite Element Method (CutFEM). The project uses the finite element library `deal.ii` [1], in addition to code developed by Simon Sticko for the CutFEM spesific implementations.


## Implemented solvers
The following solvers all use CutFEM, and hence solve the problems on an unfitted mesh.

#### Stationary problems
- The Poisson Problem: `poisson`
- Stokes Equation: `stokes`
- Generalized Stokes Equation: `stokes_gen`

#### Time dependent problems
- The Heat Equation: `heat_eqn`
- Time Dependent Stokes (Implicit Euler): `stokes_time`
- Time Dependent Stokes (BDF-2): `stokes_time2`


In addition, solvers for the Poisson problem and the Stokes Equations are also implemented on a fitted mesh, using Nitsches method for the Dirichlet boundary conditions.

## Compiling
### Compiling with CutFEM
The code written in this project depends upon a CutFEM library under development by Simon Sticko, and is therefore needed when compiling.

Endte med å måtte sette path til DealiiCutElements når jeg kjører cmake:
Create a folder `build` in the root of this project, then set the path to the project `DealiiCutElements` when running `cmake`, with the following command
```shell script
build $ cmake -DCUTFEM_DIR=~/git/path/to/dealiicutelements/install ..
build $ make -j4
```

### Compile in a Docker-container
Note that this only works for the code not using CutFEM, i.e. the solvers in the directory `src/nitsche`.

Get the Deal.II docker image from https://hub.docker.com/r/dealii/dealii/
```
$ docker pull dealii/dealii
```
Run an interactive shell in the container with this project as a volume in
 the container
```shell script
$ docker run -itv ~/path/to/microbubble:/home/dealii/microbubble
 dealii/dealii:<tag> 
```
If the project have already been compiled on the "outside" it may be
 necessary to delete all Cmake output before running the container.

Inside the container, compile by running
```shell script
$ cd microbubble/
$ cmake .
$ make
```



#### References
 [1] https://www.dealii.org/