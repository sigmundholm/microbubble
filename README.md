

### Compile in Docker-container

Get the Deal.II docker image from https://hub.docker.com/r/dealii/dealii/
```
$ docker pull dealii/dealii
```
Run an interactive shell in the container with this project as a volume in
 the container
```
$ docker run -itv ~/path/to/microbubble:/home/dealii/microbubble
 dealii/dealii:<tag> 
```
If the project have already been compiled on the "outside" it may be
 necessary to delete all Cmake output before running the container.

Inside the container, compile by running
```
$ cd microbubble/
$ cmake .
$ make
```
