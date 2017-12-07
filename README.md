athena
======

Athena++ radiation MHD code

This is a version with parallel FFT using Plimpton's domain decomposition library (http://www.sandia.gov/~sjplimp/docs/fft/README.html) and turbulence driver. 

See src/fft/turbulence.cpp for more details.

Simple test run can be done with a configuration option

./configure.py --prob=turb -fft -mpi --eos=isothermal
