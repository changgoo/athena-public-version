athena
======

Athena++ radiation MHD code

This is a version with a turbulence driver using the the Plimpton's library for parallel FFT (http://www.sandia.gov/~sjplimp/docs/fft/README.html).

See src/fft/turbulence.cpp for more details.

Simple test run can be done with a configuration option

./configure.py --prob=turb -fft -mpi --eos=isothermal
