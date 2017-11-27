//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fft.cpp
//  \brief Problem generator for complex-to-complex FFT test.
//

// C++ headers
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <ctime>
#include <iomanip>

// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../fft/athena_fft.hpp"
#include "../mesh/mesh.hpp"

#ifdef OPENMP_PARALLEL
#include "omp.h"
#endif

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin)
{
  Real x0=0.0, y0=0.0, z0=0.0, r2;
  int i,j,k;
  int nbs=nslist[Globals::my_rank];
  int nbe=nbs+nblist[Globals::my_rank]-1;
  int is=pblock->is, ie=pblock->ie;
  int js=pblock->js, je=pblock->je;
  int ks=pblock->ks, ke=pblock->ke;

  AthenaArray<Real> src, dst;
  int nx1=pblock->block_size.nx1+2*NGHOST;
  int nx2=pblock->block_size.nx2+2*NGHOST;
  int nx3=pblock->block_size.nx3+2*NGHOST;
  
  src.NewAthenaArray(nx3,nx2,nx1);
  dst.NewAthenaArray(2,nx3,nx2,nx1);

  if(FFT_ENABLED){
    FFTDriver *pfftd;
    pfftd = new FFTDriver(this, pin);
    pfftd->InitializeFFTBlock(true);
    pfftd->QuickCreatePlan();

    FFTBlock *pfft = pfftd->pmy_fb;
  // Repeating FFTs for timing
    int ncycle = pin->GetOrAddInteger("problem","ncycle",100);
    if(ncycle > 0){
    if(Globals::my_rank == 0){
      std::cout << "=====================================================" << std::endl;
      std::cout << "Initialize...                                        " << std::endl;
      std::cout << "=====================================================" << std::endl;
    }

    for(int igid=nbs;igid<=nbe;igid++){
      MeshBlock *pmb=FindMeshBlock(igid);
      if(pmb != NULL){
        Coordinates *pcoord = pmb->pcoord;
        for (k=ks; k<=ke; ++k) {
        for (j=js; j<=je; ++j) {
        for (i=is; i<=ie; ++i) {
          if (COORDINATE_SYSTEM == "cartesian") {
            Real x = pcoord->x1v(i);
            Real y = pcoord->x2v(j);
            Real z = pcoord->x3v(k);
            r2 = sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
          }
          src(k,j,i)= std::exp(-r2);
        }}}
        pfft->LoadSource(src,1,NGHOST,pmb->loc,pmb->block_size);
      }
    }

    if(Globals::my_rank == 0){
      std::cout << "=====================================================" << std::endl;
      std::cout << "End Initialization...                                " << std::endl;
      std::cout << "=====================================================" << std::endl;
    }

    if(Globals::my_rank == 0){
      std::cout << "=====================================================" << std::endl;
      std::cout << "Execute FFT " << ncycle << "                         " << std::endl;
      std::cout << "=====================================================" << std::endl;
    }

    clock_t tstart = clock();
#ifdef OPENMP_PARALLEL
    double omp_start_time = omp_get_wtime();
#endif
    for (int n=0; n <= ncycle; n++) {
      pfft->ExecuteForward();
      pfft->ExecuteBackward();
    }
#ifdef OPENMP_PARALLEL
    double omp_time = omp_get_wtime() - omp_start_time;;
#endif
    clock_t tstop = clock();
    float cpu_time = (tstop>tstart ? (float)(tstop-tstart) : 1.0)/(float)CLOCKS_PER_SEC;
    int64_t zones = GetTotalCells();
    float zc_cpus = (float)(zones*ncycle)/cpu_time;

    if(Globals::my_rank == 0){
      std::cout << std::endl << "cpu time used  = " << cpu_time << std::endl;
      std::cout << "zone-cycles/cpu_second = " << zc_cpus << std::endl;
#ifdef OPENMP_PARALLEL
      float zc_omps = (float)(zones*ncycle)/omp_time;
      std::cout << std::endl << "omp wtime used = " << omp_time << std::endl;
      std::cout << "zone-cycles/omp_wsecond = " << zc_omps << std::endl;
#endif
    }
    }
// Reset everything and do FFT once for error estimation
    for(int igid=nbs;igid<=nbe;igid++){
      MeshBlock *pmb=FindMeshBlock(igid);
      if(pmb != NULL){
        Coordinates *pcoord = pmb->pcoord;
        for (k=ks; k<=ke; ++k) {
        for (j=js; j<=je; ++j) {
        for (i=is; i<=ie; ++i) {
          if (COORDINATE_SYSTEM == "cartesian") {
            Real x = pcoord->x1v(i);
            Real y = pcoord->x2v(j);
            Real z = pcoord->x3v(k);
            r2 = sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
          }
          src(k,j,i) = std::exp(-r2);
        }}}
        pfft->LoadSource(src,1,NGHOST,pmb->loc,pmb->block_size);
      }
    }

    pfft->ExecuteForward();
    pfft->ApplyKernel(0);
    pfft->ExecuteBackward();

    Real err1_tot=0.0,err2_tot=0.0;

    for(int igid=nbs;igid<=nbe;igid++){
      MeshBlock *pmb=FindMeshBlock(igid);
      if(pmb != NULL){
        pfft->RetrieveResult(dst,2,NGHOST,pmb->loc,pmb->block_size);

        Real err1=0.0,err2=0.0;
        Coordinates *pcoord = pmb->pcoord;
        for (k=ks; k<=ke; ++k) {
        for (j=js; j<=je; ++j) {
        for (i=is; i<=ie; ++i) {
          if (COORDINATE_SYSTEM == "cartesian") {
            Real x = pcoord->x1v(i);
            Real y = pcoord->x2v(j);
            Real z = pcoord->x3v(k);
            r2 = sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
          }
          src(k,j,i) = std::exp(-r2);

          err1 += std::abs(dst(0,k,j,i) - src(k,j,i));
          err2 += std::abs(dst(1,k,j,i));
        }}}

        std::cout << std::setprecision(15) << std::scientific;
        std::cout << "=====================================================" << std::endl;
        std::cout << "Block number: " << igid << std::endl;
        std::cout << "rmax: " << r2 << " k,j,i " << k << " " << j << " " << i << std::endl;
        std::cout << "Block loc: " << pmb->loc.lx1 << " "  << pmb->loc.lx2 << " "  << pmb->loc.lx3 << std::endl;
        std::cout << "Error for Real: " << err1 <<" Imaginary: " << err2 << std::endl;
        std::cout << "=====================================================" << std::endl;

        err1_tot += err1;
        err2_tot += err2;
      }
    }

    if(Globals::my_rank == 0){
      std::cout << std::setprecision(15) << std::scientific;
      std::cout << "=====================================================" << std::endl;
      std::cout << "Error for Real: " << err1_tot <<" Imaginary: " << err2_tot << std::endl;
      std::cout << "=====================================================" << std::endl;
    }
  }

  return;
}
