//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file sne.cpp
//  \brief Problem generator for spherical blast wave problem. Works in spherical
//         coordinates. Unlike blast.cpp, this launches the blast from the center. 
//         Because of the coordinate singularity at r = 0, the blast in launched 
//         from near the center of a cone, but not the very cener. 
//         Currently, no magetic field is enabled. 
//   

// C++ headers
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"


// Problem specific macros

// k_b* 1 Kelvin in ergs. Multiply by this to get from kT to erg.
// (just kb; duh)
#define KELVIN_TIMES_KB_ERG 1.3806488e-16

// Multiply by this to get erg/cm^3 to code units
#define ERG_CM3_CODE 4.270455e13

// Multiply by this to get code units to m_p/cm3
#define CODE_RHO_MP_CM3 1.4

// Multiply by this to get code units to Myr
#define CODE_TIME_MYR 0.97779

// constants for this problem
static const Real T_PE = 1.0e4; // Kelvin
static const Real kb = 1.3806488e-16; // in cgs
static const Real mu = 0.62; // for ionized H+he
static const Real mp_g = 1.6726e-24; // multiply by this to turn mp to g
static const Real muH = CODE_RHO_MP_CM3; // for relating nH and rho for neutral gas
static const Real PC_IN_CM = 3.08568e+18;
static const Real PC3_IN_CM3 = PC_IN_CM*PC_IN_CM*PC_IN_CM;
static const Real MSUN_IN_MP = 1.188837e57;
static const Real MYR_IN_SEC = 3.15576e13;
static const Real MASS_UNIT_G = 6.8798259e31; // rho * L^3
static const Real ENERGY_UNIT_ERG = 6.879825e41; // e * L^3
static const Real CODE_TIME_SEC = CODE_TIME_MYR*MYR_IN_SEC;
static const Real max_time_myr = 40; // stop SNe after this long. 

Real T_floor, T_max, cfl_cool, gm1, Gamma0, frac_sphere, delta_t_SNe_code, E51;
Real rblast, pa, da, M_ej, r_min, dt_SN, N_dt, kappa_max, r_min_cen, lambda_dv;
Real kappa_max_rho; // such that kappa_max ~ rho/rho_kappa_thresh
std::string feedback; 
int kappa_flag;
int N_exploded, N_exploded_last, N_step_since_exploded;
bool saturation_on, cooling_on;

static Real hst_Ethermal(MeshBlock *pmb, int iout);
static Real hst_Ehot(MeshBlock *pmb, int iout);
static Real hst_Ehot_thermal(MeshBlock *pmb, int iout);
static Real prad_bubble(MeshBlock *pmb, int iout);
static Real hst_Mhot(MeshBlock *pmb, int iout);
static Real hst_Mtot(MeshBlock *pmb, int iout);
static Real hst_Etot(MeshBlock *pmb, int iout);
static Real hst_Vbubble(MeshBlock *pmb, int iout);
Real hst_CoolingLosses(MeshBlock *pmb, int iout);
static Real Lambda_T(const Real T);
static Real tcool(const Real T, const Real nH);
static Real cooling_timestep(MeshBlock *pmb);
static Real hst_Mshell(MeshBlock *pmb, int iout); 
static Real hst_Rshell(MeshBlock *pmb, int iout); 
static Real Etot_bubble(MeshBlock *pmb, int iout);
static Real cond_timestep(MeshBlock *pmb, int iout);
static Real max_T(MeshBlock *pmb, int iout);
static Real hst_Vshell(MeshBlock *pmb, int iout);

Real get_minimum_radius(Mesh *pm);
void get_values_for_injection_region(Mesh *pm, Real data_vals[]);
void insert_SNe(MeshBlock *pmb);
void Cooling(MeshBlock *pmb, const Real t, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, 
    AthenaArray<Real> &cons);
static Real parker_conductivity(Real T_K);
static Real kappa_nonlinear_code_units(Real rho_code, Real lamma_deltav);
static Real spitzer_conductivity(Real T_K, Real ne_cm3);
static Real KappaCodeUnits(const AthenaArray<Real> &prim, int k, int j, int i);
void SpitzerConduction(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke);

//========================================================================================
// InitUserMeshData. Called at the beginning of the simulation 
//========================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  T_floor = pin->GetReal("problem", "T_floor"); 
  T_max = pin->GetReal("problem", "T_max"); 
  cfl_cool = pin->GetReal("problem", "cfl_cool");
  gm1 = pin->GetReal("hydro", "gamma") - 1;
  E51 = pin->GetReal("problem","E51");
  rblast = pin->GetReal("problem","r_blast");
  
  Real pamb   = pin->GetReal("problem","pamb");
  M_ej = pin->GetReal("problem", "M_ej");
  
  pa = pamb * KELVIN_TIMES_KB_ERG * ERG_CM3_CODE; // pressure in code units 
  da   = pin->GetReal("problem","damb");
  rblast = pin->GetReal("problem", "r_blast");
  r_min = pin->GetReal("mesh","x1min");
  
  delta_t_SNe_code = pin->GetReal("problem", "delta_t_SNe")/CODE_TIME_MYR; 
  
  Real phi_min = pin->GetReal("mesh", "x3min");
  Real phi_max = pin->GetReal("mesh", "x3max");
  Real theta_min = pin->GetReal("mesh", "x2min");
  Real theta_max = pin->GetReal("mesh", "x2max");
  frac_sphere = (phi_max - phi_min) * (std::cos(theta_min) - std::cos(theta_max))/(4*PI);
  
  feedback = pin->GetOrAddString("problem", "feedback", "thermal");
  
  // set heating to balance cooling exactly 
  Gamma0 = da*Lambda_T( pamb/(da*(muH/mu)) );
  
  
  // start the counter for SNe explosions
  
  
  if (time == 0) {
      N_exploded = 1;
  } else {
      N_exploded = std::floor(time/delta_t_SNe_code) + 1;
  }
  N_exploded_last = N_exploded - 1;
  N_step_since_exploded = 0;
  
  kappa_flag = pin->GetReal("problem", "kappa_iso");
  saturation_on = pin->GetOrAddBoolean("problem", "saturation_on", true);
  cooling_on = pin->GetOrAddBoolean("problem", "cooling_on", true);
  
  N_dt = pin->GetOrAddReal("problem", "N_dt", 1);
  dt_SN = pin->GetOrAddReal("problem", "dt_SN", 1.0e-10);
  Real kappa_max_T = pin->GetReal("problem", "kappa_max_T"); // kelvin
  
  kappa_max_rho = pin->GetReal("problem", "kappa_max_rho");
  kappa_max = spitzer_conductivity(kappa_max_T, 0.01); // kappa thresh with ne = 0.01, ergs. 
  
  lambda_dv = pin->GetOrAddReal("problem", "lambda_dv", 0.0);
  
  //r_min_cen = 1e30;
  //r_min_cen = get_minimum_radius(pblock->pmy_mesh);
  r_min_cen = r_min;
  
  // Enroll source function
  if (cooling_on) {
    EnrollUserExplicitSourceFunction(Cooling);
  }
  EnrollUserTimeStepFunction(cooling_timestep);
  
  int num_hist = 12;
  if (kappa_flag > 0) {
    std::cout << "conduction is on!: " << std::endl;
    EnrollConductionCoefficient(SpitzerConduction);
    num_hist += 2;
  }

  AllocateUserHistoryOutput(num_hist);
  EnrollUserHistoryOutput(0, hst_Mhot, "Mhot");
  EnrollUserHistoryOutput(1, hst_Mtot, "Mtot");
  EnrollUserHistoryOutput(2, hst_Vbubble, "Vbubble");
  EnrollUserHistoryOutput(3, hst_Mshell, "Mshell");
  EnrollUserHistoryOutput(4, hst_Rshell, "MRshell");
  EnrollUserHistoryOutput(5, hst_Etot, "Etot");
  EnrollUserHistoryOutput(6, hst_Ethermal, "Ethermal");
  EnrollUserHistoryOutput(7, hst_Ehot, "Ehot");
  EnrollUserHistoryOutput(8, hst_Ehot_thermal, "Ehot_thermal");
  EnrollUserHistoryOutput(9, prad_bubble, "prad_bubble");
  EnrollUserHistoryOutput(10, Etot_bubble, "Etot_bubble");
  EnrollUserHistoryOutput(11, hst_Vshell, "Vshell");
  if (kappa_flag > 0) {
    EnrollUserHistoryOutput(12, cond_timestep, "cond_timestep");
    EnrollUserHistoryOutput(13, max_T, "Tmax");
  } 
}

// get the minimum cell-centered radius in the whole domain
Real get_minimum_radius(Mesh *pm) {
  MeshBlock *pmb=pm->pblock;
  while (pmb != NULL) {
    Real this_rmin = pmb->pcoord->x1v(pmb->is);
    if (this_rmin < r_min_cen) r_min_cen = this_rmin;
    pmb=pmb->next;
  }
  std::cout << "found a minimum radius cell of " << r_min_cen << std::endl;
  return r_min_cen;
}

// figure out the density we want inside a given injection region
// return an array of [rho_ej, r2dv, SNvolume, Ethermal_old]
void get_values_for_injection_region(Mesh *pm, Real data_vals[]) {
  
  MeshBlock *pmb=pm->pblock;
  
  Real M_enc = M_ej;
  
  while (pmb != NULL) {
    
    std::cout << "min radius is  " << pmb->pcoord->x1v(pmb->is) << std::endl;
    std::cout << "next is   " << pmb->next << std::endl;
    
    if (pmb->pcoord->x1v(pmb->is) > rblast) {
      pmb=pmb->next;
      continue;
    } 
    // hack but works if only injecting in one block
    r_min_cen = pmb->pcoord->x1v(pmb->is);
        
    Real SN_volume = 0;
    Real SN_mass = 0;
    Real actual_r2dv = 0;
    Real E_thermal_old_SN = 0;
  
    for (int k=pmb->ks; k<=pmb->ke; k++) {
      for (int j=pmb->js; j<=pmb->je; j++) {
        for (int i=pmb->is; i<=pmb->ie; i++) {
          Real r = pmb->pcoord->x1v(i);
          if (r < rblast) {
            Real cell_volume = pmb->pcoord->GetCellVolume(k,j,i);
            SN_volume += cell_volume/frac_sphere;
            SN_mass += pmb->phydro->u(IDN,k,j,i)*CODE_RHO_MP_CM3*cell_volume*PC3_IN_CM3/MSUN_IN_MP/frac_sphere;
            actual_r2dv += SQR((r - r_min_cen))*cell_volume/frac_sphere;
            Real this_eth = (pmb->phydro->u(IEN,k,j,i) - (SQR(pmb->phydro->u(IM1,k,j,i)) +
                 SQR(pmb->phydro->u(IM2,k,j,i))+ SQR(pmb->phydro->u(IM3,k,j,i)))/(2*pmb->phydro->u(IDN,k,j,i)));
            E_thermal_old_SN += this_eth*cell_volume/frac_sphere;
          }
        }
      }
    }
    M_enc += SN_mass;
    data_vals[1] += actual_r2dv;
    data_vals[2] += SN_volume;
    data_vals[3] += E_thermal_old_SN;
    
    pmb=pmb->next;
  }
  data_vals[0] = M_enc*MSUN_IN_MP/(data_vals[2]*PC3_IN_CM3*CODE_RHO_MP_CM3);
}

void MeshBlock::UserWorkInLoop(void) {
  
  // only call this function if a supernova is supposed to happen this timestep
  Real t = phydro->pmy_block->pmy_mesh->time;
  if (t > max_time_myr*CODE_TIME_MYR) {
    return;
  } 
  
  if ( (t/delta_t_SNe_code > N_exploded)) {
    std::cout << "injecting SN at time " << t << "; have done " << N_exploded << std::endl;
    insert_SNe(phydro->pmy_block);    
    N_exploded += 1;
  } 
    return;
  }

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin){
  AllocateUserOutputVariables(2);
  return;
}

// calculate the conductive flux and the saturation limit 
void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  if (kappa_flag > 0) {
    for(int k=ks; k<=ke; k++) {
      for(int j=js; j<=je; j++) {
        for(int i=is; i<=ie; i++) {
  		    Real kappa_primef = 0.5*(phydro->phdif->kappa(ISO,k,j,i) + 
              phydro->phdif->kappa(ISO,k,j,i-1));
          Real dTdx = (phydro->w(IPR,k,j,i)/phydro->w(IDN,k,j,i) - phydro->w(IPR,k,j,i-1)/
                  phydro->w(IDN,k,j,i-1))/pcoord->dx1v(i-1);
          Real x1flux = kappa_primef*dTdx;
          Real x1flux_cgs = x1flux*ENERGY_UNIT_ERG/CODE_TIME_SEC/(PC_IN_CM*PC_IN_CM);
        
          Real q_sat = 1.5*std::pow(phydro->w(IPR,k,j,i), 1.5)/std::sqrt(phydro->w(IDN,k,j,i));
          Real q_sat_cgs = q_sat*ENERGY_UNIT_ERG/CODE_TIME_SEC/(PC_IN_CM*PC_IN_CM);

          user_out_var(0,k,j,i) = x1flux_cgs;
          user_out_var(1,k,j,i) = q_sat_cgs;
        }
      }
    }
  }
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Spherical blast wave test problem generator
//========================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  // Only spherical coordinates for now 
  if (COORDINATE_SYSTEM != "spherical_polar") {
      std::stringstream msg;
      msg << "### FATAL ERROR in blast.cpp ProblemGenerator" << std::endl
          << "Unrecognized COORDINATE_SYSTEM= " << COORDINATE_SYSTEM << std::endl;
      throw std::runtime_error(msg.str().c_str());
  } 

  // setup uniform ambient medium 
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        phydro->u(IDN,k,j,i) = da;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        phydro->u(IEN,k,j,i) = pa/gm1; 
      }
    }
  }
  
  
  // start with a SN at t=0
  insert_SNe(phydro->pmy_block);
  
  // No magnetic fields for now 
  if (MAGNETIC_FIELDS_ENABLED) {
    std::stringstream msg;
    msg << "### FATAL ERROR in blast.cpp ProblemGenerator" << std::endl
        << "Magnetic fields not currently supported." << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
}


//========================================================================================
//! utility functions
//========================================================================================

void insert_SNe(MeshBlock *pmb) {
  
  if (pmb->pcoord->x1v(pmb->is) > rblast) return;
  
  //[rho_ej, r4dv, SNvolume, Ethermal_old]
  Real *data_vals = new Real[4];
  for (int n=0; n<4; ++n) data_vals[n]=0.0;
  get_values_for_injection_region(pmb->pmy_mesh, data_vals);
  
  // hack for now
  r_min_cen = pmb->pcoord->x1v(pmb->is);
  
  Real rho_ej = data_vals[0];
  Real actual_r2dv =  data_vals[1];
  Real SN_volume =  data_vals[2];
  Real E_thermal_old_SN = data_vals[3];
  delete [] data_vals;
  
  std::cout << "r_min_cen: " << r_min_cen << std::endl;
  std::cout << "rho_ej: " << rho_ej << std::endl;
  std::cout << "actual_r2dv: " << actual_r2dv << std::endl;
  std::cout << "SN_volume: " << SN_volume << std::endl;
  std::cout << "E_thermal_old_SN: " << E_thermal_old_SN << std::endl;
  

  Real e_hot = E51 * 1e51 * ERG_CM3_CODE / (SN_volume * PC3_IN_CM3); 
  Real v0 = 4313.0*std::sqrt(E51)*(rblast/4)/std::sqrt(actual_r2dv/2500)/std::sqrt(rho_ej);
  //Real v0 = 8626.7*std::sqrt(E51)*SQR(rblast/4)/std::sqrt(actual_r4dv/1.0e4)/std::sqrt(rho_ej);
  
  std::cout << "v0: " << v0 << std::endl;  
  
  for (int k=pmb->ks; k<=pmb->ke; k++) {
    for (int j=pmb->js; j<=pmb->je; j++) {
      for (int i=pmb->is; i<=pmb->ie; i++) {
        Real r = pmb->pcoord->x1v(i);
        if (r < rblast) {
          Real &rho = pmb->phydro->u(IDN,k,j,i);
          Real &e = pmb->phydro->u(IEN,k,j,i);
          Real &m1 =pmb->phydro->u(IM1,k,j,i);
          Real &m2 =pmb->phydro->u(IM2,k,j,i);
          Real &m3 =pmb->phydro->u(IM3,k,j,i);
          
          if (feedback == std::string ("thermal")) {
            rho = rho_ej;
            e += e_hot;   
          }  else if (feedback == std::string ("kinetic")) {
            rho = rho_ej;
            m1 += v0*((r - r_min_cen)/rblast)*rho_ej;
            //m1 += v0*(SQR((r - r_min_cen)/rblast))*rho_ej;
            e = (SQR(m1) + SQR(m2)+ SQR(m3))/(2*rho) + E_thermal_old_SN/SN_volume;
          } else if (feedback == std::string ("sedov")) { // just a 70/30 mix of thermal and kinetic
            rho = rho_ej;
            m1 += v0*((r - r_min_cen)/rblast)*rho_ej*std::sqrt(0.283);
            e = E_thermal_old_SN/SN_volume + (SQR(m1) + SQR(m2)+ SQR(m3))/(2*rho) + 0.717*e_hot;
          } else {
            std::stringstream msg;
            msg << "### FATAL ERROR bad feedback key" << std::endl;
            throw std::runtime_error(msg.str().c_str());
          }
        }
      }
    }
  }

}

// conductivity from footnote 6 of Gnat paper, or Spitzer 1962
// returns takes T in Kelvin and n_e in cm^-3, and returns a value
// in erg/s/cm/K
static Real spitzer_conductivity(Real T_K, Real ne_cm3) {
  Real lnLambda = 29.7 + std::log((T_K/1e6)/std::sqrt(ne_cm3));
  Real kappa = 1.84e-5*std::pow(T_K, 2.5)/lnLambda;
  return kappa;
}

static Real parker_conductivity(Real T_K) {
  Real kappa = 2.5e3*std::sqrt(T_K);
  return kappa;
}


// lamma_deltav is in km/s * pc, rho is in code units
static Real kappa_nonlinear_code_units(Real rho_code, Real lamma_deltav){
    Real lamma_deltav_cgs = lamma_deltav*1e5*PC_IN_CM;
    Real kappa_min_cgs = 1.5*lamma_deltav_cgs*rho_code*CODE_RHO_MP_CM3*kb/mu;
    Real this_kappa_max = kappa_max*rho_code/kappa_max_rho;
    kappa_min_cgs = std::min(kappa_min_cgs, this_kappa_max); // erg/s/cm/K
    Real kf_min_cgs = kappa_min_cgs*mu*mp_g/kb; // in cgs, kappa*mu*m_p/k_B
    Real kappa_min_code = kf_min_cgs*CODE_TIME_SEC*PC_IN_CM/MASS_UNIT_G;
    return kappa_min_code;
}

// calculates the spitzer/parker conductivity value in code units.
// this is actually kappa * mu * m_p/k_B. It has dimensions [mass]/([length]*[time])
// kappa_max is the cgs value of the threshold we assume
static Real KappaCodeUnits(const AthenaArray<Real> &prim, int k, 
    int j, int i) {
  
  Real P_over_k = prim(IPR,k,j,i)/ERG_CM3_CODE/KELVIN_TIMES_KB_ERG;
  Real T_K = P_over_k/((muH/mu)*prim(IDN,k,j,i)); // Kelvin
  
  Real n_e = 1.2 * prim(IDN,k,j,i); // ionized hydrogen and helium; 1.2 * n_H
  
  Real kappa_cgs;
  if (T_K > 6.6e4) {
    kappa_cgs = spitzer_conductivity(T_K, n_e);
  } else {
    kappa_cgs = parker_conductivity(T_K);
  }
  
  Real this_kappa_max = kappa_max*prim(IDN,k,j,i)/kappa_max_rho;
  if (kappa_cgs > this_kappa_max) {
    kappa_cgs = this_kappa_max;
  }
  Real kf_cgs = kappa_cgs*mu*mp_g/kb; // in cgs, kappa*mu*m_p/k_B
  Real kf_code = kf_cgs*CODE_TIME_SEC*PC_IN_CM/MASS_UNIT_G;
  return kf_code;
}

// modelled after the ConstConduction() function
void SpitzerConduction(HydroDiffusion *phdif, MeshBlock *pmb, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bcc, int is, int ie, int js, int je, int ks, int ke) {
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {         
        Real kappa_code = KappaCodeUnits(prim, k, j, i);
        
        Real kappa_eff;
        if (saturation_on) {
          Real dTdx = (prim(IPR,k,j,i+1)/prim(IDN,k,j,i+1) - prim(IPR,k,j,i-1)/
                  prim(IDN,k,j,i-1))/(pmb->pcoord->dx1v(i-1) + pmb->pcoord->dx1v(i));
          Real q_sat = 1.5*std::pow(prim(IPR,k,j,i), 1.5)/std::sqrt(prim(IDN,k,j,i));
          kappa_eff = 1/( std::abs(dTdx/q_sat) + 1/kappa_code ); 
        } else {
          kappa_eff = kappa_code;
        }
        
        // now the nonlinear conductivity
        Real kappa_min = kappa_nonlinear_code_units(prim(IDN,k,j,i), lambda_dv);
        kappa_eff = std::max(kappa_min, kappa_eff);
        
        phdif->kappa(ISO,k,j,i) = kappa_eff; //  dimensions [mass]/([length]*[time])
      }
    }
  }
  return;
}

// cooling
static int nfit_cool = 12;
static Real T_cooling_curve[12] = {0.99999999e2,
   1.0e+02, 6.0e+03, 1.75e+04, 
   4.0e+04, 8.7e+04, 2.30e+05, 
   3.6e+05, 1.5e+06, 3.50e+06, 
   2.6e+07, 1.0e+12};

static Real lambda_cooling_curve[12] =  { 3.720076376848256e-71,
    1.00e-27,   2.00e-26,   1.50e-22,
    1.20e-22,   5.25e-22,   5.20e-22,
    2.25e-22,   1.25e-22,   3.50e-23,
    2.10e-23,   4.12e-21};

static Real exponent_cooling_curve[12] = {1e10,
   0.73167566,  8.33549431, -0.26992783,  
   1.89942352, -0.00984338, -1.8698263 , 
  -0.41187018, -1.50238273, -0.25473349,  
   0.5000359, 0.5 };


// cooling function in erg/s/cm3 
static Real Lambda_T(const Real T) {
  int k, n=nfit_cool-1;
  // first find the temperature bin 
  for(k=n; k>=0; k--){
    if (T >= T_cooling_curve[k])
      break;
  }
  if (T > T_cooling_curve[0]){
    return (lambda_cooling_curve[k] * 
              std::pow(T/T_cooling_curve[k], exponent_cooling_curve[k]));
  } else {
    return 1.0e-50;
  }
}



// cooling timescale in seconds. T in K, nH in cm-3
static Real tcool(const Real T, const Real nH) {
  if (T < T_PE){
    return (kb * T) / (gm1*(mu/muH) * (nH*Lambda_T(T) - Gamma0) );
  } else {
    return (kb * T) / (gm1*(mu/muH) * (nH*Lambda_T(T)) );
  }
}

  
//----------------------------------------------------------------------------------------
// Source function for cooling 
// Inputs:
//   pmb: pointer to MeshBlock
//   t,dt: time (not used) and timestep
//   prim: primitives
  // bcc: magnetic fields (not used)
// Outputs:
//   cons: conserved variables updated

void Cooling(MeshBlock *pmb, const Real t, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, 
    AthenaArray<Real> &cons) {

  Real delta_e_block = 0.0;
  Real delta_e_ceil_block  = 0.0;
    
    
  Real dt_sec = dt*CODE_TIME_SEC;
  
  // Extract indices
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        
        // Extract conserved quantities
        Real &rho = cons(IDN,k,j,i);
        Real &e = cons(IEN,k,j,i);
        Real &m1 = cons(IM1,k,j,i);
        Real &m2 = cons(IM2,k,j,i);
        Real &m3 = cons(IM3,k,j,i);

        Real kinetic = (SQR(m1) + SQR(m2) + SQR(m3)) / (2.0 * rho);
        Real u = e - kinetic;
        Real P = u*gm1; 
 
        // calculate temperature in physical units before cooling
        Real nH = 1*rho;
        Real n_cm3 = CODE_RHO_MP_CM3*rho/mu; // ionized H+He
        
        Real T_before = P/ERG_CM3_CODE/KELVIN_TIMES_KB_ERG/n_cm3; // T in K        
        
        Real T_update = 0.;
        T_update += T_before;
        
        // dT/dt = - T/tcool(T,nH) ---- RK4
        Real k1 = -1.0 * (T_update/tcool(T_update, nH));
        Real k2 = -1.0 * (T_update + 0.5*dt_sec * k1)/tcool(T_update + 0.5*dt_sec * k1, nH);
        Real k3 = -1.0 * (T_update + 0.5*dt_sec * k2)/tcool(T_update + 0.5*dt_sec * k2, nH);
        Real k4 = -1.0 * (T_update + dt_sec * k3)/tcool(T_update + dt_sec * k3, nH);
        T_update += (k1 + 2.*k2 + 2.*k3 + k4)/6.0 * dt_sec; // new Temp in K

        // don't cool below cooling floor and find new internal thermal energy 
        Real u_after = n_cm3*std::max(T_update, T_floor)*KELVIN_TIMES_KB_ERG*ERG_CM3_CODE/gm1;

        // temperature ceiling 
        Real delta_e_ceil = 0.0;
        if (T_update > T_max){
          std::cout << "OH NO WE REACHED THE TEMPERATURE CEILING " << std::endl;
          Real total_energy_meshblock = hst_Etot(pmb, 0);
          std::cout << "total in ergs: " << total_energy_meshblock << std::endl;
          delta_e_ceil -= u_after;
          u_after = n_cm3*std::min(T_update, T_max)*KELVIN_TIMES_KB_ERG*ERG_CM3_CODE/gm1;
          delta_e_ceil += u_after;
          T_update = T_max;
        }

        Real delta_e = u_after - u;

        // change internal energy
        e += delta_e;          
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
// Function to calculate the timestep required to resolve cooling 
//          tcool = 3/2 P/Edot_cool
// Inputs:
//   pmb: pointer to MeshBlock

// as a hack, this function also forces the timestep to be small
// immediately following a SN injection. 

Real cooling_timestep(MeshBlock *pmb) {
  Real dt_cutoff = 1e-13;
  Real min_dt;
  if (N_exploded_last < N_exploded) {
    if (N_step_since_exploded < N_dt) {
      min_dt = dt_SN;
      N_step_since_exploded += 1;
    } else {
      N_exploded_last = N_exploded;
      N_step_since_exploded = 0;
      min_dt = 1.0;
    }
  } else {
    min_dt = 1.0;
  }
  
  if (cooling_on == false) {
    return min_dt;
  }
  
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real &P = pmb->phydro->w(IPR,k,j,i);
        Real &rho = pmb->phydro->w(IDN,k,j,i);
        Real n_cm3 = CODE_RHO_MP_CM3*rho/mu; // ionized H
        Real T_before = P/ERG_CM3_CODE/KELVIN_TIMES_KB_ERG/n_cm3;
        Real nH = 1*rho; 
        Real cooling_time_code = std::abs(tcool(T_before, nH))/MYR_IN_SEC/CODE_TIME_MYR;
        if (T_before > 1.01 * T_floor){
          min_dt = std::min(min_dt, cfl_cool * cooling_time_code);
        }
        min_dt = std::max(dt_cutoff, min_dt);
      }
    }
  }
  return min_dt;
}


//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  return;
}

static Real hst_Mhot(MeshBlock *pmb, int iout){    
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> &u = pmb->phydro->u;
  AthenaArray<Real> volume; // 1D array of cell volumes
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  volume.NewAthenaArray(ncells1);

  Real Mhot = 0;
  Real thermal_e, pressure, Temp; 
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        thermal_e = u(IEN,k,j,i) - 0.5/u(IDN,k,j,i)*(SQR(u(IM1,k,j,i))
                         + SQR(u(IM2,k,j,i))+ SQR(u(IM3,k,j,i)));
        pressure = thermal_e*gm1/ERG_CM3_CODE/KELVIN_TIMES_KB_ERG; // P/k in K/cm3
        Temp = pressure/(u(IDN,k,j,i)*(muH/mu));
        if (Temp > 1.0e5) {
          Mhot += CODE_RHO_MP_CM3*u(IDN,k,j,i)*volume(i)*PC3_IN_CM3/MSUN_IN_MP; // in Msun
        } else {
          Mhot += 0;
        } 
  }}}
  volume.DeleteAthenaArray();  
  return Mhot/frac_sphere;
}


static Real hst_Vbubble(MeshBlock *pmb, int iout){    
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> &u = pmb->phydro->u;
  AthenaArray<Real> volume; // 1D array of cell volumes
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  volume.NewAthenaArray(ncells1);

  Real Vbubble = 0;
  Real thermal_e, pressure, Temp, vr; 
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        thermal_e = u(IEN,k,j,i) - 0.5/u(IDN,k,j,i)*(SQR(u(IM1,k,j,i))
                         + SQR(u(IM2,k,j,i))+ SQR(u(IM3,k,j,i)));
        pressure = thermal_e*gm1/ERG_CM3_CODE/KELVIN_TIMES_KB_ERG; // P/k in K/cm3
        vr = u(IM1,k,j,i)/u(IDN,k,j,i);
        Temp = pressure/((muH/mu)*u(IDN,k,j,i));
        if (Temp > 1e5 || std::abs(vr) >  1.0) {
          Vbubble += volume(i); // in pc3 
        } 
  }}}
  volume.DeleteAthenaArray();  
  return Vbubble/frac_sphere;
}



static Real hst_Mtot(MeshBlock *pmb, int iout){    
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> &u = pmb->phydro->u;
  AthenaArray<Real> volume; // 1D array of cell volumes
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  volume.NewAthenaArray(ncells1);

  Real Mtot = 0;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        Mtot += CODE_RHO_MP_CM3*u(IDN,k,j,i)*volume(i)*PC3_IN_CM3/MSUN_IN_MP; // in Msun
  }}}
  volume.DeleteAthenaArray();  
  return Mtot/frac_sphere;
}

// total energy in ergs
static Real hst_Etot(MeshBlock *pmb, int iout){    
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> &u = pmb->phydro->u;
  AthenaArray<Real> volume; // 1D array of cell volumes
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  volume.NewAthenaArray(ncells1);

  Real Etot = 0;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        Etot += u(IEN,k,j,i)*volume(i)/ERG_CM3_CODE*PC3_IN_CM3; // in ergs
  }}}
  volume.DeleteAthenaArray();  
  return Etot/frac_sphere;
}

static Real hst_Ethermal(MeshBlock *pmb, int iout){    
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> &u = pmb->phydro->u;
  AthenaArray<Real> volume; // 1D array of cell volumes
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  volume.NewAthenaArray(ncells1);

  Real Ethermal = 0;
  Real e_thermal; 
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        e_thermal = u(IEN,k,j,i) - 0.5/u(IDN,k,j,i)*(SQR(u(IM1,k,j,i))
                         + SQR(u(IM2,k,j,i))+ SQR(u(IM3,k,j,i)));
        Ethermal += e_thermal*volume(i)/ERG_CM3_CODE*PC3_IN_CM3; // in ergs
  }}}
  volume.DeleteAthenaArray();  
  return Ethermal/frac_sphere;
}

static Real hst_Ehot(MeshBlock *pmb, int iout){    
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> &u = pmb->phydro->u;
  AthenaArray<Real> volume; // 1D array of cell volumes
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  volume.NewAthenaArray(ncells1);

  Real Ehot = 0;
  Real thermal_e, pressure, Temp; 
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        thermal_e = u(IEN,k,j,i) - 0.5/u(IDN,k,j,i)*(SQR(u(IM1,k,j,i))
                         + SQR(u(IM2,k,j,i))+ SQR(u(IM3,k,j,i)));
        pressure = thermal_e*gm1/ERG_CM3_CODE/KELVIN_TIMES_KB_ERG; // P/k in K/cm3
        Temp = pressure/(u(IDN,k,j,i)*(muH/mu));
        if (Temp > 1.0e5) {
          Ehot += u(IEN,k,j,i)*volume(i)/ERG_CM3_CODE*PC3_IN_CM3; // in ergs
        } 
  }}}
  volume.DeleteAthenaArray();  
  return Ehot/frac_sphere;
}

static Real hst_Ehot_thermal(MeshBlock *pmb, int iout){    
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> &u = pmb->phydro->u;
  AthenaArray<Real> volume; // 1D array of cell volumes
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  volume.NewAthenaArray(ncells1);

  Real Ehot_thermal = 0;
  Real thermal_e, pressure, Temp; 
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        thermal_e = u(IEN,k,j,i) - 0.5/u(IDN,k,j,i)*(SQR(u(IM1,k,j,i))
                         + SQR(u(IM2,k,j,i))+ SQR(u(IM3,k,j,i)));
        pressure = thermal_e*gm1/ERG_CM3_CODE/KELVIN_TIMES_KB_ERG; // P/k in K/cm3
        Temp = pressure/(u(IDN,k,j,i)*(muH/mu));
        if (Temp > 1.0e5) {
          Ehot_thermal += thermal_e*volume(i)/ERG_CM3_CODE*PC3_IN_CM3; // in ergs
        } 
  }}}
  volume.DeleteAthenaArray();  
  return Ehot_thermal/frac_sphere;
}

Real hst_CoolingLosses(MeshBlock *pmb, int iout) {
  Real delta_e = pmb->ruser_meshblock_data[0](iout);
  pmb->ruser_meshblock_data[0](iout) = 0.0;
  return delta_e/frac_sphere;
}

static Real hst_Rshell(MeshBlock *pmb, int iout){    
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> &u = pmb->phydro->u;
  AthenaArray<Real> volume; // 1D array of cell volumes
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  volume.NewAthenaArray(ncells1);

  Real r_times_M_shell = 0;
  Real thermal_e, pressure, Temp, vr, r; 
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        thermal_e = u(IEN,k,j,i) - 0.5/u(IDN,k,j,i)*(SQR(u(IM1,k,j,i))
                         + SQR(u(IM2,k,j,i))+ SQR(u(IM3,k,j,i)));
        pressure = thermal_e*gm1/ERG_CM3_CODE/KELVIN_TIMES_KB_ERG; // P/k in K/cm3
        vr = u(IM1,k,j,i)/u(IDN,k,j,i);
        r = pmb->pcoord->x1v(i); 
        Temp = pressure/((muH/mu)*u(IDN,k,j,i));
        if (Temp < 2.0e4 && std::abs(vr) >  1.0) {
          r_times_M_shell += r*CODE_RHO_MP_CM3*u(IDN,k,j,i)*volume(i)*PC3_IN_CM3/MSUN_IN_MP; // in Msun
        } else {
          r_times_M_shell += 0;
        } 
  }}}
  volume.DeleteAthenaArray();  
  return r_times_M_shell/frac_sphere;
}

static Real hst_Vshell(MeshBlock *pmb, int iout){    
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> &u = pmb->phydro->u;
  AthenaArray<Real> volume; // 1D array of cell volumes
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  volume.NewAthenaArray(ncells1);

  Real vr_times_M_shell = 0;
  Real thermal_e, pressure, Temp, vr, r; 
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        thermal_e = u(IEN,k,j,i) - 0.5/u(IDN,k,j,i)*(SQR(u(IM1,k,j,i))
                         + SQR(u(IM2,k,j,i))+ SQR(u(IM3,k,j,i)));
        pressure = thermal_e*gm1/ERG_CM3_CODE/KELVIN_TIMES_KB_ERG; // P/k in K/cm3
        vr = u(IM1,k,j,i)/u(IDN,k,j,i);
        Temp = pressure/((muH/mu)*u(IDN,k,j,i));
        if (Temp < 2.0e4 && std::abs(vr) > 1.0) {
          vr_times_M_shell += vr*CODE_RHO_MP_CM3*u(IDN,k,j,i)*volume(i)*PC3_IN_CM3/MSUN_IN_MP; // in Msun*km/s
        } else {
          vr_times_M_shell += 0;
        } 
  }}}
  volume.DeleteAthenaArray();  
  return vr_times_M_shell/frac_sphere;
}

static Real hst_Mshell(MeshBlock *pmb, int iout){    
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> &u = pmb->phydro->u;
  AthenaArray<Real> volume; // 1D array of cell volumes
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  volume.NewAthenaArray(ncells1);

  Real M_shell = 0;
  Real thermal_e, pressure, Temp, vr; 
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        thermal_e = u(IEN,k,j,i) - 0.5/u(IDN,k,j,i)*(SQR(u(IM1,k,j,i))
                         + SQR(u(IM2,k,j,i))+ SQR(u(IM3,k,j,i)));
        pressure = thermal_e*gm1/ERG_CM3_CODE/KELVIN_TIMES_KB_ERG; // P/k in K/cm3
        vr = u(IM1,k,j,i)/u(IDN,k,j,i);
        Temp = pressure/((muH/mu)*u(IDN,k,j,i));
        if (Temp < 2.0e4 && std::abs(vr) >  1.0) {
          M_shell += CODE_RHO_MP_CM3*u(IDN,k,j,i)*volume(i)*PC3_IN_CM3/MSUN_IN_MP; // in Msun
        } else {
          M_shell += 0;
        } 
  }}}
  volume.DeleteAthenaArray();  
  return M_shell/frac_sphere;
}

static Real prad_bubble(MeshBlock *pmb, int iout){    
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> &u = pmb->phydro->u;
  AthenaArray<Real> volume; // 1D array of cell volumes
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  volume.NewAthenaArray(ncells1);

  Real prad_bubble = 0;
  Real thermal_e, pressure, Temp, vr; 
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        thermal_e = u(IEN,k,j,i) - 0.5/u(IDN,k,j,i)*(SQR(u(IM1,k,j,i))
                         + SQR(u(IM2,k,j,i))+ SQR(u(IM3,k,j,i)));
        pressure = thermal_e*gm1/ERG_CM3_CODE/KELVIN_TIMES_KB_ERG; // P/k in K/cm3
        vr = u(IM1,k,j,i)/u(IDN,k,j,i);
        Temp = pressure/((muH/mu)*u(IDN,k,j,i));
        if (Temp > 1e5 || std::abs(vr) >  1.0) { // in Msun km/s
          prad_bubble += u(IM1,k,j,i)*volume(i)/CODE_RHO_MP_CM3/MSUN_IN_MP*PC3_IN_CM3; 
        } 
  }}}
  volume.DeleteAthenaArray();  
  return prad_bubble/frac_sphere;
}


static Real Etot_bubble(MeshBlock *pmb, int iout){    
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> &u = pmb->phydro->u;
  AthenaArray<Real> volume; // 1D array of cell volumes
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  volume.NewAthenaArray(ncells1);

  Real Etot_bubble = 0;
  Real thermal_e, pressure, Temp, vr; 
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,volume);
      for (int i=is; i<=ie; i++) {
        thermal_e = u(IEN,k,j,i) - 0.5/u(IDN,k,j,i)*(SQR(u(IM1,k,j,i))
                         + SQR(u(IM2,k,j,i))+ SQR(u(IM3,k,j,i)));
        pressure = thermal_e*gm1/ERG_CM3_CODE/KELVIN_TIMES_KB_ERG; // P/k in K/cm3
        vr = u(IM1,k,j,i)/u(IDN,k,j,i);
        Temp = pressure/((muH/mu)*u(IDN,k,j,i));
        if (Temp > 1e5 || std::abs(vr) >  1.0) { // in Msun km/s
          Etot_bubble += u(IEN,k,j,i)*volume(i)/ERG_CM3_CODE*PC3_IN_CM3; // in ergs 
        } 
  }}}
  volume.DeleteAthenaArray();  
  return Etot_bubble/frac_sphere;
}

// calculate the maximum allowed CFL timestep in seconds. 
// note this and the function below will produced nonsense 
// answers if more than one meshblock is used. 
static Real cond_timestep(MeshBlock *pmb, int iout){    
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;

  Real min_dt = 1e20;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
    
		   min_dt = std::min(min_dt, 0.5*SQR(pmb->pcoord->dx1v(i))*pmb->phydro->u(IDN,k,j,i)/
           pmb->phydro->phdif->kappa(ISO,k,j,i));
        } 
      }
    }
  return min_dt*CODE_TIME_SEC; 
}

// maximum temperature occuring anywhere
static Real max_T(MeshBlock *pmb, int iout){    
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> &u = pmb->phydro->u;

  Real max_T = 0;
  Real thermal_e, pressure, Temp; 
  
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        thermal_e = u(IEN,k,j,i) - 0.5/u(IDN,k,j,i)*(SQR(u(IM1,k,j,i))
                         + SQR(u(IM2,k,j,i))+ SQR(u(IM3,k,j,i)));
        pressure = thermal_e*gm1/ERG_CM3_CODE/KELVIN_TIMES_KB_ERG; // P/k in K/cm3
        Temp = pressure/((muH/mu)*u(IDN,k,j,i)); // in K
		    max_T = std::max(max_T, Temp);
        } 
      }
    }
  return max_T; 
}

