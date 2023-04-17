/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(precession/spin,FixPrecessionSpin);
// clang-format on
#else

#ifndef LMP_FIX_PRECESSION_SPIN_H
#define LMP_FIX_PRECESSION_SPIN_H

#include "fix.h"

namespace LAMMPS_NS {

class FixPrecessionSpin : public Fix {
  friend class FixPour;

 public:
  FixPrecessionSpin(class LAMMPS *, int, char **);
  ~FixPrecessionSpin() override;
  int setmask() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void setup(int) override;
  void min_setup(int) override;
  void post_force(int) override;
  void post_force_respa(int, int, int) override;
  void min_post_force(int) override;
  double compute_scalar() override;

  int zeeman_flag, stt_flag, aniso_flag, cubic_flag, hexaniso_flag, elastic_flag;
  void compute_single_precession(int, double *, double *);

  // zeeman calculations

  void compute_zeeman(int, double *);
  double compute_zeeman_energy(double *);

  // stt calculations

  void compute_stt(double *, double *);
  double compute_stt_energy(double *);

  // uniaxial aniso calculations

  void compute_anisotropy(double *, double *);
  double compute_anisotropy_energy(double *);

  // cubic aniso calculations

  void compute_cubic(double *, double *);
  double compute_cubic_energy(double *);

  // hexagonal aniso calculations

  void compute_hexaniso(double *, double *);
  double compute_hexaniso_energy(double *);

  // Magnetoelastic Calculations
 
  void compute_elastic( double[][3] , double *, double *);
  void compute_elastic_mech(int, double *, double *);
  //void compute_elastic_mech( double *, double *);
  double compute_elastic_energy( double[][3], double *); 

  // Function to invert 3x3 matricies
 
  void solve3x3exactly(double a[][3], double c[][3], double l[][3]);
  
  // storing magnetic energies

  int nlocal_max;    // max nlocal (for list size)
  double *emag;      // energy list

 protected:
  int style;    // style of the magnetic precession

  double degree2rad;
  double hbar;
  int ilevel_respa;
  int time_origin;
  int eflag;
  double eprec, eprec_all;

  int varflag;
  int magfieldstyle;
  int magvar;
  char *magstr;

  // zeeman field intensity and direction

  double H_field;
  double nhx, nhy, nhz;
  double hx, hy, hz;    // temp. force variables

  // STT intensity and direction

  double stt_field;
  double nsttx, nstty, nsttz;
  double sttx, stty, sttz;

  // magnetic anisotropy intensity and direction

  double Ka;     // aniso const. in eV
  double Kah;    // aniso const. in rad.THz
  double nax, nay, naz;
  double Kax, Kay, Kaz;    // temp. force variables

  // cubic anisotropy intensity

  double k1c, k2c;      // cubic const. in eV
  double k1ch, k2ch;    // cubic const. in rad.THz
  double nc1x, nc1y, nc1z;
  double nc2x, nc2y, nc2z;
  double nc3x, nc3y, nc3z;

  // hexagonal anisotropy
  double K6;               // hexagonal aniso const. in eV
  double K6h;              // hexagonal aniso const. in rad.THz
  double n6x, n6y, n6z;    // main axis
  double m6x, m6y, m6z;    // secondary (perpendicular) axis
  double l6x, l6y, l6z;    // =(m x n)

  // Magnetoelastic calculations
  double b1e, b2e; 	   //MagnetoElastic const. in eV
  double b1eh, b2eh;       //MagnetoElastic const. in rad.THz  
  //initalized constant length of bonds
  double l,ax,ay,az,vol; 
  bool isPopulated;
  //double ne1x, ne1y, ne1z;
  //double ne2x, ne2y, ne2z;
  //double ne3x, ne3y, ne3z;

  class NeighList *list;
  // store r0 starting atom positions used in strain (Must be in Ground State)

  double **r0;
  
  // Box dimensions for smart distancing method of r0
  double Lx,Ly,Lz;

  void set_magneticprecession();
};

}    // namespace LAMMPS_NS

#endif
#endif
