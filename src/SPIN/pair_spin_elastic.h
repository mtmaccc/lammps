/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.
   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(spin/elastic,PairSpinElastic)

#else

#ifndef LMP_PAIR_SPIN_ELASTIC_H
#define LMP_PAIR_SPIN_ELASTIC_H

#include "pair_spin.h"

namespace LAMMPS_NS {

class PairSpinElastic : public PairSpin {
 public:
  PairSpinElastic(LAMMPS *lmp) : PairSpin(lmp) {}
  virtual ~PairSpinElastic();
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  void *extract(const char *, int &);

  void compute(int, int);
  void compute_single_pair(int, double *);
  
  void compute_elastic(int, double[][3] , double *, double *, double *, double, double *);
  //void compute_elastic_mech(int, int, int, double, double, double[][3], double *, double *);
  void compute_elastic_mech(int, double *, double *, double *, double, double *);
  double compute_elastic_energy(int, double[][3], double *, double *, double, double *); 

  // Function to invert 3x3 matricies
  
  void solve3x3exactly(double a[][3], double c[][3], double l[][3]);

  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);

  double cut_spin_elastic_global;          // global neel cutoff distance

 protected:
  
   // store r0 starting atom positions used in strain (Must be in Ground State)

  double **r0;
  
  // Box dimensions for smart distancing method of r0
  double Lx,Ly,Lz;

  //initalized constant length of bonds
  double l,ax,ay,az,vol; 

  // Elastic Equation Variables

  double **b1_mag, **b1_mech;               // elastic Coeffs B1 _mag in eV _mech in rad.THz
  double **b2_mag, **b2_mech;               // elastic Coeffs B2 _mag in eV _mech in rad.THz
  double **cut_spin_elastic;               // cutoff distance exchange

  double n1x, n1y, n1z;             // x, y, z, unit vector to define "1" direction (x)
  double n2x, n2y, n2z;             // x, y, z, unit vector to define "2" direction (y)
  double n3x, n3y, n3z;             // x, y, z, unit vector to define "3" direction (z)
  
  // Previous strain value for direct magnetoelastic effect
  
  double **e0;

  void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:
 
E: Incorrect args in pair_spin command
 
Self-explanatory.
 
E: Spin simulations require metal unit style
 
Self-explanatory.

E : Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair spin requires atom attribute spin

The atom style defined does not have these attributes.
*/
