/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------
   Contributing authors: Julien Tranchida (SNL)
                         Aidan Thompson (SNL)

   Please cite the related publication:
   Tranchida, J., Plimpton, S. J., Thibaudeau, P., & Thompson, A. P. (2018).
   Massively parallel symplectic algorithm for coupled magnetic spin dynamics
   and molecular dynamics. Journal of Computational Physics.
------------------------------------------------------------------------- */

#include "pair_spin_elastic.h"

#include <iostream> //USE THIS TO PRINT AND BUGFIX REMOVE LATER

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "domain.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
/* ---------------------------------------------------------------------- */

PairSpinElastic::~PairSpinElastic()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cut_spin_elastic);
    memory->destroy(b1_mag);
    memory->destroy(b1_mech);
    memory->destroy(b2_mag);
    memory->destroy(b2_mech);
    memory->destroy(cutsq); // to be deleted
    memory->destroy(emag);
    memory->destroy(r0); // to be fixed
    memory->destroy(e0); // to be fixed

    
  }
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairSpinElastic::settings(int narg, char **arg)
{
  PairSpin::settings(narg,arg);

  cut_spin_elastic_global = utils::numeric(FLERR,arg[0],false,lmp);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++) {
      for (j = i+1; j <= atom->ntypes; j++) {
        if (setflag[i][j]) {
          cut_spin_elastic[i][j] = cut_spin_elastic_global;
        }
      }
    }
  }

}

/* ----------------------------------------------------------------------
   set coeffs for one or more type spin pairs (only one for now)
------------------------------------------------------------------------- */

void PairSpinElastic::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  // check if args correct

  if (strcmp(arg[2],"elastic") != 0)
    error->all(FLERR,"Incorrect args in pair_style command");
  if (narg != 15)
    error->all(FLERR,"Incorrect args in pair_style command");

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);
  

  // Get MagnetoElastic Constants and Directions from input command
 const double rij = utils::numeric(FLERR,arg[3],false,lmp);
 const double b1 = utils::numeric(FLERR,arg[4],false,lmp);
 const double b2 = utils::numeric(FLERR,arg[5],false,lmp);
 const double n1xc = utils::numeric(FLERR,arg[6],false,lmp);
 const double n1yc = utils::numeric(FLERR,arg[7],false,lmp);
 const double n1zc = utils::numeric(FLERR,arg[8],false,lmp);
 const double n2xc = utils::numeric(FLERR,arg[9],false,lmp);
 const double n2yc = utils::numeric(FLERR,arg[10],false,lmp);
 const double n2zc = utils::numeric(FLERR,arg[11],false,lmp);
 const double n3xc = utils::numeric(FLERR,arg[12],false,lmp);
 const double n3yc = utils::numeric(FLERR,arg[13],false,lmp);
 const double n3zc = utils::numeric(FLERR,arg[14],false,lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      cut_spin_elastic[i][j] = rij;
      b1_mag[i][j] = b1/hbar;
      b1_mech[i][j] = b1;
      b2_mag[i][j] = b2/hbar;
      b2_mech[i][j] = b2;
      setflag[i][j] = 1;
      count++;
    }
 }
  if (count == 0)
    error->all(FLERR,"Incorrect args in pair_style command");
 
  // Normalize Magnetoelastic vectors
  
  double norm2,norm;

  norm2 = (n1xc*n1xc + n1yc*n1yc + n1zc*n1zc);
  if (norm2 == 0.0)
    error->all(FLERR,"Illegal spin/elastic vector");
  norm = 1.0/sqrt(norm2);  
  n1x = n1xc * norm;
  n1y = n1yc * norm;
  n1z = n1zc * norm;
  norm2 = (n2xc*n2xc + n2yc*n2yc + n2zc*n2zc);
  if (norm == 0.0)
    error->all(FLERR,"Illegal spin/elastic vector");  
  norm = 1.0/sqrt(norm2);  
  n2x = n2xc * norm;
  n2y = n2yc * norm;
  n2z = n2zc * norm;
  norm2 = (n3xc*n3xc + n3yc*n3yc + n3zc*n3zc);
  if (norm == 0.0)
    error->all(FLERR,"Illegal spin/elastic vector");  
  norm = 1.0/sqrt(norm2);  
  n3x = n3xc * norm;
  n3y = n3yc * norm;
  n3z = n3zc * norm;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairSpinElastic::init_style()
{
  PairSpin::init_style();

  // Creates r0 vector for initial atomic positions & ghost atoms
   if(update->ntimestep == 0){
    //if(true){
    //int nall = atom->nlocal;
    int nall = atom->natoms;

     memory->grow(r0,nall,3,"pair/spin/elastic:r0");
     memory->grow(e0,nall,6,"pair/spin/elastic:e0");

     tagint *tag = atom->tag;
     double **x = atom->x;
     for (int i = 0; i < nall; i++) {
	r0[i][0]  = x[i][0];
	r0[i][1]  = x[i][1];
	r0[i][2]  = x[i][2];
	e0[i][0] = e0[i][1] = e0[i][2] = 0.0;
	e0[i][3] = e0[i][4] = e0[i][5] = 0.0; 
	}
    
   // Initialize box constants for energy computation

   ax = sqrt( (x[1][0] - x[0][0])*(x[1][0] - x[0][0]));
   ay = sqrt( (x[1][1] - x[0][1])*(x[1][1] - x[0][1]));
   az = sqrt( (x[1][2] - x[0][2])*(x[1][2] - x[0][2]));

/*	r0[0][0] = 0.0;
	r0[0][1] = 0.0;
	r0[0][2] = 0.0;
	r0[1][0] = 1.420028359;
	r0[1][1] = 1.419999109;
	r0[1][2] = 1.419999109;
	r0[2][0] = 2.840056717;
	r0[2][1] = 0.0;
	r0[2][2] = 0.0;
	r0[3][0] = 4.260085076;
	r0[3][1] = 1.419999109;
	r0[3][2] = 1.419999109;
	r0[4][0] = 0.0;
	r0[4][1] = 2.839998217;
	r0[4][2] = 0.0;
	r0[5][0] = 1.420028359;
	r0[5][1] = 4.259997326;
	r0[5][2] = 1.419999109;
	r0[6][0] = 2.840056717;
	r0[6][1] = 2.839998217;
	r0[6][2] = 0.0;
	r0[7][0] = 4.260085076;
	r0[7][1] = 4.259997326;
	r0[7][2] = 1.419999109;
	r0[8][0] = 0.0;
	r0[8][1] = 0.0;
	r0[8][2] = 2.839998217;
	r0[9][0] = 1.420028359;
	r0[9][1] = 1.419999109;
	r0[9][2] = 4.259997326;
	r0[10][0] = 2.840056717;
	r0[10][1] = 0.0;
	r0[10][2] = 2.839998217;
	r0[11][0] = 4.260085076;
	r0[11][1] = 1.419999109;
	r0[11][2] = 4.259997326;
	r0[12][0] = 0.0;
	r0[12][1] = 2.839998217;
	r0[12][2] = 2.839998217;
	r0[13][0] = 1.420028359;
	r0[13][1] = 4.259997326;
	r0[13][2] = 4.259997326;
	r0[14][0] = 2.840056717;
	r0[14][1] = 2.839998217;
	r0[14][2] = 2.839998217;
	r0[15][0] = 4.260085076;
	r0[15][1] = 4.259997326;
	r0[15][2] = 4.259997326;
   ax = 1.420028359;
   ay = az = 1.419999109;
	     
   Lx = 5.680113434;
   Ly = Lz = 5.679996434;*/
   l = sqrt(ax*ax+ay*ay+az*az);	   
   

	
   // Store box dimensions for future smart distancing in periodic scenarios
   Lx = domain ->xprd;
   Ly = domain ->yprd;
   Lz = domain ->zprd;
  //THIS ONLY WORKS FOR BCC, MUST FIND AN ALTERNATE METHOD FOR FCC 
   vol = 2*ax*2*ay*2*az;
   printf("rij = %f ax = %f  ay=%f az=%f vol = %f lx = %f ly = %f lz = %f \n ",l,ax,ay,az,vol,Lx,Ly,Lz);
   }

}


/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairSpinElastic::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  b1_mag[j][i] = b1_mag[i][j];
  b1_mech[j][i] = b1_mech[i][j];
  b2_mag[j][i] = b2_mag[i][j];
  b2_mech[j][i] = b2_mech[i][j];
  cut_spin_elastic[j][i] = cut_spin_elastic[i][j];

  return cut_spin_elastic_global;
}

/* ----------------------------------------------------------------------
   extract the larger cutoff
------------------------------------------------------------------------- */

void *PairSpinElastic::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str,"cut") == 0) return (void *) &cut_spin_elastic_global;
  return nullptr;
}

/* ---------------------------------------------------------------------- */

void PairSpinElastic::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype,icomp,jcomp;
  double evdwl,ecoul,nearest;
  double xi[3], sio[3], rij[3], rijo[3], sjo[3], spi[3]; 
  double a[3][3], c[3][3], l[3][3], eij[3][3];
  double fi[3], fmi[3];
  double local_cut2;
  double delx,dely,delz;
  double rsq, inorm;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = ecoul = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  double **fm = atom->fm;
  double **sp = atom->sp;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  int natoms = atom->natoms;
  int *sametag = atom->sametag;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  tagint *tag = atom->tag;

  // checking size of emag

  if (nlocal_max < nlocal) {                    // grow emag lists if necessary
    nlocal_max = nlocal;
    memory->grow(emag,nlocal_max,"pair/spin:emag");
  }
  
  // computation of the elastic interaction
  // loop over atoms and their neighbors

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];

    jlist = firstneigh[i];
    jnum = numneigh[i];

    xi[0] = x[i][0];
    xi[1] = x[i][1];
    xi[2] = x[i][2];

    spi[0] = sp[i][0];
    spi[1] = sp[i][1];
    spi[2] = sp[i][2];

    emag[i] = 0.0;

    //Re-define old variables from two atom method OUTSIDE j loop
    nearest = 0.0;
    evdwl = 0.0;
    fi[0] = fi[1] = fi[2] = 0.0;
    fmi[0] = fmi[1] = fmi[2] = 0.0;
    icomp = i;
    // set up variables for Strain Calculation to zero
    for (int cx = 0; cx<3; cx++){
      for (int cy = 0; cy<3; cy++){
	a[cx][cy] = 0.0;
	c[cx][cy] = 0.0;
	l[cx][cy] = 0.0;
	eij[cx][cy] = 0.0;
      }
    }

    // loop on neighbors to create strain tensor
	// IMPOSSIBLE TO KNOW STRAIN TENSOR a priori SO MUST CONSTRUCT USING CURRENT TENSOR
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      // define itype & jtype for cuttof calculation
      jtype = type[j];
      
      rijo[0] = rijo[1] = rijo[2] = 0.0; //old Rij vector (ref initial postion)
      rij[0] = rij[1] = rij[2] = 0.0;  // current Rij vector (current i j positions)
      sio[0] = sio[1] = sio[2] = 0.0;
      sjo[0] = sjo[1] = sjo[2] = 0.0;
 
      //create rij vector using CURRENT atomic positions

      rij[0] = x[j][0] - xi[0];
      rij[1] = x[j][1] - xi[1];
      rij[2] = x[j][2] - xi[2];
      	  
      // Create rij vector from atomic positions at begining of fix
      // get tag number corresponding to j
      
      if(j != tag[j] - 1) //ensures j atoms are constircted with tags defined during init /
      	j = tag[j] - 1; // j = sametag[j] //Tag is N number of atoms while j must be N-1
      if(i != tag[i] - 1)
	icomp = tag[i]-1;
	
      // Create rij vector from atomic positions at begining of fix
      
      // SMART DISTANCE
      // send old atoms through smart distance to ensure pointing rij vector is correct only works on cubes? (I think)
      // scale sio & sjo by simuulation size
	  
	 if(domain->nonperiodic == 0){
	   sio[0] = r0[icomp][0]/Lx;
 	   sio[1] = r0[icomp][1]/Ly;
	   sio[2] = r0[icomp][2]/Lz;
	   sjo[0] = r0[j][0]/Lx;
	   sjo[1] = r0[j][1]/Ly;
	   sjo[2] = r0[j][2]/Lz;
      
	   // Verify smart distance direction, isnt smart enough if atom is located EXACTLY mid simulation at 0.50 and simulation is small
	   // Preform Smart distance to ensure Pointing OLD rij vector is pointing to CORRECT atom
	   rijo[0] = sjo[0] - sio[0];
	   rijo[0] += 0.5;
	   rijo[0] -= floor(rijo[0]);
	   rijo[0] -= 0.5;
	   rijo[0] *= Lx;

	   rijo[1] = sjo[1] - sio[1];
	   rijo[1] += 0.5;
	   rijo[1] -= floor(rijo[1]);
	   rijo[1] -= 0.5;
	   rijo[1] *= Ly;
		   
 	   rijo[2] = sjo[2] - sio[2];
	   rijo[2] += 0.5;
	   rijo[2] -= floor(rijo[2]);
	   rijo[2] -= 0.5;
	   rijo[2] *= Lz;
	  }
      	  else{
	   rijo[0] = r0[j][0] - r0[icomp][0];
	   rijo[1] = r0[j][1] - r0[icomp][1];
	   rijo[2] = r0[j][2] - r0[icomp][2];
	  }	
      //define rsq & localcut 2 for cutoff criteria
      rsq = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]; 
      local_cut2 = cut_spin_elastic[itype][jtype]*cut_spin_elastic[itype][jtype];

	  
      // Check rsq to ensure atom j is within cutoff 
      // If failed, atom j isnt a major contributor to strain on atom i  
      if (rsq <= local_cut2) {
        
	//Ensure that Newtonian atom force is divided evenly
	nearest++;
		
    // printf("#%f neighbor of atom id = %d is atom %d x = %f y = %f z = %f rij away \n", nearest,i,j,rij[0],rij[1],rij[2] );
    // printf("atom %d is located at x = %f y = %f z = %f and atom %d is located at x = %f y = %f z = %f \n",i,xi[0],xi[1],xi[2],j,x[j][0],x[j][1],x[j][2]);
// Create Vi & Wi Matrixes to create Transformatoin Matrix J

        // Vi = SUM j E Jnum ( rijo' * rijo )
    	for (int ax = 0; ax<3; ax++){
          for (int ay = 0; ay<3; ay++){
		a[ax][ay] += (rijo[ax] * rijo[ay]);
		c[ax][ay] += (rijo[ax] * rij[ay]);
          }
        }
      }
    }	
    
    // Compute Transformation Matrix J (Ji = Vi^(-1)*Wi
    // *NOTE* Will throw an error if there are less than 3 neighboring atoms or only 3 atoms are coplanar  
    // Error Check? Ensure that nearest has ATLEAST two neighboring atoms
   
     //printf("Nearest atoms of atom id = %d = %f \n",i,nearest);
    if (nearest >= 3)
    { 
    	solve3x3exactly(a,c,l);

    	// compute strain matrix eij = 0.5(Ji*Ji' - I)
          
    	//ji*ji' 
    	for (int k=0; k<3; k++){
          for (int n=0; n<3; n++){
            for (int m=0; m<3; m++){
		eij[k][n] += (l[k][m] * l[n][m]);
	    }
          }
    	}  
    	//Subtract identity matrix from diagonal elements; then divide entire matrix by half
    	eij[0][0] -= 1;
    	eij[1][1] -= 1;
    	eij[2][2] -= 1;

    	// Divide matrix in half to get full strain tensor
    	for (int cx = 0; cx<3; cx++){
      	  for (int cy = 0; cy<3; cy++){
              eij[cx][cy] *= 0.5;
	  }
        }
    }
  
  // Compute elastic interaction
  // loop through all neighbors to define energies per atom
    
   for (jj = 0; jj < jnum; jj++) {
     j = jlist[jj];
     j &= NEIGHMASK;

     // define itype & jtype for cuttof calculation
     jtype = type[j];
     
     //initialize jcomp for newtoniain forces at end
     jcomp = j;

     rijo[0] = rijo[1] = rijo[2] = 0.0; //old Rij vector (ref initial postion)
     rij[0] = rij[1] = rij[2] = 0.0;  // current Rij vector (current i j positions)
     sio[0] = sio[1] = sio[2] = 0.0;
     sjo[0] = sjo[1] = sjo[2] = 0.0;
     //create rij vector using CURRENT atomic positions

     rij[0] = x[j][0] - xi[0];
     rij[1] = x[j][1] - xi[1];
     rij[2] = x[j][2] - xi[2];
      	  
     // Create rij vector from atomic positions at begining of fix
     // get tag number corresponding to j
      
     if(j != tag[j] - 1) //ensures j atoms are constircted with tags defined during init /
     	j = tag[j] - 1; // j = sametag[j] //Tag is N number of atoms while j must be N-1
     if(i != tag[i] - 1)
	icomp = tag[i]-1;
	
     // Create rij vector from atomic positions at begining of fix
      
     // SMART DISTANCE
     // send old atoms through smart distance to ensure pointing rij vector is correct only works on cubes? (I think)
     // scale sio & sjo by simuulation size
	  
	 if(domain->nonperiodic == 0){
	   sio[0] = r0[icomp][0]/Lx;
 	   sio[1] = r0[icomp][1]/Ly;
	   sio[2] = r0[icomp][2]/Lz;
	   sjo[0] = r0[j][0]/Lx;
	   sjo[1] = r0[j][1]/Ly;
	   sjo[2] = r0[j][2]/Lz;
      
	   // Verify smart distance direction, isnt smart enough if atom is located EXACTLY mid simulation at 0.50 and simulation is small
	   // Preform Smart distance to ensure Pointing OLD rij vector is pointing to CORRECT atom
	   rijo[0] = sjo[0] - sio[0];
	   rijo[0] += 0.5;
	   rijo[0] -= floor(rijo[0]);
	   rijo[0] -= 0.5;
	   rijo[0] *= Lx;

	   rijo[1] = sjo[1] - sio[1];
	   rijo[1] += 0.5;
	   rijo[1] -= floor(rijo[1]);
	   rijo[1] -= 0.5;
	   rijo[1] *= Ly;
		   
 	   rijo[2] = sjo[2] - sio[2];
	   rijo[2] += 0.5;
	   rijo[2] -= floor(rijo[2]);
	   rijo[2] -= 0.5;
	   rijo[2] *= Lz;
	  }
      	  else{
	   rijo[0] = r0[j][0] - r0[icomp][0];
	   rijo[1] = r0[j][1] - r0[icomp][1];
	   rijo[2] = r0[j][2] - r0[icomp][2];
	  }	
      //define rsq & localcut 2 for cutoff criteria
     
     rsq = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]; 
     local_cut2 = cut_spin_elastic[itype][jtype]*cut_spin_elastic[itype][jtype];

     // compute elastic interaction
     if (rsq <= local_cut2) {
     //printf("Pre Elastice Force on atom i = %d from atom j = %d  fx = %.16f, fy = %.16f fz=%.16f \n",i,j,f[i][0],f[i][1],f[i][2]);
     //printf("current spin orientation for atom i = %d sx = %.16f, sy = %.16f sz=%.16f \n",i,j,spi[0],spi[1],spi[2]);
    // printf("current values for magnetoelastic force between atom i = %d from atom j = %d b = %f ax = %f ay = %f az = %f vol = %f  \n",i,j,(vol*b1_mech[itype][itype])/(8*2),ax,ay,az,vol);
	//normalize seperation length
    
	compute_elastic(icomp,eij,fmi,spi,rij,nearest,rijo);     
     
      if (lattice_flag)
	 compute_elastic_mech(icomp,fi,spi,rij,nearest,rijo);

     //printf("Calculated Magnetoelastic  Force on atom i = %d fx = %.16f, fy = %.16f fz=%.16f \n",i,fi[0],fi[1],fi[2]);
      if (eflag){
	 evdwl -= compute_elastic_energy(icomp,eij,spi,rij,nearest,rijo);
//	 printf("Magnetoelastic energy on atom i = %d and atom j =%d = %.16f \n",i,j,evdwl);
	 emag[i] =+ evdwl;
       } else evdwl = 0.0;

     f[i][0] += fi[0];
     f[i][1] += fi[1];
     f[i][2] += fi[2];
     
     if (newton_pair || j < nlocal) {
       f[jcomp][0] -= fi[0];
       f[jcomp][1] -= fi[1];
       f[jcomp][2] -= fi[2];
     }
     
//	 printf("SECOND Magnetoelastic energy on atom i = %d and atom j =%d = %.16f \n",i,j,evdwl);
  // printf("Atom %d original rij with atom j= %d x=%f y=%f z=%f \n ",icomp,j, rijo[0], rijo[1], rijo[2] );
  //printf("Atom %d current rij with atom j = %d  x=%f y=%f z=%f with %f neighbors \n ",icomp,j, rij[0], rij[1], rij[2], nearest);
  // printf("Atom %d effective field fmx=%f fmy=%f fmz=%f \n ",icomp,fmi[0],fmi[1],fmi[2] );
    //   printf("nearest between at i = %d and atom j = %d is %f for trad distance is =%f \n", icomp,j,nearest,sqrt(rijo[0]*rijo[0] + rijo[1]*rijo[1]+rijo[2]*rijo[2])); 
     fm[i][0] += fmi[0];
     fm[i][1] += fmi[1];
     fm[i][2] += fmi[2];

    if (evflag) ev_tally_xyz(i,j,nlocal,newton_pair,evdwl,ecoul,fi[0],fi[1],fi[2],rij[0],rij[1],rij[2]);

    // printf("Post Elastice Force on atom i = %d from atom j = %d  fx = %.16f, fy = %.16f fz=%.16f \n",i,j,f[i][0],f[i][1],f[i][2]);
    }
   }  	
 //    printf(" Total Calculated Magnetoelastic  Force on atom i = %d fx = %.16f, fy = %.16f fz=%.16f \n",i,fi[0],fi[1],fi[2]);
//	printf("Current strain on atom i = %d e1 =%.16f e2 =%.16f e3 =%.16f e4 =%.16f e5 =%.16f e6 =%.16f   \n ",icomp ,eij[0][0] ,eij[1][1],eij[2][2],eij[2][1], eij[2][0], eij[1][0]);
  }
  
  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   update the pair interactions fmi acting on the spin ii
   ------------------------------------------------------------------------- */

void PairSpinElastic::compute_single_pair(int ii, double fmi[3])
{
  int *type = atom->type;
  double **x = atom->x;
  double **f = atom->f;
  double **sp = atom->sp;
  double local_cut2;

  double xi[3], rij[3], spi[3], rijo[3], sio[3], sjo[3];
  double eij[3][3], a[3][3], l[3][3], c[3][3];
 
  int j,jnum,itype,jtype,ntypes,icomp,jcomp;
  int k,locflag;
  int *jlist,*numneigh,**firstneigh;
  int natoms = atom->natoms;
  int *sametag = atom->sametag;


  double rsq, inorm, nearest;

  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  tagint *tag = atom->tag; 

  // check if interaction applies to type of ii

  itype = type[ii];
  ntypes = atom->ntypes;
  locflag = 0;
  k = 1;
  while (k <= ntypes) {
    if (k <= itype) {
      if (setflag[k][itype] == 1) {
        locflag =1;
        break;
      }
      k++;
    } else if (k > itype) {
      if (setflag[itype][k] == 1) {
        locflag =1;
        break;
      }
      k++;
    } else error->all(FLERR,"Wrong type number");
  }

  // if interaction applies to type ii,
  // locflag = 1 and compute pair interaction

  if (locflag == 1) {
    // set values of atom ii
    spi[0] = sp[ii][0];
    spi[1] = sp[ii][1];
    spi[2] = sp[ii][2];

    xi[0] = x[ii][0];
    xi[1] = x[ii][1];
    xi[2] = x[ii][2];

    // set up variables for Strain Calculation to zero
    for (int cx = 0; cx<3; cx++){
      for (int cy = 0; cy<3; cy++){
	a[cx][cy] = 0.0;
	c[cx][cy] = 0.0;
	l[cx][cy] = 0.0;
	eij[cx][cy] = 0.0;
      }
    }

    jlist = firstneigh[ii];
    jnum = numneigh[ii];
    
    icomp = ii;
    // Loop on neightbors of atom ii to build the strain matrix
     
    nearest = 0; 
    for (int jj = 0; jj < jnum; jj++) {

      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];
    
      rijo[0] = rijo[1] = rijo[2] = 0.0; //old Rij vector (ref initial postion)
      rij[0] = rij[1] = rij[2] = 0.0;  // current Rij vector (current i j positions)
      sio[0] = sio[1] = sio[2] = 0.0;
      sjo[0] = sjo[1] = sjo[2] = 0.0;
      // Create rij vector using CURRENT atomic positions      

      rij[0] = x[j][0] - xi[0];
      rij[1] = x[j][1] - xi[1];
      rij[2] = x[j][2] - xi[2];
		

      // Account for Ghost & re-indexed neighborlist atoms via utilzing j = tag[j]-1 
      if(j != tag[j] - 1) 
       j = tag[j] - 1; // j = sametag[j] //Tag is N number of atoms while j must be N-1          
      if(ii != tag[ii] - 1)
	 icomp = tag[ii]-1;
      // Create old rij vector from atomic positions at begining of fix
      // Use smart distance to ensure rij pointing vector is in the correct direction

      //Create scaled sio & sjo vectors by simulation size
      if(domain->nonperiodic == 0){
	 sio[0] = r0[icomp][0]/Lx;
	 sio[1] = r0[icomp][1]/Ly;
	 sio[2] = r0[icomp][2]/Lz;
	 sjo[0] = r0[j][0]/Lx;
	 sjo[1] = r0[j][1]/Ly;
	 sjo[2] = r0[j][2]/Lz;
      
	// Verify smart distance direction, isnt smart enough if atom is located EXACTLY mid simulation at 0.50
	  // Preform Smart distance to ensure Pointing OLD rij vector is pointing to CORRECT atom
	 rijo[0] = sjo[0] - sio[0];
	 rijo[0] += 0.5;
	 rijo[0] -= floor(rijo[0]);
	 rijo[0] -= 0.5;
	 rijo[0] *= Lx;
 
	 rijo[1] = sjo[1] - sio[1];
	 rijo[1] += 0.5;
	 rijo[1] -= floor(rijo[1]);
	 rijo[1] -= 0.5;
	 rijo[1] *= Ly;
		 
	 rijo[2] = sjo[2] - sio[2];
	 rijo[2] += 0.5;
	 rijo[2] -= floor(rijo[2]);
	 rijo[2] -= 0.5;
	 rijo[2] *= Lz;
	 }
      else{
	 rijo[0] = r0[j][0] - r0[icomp][0];
	 rijo[1] = r0[j][1] - r0[icomp][1];
	 rijo[2] = r0[j][2] - r0[icomp][2];
     	 }	      

     //Define rsq & localcut 2 for cutoff criteria    
	  
      local_cut2 = cut_spin_elastic[itype][jtype]*cut_spin_elastic[itype][jtype];
      rsq = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2];
      
      // Check rsq to ensure atom j is within cutoff
      // If failed, atom j isnt a major contributor to strain on atom i 
      
      if (rsq <= local_cut2) {
	nearest ++;
        // Create Vi & Wi Matrixes used in Transformation matrix
	
	// Ai = SUM j E Jnum (rijo' * rijo) & Ci = SUM j E jnum ( rijo' * rij )

    	for (int ax = 0; ax<3; ax++){
          for (int ay = 0; ay<3; ay++){
	    a[ax][ay] += (rijo[ax] * rijo[ay]);
	    c[ax][ay] += (rijo[ax] * rij[ay]);
	  }	
	}
      }
    }
    // START CALCULATIONS HERE
    // Compute Transformation Matrix J (Ji = Vi^-1*Wi)
    // *NOTE* WILL throw an error if there are less than 3 neighboring atoms, Error check?

   //  printf("Nearest (half step) atoms = %f \n",nearest);
   if (nearest >= 3){ 
      solve3x3exactly(a,c,l);

      // Compute Strain Matrix eij = 0.5(Ji*Ji' - I)
      // Ji*Ji'
    
      for (int k=0; k<3; k++){
        for (int n=0; n<3; n++){
          for (int m=0; m<3; m++){
            eij[k][n] += (l[k][m] * l[n][m]);
	    }
	  }
	}
  
      // Subtract identity matrix from Diagonal elements; Then divide by half
  
      eij[0][0] -= 1;
      eij[1][1] -= 1;
      eij[2][2] -= 1;

      // Divide matrix in half ot get full strain matrix
      for (int cx = 0; cx<3; cx++){
        for (int cy = 0; cy<3; cy++){
          eij[cx][cy] *= 0.5;
        }
      }
   }
    //loop through all atoms to calculate per atom elastic interaction
  
    for (int jj = 0; jj < jnum; jj++) {

      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];
    
      rijo[0] = rijo[1] = rijo[2] = 0.0; //old Rij vector (ref initial postion)
      rij[0] = rij[1] = rij[2] = 0.0;  // current Rij vector (current i j positions)
      sio[0] = sio[1] = sio[2] = 0.0;
      sjo[0] = sjo[1] = sjo[2] = 0.0;
      // Create rij vector using CURRENT atomic positions      

      rij[0] = x[j][0] - xi[0];
      rij[1] = x[j][1] - xi[1];
      rij[2] = x[j][2] - xi[2];
		

      // Account for Ghost & re-indexed neighborlist atoms via utilzing j = tag[j]-1 
      if(j != tag[j] - 1) 
       j = tag[j] - 1; // j = sametag[j] //Tag is N number of atoms while j must be N-1          
      if(ii != tag[ii] - 1)
	 icomp = tag[ii]-1;
      // Create old rij vector from atomic positions at begining of fix
      // Use smart distance to ensure rij pointing vector is in the correct direction

      //Create scaled sio & sjo vectors by simulation size
      if(domain->nonperiodic == 0){
	 sio[0] = r0[icomp][0]/Lx;
	 sio[1] = r0[icomp][1]/Ly;
	 sio[2] = r0[icomp][2]/Lz;
	 sjo[0] = r0[j][0]/Lx;
	 sjo[1] = r0[j][1]/Ly;
	 sjo[2] = r0[j][2]/Lz;
      
	// Verify smart distance direction, isnt smart enough if atom is located EXACTLY mid simulation at 0.50
	  // Preform Smart distance to ensure Pointing OLD rij vector is pointing to CORRECT atom
	 rijo[0] = sjo[0] - sio[0];
	 rijo[0] += 0.5;
	 rijo[0] -= floor(rijo[0]);
	 rijo[0] -= 0.5;
	 rijo[0] *= Lx;
 
	 rijo[1] = sjo[1] - sio[1];
	 rijo[1] += 0.5;
	 rijo[1] -= floor(rijo[1]);
	 rijo[1] -= 0.5;
	 rijo[1] *= Ly;
		 
	 rijo[2] = sjo[2] - sio[2];
	 rijo[2] += 0.5;
	 rijo[2] -= floor(rijo[2]);
	 rijo[2] -= 0.5;
	 rijo[2] *= Lz;
	 }
      else{
	 rijo[0] = r0[j][0] - r0[icomp][0];
	 rijo[1] = r0[j][1] - r0[icomp][1];
	 rijo[2] = r0[j][2] - r0[icomp][2];
     	 }	      

     //Define rsq & localcut 2 for cutoff criteria    
	  
      local_cut2 = cut_spin_elastic[itype][jtype]*cut_spin_elastic[itype][jtype];
      rsq = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2];
      
      // Check rsq to ensure atom j is within cutoff
      // If failed, atom j isnt a major contributor to strain on atom i 
     
      //calculate elastic interaction
      
      if (rsq <= local_cut2) {
	//normalize seperation length
	     
	//printf("Strain on atom i = %d e1 =%f e2 =%f e3 =%f e4 =%f e5 =%f e6 =%f Lx =%f Ly =%f Lz =%f r0 atom 0 x =%f  \n ",icomp ,eij[0][0] ,eij[1][1],eij[2][2],eij[2][1], eij[2][0], eij[1][0], Lx, Ly, Lz, r0[0][0]);
        //printf("nearest between at i = %d and atom j = %d is %f for single pair distance is =%f \n", icomp,j,nearest,sqrt(rijo[0]*rijo[0] + rijo[1]*rijo[1]+rijo[2]*rijo[2])); 
	// Compute Elastic Interaction
    	compute_elastic(ii,eij,fmi,spi,rij,nearest,rijo);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   Compute elastic energy once strain on atom i is calculated
 ---------------------------------------------------------------------- */

void PairSpinElastic::compute_elastic(int i, double eij[3][3], double fmi[3], double spi[3], double rij[3], double nearest, double rijo[3])
{
  int *type = atom->type;
  int itype;
  itype = type[i];
  
  double skx,sky,skz;
  double first1,first2,first3;
  double firstx,firsty,firstz;
  double second1,second2,second3;
  double secondx,secondy,secondz;
  double b1bar,b2bar;

  //create x,y,z components of spin (si * ni) = alphi
  skx = spi[0]*n1x+spi[1]*n1y+spi[2]*n1z;
  sky = spi[0]*n2x+spi[1]*n2y+spi[2]*n2z;
  skz = spi[0]*n3x+spi[1]*n3y+spi[2]*n3z;

  //dimmensionalize energy constant B
  //ONLY WORKS FOR BCC MUST UPDATE TO INCLUDE FCC ATOMS
  //ORIGINAL
  //b1bar =(vol*b1_mag[itype][itype])/(nearest * 2);
  //b2bar =(vol*b2_mag[itype][itype])/(nearest * 2);


  //Additions done to account for removal of atoms
  b1bar =(vol*b1_mech[itype][itype])/(nearest * (2*(nearest/8)));
  b2bar =(vol*b2_mech[itype][itype])/(nearest * (2*(nearest/8)));
 /* //calculate first part of equation 2b1*[alphi*epsiii]
  //ASSUMES MONOTYPE SYSTEM FIX LATER
  firstx = 2.0*b1_mag[itype][itype]*(skx * eij[0][0]);
  firsty = 2.0*b1_mag[itype][itype]*(sky * eij[1][1]);
  firstz = 2.0*b1_mag[itype][itype]*(skz * eij[2][2]);

  //Calculate 2nd part of equation b2[alphj * epsiij + alphk * epsiik
  //ASSUMES MONOTYPE SYSTEM FIX LATER
  secondx = 2.0*b2_mag[itype][itype]*(sky*eij[0][1] + skz*eij[0][2]);
  secondy = 2.0*b2_mag[itype][itype]*(skx*eij[0][1] + skz*eij[1][2]);
  secondz = 2.0*b2_mag[itype][itype]*(skx*eij[0][2] + sky*eij[1][2]);
*/


 /*firstx = 2.0*b1bar*(skx * eij[0][0]);
  firsty = 2.0*b1bar*(sky * eij[1][1]);
  firstz = 2.0*b1bar*(skz * eij[2][2]);*/
  
  // adjust case for only one neighbor
  if(nearest < 3){

  first1 = 2.0*skx*((rij[0]-rijo[0])/rijo[0]);
  first2 = 2.0*sky*((rij[1]-rijo[1])/rijo[1]);
  first3 = 2.0*skz*((rij[2]-rijo[2])/rijo[2]);
  }
  else{
  first1 = 2.0*skx*eij[0][0];
  first2 = 2.0*sky*eij[1][1];
  first3 = 2.0*skz*eij[2][2];
  }

  firstx = b1bar*(n1x*first1 + n2x*first2 + n3x*first3);
  firsty = b1bar*(n1y*first1 + n2y*first2 + n3y*first3);
  firstz = b1bar*(n1z*first1 + n2z*first2 + n3z*first3);

  //Calculate 2nd part of equation b2[alphj * epsiij + alphk * epsiik
  //ASSUMES MONOTYPE SYSTEM FIX LATER

  second1 = second2 = second3 = 0.0;  
 
  if(nearest >= 3){
    second1 = 2.0*(sky*eij[0][1] + skz*eij[0][2]);
    second2 = 2.0*(skx*eij[0][1] + skz*eij[1][2]);
    second3 = 2.0*(skx*eij[0][2] + sky*eij[1][2]);
  }

  secondx = b2bar*(n1x*second1 + n2x*second2 + n3x*second3);
  secondy = b2bar*(n1y*second1 + n2y*second2 + n3y*second3);
  secondz = b2bar*(n1z*second1 + n2z*second2 + n3z*second3);
  
 /* secondx = 2.0*b2bar*(sky*eij[0][1] + skz*eij[0][2]);
  secondy = 2.0*b2bar*(skx*eij[0][1] + skz*eij[1][2]);
  secondz = 2.0*b2bar*(skx*eij[0][2] + sky*eij[1][2]);*/
  
  //Sum two parts together and add into forces
  //Note** assumes n1,n2, and n3 are cardinal directions. Fix later?
  fmi[0] += (firstx + secondx);
  fmi[1] += (firsty + secondy);
  fmi[2] += (firstz + secondz);
}

/* ----------------------------------------------------------------------*/ 

//void PairSpinElastic::compute_elastic_mech(int i, int dir, int nearest, double rij, double rijPrevious, double eij[3][3], double fi[3],  double spi[3])
void PairSpinElastic::compute_elastic_mech(int i, double fi[3],  double spi[3], double rij[3], double nearest, double rijo[3])
{
  int *type = atom->type;
  int itype, jtype;
  itype = type[i];

  double skx,sky,skz,skx2,sky2,skz2;
 // double de1,de2,de3,de4,de5,de6;
  //double dedir1,dedir2,dedir3;
  //double dedir4,dedir5,dedir6;
  double fx,fy,fz,fx1,fx2,fy1,fy2,fz1,fz2;
  double b1bar,b2bar;
  
  //dimmensionalize energy constant B
  //ONLY WORKS FOR BCC CYSTALS MUST UPDATE FOR FCC
  //ORIGINAL VERSION
 // b1bar =(vol*b1_mech[itype][itype])/(nearest * 2);
 // b2bar =(vol*b2_mech[itype][itype])/(nearest * 2);

  //Additions done to account for removal of atoms
  b1bar =(vol*b1_mech[itype][itype])/(nearest * (2*(nearest/8)));
  b2bar =(vol*b2_mech[itype][itype])/(nearest * (2*(nearest/8)));
 //printf("b1 bar = %f \n",b1bar);
  //create x,y,z components of spin (si * ni)
  skx = spi[0]*n1x+spi[1]*n1y+spi[2]*n1z;
  sky = spi[0]*n2x+spi[1]*n2y+spi[2]*n2z;
  skz = spi[0]*n3x+spi[1]*n3y+spi[2]*n3z;

  //create squared x,y,z components of spin (si * ni)^2
  skx2 = skx*skx;
  sky2 = sky*sky;
  skz2 = skz*skz;

  fx = (b1bar*skx2)/ax;
  fy = (b1bar*sky2)/ay;
  fz = (b1bar*skz2)/az;
  
  fx1 = (b2bar*skx*skz)/az;
  fy1 = (b2bar*sky*skz)/az;
  fz1 = (b2bar*sky*skz)/ay;

  fx2 = (b2bar*skx*sky)/ay;
  fy2 = (b2bar*skx*sky)/ax;
  fz2 = (b2bar*skx*skz)/ax;
 // fi[0] -= (fx1+fx2+fx3); 
 // fi[1] -= (fy1+fy2+fy3); 
 // fi[2] -= (fz1+fz2+fz3); 
 //Do you need to multiply by half here? take note and remember if something odd is going on.
 //fi[0] += 0.5*(fx + fx1 + fx2);
 //fi[1] += 0.5*(fy + fy1 + fy2);
 //fi[2] += 0.5*(fz + fz1 + fz2);
 fi[0] +=(fx + fx1 + fx2);
 fi[1] +=(fy + fy1 + fy2);
 fi[2] +=(fz + fz1 + fz2);


}

/* ---------------------------------------------------------------------- */

double PairSpinElastic::compute_elastic_energy(int i, double eij[3][3], double spi[3], double rij[3], double nearest, double rijo[3] ) 
{
  
  int *type = atom->type;
  int itype;
  itype = type[i];

  double skx,sky,skz;
  double energy = 0.0;
  double b1bar, b2bar;

  //create x,y,z components of spin (si * ni) = alphi
  skx = spi[0]*n1x+spi[1]*n1y+spi[2]*n1z;
  sky = spi[0]*n2x+spi[1]*n2y+spi[2]*n2z;
  skz = spi[0]*n3x+spi[1]*n3y+spi[2]*n3z;

  //dimmensionalize energy constant B
  //ONLY WORKS FOR BCC CYSTALS MUST UPDATE FOR FCC
  //ORIGINAL IMPLEMENTATION
  //b1bar =(vol*b1_mag[itype][itype])/(nearest * 2);
  //b2bar =(vol*b2_mag[itype][itype])/(nearest * 2);
 
 //Additions done to account for removal of atoms
  b1bar =(vol*b1_mech[itype][itype])/(nearest * (2*(nearest/8)));
  b2bar =(vol*b2_mech[itype][itype])/(nearest * (2*(nearest/8)));
 

// printf("b1 bar equals =%f b2 bar equals %f \n",vol*b1_mech[itype][itype]/2,vol*b2_mech[itype][itype]/2);
  //b2bar = b2_mech[itype][itype]/(nearest * l);
  /*Calculate Energy
  //Note, if Energy looks odd, THIS is the suspect step
  //should energy var be positive or negative????
  //also only works in a monotype system FIX THIS
  */


//printf("skx2 = %0.16f sky2 = %0.16f skz = %0.16f rijo x = %0.16f rijo y %0.16f rijoz = %0.16f e1 =%f e2 =%f e3 =%f   \n ", skx*skx,sky*sky,skz*skz,rijo[0],rijo[1],rijo[2],eij[0][0],eij[1][1],eij[2][2]);
  if(nearest < 3){
    eij[0][0] = ((rij[0] - rijo[0])/rijo[0]);      
    eij[1][1] = ((rij[1] - rijo[1])/rijo[1]);      
    eij[2][2] = ((rij[2] - rijo[2])/rijo[2]);
    eij[1][2] = eij[0][2] = eij[0][1] = 0.0;
  }     

  energy = b1bar*(skx*skx*eij[0][0]+ sky*sky*eij[1][1] + skz*skz*eij[2][2]);
  //energy -= ((b1_mech[itype][itype]*eij[0][0])/3);
  //energy -= ((b1_mech[itype][itype]*eij[1][1])/3);
  //energy -= ((b1_mech[itype][itype]*eij[2][2])/3);
  energy += 2.0*b2bar*(sky*skz*eij[1][2] + skx*skz*eij[0][2] + skx*sky*eij[0][1]);
  //printf("energy = %0.16f \n",energy);
  //return 0.5*(energy);
  return energy;
}

/* ---------------------------------------------------------------------- */

void PairSpinElastic::solve3x3exactly(double a[][3],
                                double c[][3], double l[][3])
{
  double ai[3][3];
  double determ, determinv;
 
  //TO USE IN STRAIN CALC: a = Vi (to be inversed). c = Wi. l = Ji (output)  

  // calculate the determinant of the matrix

  determ = a[0][0]*a[1][1]*a[2][2] + a[0][1]*a[1][2]*a[2][0] +
    a[0][2]*a[1][0]*a[2][1] - a[0][0]*a[1][2]*a[2][1] -
    a[0][1]*a[1][0]*a[2][2] - a[0][2]*a[1][1]*a[2][0];

  // check if matrix is actually invertible

 //std::cout << "DETERMINANT ="<< determ <<  std::endl;
  if (determ == 0.0) 
    error->one(FLERR,"Strain Tensor determinant = 0.0");

  // calculate the inverse 3x3 matrix: A^(-1) = (ai_jk)

  determinv = 1.0/determ;
  ai[0][0] =  determinv * (a[1][1]*a[2][2] - a[1][2]*a[2][1]);
  ai[0][1] = -determinv * (a[0][1]*a[2][2] - a[0][2]*a[2][1]);
  ai[0][2] =  determinv * (a[0][1]*a[1][2] - a[0][2]*a[1][1]);
  ai[1][0] = -determinv * (a[1][0]*a[2][2] - a[1][2]*a[2][0]);
  ai[1][1] =  determinv * (a[0][0]*a[2][2] - a[0][2]*a[2][0]);
  ai[1][2] = -determinv * (a[0][0]*a[1][2] - a[0][2]*a[1][0]);
  ai[2][0] =  determinv * (a[1][0]*a[2][1] - a[1][1]*a[2][0]);
  ai[2][1] = -determinv * (a[0][0]*a[2][1] - a[0][1]*a[2][0]);
  ai[2][2] =  determinv * (a[0][0]*a[1][1] - a[0][1]*a[1][0]);

  // calculate the solution:  L = A^(-1) * C

  for (int i=0; i<3; i++){
    for (int j=0; j<3; j++){
      for (int k=0; k<3; k++){
       l[i][j] += (ai[i][k] * c[k][j]);
	  }
	}
  }
}

/* ----------------------------------------------------------------------
 i  allocate all arrays
------------------------------------------------------------------------- */

void PairSpinElastic::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cut_spin_elastic,n+1,n+1,"pair/spin/soc/elastic:cut_spin_elastic");

  memory->create(b1_mag,n+1,n+1,"pair/spin/soc/elastic:b1_mag");
  memory->create(b1_mech,n+1,n+1,"pair/spin/soc/elastic:b1_mech");
  memory->create(b2_mag,n+1,n+1,"pair/spin/soc/elastic:b2_mag");
  memory->create(b2_mech,n+1,n+1,"pair/spin/soc/elastic:b2_mech");
  memory->create(r0,n+1,n+1,"pair/spin/soc/elastic:r0");
  memory->create(e0,n+1,n+1,"pair/spin/soc/elastic:e0");
  memory->create(cutsq,n+1,n+1,"pair/spin/soc/elastic:cutsq");
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairSpinElastic::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&b1_mag[i][j],sizeof(double),1,fp);
        fwrite(&b1_mech[i][j],sizeof(double),1,fp);
        fwrite(&b2_mag[i][j],sizeof(double),1,fp);
        fwrite(&b2_mech[i][j],sizeof(double),1,fp);
        fwrite(&r0[i][j],sizeof(double),1,fp);		
	fwrite(&e0[i][j],sizeof(double),1,fp);		
        fwrite(&cut_spin_elastic[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairSpinElastic::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++) {
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR,&setflag[i][j],sizeof(int),1,fp,nullptr,error);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR,&b1_mag[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&b1_mech[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&b2_mag[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&b2_mech[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&r0[i][j],sizeof(double),1,fp,nullptr,error);
	  utils::sfread(FLERR,&e0[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&cut_spin_elastic[i][j],sizeof(double),1,fp,nullptr,error);
        }
        MPI_Bcast(&b1_mag[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&b1_mech[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&b2_mag[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&b2_mech[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&r0[i][j],1,MPI_DOUBLE,0,world);
	MPI_Bcast(&e0[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut_spin_elastic[i][j],1,MPI_DOUBLE,0,world);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairSpinElastic::write_restart_settings(FILE *fp)
{
  fwrite(&cut_spin_elastic_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairSpinElastic::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR,&cut_spin_elastic_global,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&offset_flag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&mix_flag,sizeof(int),1,fp,nullptr,error);
  }
  MPI_Bcast(&cut_spin_elastic_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}
