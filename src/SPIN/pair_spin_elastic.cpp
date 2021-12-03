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
	//memory->destroy(r0); // to be fixed

    
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

  double norm = 1.0/sqrt(n1xc*n1xc + n1yc*n1yc + n1zc*n1zc);
  n1x = n1xc * norm;
  n1y = n1yc * norm;
  n1z = n1zc * norm;
  norm = 1.0/sqrt(n2xc*n2xc + n2yc*n2yc + n2zc*n2zc);
  n2x = n2xc * norm;
  n2y = n2yc * norm;
  n2z = n2zc * norm;
  norm = 1.0/sqrt(n3xc*n3xc + n3yc*n3yc + n3zc*n3zc);
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
  //if(atom -> natoms == atom->natoms + atom->nghost){
	if(!allocated){
	// int nlocal = atom->nlocal;
	int nall = atom->natoms + atom->nghost;
	
	memory->grow(r0,nall,3,"pair/spin/elastic:r0");
  
	tagint *tag = atom->tag;
	double **x = atom->x;
	// for (int i = 0; i < nlocal; i++) {
	for (int i = 0; i < nall; i++) {
		r0[i][0] = x[i][0];
		r0[i][1] = x[i][1];
		r0[i][2] = x[i][2];

	}
	
   //Store box dimensions for future smart distancing in periodic case
      Lx = domain->xprd;
	  Ly = domain->yprd;
	  Lz = domain->zprd;
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
  int i,j,ii,jj,inum,jnum,itype,jtype,icomp;
  double evdwl,ecoul;
  double xi[3], sio[3], rij[3], rijo[3], sjo[3];
  double spi[3];
  double a[3][3], c[3][3], l[3][3], eij[3][3];
  double fi[3], fmi[3];
  double local_cut2;
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
    // loop on neighbors
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
    // Error Check?
    
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
   
  // compute elastic interaction 
  // Compute strain, force, and energy here AFTER summing vectors for strain calculation
 
  //TEST ENERGY CALCULATION
 
    //printf("Strain on atom i = %d e1 =%f e2 =%f e3 =%f e4 =%f e5 =%f e6 =%f Lx =%f Ly =%f Lz =%f r0 atom 0 x =%f  \n ",icomp ,eij[0][0] ,eij[1][1],eij[2][2],eij[2][1], eij[2][0], eij[1][0], Lx, Ly, Lz, r0[0][0]);
    compute_elastic(icomp,eij,fmi,spi);

    if (lattice_flag) {
	//Loop on Neighbors of atom I to create elastic mech forces


        // THIS CURRENTLY DOESNT DO ANYTHING NEED TO FIGURE OUT BETTER IMPLEMENTATION:	
	//Pair of atom Ij see if can find express corresponding to the force h mag by rij
	//Put ^^^^^ into the elastic mech function
	//works??????



       for (jj = 0; jj < jnum; jj++) {
         j = jlist[jj];
         j &= NEIGHMASK;
     
         // // get tag number corresponding to j

         // int natoms = atom->natoms;
         // int *sametag = atom->sametag;
         // int k;
         // if (sametag[j] >= 0) {
         //   k = sametag[j];
         //   while (k >= 0) {
         //     k = sametag[k];
         //   }
         // }


         // Create rij vector from atomic positions at begining of fix
         
         rijo[0] = r0[j][0] - r0[i][0];
         rijo[1] = r0[j][1] - r0[i][1];
         rijo[2] = r0[j][2] - r0[i][2];
         // rijo[0] = r0[k][0] - r0[i][0];
         // rijo[1] = r0[k][1] - r0[i][1];
         // rijo[2] = r0[k][2] - r0[i][2];
         
         // create rij vector using CURRENT atomic positions

         rij[0] = x[j][0] - xi[0];
         rij[1] = x[j][1] - xi[1];
         rij[2] = x[j][2] - xi[2];
         
         // compute_elastic_mech(i,j,rsq,eij,fi,spi,spj); // fix eventually
    
         if (evflag) ev_tally_xyz(i,j,nlocal,newton_pair,
             evdwl,ecoul,fi[0],fi[1],fi[2],rij[0],rij[1],rij[2]);
       }
    }


    f[i][0] += fi[0]; //mechancial force
    f[i][1] += fi[1];
    f[i][2] += fi[2];
    fm[i][0] += fmi[0]; //Magnetic Force
    fm[i][1] += fmi[1];
    fm[i][2] += fmi[2];
    
    //VERYFY THIS DOESNT NEED TO BE PLACED WITHIN J LOOP
    if (eflag) {
      evdwl -= compute_elastic_energy(i,eij,spi);
      // evdwl -= (spi[0]*fmi[0] + spi[1]*fmi[1] + spi[2]*fmi[2]);
      // evdwl -= compute_elastic_energy();
      // evdwl *= 0.5*hbar;
      emag[i] += evdwl;
    } else evdwl = 0.0;
   

    // if (evflag) ev_tally_xyz(i,j,nlocal,newton_pair,   //Need to fix implementation with strain what rij to use?
    // evdwl,ecoul,fi[0],fi[1],fi[2],rij[0],rij[1],rij[2]);
	
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
 
  int j,jnum,itype,jtype,ntypes,icomp;
  int k,locflag;
  int *jlist,*numneigh,**firstneigh;
  int natoms = atom->natoms;
  int *sametag = atom->sametag;


  double rsq, inorm;

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
    // Loop on neightbors of atom ii
    
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
	//printf("Strain on atom i = %d e1 =%f e2 =%f e3 =%f e4 =%f e5 =%f e6 =%f Lx =%f Ly =%f Lz =%f r0 atom 0 x =%f  \n ",icomp ,eij[0][0] ,eij[1][1],eij[2][2],eij[2][1], eij[2][0], eij[1][0], Lx, Ly, Lz, r0[0][0]);
    // Compute Elastic Interaction
    compute_elastic(ii,eij,fmi,spi);
  }
}

/* ----------------------------------------------------------------------

void PairSpinElastic::compute_strain() // ADD VARIABLE DEFINITON
{
  /* WORKFLOW
 *	1. Get current atom i being scanned from :compute: get positions
 *	2. Get Neighborlist for atom i 	(jlist in :compute: get positions
 *	3. Create two displacement vectors
 *	 3.1 One vector (dij = xyzj - xyzi) the CURRENT seperation between atoms i & J
 *	 3.2 Second vector(dijo) the OLD seperation between atoms i & J
 *	  3.2* Should old vector be previous seperation(a timestep back) or a reference seperation (at initi)?
 *	4.  Create Vi and Wi matrixes (should be 3x3 matrixes)
 *	 4.1 Vi = Sum of all neighbors of atom i djio'*djio
 *	 4.2 Wi = Sum of all neighbors of atom i Djiot*dji
 *	 5. Create Transformation Matrix J
 *	  5.1 Ji = Vi^inv*Wi (REF SOLVE3x3EXACT)q
 *	 6. Create Strain Matrix (TENSOR) 
 *	  6.1 Eta = 0.5*(JiJi^t-I)
 *	 Notes:
 *	 Are Hydrostatic invariante and local shear invariants needed if so ref li ref. 
/

}

 ---------------------------------------------------------------------- */

void PairSpinElastic::compute_elastic(int i, double eij[3][3], double fmi[3], double spi[3])
{
  int *type = atom->type;
  int itype;
  itype = type[i];
  
  double skx,sky,skz;
  double firstx, firsty, firstz;
  double secondx, secondy, secondz;

  //create x,y,z components of spin (si * ni) = alphi
  skx = spi[0]*n1x+spi[1]*n1y+spi[2]*n1z;
  sky = spi[0]*n2x+spi[1]*n2y+spi[2]*n2z;
  skz = spi[0]*n3x+spi[1]*n3y+spi[2]*n3z;

  //calculate first part of equation 2b1*[alphi*epsiii]
  //ASSUMES MONOTYPE SYSTEM FIX LATER
  firstx = 2.0*b1_mag[itype][itype]*(skx * eij[0][0]);
  firsty = 2.0*b1_mag[itype][itype]*(sky * eij[1][1]);
  firstz = 2.0*b1_mag[itype][itype]*(skz * eij[2][2]);

  //Calculate 2nd part of equation b2[alphj * epsiij + alphk * epsiik
  //ASSUMES MONOTYPE SYSTEM FIX LATER
  secondx = b2_mag[itype][itype]*(sky*eij[0][1] + skz*eij[0][2]);
  secondy = b2_mag[itype][itype]*(skx*eij[0][1] + skz*eij[1][2]);
  secondz = b2_mag[itype][itype]*(skx*eij[0][2] + sky*eij[1][2]);

  //Sum two parts together and add into forces
  //Note** assumes n1,n2, and n3 are cardinal directions. Fix later?
  fmi[0] += (firstx + secondx);
  fmi[1] += (firsty + secondy);
  fmi[2] += (firstz + secondz);

/* OLD NEEL ELASTIC COMPUTE
  double qr,gr,g1r,q1r,q2r,ra;
  double pdx, pdy, pdz;
  double pq1x, pq1y, pq1z;
  double pq2x, pq2y, pq2z;
  double eij_si,eij_sj,si_sj,eij_si_2,eij_sj_3,coeff1;

  // compute Elastic's functions

  ra = rsq/g3[itype][jtype]/g3[itype][jtype];
  gr = 4.0*g1[itype][jtype]*ra;
  gr *= (1.0-g2[itype][jtype]*ra);
  gr *= exp(-ra);

  ra = rsq/q3[itype][jtype]/q3[itype][jtype];
  qr = 4.0*q1[itype][jtype]*ra;
  qr *= (1.0-q2[itype][jtype]*ra);
  qr *= exp(-ra);

  g1r = (gr + 12.0*qr/35.0);
  q1r = 9.0*qr/5.0;
  q2r = -2.0*qr/5.0;

  // pseudo-dipolar component

  eij_si = eij[0]*spi[0] + eij[1]*spi[1] + eij[2]*spi[2];
  eij_sj = eij[0]*spj[0] + eij[1]*spj[1] + eij[2]*spj[2];
  si_sj = spi[0]*spj[0] + spi[1]*spj[1] + spi[2]*spj[2];

  pdx = g1r*(eij_sj*eij[0] - spj[0]/3.0);
  pdy = g1r*(eij_sj*eij[1] - spj[1]/3.0);
  pdz = g1r*(eij_sj*eij[2] - spj[2]/3.0);

  // pseudo-quadrupolar components

  eij_si_2 = eij_si*eij_si;
  pq1x = -(eij_si_2 - si_sj/3.0)*spj[0]/3.0;
  pq1y = -(eij_si_2 - si_sj/3.0)*spj[1]/3.0;
  pq1z = -(eij_si_2 - si_sj/3.0)*spj[2]/3.0;

  coeff1 = (eij_sj*eij_sj-si_sj/3.0);
  pq1x += coeff1*(2.0*eij_si*eij[0] - spj[0]/3.0);
  pq1y += coeff1*(2.0*eij_si*eij[1] - spj[1]/3.0);
  pq1z += coeff1*(2.0*eij_si*eij[2] - spj[2]/3.0);

  pq1x *= q1r;
  pq1y *= q1r;
  pq1z *= q1r;

  eij_sj_3 = eij_sj*eij_sj*eij_sj;
  pq2x = 3.0*eij_si_2*eij_sj*eij[0] + eij_sj_3*eij[0];
  pq2y = 3.0*eij_si_2*eij_sj*eij[1] + eij_sj_3*eij[1];
  pq2z = 3.0*eij_si_2*eij_sj*eij[2] + eij_sj_3*eij[2];

  pq2x *= q2r;
  pq2y *= q2r;
  pq2z *= q2r;

  // adding three contributions

  fmi[0] += (pdx + pq1x + pq2x);
  fmi[1] += (pdy + pq1y + pq2y);
  fmi[2] += (pdz + pq1z + pq2z);
*/
}

/* ---------------------------------------------------------------------- 

void PairSpinElastic::compute_elastic_mech(int i, int j, double rsq, double eij[3], double fi[3],  double spi[3], double spj[3])
{
/*
  int *type = atom->type;
  int itype, jtype;
  itype = type[i];
  jtype = type[j];

  double g_mech, gij, dgij;
  double q_mech, q1ij, dq1ij;
  double q2ij, dq2ij;
  double pdx, pdy, pdz;
  double pq1x, pq1y, pq1z;
  double pq2x, pq2y, pq2z;
  double ra, rr, drij, ig3, iq3;

  drij = sqrt(rsq);

  double scalar_si_sj = spi[0]*spj[0]+spi[1]*spj[1]+spi[2]*spj[2];
  double scalar_eij_si = eij[0]*spi[0]+eij[1]*spi[1]+eij[2]*spi[2];
  double scalar_eij_sj = eij[0]*spj[0]+eij[1]*spj[1]+eij[2]*spj[2];

  // pseudo-dipolar component

  g_mech = g1_mech[itype][jtype];
  ig3 = 1.0/(g3[itype][jtype]*g3[itype][jtype]);

  ra = rsq*ig3;
  rr = drij*ig3;

  gij = 4.0*g_mech*ra;
  gij *= (1.0-g2[itype][jtype]*ra);
  gij *= exp(-ra);

  dgij = 1.0-ra-g2[itype][jtype]*ra*(2.0-ra);
  dgij *= 8.0*g_mech*rr*exp(-ra);

  double pdt1 = (dgij-2.0*gij/drij)*scalar_eij_si*scalar_eij_sj;
  pdt1 -= scalar_si_sj*dgij/3.0;
  double pdt2 = scalar_eij_sj*gij/drij;
  double pdt3 = scalar_eij_si*gij/drij;
  pdx = -(pdt1*eij[0] + pdt2*spi[0] + pdt3*spj[0]);
  pdy = -(pdt1*eij[1] + pdt2*spi[1] + pdt3*spj[1]);
  pdz = -(pdt1*eij[2] + pdt2*spi[2] + pdt3*spj[2]);

  // pseudo-quadrupolar component

  q_mech = q1_mech[itype][jtype];
  iq3 = 1.0/(q3[itype][jtype]*q3[itype][jtype]);

  ra = rsq*iq3;
  rr = drij*iq3;

  q1ij = 4.0*q_mech*ra;
  q1ij *= (1.0-q2[itype][jtype]*ra);
  q1ij *= exp(-ra);
  q2ij = -2.0*q1ij/9.0;

  dq1ij = 1.0-ra-q2[itype][jtype]*ra*(2.0-ra);
  dq1ij *= 8.0*q_mech*rr*exp(-ra);
  dq2ij = -2.0*dq1ij/9.0;

  double scalar_eij_si_2 = scalar_eij_si*scalar_eij_si;
  double scalar_eij_sj_2 = scalar_eij_sj*scalar_eij_sj;
  double pqt1 = scalar_eij_si_2 - scalar_si_sj/3.0;
  double pqt2 = scalar_eij_sj_2 - scalar_si_sj/3.0;
  pq1x = dq1ij * pqt1 * pqt2 * eij[0];
  pq1y = dq1ij * pqt1 * pqt2 * eij[1];
  pq1z = dq1ij * pqt1 * pqt2 * eij[2];

  double scalar_eij_si_3 = scalar_eij_si*scalar_eij_si*scalar_eij_si;
  double scalar_eij_sj_3 = scalar_eij_sj*scalar_eij_sj*scalar_eij_sj;
  double scalar_si_sj_2 = scalar_si_sj*scalar_si_sj;
/*  double pqt3 = 2.0*scalar_eij_si*scalar_eij_sj_3/drij;
  double pqt4 = 2.0*scalar_eij_sj*scalar_eij_si_3/drij;
  double pqt5 = -2.0*scalar_si_sj*scalar_eij_si/(3.0*drij);
  double pqt6 = -2.0*scalar_si_sj*scalar_eij_sj/(3.0*drij);
//  pq1x += q1ij*(spi[0]*(pqt3+pqt6) + spj[0]*(pqt4+));
  pq1x += q1ij*(pqt3*spi[0]+pqt4*spj[0]+pqt5*spi[0]+pqt6*spi[0]);
  pq1y += q1ij*(pqt3*spi[1]+pqt4*spj[1]+pqt5*spi[1]+pqt6*spj[1]);
  pq1z += q1ij*(pqt3*spi[2]+pqt4*spj[2]+pqt5*spi[2]+pqt6*spj[2]);
*/

/*
  double pqt3 = 2.0*scalar_eij_si*(scalar_eij_sj_2-scalar_si_sj/3.0)/drij;
  double pqt4 = 2.0*scalar_eij_sj*(scalar_eij_si_2-scalar_si_sj/3.0)/drij;
  pq1x += q1ij*(pqt3*spi[0] + pqt4*spj[0]);
  pq1y += q1ij*(pqt3*spi[1] + pqt4*spj[1]);
  pq1z += q1ij*(pqt3*spi[2] + pqt4*spj[2]);
  double pqt7 = 4.0*scalar_eij_si_2*scalar_eij_sj_2/drij;
  double pqt8 = 2.0*scalar_si_sj_2*scalar_eij_sj/(3.0*drij);
  double pqt9 = 2.0*scalar_si_sj_2*scalar_eij_si/(3.0*drij);
  pq1x -= q1ij*(pqt7 + pqt8 + pqt9)*eij[0];
  pq1y -= q1ij*(pqt7 + pqt8 + pqt9)*eij[1];
  pq1z -= q1ij*(pqt7 + pqt8 + pqt9)*eij[2];

/*
  double pqt3 = 2.0*scalar_eij_si*(scalar_eij_sj_2-scalar_si_sj/3.0)/drij;
  double pqt4 = 2.0*scalar_eij_sj*(scalar_eij_si_2-scalar_si_sj/3.0)/drij;
  pq1x += q1ij*(pqt3*spi[0] + pqt4*spj[0]);
  pq1y += q1ij*(pqt3*spi[1] + pqt4*spj[1]);
  pq1z += q1ij*(pqt3*spi[2] + pqt4*spj[2]);
*/

  //double scalar_eij_si_3 = scalar_eij_si*scalar_eij_si*scalar_eij_si;
  //double scalar_eij_sj_3 = scalar_eij_sj*scalar_eij_sj*scalar_eij_sj;
 /* double pqt10 = scalar_eij_sj*scalar_eij_si_3;
  double pqt11 = scalar_eij_si*scalar_eij_sj_3;
  pq2x = dq2ij*(pqt10 + pqt11)*eij[0];
  pq2y = dq2ij*(pqt10 + pqt11)*eij[1];
  pq2z = dq2ij*(pqt10 + pqt11)*eij[2];

  double pqt12 = scalar_eij_si_3/drij;
  double pqt13 = scalar_eij_sj_3/drij;
  double pqt14 = 3.0*scalar_eij_sj*scalar_eij_si_2/drij;
  double pqt15 = 3.0*scalar_eij_si*scalar_eij_sj_2/drij;
  pq2x += q2ij*((pqt12+pqt15)*spj[0]+(pqt13+pqt14)*spi[0]);
  pq2y += q2ij*((pqt12+pqt15)*spj[1]+(pqt13+pqt14)*spi[1]);
  pq2z += q2ij*((pqt12+pqt15)*spj[2]+(pqt13+pqt14)*spi[2]);
  double pqt16 = 4.0*scalar_eij_sj*scalar_eij_si_3/drij;
  double pqt17 = 4.0*scalar_eij_si*scalar_eij_sj_3/drij;
  pq2x -= q2ij*(pqt16 + pqt17)*eij[0];
  pq2y -= q2ij*(pqt16 + pqt17)*eij[1];
  pq2z -= q2ij*(pqt16 + pqt17)*eij[2];

  // adding three contributions

  fi[0] = pdx + pq1x + pq2x;
  fi[1] = pdy + pq1y + pq2y;
  fi[2] = pdz + pq1z + pq2z;

}

/* ---------------------------------------------------------------------- */

double PairSpinElastic::compute_elastic_energy(int i, double eij[3][3], double spi[3])
{
  
  int *type = atom->type;
  int itype;
  itype = type[i];

  double skx,sky,skz;
  double energy = 0.0;

  //create x,y,z components of spin (si * ni) = alphi
  skx = spi[0]*n1x+spi[1]*n1y+spi[2]*n1z;
  sky = spi[0]*n2x+spi[1]*n2y+spi[2]*n2z;
  skz = spi[0]*n3x+spi[1]*n3y+spi[2]*n3z;

  /*Calculate Energy
  //Note, if Energy looks odd, THIS is the suspect step
  //should energy var be positive or negative????
  //also only works in a monotype system FIX THIS
  */
  energy = b1_mech[itype][itype]*(skx*skx*eij[0][0] + sky*sky*eij[1][1] + skx*skx*eij[2][2]);
  energy += b2_mech[itype][itype]*(sky*skz*eij[1][2] + skx*skz*eij[0][2] + skx*sky*eij[0][1]);

  //return 0.5*(energy);
  return energy;
/*
  int *type = atom->type;
  int itype, jtype;
  itype = type[i];
  jtype = type[j];

  double qr,gr,g1r,q1r,q2r,ra;
  double epd,epq1,epq2;
  double eij_si,eij_sj,si_sj;
  double eij_si_2,eij_sj_2,eij_si_3,eij_sj_3;

  // compute Elastic's functions

  ra = rsq/g3[itype][jtype]/g3[itype][jtype];
  gr = 4.0*g1[itype][jtype]*ra;
  gr *= (1.0-g2[itype][jtype]*ra);
  gr *= exp(-ra);

  ra = rsq/q3[itype][jtype]/q3[itype][jtype];
  qr = 4.0*q1[itype][jtype]*ra;
  qr *= (1.0-q2[itype][jtype]*ra);
  qr *= exp(-ra);

  g1r = (gr + 12.0*qr/35.0);
  q1r = 9.0*qr/5.0;
  q2r = -2.0*qr/5.0;

  eij_si = eij[0]*spi[0] + eij[1]*spi[1] + eij[2]*spi[2];
  eij_sj = eij[0]*spj[0] + eij[1]*spj[1] + eij[2]*spj[2];
  si_sj = spi[0]*spj[0] + spi[1]*spj[1] + spi[2]*spj[2];
  epd = g1r*(eij_si*eij_sj-si_sj/3.0);
  eij_si_2 = eij_si*eij_si;
  eij_sj_2 = eij_sj*eij_sj;
  epq1 = q1r*(eij_si_2-si_sj/3.0)*(eij_sj_2-si_sj/3.0);
  eij_si_3 = eij_si*eij_si_2;
  eij_sj_3 = eij_sj*eij_sj_2;
  epq2 = q2r*(eij_si*eij_sj_3+eij_sj*eij_si_3);

  return (epd+epq1+epq2);
*/
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
          utils::sfread(FLERR,&cut_spin_elastic[i][j],sizeof(double),1,fp,nullptr,error);
        }
        MPI_Bcast(&b1_mag[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&b1_mech[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&b2_mag[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&b2_mech[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&r0[i][j],1,MPI_DOUBLE,0,world);
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
