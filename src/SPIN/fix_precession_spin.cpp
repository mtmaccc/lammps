// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
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

#include "fix_precession_spin.h"

#include <cmath>
#include <cstring>
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "input.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "respa.h"
#include "update.h"
#include "variable.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "domain.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum{CONSTANT,EQUAL};

/* ---------------------------------------------------------------------- */

FixPrecessionSpin::FixPrecessionSpin(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), emag(nullptr), list(nullptr)
{
  if (narg < 7) error->all(FLERR,"Illegal precession/spin command");

  // magnetic interactions coded for cartesian coordinates

  hbar = force->hplanck/MY_2PI;

  dynamic_group_allow = 1;
  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 3;
  global_freq = 1;
  extscalar = 1;
  extvector = 1;
  energy_global_flag = 1;
  virial_global_flag = virial_peratom_flag = 1;
  respa_level_support = 1;
  ilevel_respa = 0;

  magstr = nullptr;
  magfieldstyle = CONSTANT;

  H_field = 0.0;
  nhx = nhy = nhz = 0.0;
  hx = hy = hz = 0.0;
  stt_field = 0.0;
  nsttx = nstty = nsttz = 0.0;
  sttx = stty = sttz = 0.0;
  Ka = 0.0;
  nax = nay = naz = 0.0;
  Kax = Kay = Kaz = 0.0;
  k1c = k2c = 0.0;
  nc1x = nc1y = nc1z = 0.0;
  nc2x = nc2y = nc2z = 0.0;
  nc3x = nc3y = nc3z = 0.0;
//  ne1x = ne1y = ne1z = 0.0;
//  ne2x = ne2y = ne2z = 0.0;
//  ne3x = ne3y = ne3z = 0.0;
  b1e = b2e = 0.0;
  K6 = 0.0;
  n6x = n6y = n6z = 0.0;
  m6x = m6y = m6z = 0.0;

  zeeman_flag = stt_flag = aniso_flag = cubic_flag = hexaniso_flag = elastic_flag = 0;

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"zeeman") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix precession/spin command");
      zeeman_flag = 1;
      H_field = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      nhx = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      nhy = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      nhz = utils::numeric(FLERR,arg[iarg+4],false,lmp);
      iarg += 5;
    } else if (strcmp(arg[iarg],"stt") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix precession/spin command");
      stt_flag = 1;
      stt_field = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      nsttx = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      nstty = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      nsttz = utils::numeric(FLERR,arg[iarg+4],false,lmp);
      iarg += 5;
    } else if (strcmp(arg[iarg],"anisotropy") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix precession/spin command");
      aniso_flag = 1;
      Ka = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      nax = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      nay = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      naz = utils::numeric(FLERR,arg[iarg+4],false,lmp);
      iarg += 5;
    } else if (strcmp(arg[iarg],"cubic") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix precession/spin command");
      cubic_flag = 1;
      k1c = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      k2c = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      nc1x = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      nc1y = utils::numeric(FLERR,arg[iarg+4],false,lmp);
      nc1z = utils::numeric(FLERR,arg[iarg+5],false,lmp);
      nc2x = utils::numeric(FLERR,arg[iarg+6],false,lmp);
      nc2y = utils::numeric(FLERR,arg[iarg+7],false,lmp);
      nc2z = utils::numeric(FLERR,arg[iarg+8],false,lmp);
      nc3x = utils::numeric(FLERR,arg[iarg+9],false,lmp);
      nc3y = utils::numeric(FLERR,arg[iarg+10],false,lmp);
      nc3z = utils::numeric(FLERR,arg[iarg+11],false,lmp);
      iarg += 12;
    } else if (strcmp(arg[iarg],"hexaniso") == 0) {
      if (iarg+7 > narg) error->all(FLERR,"Illegal fix precession/spin command");
      hexaniso_flag = 1;
      K6 = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      n6x = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      n6y = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      n6z = utils::numeric(FLERR,arg[iarg+4],false,lmp);
      m6x = utils::numeric(FLERR,arg[iarg+5],false,lmp);
      m6y = utils::numeric(FLERR,arg[iarg+6],false,lmp);
      m6z = utils::numeric(FLERR,arg[iarg+7],false,lmp);
      iarg += 8;
    } else if (strcmp(arg[iarg],"elastic") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix precession/spin command");
      elastic_flag = 1;
      b1e = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      b2e = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      iarg += 3;
    } else error->all(FLERR,"Illegal precession/spin command");
  }

  // normalize vectors

  double norm2,inorm;
  if (zeeman_flag) {
    norm2 = nhx*nhx + nhy*nhy + nhz*nhz;
    if (norm2 == 0.0)
      error->all(FLERR,"Illegal precession/spin command");
    inorm = 1.0/sqrt(norm2);
    nhx *= inorm;
    nhy *= inorm;
    nhz *= inorm;
  }

  if (stt_flag) {
    norm2 = nsttx*nsttx + nstty*nstty + nsttz*nsttz;
    if (norm2 == 0.0)
      error->all(FLERR,"Illegal precession/spin command");
    inorm = 1.0/sqrt(norm2);
    nsttx *= inorm;
    nstty *= inorm;
    nsttz *= inorm;
  }

  if (aniso_flag) {
    norm2 = nax*nax + nay*nay + naz*naz;
    if (norm2 == 0.0)
      error->all(FLERR,"Illegal precession/spin command");
    inorm = 1.0/sqrt(norm2);
    nax *= inorm;
    nay *= inorm;
    naz *= inorm;
  }

  if (cubic_flag) {
    norm2 = nc1x*nc1x + nc1y*nc1y + nc1z*nc1z;
    if (norm2 == 0.0)
      error->all(FLERR,"Illegal precession/spin command");
    inorm = 1.0/sqrt(norm2);
    nc1x *= inorm;
    nc1y *= inorm;
    nc1z *= inorm;

    norm2 = nc2x*nc2x + nc2y*nc2y + nc2z*nc2z;
    if (norm2 == 0.0)
      error->all(FLERR,"Illegal precession/spin command");
    inorm = 1.0/sqrt(norm2);
    nc2x *= inorm;
    nc2y *= inorm;
    nc2z *= inorm;

    norm2 = nc3x*nc3x + nc3y*nc3y + nc3z*nc3z;
    if (norm2 == 0.0)
      error->all(FLERR,"Illegal precession/spin command");
    inorm = 1.0/sqrt(norm2);
    nc3x *= inorm;
    nc3y *= inorm;
    nc3z *= inorm;
  }

  if (hexaniso_flag) {
    norm2 = n6x*n6x + n6y*n6y + n6z*n6z;
    if (norm2 == 0.0)
      error->all(FLERR,"Illegal precession/spin command");
    inorm = 1.0/sqrt(norm2);
    n6x *= inorm;
    n6y *= inorm;
    n6z *= inorm;

    norm2 = m6x*m6x + m6y*m6y + m6z*m6z;
    if (norm2 == 0.0)
      error->all(FLERR,"Illegal precession/spin command");
    inorm = 1.0/sqrt(norm2);
    m6x *= inorm;
    m6y *= inorm;
    m6z *= inorm;
    l6x = (n6z*m6y-n6y*m6z);
    l6y = (n6x*m6z-n6z*m6x);
    l6z = (n6y*m6x-n6x*m6y);

    norm2 = l6x*l6x + l6y*l6y + l6z*l6z;
    if (norm2 == 0.0)
      error->all(FLERR,"Illegal precession/spin command");
    inorm = 1.0/sqrt(norm2);
    l6x *= inorm;
    l6y *= inorm;
    l6z *= inorm;
    m6x = (l6z*n6y-l6y*n6z);
    m6y = (l6x*n6z-l6z*n6x);
    m6z = (l6y*n6x-l6x*n6y);
  }

  degree2rad = MY_PI/180.0;
  time_origin = update->ntimestep;

  eflag = 0;
  eprec = 0.0;
}

/* ---------------------------------------------------------------------- */

FixPrecessionSpin::~FixPrecessionSpin()
{
  delete [] magstr;
  memory->destroy(emag);
}

/* ---------------------------------------------------------------------- */

int FixPrecessionSpin::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= MIN_POST_FORCE;
  mask |= POST_FORCE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixPrecessionSpin::init()
{
  const double hbar = force->hplanck/MY_2PI;    // eV/(rad.THz)
  const double mub = 5.78901e-5;                // in eV/T
  const double gyro = 2.0*mub/hbar;             // in rad.THz/T

  // convert field quantities to rad.THz
  H_field *= gyro;
  Kah = Ka/hbar;
  k1ch = k1c/hbar;
  k2ch = k2c/hbar;
  b1eh = b1e/hbar;
  b2eh = b2e/hbar;
  K6h = K6/hbar;
 

  if (utils::strmatch(update->integrate_style,"^respa")) {
    ilevel_respa = (dynamic_cast<Respa *>( update->integrate))->nlevels-1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level,ilevel_respa);
  }

  if (magstr) {
  magvar = input->variable->find(magstr);
  if (magvar < 0)
        error->all(FLERR,"Illegal precession/spin command");
  if (!input->variable->equalstyle(magvar))
        error->all(FLERR,"Illegal precession/spin command");
  }

  varflag = CONSTANT;
  if (magfieldstyle != CONSTANT) varflag = EQUAL;

  // set magnetic field components

  if (varflag == CONSTANT) set_magneticprecession();

  // init. size of energy stacking lists

  nlocal_max = atom->nlocal;
  memory->grow(emag,nlocal_max,"pair/spin:emag");

  if(elastic_flag){

    //get full neighbor listi
    //built whenever re-neighboring occours
    neighbor->add_request(this, NeighConst::REQ_FULL);
    //neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
 	  
    // Creates r0 vector for initial atomic positions & ghost atoms
   if(update->ntimestep == 0){
    //int nall = atom->nlocal;
    int nall = atom->natoms;

  /*  //clear old values from memory
   if(update->ntimestep != 0)
    memory->destroy(r0);
    */
    
    //create new r0 matrix
    memory->create(r0,nall,3,"precession/spin:r0");
     //memory->grow(r0,nall,3,"pair/spin/elastic:r0");

    double **x = atom->x;

     //get initial positions
    for (int i = 0; i < nall; i++) {
      r0[i][0]  = x[i][0];
      r0[i][1]  = x[i][1];
      r0[i][2]  = x[i][2];
      printf("atom i = %d x = %f y = %f z = %f \n",i,r0[i][0],r0[i][1],r0[i][2]);
      }     
     // Initialize box constants for energy computation

     //l = 2.4609; //2.851;
     ax = sqrt( (x[1][0] - x[0][0])*(x[1][0] - x[0][0]));
     ay = sqrt( (x[1][1] - x[0][1])*(x[1][1] - x[0][1]));
     az = sqrt( (x[1][2] - x[0][2])*(x[1][2] - x[0][2]));

     l = sqrt(ax*ax+ay*ay+az*az);	   
   

  //THIS ONLY WORKS FOR BCC, MUST FIND AN ALTERNATE METHOD FOR FCC 
   vol = 2*ax*2*ay*2*az;
    // Store box dimensions for future smart distancing in periodic scenarios
     Lx = domain ->xprd;
     Ly = domain ->yprd;
     Lz = domain ->zprd;
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixPrecessionSpin::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixPrecessionSpin::setup(int vflag)
{
  if (utils::strmatch(update->integrate_style,"^verlet"))
    post_force(vflag);
  else {
    (dynamic_cast<Respa *>( update->integrate))->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag,ilevel_respa,0);
    (dynamic_cast<Respa *>( update->integrate))->copy_f_flevel(ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixPrecessionSpin::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixPrecessionSpin::post_force(int  vflag )
{
  // update mag field with time (potential improvement)

  if (varflag != CONSTANT) {
    modify->clearstep_compute();
    modify->addstep_compute(update->ntimestep + 1);
    set_magneticprecession();           // update mag. field if time-dep.
  }

  int *mask = atom->mask;
  double **fm = atom->fm;
  double **sp = atom->sp;
  const int nlocal = atom->nlocal;
  double spi[4], fmi[3], epreci;

  // checking size of emag

  if (nlocal_max < nlocal) {                    // grow emag lists if necessary
    nlocal_max = nlocal;
    memory->grow(emag,nlocal_max,"pair/spin:emag");
  }

  eflag = 0;
  eprec = 0.0;

  //loop over all atoms
    
   
  if(elastic_flag){
    v_init(vflag);
  }


 /* int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  tagint *tag = atom->tag;*/

  //for (int ii = 0; ii < inum; ii++) {
  for (int i = 0; i < nlocal; i++) {
    //int i = ilist[ii];  
    emag[i] = 0.0;
    if (mask[i] & groupbit) {
      epreci = 0.0;
      spi[0] = sp[i][0];
      spi[1] = sp[i][1];
      spi[2] = sp[i][2];
      spi[3] = sp[i][3];
      fmi[0] = fmi[1] = fmi[2] = 0.0;

      if (zeeman_flag) {          // compute Zeeman interaction
        compute_zeeman(i,fmi);
        epreci -= compute_zeeman_energy(spi);
      }

      if (stt_flag) {             // compute Spin Transfer Torque
        compute_stt(spi,fmi);
        epreci -= compute_stt_energy(spi);
      }

      if (aniso_flag) {           // compute magnetic anisotropy
        compute_anisotropy(spi,fmi);
        epreci -= compute_anisotropy_energy(spi);
      }

      if (cubic_flag) {           // compute cubic anisotropy
        compute_cubic(spi,fmi);
        epreci -= compute_cubic_energy(spi);
      }

      if (hexaniso_flag) {        // compute hexagonal anisotropy
        compute_hexaniso(spi,fmi);
        epreci -= compute_hexaniso_energy(spi);
      }

      if (elastic_flag) {        // compute MagnetoElastic anisotropy
	int j,jj,jnum,itype,jtype,icomp,nearest;
	double xi[3], sio[3], sjo[3], rij[3], rijo[3];	
	double a[3][3], c[3][3], l[3][3], eij[3][3];
	double fi[3], v[6], unwrap[3]; 
	double delx, dely, delz;
	double local_cut2, rsq;
	int *jlist;

	double **x = atom->x;
	double **f = atom->f;
	int newton_pair = force->newton_pair;
	int natoms = atom->natoms;
	int *sametag = atom->sametag;
	imageint *image = atom->image;
	tagint *tag = atom->tag;
	
     
        int *numneigh,**firstneigh;
    
        numneigh = list->numneigh;
        firstneigh = list->firstneigh;
	
	//neighbor->build_one(list,1);
	//itype = type[i];
	xi[0] = x[i][0];
        xi[1] = x[i][1];
        xi[2] = x[i][2];

	jlist = firstneigh[i];
	jnum = numneigh[i];
	icomp = i;	
	// zero out inversion matrix for atomistic strain tensor
        for (int cx = 0; cx<3; cx++){
          for (int cy = 0; cy<3; cy++){
             a[cx][cy] = 0.0;
	     c[cx][cy] = 0.0;
	     l[cx][cy] = 0.0;
	     eij[cx][cy] = 0.0;
             }
        }	
	
	//zero out v array
	v[0] = v[1] = v[2] = v[3] = v[4] = v[5] = 0.0;

	//zero out temp fi array
	fi[0] = fi[1] = fi[2] = 0.0;
	
	//Begin loop of neighboring atoms
	for (jj = 0; jj < jnum; jj++) {
	    j = jlist[jj];
	    j &= NEIGHMASK;
		
		// define itype & jtype for cuttof calculation
		
	    rijo[0] = rijo[1] = rijo[2] = 0.0; //old Rij vector (ref initial postion)
	    rij[0] = rij[1] = rij[2] = 0.0;  // current Rij vector (current i j positions)
	    sio[0] = sio[1] = sio[2] = 0.0;
	    sjo[0] = sjo[1] = sjo[2] = 0.0;
			
	//create rij vector using current atomic positions
			
	   rij[0] = x[j][0] - xi[0];
	   rij[1] = x[j][1] - xi[1];
	   rij[2] = x[j][2] - xi[2];
			
	// Create rij vector from atomic positions at begining of fix
	// get tag number corresponding to j
			
	    //ensures j atoms are constircted with tags defined during init /
	    if(j != tag[j] - 1) j = tag[j] - 1; 
	    // j = sametag[j] //Tag is N number of atoms while j must be N-1
    	    if(i != tag[i] - 1)	icomp = tag[i]-1;
	
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
         
	  //define nearest neighbor cutoff distance
	  local_cut2 = 2.6*2.6;

	  
          // Check rsq to ensure atom j is within cutoff 
          // If failed, atom j isnt a major contributor to strain on atom i  
          if (rsq <= local_cut2) {
        
	     //Ensure that Newtonian atom force is divided evenly
	    nearest++;
		
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
	
	//printf("Atom %d previous location x=%f y=%f z=%f \n ",icomp, r0[icomp][0],r0[icomp][1],r0[icomp][2]);
	//printf("Atom %d current location  x=%f y=%f z=%f \n ",icomp, xi[0],xi[1],xi[2]);
//        printf("Current strain on atom i = %d e1 =%.16f e2 =%.16f e3 =%.16f e4 =%.16f e5 =%.16f e6 =%.16f   \n ",icomp ,eij[0][0] ,eij[1][1],eij[2][2],eij[2][1], eij[2][0], eij[1][0]);
     
	compute_elastic(eij,spi,fmi);
	epreci -= compute_elastic_energy(eij,spi);
	
	compute_elastic_mech(icomp,spi,fi); // fix eventually
	//compute_elastic_mech(spi,fi); // fix eventually 
	
	//printf("Forces on atom i = %d ,xfi=%.16f, yfi =%.16f, zfi =%.16f  \n ",icomp,fi[0],fi[1],fi[2]);
	//printf("Spin direction on atom i = %d ,xfi=%.16f, yfi =%.16f, zfi =%.16f  \n ",icomp,spi[0],spi[1],spi[2]);
	//printf("Mechanical Forces between atom i = %d ,xfi=%.16f, yfi =%.16f, zfi =%.16f  \n ",icomp,f[i][0],f[i][1],f[i][2]);
	f[i][0] += fi[0];
	f[i][1] += fi[1];
	f[i][2] += fi[2];
         
        if (newton_pair || j <nlocal) {
	  f[j][0] -= fi[0];
	  f[j][1] -= fi[1];
	  f[j][2] -= fi[2];
	}
	//compute virial continbutions due to added force
	
	if(evflag) {
          domain->unmap(x[i],image[i],unwrap);
	  v[0] = fi[0] * (unwrap[0]*eij[0][0]);
          v[1] = fi[1] * (unwrap[1]*eij[1][1]);
	  v[2] = fi[2] * (unwrap[2]*eij[2][2]);
	  v[3] = fi[0] * (unwrap[1]*eij[1][0]);
	  v[4] = fi[0] * (unwrap[2]*eij[2][0]);
	  v[5] = fi[1] * (unwrap[2]*eij[2][1]);
	  v_tally(i,v);
	}	
       }
      
      //Add computed magnetic energy & effective field to global magnetic energy & effective field
      //
      emag[i] += epreci;
      eprec += epreci;
      fm[i][0] += fmi[0];
      fm[i][1] += fmi[1];
      fm[i][2] += fmi[2];
	  
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixPrecessionSpin::compute_single_precession(int i, double spi[3], double fmi[3])
{
  int *mask = atom->mask;
  if (mask[i] & groupbit) {
    if (zeeman_flag) compute_zeeman(i,fmi);
    if (stt_flag) compute_stt(spi,fmi);
    if (aniso_flag) compute_anisotropy(spi,fmi);
    if (cubic_flag) compute_cubic(spi,fmi);
    if (hexaniso_flag) compute_hexaniso(spi,fmi);
    if (elastic_flag){

    double **x = atom->x;
    double **f = atom->f;
    double **sp = atom->sp;
    double local_cut2;

    double xi[3], rij[3], rijo[3], sio[3], sjo[3];
    double eij[3][3], a[3][3], l[3][3], c[3][3];
 
    int j,jnum,ntypes,icomp;
    int *jlist,*numneigh,**firstneigh;
    int natoms = atom->natoms;
    int *sametag = atom->sametag;
    tagint *tag = atom->tag;
    double rsq, inorm;

    numneigh = list->numneigh;
    firstneigh = list->firstneigh;
    
    xi[0] = x[i][0];
    xi[1] = x[i][1];
    xi[2] = x[i][2];

    jlist = firstneigh[i];
    jnum = numneigh[i];
    
    for (int cx = 0; cx<3; cx++){
      for (int cy = 0; cy<3; cy++){
          a[cx][cy] = 0.0;
          c[cx][cy] = 0.0;
          l[cx][cy] = 0.0;
          eij[cx][cy] = 0.0;
      }
    }	
    icomp = i;
    //loop on neighbors of atom icomp

    for (int jj = 0; jj < jnum; jj++) {

      j = jlist[jj];
      j &= NEIGHMASK;
    
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
      if(i != tag[i] - 1)
	 icomp = tag[i]-1;
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
	  
      local_cut2 =2.6*2.6; 
      //local_cut2 =4*4;  //adjust if using for non first nearest neihbors 
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

    compute_elastic(eij,spi,fmi);
    }
  }
}

/* ----------------------------------------------------------------------
   Zeeman
------------------------------------------------------------------------- */

void FixPrecessionSpin::compute_zeeman(int i, double fmi[3])
{
  double **sp = atom->sp;
  fmi[0] += sp[i][3]*hx;
  fmi[1] += sp[i][3]*hy;
  fmi[2] += sp[i][3]*hz;
}

/* ---------------------------------------------------------------------- */

double FixPrecessionSpin::compute_zeeman_energy(double spi[4])
{
  double energy = 0.0;
  double scalar = nhx*spi[0]+nhy*spi[1]+nhz*spi[2];
  energy = hbar*H_field*spi[3]*scalar;
  return energy;
}

/* ----------------------------------------------------------------------
   STT
------------------------------------------------------------------------- */

void FixPrecessionSpin::compute_stt(double spi[3], double fmi[3])
{
  double sx = spi[0];
  double sy = spi[1];
  double sz = spi[2];
  fmi[0] += 1.0*stt_field*( sy*nsttz-sz*nstty);
  fmi[1] += 1.0*stt_field*(-sx*nsttz+sz*nsttx);
  fmi[2] += 1.0*stt_field*( sx*nstty-sy*nsttx);
}

/* ---------------------------------------------------------------------- */

double FixPrecessionSpin::compute_stt_energy(double * /* spi */)
{
  double energy = 0.0;  // Non-conservative force
  return energy;
}

/* ----------------------------------------------------------------------
   compute uniaxial anisotropy interaction for spin i
------------------------------------------------------------------------- */

void FixPrecessionSpin::compute_anisotropy(double spi[3], double fmi[3])
{
  double scalar = nax*spi[0] + nay*spi[1] + naz*spi[2];
  fmi[0] += scalar*Kax;
  fmi[1] += scalar*Kay;
  fmi[2] += scalar*Kaz;
}

/* ---------------------------------------------------------------------- */

double FixPrecessionSpin::compute_anisotropy_energy(double spi[3])
{
  double energy = 0.0;
  double scalar = nax*spi[0] + nay*spi[1] + naz*spi[2];
  energy = Ka*scalar*scalar;
  return energy;
}

/* ----------------------------------------------------------------------
   compute cubic anisotropy interaction for spin i
------------------------------------------------------------------------- */

void FixPrecessionSpin::compute_cubic(double spi[3], double fmi[3])
{
  double skx,sky,skz,skx2,sky2,skz2;
  double four1,four2,four3,fourx,foury,fourz;
  double six1,six2,six3,sixx,sixy,sixz;

  skx = spi[0]*nc1x+spi[1]*nc1y+spi[2]*nc1z;
  sky = spi[0]*nc2x+spi[1]*nc2y+spi[2]*nc2z;
  skz = spi[0]*nc3x+spi[1]*nc3y+spi[2]*nc3z;

  skx2 = skx*skx;
  sky2 = sky*sky;
  skz2 = skz*skz;

  four1 = 2.0*skx*(sky2+skz2);
  four2 = 2.0*sky*(skx2+skz2);
  four3 = 2.0*skz*(skx2+sky2);

  fourx = k1ch*(nc1x*four1 + nc2x*four2 + nc3x*four3);
  foury = k1ch*(nc1y*four1 + nc2y*four2 + nc3y*four3);
  fourz = k1ch*(nc1z*four1 + nc2z*four2 + nc3z*four3);

  six1 = 2.0*skx*sky2*skz2;
  six2 = 2.0*sky*skx2*skz2;
  six3 = 2.0*skz*skx2*sky2;

  sixx = k2ch*(nc1x*six1 + nc2x*six2 + nc3x*six3);
  sixy = k2ch*(nc1y*six1 + nc2y*six2 + nc3y*six3);
  sixz = k2ch*(nc1z*six1 + nc2z*six2 + nc3z*six3);

  fmi[0] += (fourx + sixx);
  fmi[1] += (foury + sixy);
  fmi[2] += (fourz + sixz);
  
}

/* ---------------------------------------------------------------------- */

double FixPrecessionSpin::compute_cubic_energy(double spi[3])
{
  double energy = 0.0;
  double skx,sky,skz;

  skx = spi[0]*nc1x+spi[1]*nc1y+spi[2]*nc1z;
  sky = spi[0]*nc2x+spi[1]*nc2y+spi[2]*nc2z;
  skz = spi[0]*nc3x+spi[1]*nc3y+spi[2]*nc3z;

  energy = k1c*(skx*skx*sky*sky + sky*sky*skz*skz + skx*skx*skz*skz);
  energy += k2c*skx*skx*sky*sky*skz*skz;

  return energy;
}

/* ----------------------------------------------------------------------
   compute hexagonal anisotropy interaction for spin i
------------------------------------------------------------------------- */

void FixPrecessionSpin::compute_hexaniso(double spi[3], double fmi[3])
{
  double s_x,s_y;
  double pf, phi, ssint2;

  // changing to the axes' frame

  s_x = l6x*spi[0]+l6y*spi[1]+l6z*spi[2];
  s_y = m6x*spi[0]+m6y*spi[1]+m6z*spi[2];

  // hexagonal anisotropy in the axes' frame

  phi = atan2(s_y,s_x);
  ssint2 = s_x*s_x + s_y*s_y;                 // s^2sin^2(theta)
  pf = 6.0 * K6h * ssint2*ssint2*sqrt(ssint2);   // 6*K_6*s^5*sin^5(theta)
  double fm_x =  pf*cos(5*phi);
  double fm_y = -pf*sin(5*phi);
  double fm_z =  0;

  // back to the lab's frame

  fmi[0] += fm_x*l6x+fm_y*m6x+fm_z*n6x;
  fmi[1] += fm_x*l6y+fm_y*m6y+fm_z*n6y;
  fmi[2] += fm_x*l6z+fm_y*m6z+fm_z*n6z;
}

/* ----------------------------------------------------------------------
   compute hexagonal aniso energy of spin i
------------------------------------------------------------------------- */

double FixPrecessionSpin::compute_hexaniso_energy(double spi[3])
{
  double energy = 0.0;
  double s_x,s_y,s_z, phi,ssint2;

  // changing to the axes' frame

  s_x = l6x*spi[0]+l6y*spi[1]+l6z*spi[2];
  s_y = m6x*spi[0]+m6y*spi[1]+m6z*spi[2];
  s_z = n6x*spi[0]+n6y*spi[1]+n6z*spi[2];

  // hexagonal anisotropy in the axes' frame

  phi = atan2(s_y,s_z);
  ssint2 = s_x*s_x + s_y*s_y;

  energy = K6 * ssint2*ssint2*ssint2*cos(6*phi);

  return 2.0*energy;
}

/* ---------------------------------------------------------------------- */

void FixPrecessionSpin::compute_elastic(double eij[3][3], double spi[3], double fmi[3])
{
  double skx,sky,skz;
  double first1, first2, first3;
  double firstx, firsty, firstz;
  double second1,second2,second3;
  double secondx, secondy, secondz;
  double b1bar,b2bar;

  //create x,y,z components of spin (si * ni) = alphi
  skx = spi[0]*nc1x+spi[1]*nc1y+spi[2]*nc1z;
  sky = spi[0]*nc2x+spi[1]*nc2y+spi[2]*nc2z;
  skz = spi[0]*nc3x+spi[1]*nc3y+spi[2]*nc3z;

  //dimmensionalize energy constant B
  //ONLY WORKS FOR BCC MUST UPDATE TO INCLUDE FCC ATOMS
  b1bar =(vol*b1eh)/2;
  b2bar =(vol*b2eh)/2;
  //printf("skx =%.16f sky =%.16f, skz=%.16f, b1eh=%f b2eh=%f , \n ",skx,sky,skz,b1eh,b2eh);
  //calculate first part of equation 2b1*[alphi*epsiii]
  //ASSUMES MONOTYPE SYSTEM FIX LATER
  firstx = 2.0*b1bar*(skx * eij[0][0]);
  firsty = 2.0*b1bar*(sky * eij[1][1]);
  firstz = 2.0*b1bar*(skz * eij[2][2]);

  //Calculate 2nd part of equation b2[alphj * epsiij + alphk * epsiik
  //ASSUMES MONOTYPE SYSTEM FIX LATER
  secondx = 2.0*b2bar*(sky*eij[0][1] + skz*eij[0][2]);
  secondy = 2.0*b2bar*(skx*eij[0][1] + skz*eij[1][2]);
  secondz = 2.0*b2bar*(skx*eij[0][2] + sky*eij[1][2]);

  //Sum two parts together and add into forces
  //Note** assumes n1,n2, and n3 are cardinal directions. Fix later?
  //printf("effective field on atom ,xfi=%.16f, yfi =%.16f, zfi =%.16f  \n ",(firstx+secondx),(firsty+secondy),(firstz+secondz));
  fmi[0] += (firstx + secondx);
  fmi[1] += (firsty + secondy);
  fmi[2] += (firstz + secondz);
}

/* ---------------------------------------------------------------------- */
//void FixPrecessionSpin::compute_elastic_mech(double spi[3], double fi[3])
void FixPrecessionSpin::compute_elastic_mech(int i, double spi[3], double fi[3] )
{
  double skx,sky,skz,skx2,sky2,skz2;
  double fx1,fx2,fy1,fy2,fz1,fz2;
  double invnear,detotal;
  double b1bar,b2bar;
  
  //dimmensionalize energy constant B
  //ONLY WORKS FOR BCC MUST UPDATE TO INCLUDE FCC ATOMS
  b1bar =(vol*b1e)/2;
  b2bar =(vol*b2e)/2;
  
  //create x,y,z components of spin (si * ni)
  skx = spi[0]*nc1x+spi[1]*nc1y+spi[2]*nc1z;
  sky = spi[0]*nc2x+spi[1]*nc2y+spi[2]*nc2z;
  skz = spi[0]*nc3x+spi[1]*nc3y+spi[2]*nc3z;

  //create squared x,y,z components of spin (si * ni)^2
  skx2 = skx*skx;
  sky2 = sky*sky;
  skz2 = skz*skz;

  fx1 = (b1bar*skx2)/ax;
  fy1 = (b1bar*sky2)/ay;
  fz1 = (b1bar*skz2)/az;
  
  /*//assume all atoms have equal contribution to forces
  //Be careful, may not work if volume is not conserved
  invnear = 1/((double)nearest);

  //add forces to total force component
  detotal = de1+de2+de3+de4+de5+de6;

  fi[dir] -= invnear*detotal;
  */

//  fi[0] -= (fx1+fx2+fx3); 
//  fi[1] -= (fy1+fy2+fy3); 
//  fi[2] -= (fz1+fz2+fz3); 
 fi[0] -= fx1;
 fi[1] -= fy1;
 fi[2] -= fz2;

}
/* ----------------------------------------------------------------------*/ 
double FixPrecessionSpin::compute_elastic_energy(double eij[3][3], double spi[3])
{

  double skx,sky,skz;
  double b1bar,b2bar;
  double energy = 0.0;

  //dimmensionalize energy constant B
  //ONLY WORKS FOR BCC MUST UPDATE TO INCLUDE FCC ATOMS
  b1bar =(vol*b1e)/2;
  b2bar =(vol*b2e)/2;
  
  //create x,y,z components of spin (si * ni) = alphi
  skx = spi[0]*nc1x+spi[1]*nc1y+spi[2]*nc1z;
  sky = spi[0]*nc2x+spi[1]*nc2y+spi[2]*nc2z;
  skz = spi[0]*nc3x+spi[1]*nc3y+spi[2]*nc3z;

  /*Calculate Energy
  //Note, if Energy looks odd, THIS is the suspect step
  //should energy var be positive or negative????
  //also only works in a monotype system FIX THIS
  */
  energy = b1bar*(skx*skx*eij[0][0] + sky*sky*eij[1][1] + skz*skz*eij[2][2]);
  //energy -= ((b1_mech[itype][itype]*eij[0][0])/3);
  //energy -= ((b1_mech[itype][itype]*eij[1][1])/3);
  //energy -= ((b1_mech[itype][itype]*eij[2][2])/3);
  energy += 2.0*b2bar*(sky*skz*eij[1][2] + skx*skz*eij[0][2] + skx*sky*eij[0][1]);

//  printf("Magnetoelastic energy =%.16f  \n ",energy);
  //return 0.5*(energy);
  return energy;
}

/* ---------------------------------------------------------------------- */

void FixPrecessionSpin::solve3x3exactly(double a[][3],
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

/* ---------------------------------------------------------------------- */

void FixPrecessionSpin::set_magneticprecession()
{
  if (zeeman_flag) {
    hx = H_field*nhx;
    hy = H_field*nhy;
    hz = H_field*nhz;
  }

  if (stt_flag) {
    sttx = stt_field*nsttx;
    stty = stt_field*nstty;
    sttz = stt_field*nsttz;
  }

  if (aniso_flag) {
    Kax = 2.0*Kah*nax;
    Kay = 2.0*Kah*nay;
    Kaz = 2.0*Kah*naz;
  }
}

/* ----------------------------------------------------------------------
   potential energy in magnetic field
------------------------------------------------------------------------- */

double FixPrecessionSpin::compute_scalar()
{
  // only sum across procs one time

  if (eflag == 0) {
    MPI_Allreduce(&eprec,&eprec_all,1,MPI_DOUBLE,MPI_SUM,world);
    eflag = 1;
  }
  return eprec_all;
}

/* ---------------------------------------------------------------------- */

void FixPrecessionSpin::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixPrecessionSpin::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}
