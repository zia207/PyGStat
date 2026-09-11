C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C                                                                      %
C Copyright (C) 2003, Statios Software and Services Incorporated.  All %
C rights reserved.                                                     %
C                                                                      %
C This program has been modified from the one distributed in 1996 (see %
C below).  This version is also distributed in the hope that it will   %
C be useful, but WITHOUT ANY WARRANTY. Compiled programs based on this %
C code may be redistributed without restriction; however, this code is %
C for one developer only. Each developer or user of this source code   %
C must purchase a separate copy from Statios.                          %
C                                                                      %
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C                                                                      %
C Copyright (C) 1996, The Board of Trustees of the Leland Stanford     %
C Junior University.  All rights reserved.                             %
C                                                                      %
C The programs in GSLIB are distributed in the hope that they will be  %
C useful, but WITHOUT ANY WARRANTY.  No author or distributor accepts  %
C responsibility to anyone for the consequences of using them or for   %
C whether they serve any particular purpose or work at all, unless he  %
C says so in writing.  Everyone is granted permission to copy, modify  %
C and redistribute the programs in GSLIB, but only under the condition %
C that this notice and the above copyright notice remain intact.       %
C                                                                      %
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c
c Module to declare dynamic arrays in multiple subroutines:
c
      module geostat
      
      real,allocatable :: x(:),y(:),z(:),vr(:)
      
      end module
c
c
c
      program main
c-----------------------------------------------------------------------
c
c           Conditional Simulation of a 3-D Rectangular Grid
c           ************************************************
c
c This is a template driver program for GSLIB's "lusim" subroutine.
c 2-D realizations of a Gaussian process with a given autocovariance
c model and conditional to input Gaussian data are created. The
c conditional simulation is achieved by one matrix inversion using
c the Cholesky decomposition technique described in M. Davis, Math.
c Geol. 19(2), 1987 , and in F. Alabert , Math. Geol. 19(5) 1987
c
c The program is executed with no command line arguments.  The user
c will be prompted for the name of a parameter file.  The parameter
c file is described in the documentation (see the example lusim.par)
c and should contain the following information:
c
c   -  Name of the data file (GEOEAS format)
c   -  column numbers for x, y, wt, and variable (if no wt set to 0)
c   -  Data trimming limits
c   -  An output file (may be overwritten)
c   -  debugging level
c   -  debugging output file
c   -  Random Number Seed
c   -  The number of simulations
c   -  X grid definition (number, minimum, size): nx,xmn,xsiz
c   -  Y grid definition (number, minimum, size): ny,ymn,ysiz
c   -  Z grid definition (number, minimum, size): nz,zmn,zsiz
c   -  Variogram Definition:  number of structures, and nugget effect
c   -  The next "nst" lines requires the following (see manual):
c        a) an integer code specifying the variogram type:
c        b) the "a" parameter for the structure.
c        b) the "c" parameter for the structure.
c        c) three angles
c        d) two anisotropy ratios
c
c
c
c The output file will be a GEOEAS file containing the simulated values
c The file is ordered by x and then y then simulation (i.e., x cycles
c fastest, then y, then simulation number).
c
c
c
c AUTHOR:   F. Alabert                                        DATE: 1986
c Modified: Clayton V. Deutsch                                 1989-1999
c-----------------------------------------------------------------------
      use geostat
      include  'lusim.inc'
c
c Read the Parameter File and the Data:
c
      call readparm(MAXDAT,MAXXY)
c
c Call lusim for the simulation:
c
      call lusim(MAXDAT,MAXXY)
c
c Finished:
c
      close(lout)
      close(ldbg)
      write(*,9998) VERSION
 9998 format(/' LUSIM Version: ',f5.3, ' Finished'/)
      stop
      end
 
 
      subroutine readparm(MAXDAT,MAXXY)
c-----------------------------------------------------------------------
c
c                  Initialization and Read Parameters
c                  **********************************
c
c The input parameters and data are read in from their files. Some quick
c error checking is performed and the statistics of all the variables
c being considered are written to standard output.
c
c
c
c-----------------------------------------------------------------------
      use       msflib
      use       geostat
      include  'lusim.inc'
      character str*512
      real      var(20)
      real*8    acorni
      logical   testfl
c
c Input/Output Units:
c
      lin  = 1
      ldbg = 3
      lout = 4
c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' LUSIM Version: ',f5.3/)
c
c Get the name of the parameter file - try the default name if no input:
c
      do i=1,512
            str(i:i) = ' '
      end do
      call getarg(1,str)
      if(str(1:1).eq.' ')then
            write(*,*) 'Which parameter file do you want to use?'
            read (*,'(a)') str
      end if
      if(str(1:1).eq.' ') str(1:20) = 'lusim.par           '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'lusim.par           ') then
                  write(*,*) '        creating a blank parameter file'
                  call makepar
                  write(*,*)
            end if
            stop
      endif
      open(lin,file=str,status='OLD')
c
c Find Start of Parameters:
c
 1    read(lin,'(a4)',end=98) str(1:4)
      if(str(1:4).ne.'STAR') go to 1
c
c Read Input Parameters:
c
      read(lin,'(a512)',err=98) datafl
      call chknam(datafl,512)
      write(*,*) ' data file = ',datafl(1:40)

      read(lin,*,err=98) ix,iy,iz,ivr
      write(*,*) ' columns = ',ix,iy,iz,ivr

      read(lin,*,err=98) tmin,tmax
      write(*,*) ' trimming limits = ',tmin,tmax

      read(lin,*,err=98) idbg
      write(*,*) ' debugging option = ',idbg

      read(lin,'(a512)',err=98) dbgfl
      call chknam(dbgfl,512)
      write(*,*) ' debug file = ',dbgfl(1:40)

      read(lin,'(a512)',err=98) outfl
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,*,err=98) nsim
      write(*,*) ' number of simulations = ',nsim

      read(lin,*,err=98) nx,xmn,xsiz
      write(*,*) ' nx, xmn, xsiz = ',nx,xmn,xsiz

      read(lin,*,err=98) ny,ymn,ysiz
      write(*,*) ' ny, ymn, ysiz = ',ny,ymn,ysiz

      read(lin,*,err=98) nz,zmn,zsiz
      write(*,*) ' nz, zmn, zsiz = ',nz,zmn,zsiz

      read(lin,*,err=98) ixv(1)
      write(*,*) ' random number seed = ',ixv(1)
      do i=1,1000
             p = real(acorni(idum))
      end do

      read(lin,*,err=98) nst(1),c0(1)
      write(*,*) ' nst, c0 = ',nst(1),c0(1)

      if(nst(1).le.0) then
            write(*,9997) nst(1)
 9997       format(' nst must be at least 1, it has been set to ',i4,/,
     +             ' The c or a values can be set to zero')
            stop
      endif
      do i=1,nst(1)
            read(lin,*,err=98) it(i),cc(i),ang1(i),ang2(i),ang3(i)
            read(lin,*,err=98) aa(i),aa1,aa2
            anis1(i) = aa1 / max(aa(i),EPSLON)
            anis2(i) = aa2 / max(aa(i),EPSLON)
            write(*,*) ' it, aa, cc = ',it(i),aa(i),cc(i)
            write(*,*) ' ang1,2,3, anis1,2 = ',ang1(i),ang2(i),
     +                   ang3(i),anis1(i),anis2(i)
            if(it(i).eq.4) stop 'Power model not allowed'
      end do
      close(lin)
c
c Find the needed parameters:
c
      MAXX = nx
      MAXY = ny
      MAXZ = nz
      MAXXY = MAXX * MAXY * MAXZ
c
      open(ldbg,file=dbgfl,status='UNKNOWN')
      write(ldbg,100)
 100  format(/,'LUSIM Debugging file',/)
      open(lout,file=outfl,status='UNKNOWN')

      write(lout,201)
 201  format('LUSIM Output')
      write(lout,202) 1,nx,ny,nz,xmn,ymn,zmn,xsiz,ysiz,zsiz,nsim
 202  format(i2,3(1x,i4),3(1x,g14.8),3(1x,g12.6),i4)
      write(lout,203)
 203  format('simulated values')

c
c Check to make sure the data file exists, then either read in the
c data or write an error message and stop:
c
      inquire(file=datafl,exist=testfl)
      nd = 0
      if(.not.testfl) then
            itrans = 1
            write(*,*) 'WARNING data file ',datafl,' does not exist!'
            write(*,*) '        creating unconditional simulations.'
            return
      endif
c
c The data file exists so open the file and read in the header
c information. Initialize the storage that will be used to summarize
c the data found in the file:
c
      open(lin,file=datafl,status='OLD')
      read(lin,*,err=99)
      read(lin,*,err=99)       nvari
      do i=1,nvari
            read(lin,*)
      end do
      MAXDAT = 0
 22   read(lin,*,end=44,err=99)(var(j),j=1,nvari)
      if(var(ivr).lt.tmin.or.var(ivr).ge.tmax)go to 22
      MAXDAT = MAXDAT + 1
      go to 22
 44   continue
c
c Allocate the needed memory:
c
      allocate(x(MAXDAT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
      allocate(y(MAXDAT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
      allocate(z(MAXDAT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
      allocate(vr(MAXDAT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
      rewind(lin)
      read(lin,'(a40)',err=99) str
      read(lin,*,err=99)       nvari
      av = 0.0
      ss = 0.0
      do i=1,nvari
            read(lin,*,err=99)
      end do
c
c Read all the data until the end of the file:
c
 2    read(lin,*,end=3,err=99) (var(j),j=1,nvari)
      vrt = var(ivr)
      if(vrt.lt.tmin.or.vrt.ge.tmax) go to 2
      nd = nd + 1
      if(ix.ge.1) then
            x(nd)  = var(ix)
      else
            x(nd)  = xmn
      end if
      if(iy.ge.1) then
            y(nd)  = var(iy)
      else
            y(nd)  = ymn
      end if
      if(iz.ge.1) then
            z(nd)  = var(iz)
      else
            z(nd)  = zmn
      end if
      vr(nd) = vrt
c
c Check for co-located:
c
      do i=1,nd-1
            test = abs(x(nd)-x(i)) + abs(y(nd)-y(i)) + abs(z(nd)-z(i))
            if(test.le.EPSLON) then
                  write(*,*) ' ERROR: co-located data ',i,nd
                  stop
            end if
      end do
      av     = av + vrt
      ss     = ss + vrt*vrt
      go to 2
 3    close(lin)
c
c Compute the averages and variances as an error check for the user:
c
      av = av / max(real(nd),1.0)
      ss =(ss / max(real(nd),1.0)) - av * av
      write(*,102)    ivr,nd,av,ss
      write(ldbg,102) ivr,nd,av,ss
 102  format(/,'  Data for LUSIM: Variable number ',i2,/,
     +         '  Number   = ',i4,/,
     +         '  Average  = ',f12.4,/,
     +         '  Variance = ',f12.4)
      return
c
c Error in an Input File Somewhere:
c
 98   stop 'ERROR in parameter file!'
 99   stop 'ERROR in data file!'
      end


      subroutine lusim(MAXDAT,MAXXY)
c-----------------------------------------------------------------------
c
c           Conditional Simulation of a 3-D Rectangular Grid
c           ************************************************
c
c This subroutine generates 3-D realizations of a Gaussian process with
c a given autocovariance model, and conditional to input Gaussian data.
c The conditional simulation is achieved by a single matrix inversion
c using the Cholesky decomposition technique described in M. Davis, Math
c Geol. 19(2), 1987 , and in F. Alabert , Math. Geol. 19(5) 1987
c
c NOTE: This algorithm should only be used for small simulations
c       (nd+nx*ny<200) unless CPU time is not an issue and a check is
c       made for machine precision problems.
c
c
c INPUT VARIABLES:
c
c   nd               Number of data (no missing values)
c   x(nd)            X coordinates of the data
c   y(nd)            Y coordinates of the data
c   z(nd)            Z coordinates of the data
c   vr(nd)           Data values
c   nx               Number of blocks in X
c   ny               Number of blocks in Y
c   nz               Number of blocks in Z
c   xmn              X Coordinate of the first node
c   ymn              Y Coordinate of the first node
c   zmn              Z Coordinate of the first node
c   xsiz             X spacing of the grid nodes
c   ysiz             Y spacing of the grid nodes
c   zsiz             Z spacing of the grid nodes
c   nst              Number of variogram structures
c   c0               Nugget effect
c   it(nst)          Variogram type (1=sph,2=exp,3=gaus,4=powr)
c   aa(nst)          Range except for the power model where "aa" is
c                      the power
c   cc(nst)          Contribution of each nested structure except for
c                      the power model where "cc" is the slope
c   ang1,2,3(nst)        Azimuth angle for anisotropy
c   anis1,2(nst)        Anisotropy ratio
c   seed             Random number seed
c   lout             Fortran output unit for the simulations
c   nsim             number of simulations
c
c
c OUTPUT VARIABLES:  Simulated Values are written to "lout"
c
c
c
c WORKING VARIABLES:
c
c   xg,yg,zg(nxyz)  Location of grid nodes
c   c11(nd,nd)      Covariance matrix of the conditioning data
c   c22i(nxy,nxyz)  Covariance matrix of the grid area
c   c12(nd,nxyz)    Covariance matrix between conditioning data
c                      and simulated points
c   l11(nd,nd)      Lower triangular matrix resulting from the LU
c                      decomposition of C11
c   l22(nxyz,nxyz)  Lower triangular matrix obtained by LU decomposition
c                      of C22i - L21.U12 (see M.Davis)
c   w2(nxyz)        Vector of gaussian random numbers
c   y2(nxyz)        Vector containing the simulated values of the grid
c
c
c Original:  F.G. Alabert                                      Nov. 1985
c-----------------------------------------------------------------------
      use geostat
      include  'lusim.inc'
      real*8    p,rotmat(MAXNST,3,3),acorni
      logical   coloc
      real,allocatable :: xg(:),yg(:),zg(:),y2(:),w2(:),c11(:,:),
     +                        c12(:,:),c22(:,:),u12(:,:),c22i(:,:),
     +                        l11(:,:),l11inv(:,:),l21v1(:),l22(:,:),
     +                        le(:),slsk(:),v1(:)
c
c Allocate the needed memory:
c
      allocate(xg(MAXXY),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
      allocate(yg(MAXXY),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
      allocate(zg(MAXXY),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
      allocate(y2(MAXXY),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
      allocate(w2(MAXXY),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
      allocate(c11(MAXDAT,MAXDAT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
      allocate(c12(MAXDAT,MAXXY),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
      allocate(c22(MAXXY,MAXXY),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
      allocate(u12(MAXDAT,MAXXY),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
      allocate(c22i(MAXXY,MAXXY),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
      allocate(l11(MAXDAT,MAXDAT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
      allocate(l11inv(MAXDAT,MAXDAT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
      allocate(l21v1(MAXXY),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
      allocate(l22(MAXXY,MAXXY),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
      allocate(le(MAXDAT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
      allocate(slsk(MAXXY),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
      allocate(v1(MAXDAT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
c
c Set up the rotation/anisotropy matrices that are needed for the
c variogram:
c
      do is=1,nst(1)
            call setrot(ang1(is),ang2(is),ang3(is),anis1(is),anis2(is),
     +                  is,MAXNST,rotmat)
      end do
      nxy  = nx*ny
      nxyz = nx*ny*nz
c
c Establish positions of grid points (may be fewer than nx*ny*nz):
c
c                 nxyzu = number actually used
c
      nxyzu = 0
      do iz = 1,nz
      do iy = 1,ny
      do ix = 1,nx
            zz = zmn + (iz-1)*zsiz
            yy = ymn + (iy-1)*ysiz
            xx = xmn + (ix-1)*xsiz
            coloc = .false.
            do i=1,nd
                  test = abs(xx-x(i)) + abs(yy-y(i)) + abs(zz-z(i))
                  if(test.le.EPSLON) coloc = .true.
            end do
c
c Only simulate this grid node if not colocated:
c
            if(.not.coloc) then
                  nxyzu     = nxyzu + 1
                  xg(nxyzu) = xx
                  yg(nxyzu) = yy
                  zg(nxyzu) = zz
                  if(idbg.ge.3) write(ldbg,101) nxyzu,xg(nxyzu),
     +                                          yg(nxyzu),zg(nxyzu)
 101              format(' Node ',i4,': x = ',f9.3,' y = ',f9.3,
     +                                             ' z = ',f9.3)
            endif
      end do
      end do
      end do
c
c Compute C22i: first get all the covariances:
c
      call cova3(xg(1),yg(1),zg(1),xg(1),yg(1),zg(1),1,nst,MAXNST,
     +           c0,it,cc,aa,1,MAXNST,rotmat,cmax,sill)
      do i = 1,nxyzu
      do j = i,nxyzu
            call cova3(xg(i),yg(i),zg(i),xg(j),yg(j),zg(j),1,nst,MAXNST,
     +                 c0,it,cc,aa,1,MAXNST,rotmat,cmax,cov)
            c22i(i,j) = cov
            c22i(j,i) = cov
            if(idbg.ge.3) write(ldbg,102) i,j,cov
 102        format(' Covariance between Node ',i4,' and ',i4,' = ',f9.6)
      end do
      end do
c
c Compute C11, L11, inv(L11) AND inv(L11).Z1: First compute c11:
c
      do i=1,nd
            c11(i,i) = sill
            do j=i+1,nd
                  call cova3(x(i),y(i),z(i),x(j),y(j),z(j),1,nst,MAXNST,
     +                       c0,it,cc,aa,1,MAXNST,rotmat,cmax,cov)
                  c11(i,j) = cov
                  c11(j,i) = cov
                  if(idbg.ge.3) write(ldbg,103) i,j,cov
            end do
      end do
 103  format(' Data-Data Covariance: ',i4,' and ',i4,' = ',f9.6)
c
c Compute l11:
c
      call chol(c11,l11,nd,MAXDAT,ierr)
c
c Compute inv(l11):
c
      call linv(l11,l11inv,nd,MAXDAT)
c
c Compute l11inv.z1:
c
      do i=1,nd
            v1(i) = 0
            do k=1,i
                  v1(i) = v1(i) + l11inv(i,k)*vr(k)
            end do
      end do
c
c Computation of LAMBDAe:
c
      sle = 0.
      do i = 1,nd
            le(i) = 0.
            do k = 1,nd
                  do j = 1,nd
                        le(i) = le(i) + l11inv(j,i)*l11inv(j,k)
                  end do
            end do
            sle = sle + le(i)
      end do
c
c Compute C12 :
c
      do i = 1,nd
            do j = 1,nxyzu
                  call cova3(x(i),y(i),z(i),xg(j),yg(j),zg(j),1,nst,
     +                 MAXNST,c0,it,cc,aa,1,MAXNST,rotmat,cmax,cov)
                  c12(i,j) = cov
                  if(idbg.ge.3) write(ldbg,104) i,j,cov
      end do
      end do
 104  format(' Covariance between data ',i4,' and node ',i4,' = ',f9.6)
c
c Compute U12 = L11INV.C12:
c
      do i = 1,nd
            do j = 1,nxyzu
                  u12(i,j) = 0
                  do k = 1,i
                        u12(i,j) = u12(i,j) + l11inv(i,k)*c12(k,j)
                  end do
            end do
      end do
c
c Compute C22 - L21.U12 = C22 - U12'.U12 ---> C22:
c
      do i = 1,nxyzu
            do j = 1,nxyzu
                  c22(i,j) = c22i(i,j)
                  do k = 1,nd
                        c22(i,j) = c22(i,j) - u12(k,i)*u12(k,j)
                  end do
            end do
      end do
c
c Compute L22:
c
      call chol(c22,l22,nxyzu,MAXXY,ierr)
c
c Compute L21.L11INV.Z1 = U12'.V1 ---> L21V1:
c
      do i = 1,nxyzu
            l21v1(i) = 0
            do k = 1,nd
                l21v1(i) = l21v1(i) + u12(k,i)*v1(k)
            end do
      end do
c
c Compute LAMBDAsk:
c
      do i = 1,nxyzu
            slsk(i) = 0.
            do k = 1,nd
                  do j = k,nd
                        slsk(i) = slsk(i) + u12(j,i)*l11inv(j,k)
                  end do
            end do
      end do
c
c Local bias correction:
c
      do i = 1,nxyzu
            do j = 1,nd
                  l21v1(i) = l21v1(i) + (1-slsk(i))*le(j)*vr(j)/sle
            end do
      end do
c
c LOOP OVER THE NUMBER OF SIMULATIONS
c
      do 100 isim=1,nsim
c
c Generate (nxyzu) Gaussian Random Numbers:
c
            do i=1,nxyzu
                  w2(i) = real(acorni(idum))
            end do
            do i = 1,nxyzu
 1                p = dble(w2(i))
                  call gauinv(p,xp,ierr)
                  w2(i) = xp
                  if(w2(i).lt.(-6.).or.w2(i).gt.(6.)) then
                        w2(i) = real(acorni(idum))
                        go to 1
                  endif
            end do
c
c Compute the simulation at each grid point:
c
            do i=1,nxyzu
                  y2(i) = l21v1(i)
                  do k=1,i
                        y2(i) = y2(i) + l22(i,k)*w2(k)
                  end do
            end do
c
c Write the simulation out to file:
c
            ic = 0
            do iz=1,nz
                  do iy=1,ny
                        do ix=1,nx
                              zz = zmn + (iz-1)*zsiz
                              yy = ymn + (iy-1)*ysiz
                              xx = xmn + (ix-1)*xsiz
                              ic = ic + 1
                              test = abs(xx-xg(ic)) + 
     +                               abs(yy-yg(ic)) +
     +                               abs(zz-zg(ic))
c
c Was this grid node simulated or was it co-located with a datum:
c
                              if(test.le.EPSLON) then
                                    simval = y2(ic)
                              else
                                    ic = ic - 1
                                    do i=1,nd
                                          test = abs(xx-x(i)) + 
     +                                           abs(yy-y(i)) +
     +                                           abs(zz-z(i))
                                          if(test.le.EPSLON) then
                                                simval = vr(i)
                                                go to 2
                                          end if
                                    end do
 2                                  continue
                              end if
                              write(lout,'(g14.8)') simval
            end do
            end do
            end do
c
c END LOOP OVER SIMULATIONS:
c
 100  continue
      return
      end
 
 
 
      subroutine chol(a,t,n,ndim,ierr)
c-----------------------------------------------------------------------
c
c                      Cholesky Decomposition
c                      **********************
c
c This subroutine calculates the lower triangular matrix T which, when
c multiplied by its own transpose, gives the symmetric matrix A. (from
c "Numerical Analysis of Symmetric Matrices,"  H.R. Schwarz et al.,
c p. 254)
c
c
c
c INPUT VARIABLES:
c
c   a(n,n)           Symmetric positive definite matrix to be
c                      decomposed (destroyed in the calculation of t)
c   t(n,n)           Lower triangular matrix solution
c   n                Dimension of the system you're decomposing
c   ndim             Dimension of matrix a (Note: In the main program,
c                      matrix a may have been dimensioned larger than
c                      necessary, i.e. n, the size of the system you're
c                      decomposing, may be smaller than ndim.)
c   ierr             Error code:  ierr=0 - no errors; ierr=1 - matrix a
c                      is not positive definite
c
c
c
c NO EXTERNAL REFERENCES:
c-----------------------------------------------------------------------
      dimension a(ndim,ndim),t(ndim,ndim)
      ierr = 0
c
c Check for positive definiteness:
c
      do ip=1,n
            if(a(ip,ip).le.0.0) then
                  write(*,'(a)') 'WARNING: chol - not positive definite'
                  ierr = 1
                  go to 1
            endif
            t(ip,ip) = sqrt (a(ip,ip))
            if(ip.ge.n) return
            do k = ip+1,n
                  t(k,ip) = a(ip,k)/t(ip,ip)
            end do
            do i = ip+1,n
                  do k = i,n
                        a(i,k) = a(i,k) - t(i,ip) * t(k,ip)
                  end do
            end do
 1          continue
      end do
c
c Finished:
c
      return
      end
 
 
 
      subroutine linv(a,b,n,ndim)
c-----------------------------------------------------------------------
c
c                Inverse of a Lower Triangular Matrix
c                ************************************
c
c This subroutine finds the inverse of a lower triangular matrix A and
c stores the answer in B. (from "Numerical Analysis of Symmetric
c Matrices,"  H.R. Schwarz et al.,)
c
c
c
c INPUT VARIABLES:
c
c   a(n,n)           Lower triangular matrix to be inverted
c   b(n,n)           the inverse
c   n                Dimension of the matrix you're inverting
c   ndim             Dimension of matrix a (Note: In the main program,
c                      matrix a may have been dimensioned larger than
c                      necessary, i.e. n, the size of the system you're
c                      decomposing, may be smaller than ndim.)
c
c-----------------------------------------------------------------------
      dimension a(ndim,ndim),b(ndim,ndim)
      do i = 1,n
            if(i.gt.1) then
                  do k = 1,i-1
                        sum=0.
                        do j = k,i-1
                              sum = sum + a(i,j)*b(j,k)
                        end do
                        b(i,k) = -sum/a(i,i)
                  end do
            end if
            b(i,i) = 1./a(i,i)
      end do
c
c Finished:
c
      return
      end



      subroutine makepar
c-----------------------------------------------------------------------
c
c                      Write a Parameter File
c                      **********************
c
c
c
c-----------------------------------------------------------------------
      lun = 99
      open(lun,file='lusim.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for LUSIM',/,
     +       '                  ********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('parta.dat                        ',
     +       '-file with data')
      write(lun,12)
 12   format('1   2   0    3                   ',
     +       '-   columns for X,Y,Z, normal scores')
      write(lun,13)
 13   format('-1.0e21    1.0e21                ',
     +       '-   trimming limits')
      write(lun,14)
 14   format('3                                ',
     +       '-debugging level: 0,1,2,3')
      write(lun,15)
 15   format('lusim.dbg                        ',
     +       '-file for debugging output')
      write(lun,16)
 16   format('lusim.out                        ',
     +       '-file for realization(s)')
      write(lun,17)
 17   format('100                              ',
     +       '-number of realizations')
      write(lun,18)
 18   format('4   40.25   0.5                  ',
     +       '-nx,xmn,xsiz')
      write(lun,19)
 19   format('4   28.25   0.5                  ',
     +       '-ny,ymn,ysiz')
      write(lun,20)
 20   format('1    0.00   1.0                  ',
     +       '-nz,zmn,zsiz')
      write(lun,21)
 21   format('112063                           ',
     +       '-random number seed')
      write(lun,22)
 22   format('1    0.2                         ',
     +       '-nst, nugget effect')
      write(lun,23)
 23   format('1    0.8  0.0   0.0   0.0        ',
     +       '-it,cc,ang1,ang2,ang3')
      write(lun,24)
 24   format('         10.0  10.0  10.0        ',
     +       '-a_hmax, a_hmin, a_vert')

      close(lun)
      return
      end
