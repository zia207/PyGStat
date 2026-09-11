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
      module  geostat
      
      integer,allocatable ::  ixnode(:),iynode(:),iznode(:),
     +                        icnode(:),ixsbtosr(:),
     +                        iysbtosr(:),izsbtosr(:),
     +                        it(:),nst(:),nviol(:),nisb(:) 
      real,allocatable    ::  x(:),y(:),z(:),vr(:,:),
     +                        actloc(:),tmpdat(:),zcut(:),
     +                        calib(:,:),sim(:),gcdf(:),vcut(:),
     +                        cdf(:),ccdf(:),beez(:),c0(:),
     +                        order(:),tmp(:),gcut(:),cmax(:),
     +                        ccdfo(:),cc(:),ang1(:),ang2(:),
     +                        ang3(:),anis1(:),anis2(:),aviol(:),
     +                        xviol(:),covtab(:,:,:,:),cnodex(:),
     +                        cnodey(:),cnodev(:),cnodet(:),
     +                        vra(:),sec(:),xold(:),yold(:),
     +                        zold(:),close(:),cnodez(:),aa(:)
      real*8,allocatable  ::  rotmat(:,:,:),r(:),rr(:),s(:),a(:)
      logical,allocatable ::  atnode(:),softdat(:)
      
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
c The output file will be a GEOEAS file containing the simulated values
c The file is ordered by x,y,z, and then simulation (i.e., x cycles
c fastest, then y, then z, then simulation number).
c
c
c
c AUTHOR:   Clayton V. Deutsch                           DATE: 1989-1999
c Modified: Phaedon C. Kyriakidis                          November 1996
c-----------------------------------------------------------------------
      use       geostat
      include  'sisim_gs.inc'
c
c Read the Parameter File and the Data:
c
      call readparm(MAXX,MAXY,MAXZ,MAXCTX,MAXCTY,MAXCTZ,MAXSBX,MAXSBY,
     +              MAXSBZ,MAXDAT,MAXTAB,MAXZCUT,MAXNOD,MAXSAM,MAXVCUT,
     +              MXYZ,MAXKR1,MAXKR2,MAXROT,MAXCXY,MAXXYZ,MAXSB,
     +              MXSXY,MXSX,MXZCUT)
c
c Call sisim for the simulation:
c
      call sisim   (MAXCTX,MAXCTY,MAXCTZ,MAXSBX,MAXSBY,MAXSBZ,
     +              MAXNOD,MAXVCUT,MAXKR1,MAXROT,MAXCXY,MXZCUT)
     +           
c
c Finished:
c
      close(lout)
      close(ldbg)
      write(*,9998) VERSION
 9998 format(/' SISIM_GS Version: ',f5.3, ' Finished'/)
      stop
      end
 
 

      subroutine readparm(MAXX,MAXY,MAXZ,MAXCTX,MAXCTY,MAXCTZ,MAXSBX,
     +                    MAXSBY,MAXSBZ,MAXDAT,MAXTAB,MAXZCUT,MAXNOD,
     +                    MAXSAM,MAXVCUT,MXYZ,MAXKR1,MAXKR2,MAXROT,
     +                    MAXCXY,MAXXYZ,MAXSB,MXSXY,MXSX,MXZCUT)
c-----------------------------------------------------------------------
c
c                  Initialization and Read Parameters
c                  **********************************
c
c The input parameters and data are read in from their files. Some quick
c error checking is performed and the statistics of all the variables
c being considered are written to standard output.
c
c NOTE: 1. The variables and the data are allocated in common blocks
c          (sisim.inc)
c
c
c-----------------------------------------------------------------------
      use       msflib
      use       geostat
      include  'sisim_gs.inc'
      parameter(MV=20)
      real      var(MV)
      real*8    p,acorni
      character datafl*512,tabfl*512,outfl*512,dbgfl*512,
     +          str*512,title*80,calfl*512,softfl*512
      logical   testfl
c
c Fortran unit numbers needed:
c
      lin  = 1
      lout = 2
      ldbg = 3
c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' SISIM_GS Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'sisim_gs.par        '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'sisim_gs.par        ') then
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

      read(lin,*,err=98) ivtype
      write(*,*) ' variable type (1=continuous, 0=categorical)= ',ivtype

      read(lin,*,err=98) ncutz
      write(*,*) ' number of primary thresholds / categories = ',ncutz
c
c Find the needed parameters:
c
      MAXZCUT = ncutz
      MAXROT = MAXZCUT * MAXNST + 1
      MXZCUT = MAXZCUT + 1
c
c Allocate the needed memory:
c19
      allocate(zcut(MAXZCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 19: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c27
      allocate(cdf(MAXZCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 27: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c28
      allocate(ccdf(MAXZCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 28: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c29
      allocate(beez(MAXZCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 29: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c30
      allocate(c0(MAXZCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 30: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c31
      allocate(cmax(MAXZCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 31: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c32
      allocate(ccdfo(MAXZCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 32: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c     

      read(lin,*,err=98) (zcut(i),i=1,ncutz)
      write(*,*) ' thresholds / categories = ',(zcut(i),i=1,ncutz)

      read(lin,*,err=98) (cdf(i),i=1,ncutz)
      write(*,*) ' global cdf / pdf        = ',(cdf(i),i=1,ncutz)

      read(lin,'(a512)',err=98) datafl
      call chknam(datafl,512)
      write(*,*) ' data file = ',datafl(1:40)

      read(lin,*,err=98) ixl,iyl,izl,ivrl
      write(*,*) ' input columns = ',ixl,iyl,izl,ivrl

      read(lin,'(a512)',err=98) softfl
      call chknam(softfl,512)
      write(*,*) ' soft data file = ',softfl(1:40)
      inquire(file=softfl,exist=testfl)

      if(testfl) then
            read(lin,*,err=98) ivrs
            write(*,*) ' column for gridded soft data = ',ivrs
      else
            write(*,*) 'ERROR - the soft data file does not exist,'
            write(*,*) '        check for the file and try again  '
            stop
      end if

      read(lin,'(a512)',err=98) calfl
      call chknam(calfl,512)
      write(*,*) ' Calibration table file = ',calfl(1:40)
      inquire(file=calfl,exist=testfl)

      if(.not.testfl) then
            write(*,*) 'ERROR - the calibration table does not exist,'
            write(*,*) '        check for the file and try again  '
            stop
      end if

      read(lin,*,err=98) tmin,tmax
      write(*,*) ' trimming limits      ',tmin,tmax

      read(lin,*,err=98) zmin,zmax
      write(*,*) ' data limits (tails)  ',zmin,zmax

      read(lin,*,err=98) ltail,ltpar
      write(*,*) ' lower tail = ',ltail,ltpar

      read(lin,*,err=98) middle,mpar
      write(*,*) ' middle = ',middle,mpar

      read(lin,*,err=98) utail,utpar
      write(*,*) ' upper tail = ',utail,utpar

      read(lin,'(a512)',err=98) tabfl
      call chknam(tabfl,512)
      write(*,*) ' file for tab. quant. ',tabfl(1:40)

      read(lin,*,err=98) itabvr,itabwt
      write(*,*) ' columns for vr wt = ',itabvr,itabwt

      read(lin,*,err=98) idbg
      write(*,*) ' debugging level = ',idbg

      read(lin,'(a512)',err=98) dbgfl
      call chknam(dbgfl,512)
      write(*,*) ' debugging file = ',dbgfl(1:40)

      read(lin,'(a512)',err=98) outfl
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,*,err=98) nsim
      write(*,*) ' number of simulations = ',nsim

      read(lin,*,err=98) nx,xmn,xsiz
      write(*,*) ' X grid specification = ',nx,xmn,xsiz

      read(lin,*,err=98) ny,ymn,ysiz
      write(*,*) ' Y grid specification = ',ny,ymn,ysiz

      read(lin,*,err=98) nz,zmn,zsiz
      write(*,*) ' Z grid specification = ',nz,zmn,zsiz
      nxy  = nx*ny
      nxyz = nx*ny*nz

      read(lin,*,err=98) ixv(1)
      write(*,*) ' random number seed = ',ixv(1)
      do i=1,1000
             p = acorni(idum)
      end do 

      read(lin,*,err=98) ndmax
      write(*,*) ' ndmax = ',ndmax

      read(lin,*,err=98) nodmax
      write(*,*) ' max prev sim nodes = ',nodmax

      read(lin,*,err=98) maxsec
      write(*,*) ' max soft indicator data = ',maxsec

      read(lin,*,err=98) sstrat
      write(*,*) ' search strategy = ',sstrat

      read(lin,*,err=98) mults,nmult
      write(*,*) ' multiple grid search flag = ',mults,nmult

      read(lin,*,err=98) noct
      write(*,*) ' max per octant = ',noct

      read(lin,*,err=98) radius,radius1,radius2
      write(*,*) ' search radii = ',radius,radius1,radius2
      if(radius.lt.EPSLON) stop 'radius must be greater than zero'
      radsqd = radius  * radius
      sanis1 = radius1 / radius
      sanis2 = radius2 / radius

      read(lin,*,err=98) sang1,sang2,sang3
      write(*,*) ' search anisotropy angles = ',sang1,sang2,sang3

      read(lin,*,err=98) mxctx,mxcty,mxctz
      write(*,*) ' size of covariance lookup = ',mxctx,mxcty,mxctz
      
      read(lin,*,err=98) mik,cutmik
      write(*,*) ' median IK switch = ',mik,cutmik

      read(lin,*,err=98) ktype
      write(*,*) ' kriging type switch = ',ktype
c
c Allocate the needed memory:
c9
      allocate(it(MAXZCUT*MAXNST),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 9: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c10
      allocate(nst(MAXZCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 10: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c33
      allocate(cc(MAXNST * MAXZCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 33: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c34
      allocate(aa(MAXNST * MAXZCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 34: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c35
      allocate(ang1(MAXNST * MAXZCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 35: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c36
      allocate(ang2(MAXNST * MAXZCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 36: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c37
      allocate(ang3(MAXNST * MAXZCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 37: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c38
      allocate(anis1(MAXNST * MAXZCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 38: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c39
      allocate(anis2(MAXNST * MAXZCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 39: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c
c Output now goes to debugging file:
c
      open(ldbg,file=dbgfl,status='UNKNOWN')
      do i=1,ncutz
            read(lin,*,err=98) nst(i),c0(i)
            if(ivtype.eq.0)
     +      write(ldbg,100)  i,zcut(i),cdf(i),nst(i),c0(i)
            if(ivtype.eq.1)
     +      write(ldbg,101)  i,zcut(i),cdf(i),nst(i),c0(i)
            if(nst(i).gt.MAXNST) stop 'nst is too big'
            istart = 1 + (i-1)*MAXNST
            do j=1,nst(i)
                  index = istart + j - 1
                  read(lin,*,err=98) it(index),cc(index),ang1(index),
     +                               ang2(index),ang3(index)
                  if(it(index).eq.3) STOP 'Gaussian Model Not Allowed!'
                  read(lin,*,err=98) aa(index),aa1,aa2
                  write(ldbg,102)  j,it(index),aa(index),cc(index)
                  anis1(index) = aa1 / max(EPSLON,aa(index))
                  anis2(index) = aa2 / max(EPSLON,aa(index))
                  write(ldbg,103) ang1(index),ang2(index),ang3(index),
     +                            anis1(index),anis2(index)
            end do
      end do
      close(lin)
c
c Find the needed parameters:
c
      MAXCTX = mxctx
      MAXCTY = mxcty
      MAXCTZ = mxctz
      MAXCXY = MAXCTX * MAXCTY
      MAXXYZ = MAXCTX * MAXCTY * MAXCTZ
      MAXX   = nx
      MAXY   = ny
      MAXZ   = nz
      MXYZ   = MAXX * MAXY * MAXZ
      MAXNOD = nodmax
      MAXSAM = ndmax
      MAXKR1 = 2 * MAXNOD + 2 * MAXSAM + 1
      MAXKR2 = MAXKR1 * MAXKR1
      MAXSBX = 1
      if(nx.gt.1)then
            MAXSBX = int(nx/2)
            if(MAXSBX.gt.50)MAXSBX=50
      end if
c
      MAXSBY = 1
      if(ny.gt.1)then
            MAXSBY = int(ny/2)
            if(MAXSBY.gt.50)MAXSBY=50
      end if
c
      MAXSBZ = 1
      if(nz.gt.1)then
            MAXSBZ = int(nz/2)
            if(MAXSBZ.gt.50)MAXSBZ=50
      end if
c
      MAXSB = MAXSBX*MAXSBY*MAXSBZ
      MXSXY = 4 * MAXSBX * MAXSBY
      MXSX = 2 * MAXSBX
c
c Allocate the needed memory:
c1
      allocate(ixnode(MAXXYZ),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 1: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c2
      allocate(iynode(MAXXYZ),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 2: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c3
      allocate(iznode(MAXXYZ),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 3: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c4
      allocate(nisb(MAXSB),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 4: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c5
      allocate(icnode(MAXNOD),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 5: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c6
      allocate(ixsbtosr(8*MAXSB),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 1: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c7
      allocate(iysbtosr(8*MAXSB),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 7: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c8
      allocate(izsbtosr(8*MAXSB),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 8: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c11
      allocate(nviol(MAXZCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 11: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c21
      allocate(sim(MXYZ),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 21: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c22
      MAXORD = MXYZ
      if(MXYZ.lt.MAXCXY) MAXORD=MAXCXY
      allocate(order(MAXORD),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 22: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c23
      allocate(tmp(MAXORD),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 23: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c40
      allocate(aviol(MAXZCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 40: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c41
      allocate(xviol(MAXZCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 41: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c42
      allocate(covtab(MAXCTX,MAXCTY,MAXCTZ,MAXZCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 42: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c43
      allocate(cnodex(MAXNOD),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 43: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c44
      allocate(cnodey(MAXNOD),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 44: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c45
      allocate(cnodez(MAXNOD),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 45: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c46
      allocate(cnodev(MAXNOD),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 46: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c47
      allocate(cnodet(MAXNOD),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 47: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c48
      allocate(vra(MAXKR1),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 48: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c49
      allocate(sec(MXYZ),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 49: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c53
      allocate(rotmat(MAXROT,3,3),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 53: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c54
      allocate(r(MAXKR1),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 54: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c55
      allocate(rr(MAXKR1),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 55: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c56
      allocate(s(MAXKR1),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 56: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c57
      allocate(a(MAXKR2),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 57: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c59
      allocate(softdat(MAXKR1),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 54: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
 100  format(/,' Category  number ',i2,' = ',f12.3,/,
     +         '           global prob value = ',f8.4,/,
     +         '           number of structures = ',i3,/,
     +         '           nugget effect        = ',f8.4)
 101  format(/,' Threshold number ',i2,' = ',f12.3,/,
     +         '           global prob value = ',f8.4,/,
     +         '           number of structures = ',i3,/,
     +         '           nugget effect        = ',f8.4)
 102  format(  '           type of structure ',i3,' = ',i3,/,
     +         '           aa parameter         = ',f12.4,/,
     +         '           cc parameter         = ',f12.4)
 103  format(  '           ang1, ang2, ang3     = ',3f6.2,/,
     +         '           anis1, anis2         = ',2f12.4)
c
c Read in the calibration file
c
      write(*,*)
      write(*,*) 'Reading calibration table'
      open(lin,file=calfl,status='OLD')
      read(lin,*,err=95)
      read(lin,*,err=95) ncutv
      do i=1,ncutv
           read(lin,*)
      end do
      write(*,*)'ncutv = ',ncutv
c
c Find the needed parameters:
c Allocate the needed memory:
c
      MAXVCUT = ncutv+1
c20
      allocate(calib(MAXVCUT,MAXZCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 20: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c26
      allocate(vcut(MAXVCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR 26: Allocation failed due to ',
     +                        'insufficient memory.'
                  stop
            end if
c
      rewind(lin)
      read(lin,*,err=95)
      read(lin,*,err=95) ncutv
      do i=1,ncutv
           read(lin,*,err=95) cutoff
           vcut(i) = cutoff
           write(*,*) 'Secondary threshold',vcut(i)
      end do
      read(lin,*,err=95)
      write(*,*) 'The calibration table'
      do i=1,ncutv+1
           read(lin,*)  (calib(i,j),j=1,ncutz)
           write(*,*) (calib(i,j),j=1,ncutz)
      end do
      read(lin,*,err=95)
      do i=1,ncutz
           read(lin,*,err=95) bz
           if(bz.lt.0.0.or.bz.gt.1.0) then
                write(*,*) 'Check B(z) values to be in [0,1]'
                stop
           end if
           beez(i) = bz
           write(*,*) 'Hardness index',beez(i)
      end do
      close(lin)
c
c Check to make sure the data file exists, then either read in the
c data or write a warning:
c
      title = 'SISIM_GS SIMULATIONS:                      '//
     +        '                                        '
      nd = 0
      inquire(file=datafl,exist=testfl)
      if(.not.testfl) then
            write(*,113) datafl
 113        format('WARNING data file ',a40,' does not exist!',/,
     +             ' Hope your intention was to create an',
     +             ' unconditional simulation.')
      else
c
c The data file exists so open the file and read in the header
c information.
c
            write(*,*) 'Reading input data'
            av = 0.0
            ss = 0.0
c
c Find MAXDAT:
c
            open(lin,file=datafl,status='OLD')
            read(lin,*,err=98)
            read(lin,*,err=99) nvari
            do i=1,nvari
                  read(lin,*)
            end do
            MAXDAT = 0
 33         read(lin,*,end=66,err=98)(var(j),j=1,nvari)
            if(var(ivrl).lt.tmin.or.var(ivrl).ge.tmax)go to 33
            MAXDAT = MAXDAT + 1
            go to 33
 66         continue
            write(*,*)'nvari = ',nvari
c
c Allocate the needed memory:
c12
            allocate(x(MAXDAT))
                  if(test.ne.0)then
                        write(*,*)'ERROR 12: Allocation failed ',
     +                              'due to insufficient memory.'
                        stop
                  end if
c13
            allocate(y(MAXDAT),stat = test)
                  if(test.ne.0)then
                        write(*,*)'ERROR 13: Allocation failed ',
     +                              'due to insufficient memory.'
                        stop
                  end if
c14   
            allocate(z(MAXDAT),stat = test)
                  if(test.ne.0)then
                        write(*,*)'ERROR 14: Allocation failed ',
     +                              'due to insufficient memory.'
                        stop
                  end if
c15   
            allocate(vr(MAXDAT,MXZCUT),stat = test)
                  if(test.ne.0)then
                        write(*,*)'ERROR 15: Allocation failed ',
     +                              'due to insufficient memory.'
                        stop
                  end if
c16   
            allocate(close(MAXDAT),stat = test)
                  if(test.ne.0)then
                        write(*,*)'ERROR 16: Allocation failed ',
     +                              'due to insufficient memory.'
                        stop
                  end if
c17
            allocate(actloc(MAXDAT),stat = test)
                  if(test.ne.0)then
                        write(*,*)'ERROR 17: Allocation failed ',
     +                              'due to insufficient memory.'
                        stop
                  end if
c18   
            allocate(tmpdat(MAXDAT),stat = test)
                  if(test.ne.0)then
                        write(*,*)'ERROR 18: Allocation failed ',
     +                              'due to insufficient memory.'
                        stop
                  end if
c50
            allocate(xold(MAXDAT),stat = test)
                  if(test.ne.0)then
                        write(*,*)'ERROR 50: Allocation failed ',
     +                              'due to insufficient memory.'
                        stop
                  end if
c51   
            allocate(yold(MAXDAT),stat = test)
                  if(test.ne.0)then
                        write(*,*)'ERROR 51: Allocation failed ',
     +                              'due to insufficient memory.'
                        stop
                  end if
c52
            allocate(zold(MAXDAT),stat = test)
                  if(test.ne.0)then
                        write(*,*)'ERROR 52: Allocation failed ',
     +                              'due to insufficient memory.'
                        stop
                  end if
c58
            allocate(atnode(MAXDAT),stat = test)
                  if(test.ne.0)then
                        write(*,*)'ERROR 58: Allocation failed ',
     +                              'due to insufficient memory.'
                        stop
                  end if
c 
            rewind(lin)
            read(lin,'(a60)',err=99) title(21:80)
            read(lin,*,err=99)       nvari
            do i=1,nvari
                  read(lin,'()',err=99)
            end do
c
c Read all the data until the end of the file:
c
 5          read(lin,*,end=6,err=99) (var(j),j=1,nvari)
            vrt = var(ivrl)
            if(vrt.lt.tmin.or.vrt.ge.tmax) go to 5
            nd  = nd + 1
            x(nd) = xmn
            y(nd) = ymn
            z(nd) = zmn
            if(ixl.gt.0) x(nd) = var(ixl)
            if(iyl.gt.0) y(nd) = var(iyl)
            if(izl.gt.0) z(nd) = var(izl)
            av = av + vrt
            ss = ss + vrt*vrt
c
c The indicator data are constructed knowing the thresholds and the
c data value.
c
            if(ivtype.eq.0) then
                  do ic=1,ncutz
                        vr(nd,ic) = 0.0
                        if(int(vrt+0.5).eq.int(zcut(ic)+0.5))
     +                  vr(nd,ic) = 1.0
                  end do
            else
                  do ic=1,ncutz
                        vr(nd,ic) = 1.0
                        if(vrt.gt.zcut(ic)) vr(nd,ic) = 0.0
                  end do
            end if
            vr(nd,MXZCUT) = vrt
            go to 5
 6          close(lin)
c
c Compute the averages and variances as an error check for the user:
c
            xd = max(real(nd),1.0)
            av = av / xd
            ss =(ss / xd ) - av * av
            write(*,120)    ivrl,nd,av,ss
            write(ldbg,120) ivrl,nd,av,ss
 120        format(/,'  Data for SISIM_GS: Variable number ',i2,
     +             /,'  Number of acceptable data  = ',i8,
     +             /,'  Equal Weighted Average     = ',f12.4,
     +             /,'  Equal Weighted Variance    = ',f12.4,/)
c
c Check to make sure that the grid is compatible with the data:
c
            if(ixl.le.0.and.nx.gt.1) then
               write(*,*) 'ERROR there is no X coordinate in data file'
               write(*,*) '      nx must be set to 1'
               stop
            end if
            if(iyl.le.0.and.ny.gt.1) then
               write(*,*) 'ERROR there is no Y coordinate in data file'
               write(*,*) '      ny must be set to 1'
               stop
            end if
            if(izl.le.0.and.nz.gt.1) then
               write(*,*) 'ERROR there is no Z coordinate in data file'
               write(*,*) '      nz must be set to 1'
               stop
            end if
      endif
c
c Now, if required, read in the tabulated values for details of the dist
c
      if(ltail.eq.3.or.middle.eq.3.or.utail.eq.3) then
            ng = 0
            inquire(file=tabfl,exist=testfl)
            if(.not.testfl) stop 'ERROR tabfl does not exist'
c
            open(lin,file=tabfl,status='OLD')
            read(lin,*,err=98)
            read(lin,*,err=97) nvari
            do i=1,nvari
                  read(lin,*)
            end do
            MAXTAB = 0
 23         read(lin,*,end=55,err=97)(var(j),j=1,nvari)
            MAXTAB = MAXTAB + 1
            go to 23
 55         continue
c
c Allocate the needed memory:
c24
            allocate(gcut(MAXTAB),stat = test)
                  if(test.ne.0)then
                        write(*,*)'ERROR 24: Allocation failed ',
     +                              'due to insufficient memory.'
                        stop
                  end if
c25
            allocate(gcdf(MAXTAB),stat = test)
                  if(test.ne.0)then
                        write(*,*)'ERROR 25: Allocation failed ',
     +                              'due to insufficient memory.'
                        stop
                  end if
c
            rewind(lin)
            read(lin,*,err=97)
            read(lin,*,err=97) nvari
            do i=1,nvari
                  read(lin,*,err=97)
            end do
            tcdf = 0.0
            ng   = 0
 21         read(lin,*,end=22,err=97) (var(j),j=1,nvari)
            if(var(itabvr).lt.tmin.or.var(itabvr).ge.tmax) go to 21
            ng = ng + 1
            gcut(ng) = var(itabvr)
            gcdf(ng) = 1.0
            if(itabwt.gt.0) gcdf(ng) = var(itabwt)
            tcdf = tcdf + gcdf(ng)
            go to 21
 22         close(lin)
c
c Sort in ascending order and keep track of where the tabulated values
c switch classes:
c
            if(tcdf.le.0.0) then
                  write(*,*) 'ERROR: either the weights are zero or'
                  write(*,*) '       there are no tabulated data.'
                  stop
            endif
            call sortem(1,ng,gcut,1,gcdf,c,d,e,f,g,h)
c
c Set up gcdf for tabulated quantiles:
c
            oldcp = 0.0
            cp    = 0.0
            tcdf  = 1.0 / tcdf
            do i=1,ng
                  cp      = cp + gcdf(i) * tcdf
                  gcdf(i) =(cp + oldcp) * 0.5
                  oldcp   = cp
            end do
      end if
 13   close(lin)
c
c Read in the secondary data and assign them to the grid nodes
c
      write(*,*)
      write(*,*) 'Reading soft indicator data'
      open(lin,file=softfl,status='OLD')
      read(lin,'()',err=96)
      read(lin,*,   err=96) nvari
      do i=1,nvari
           read(lin,'(a40)',err=96) str
      end do
      nds = 0
      do iz=1,nz
          do iy=1,ny
              do ix=1,nx
                  nds = nds + 1
                  read(lin,*,err=96) (var(j),j=1,nvari)
                  if(var(ivrs).lt.tmin.or.
     +               var(ivrs).gt.tmax) then
                     sec(nds) = UNEST
                  else
                     sec(nds) = var(ivrs)
                  end if
              end do
          end do
      end do
      if(nds.ne.nxyz) then
             write(*,*) ' ERROR: Secondary data grid'
             write(*,*) '        not compatible with'
             write(*,*) '        simulation grid'
      stop
      end if
      close(lin)
c
c Load the right variogram as the first one if performing median IK:
c
      if(mik.eq.1) then
            icut = 1
            clos = abs(cutmik-zcut(1))
            do ic=2,ncutz
                  test = abs(cutmik-zcut(ic))
                  if(test.lt.clos) then
                        icut = ic
                        clos = test
                  end if
            end do
            c0(1)   = c0(icut)
            nst(1)  = nst(icut)
            istart1 = 1
            istarti = 1 + (icut-1)*MAXNST
            do ist=1,nst(1)
                  index1        = istart1 + ist - 1
                  indexi        = istarti + ist - 1
                  it(index1)    = it(indexi)
                  aa(index1)    = aa(indexi)
                  cc(index1)    = cc(indexi)
                  ang1(index1)  = ang1(indexi)
                  ang2(index1)  = ang2(indexi)
                  ang3(index1)  = ang3(indexi)
                  anis1(index1) = anis1(indexi)
                  anis2(index1) = anis2(indexi)
            end do
      end if
c
c Open the output file and return:
c
      open(lout,file=outfl,status='UNKNOWN')
      write(lout,104) title
 104  format(a80)
      write(lout,105) 1,nx,ny,nz,xmn,ymn,zmn,xsiz,ysiz,zsiz,nsim
 105  format(i2,3(1x,i4),3(1x,g14.8),3(1x,g12.6),i4)  
      write(lout,106)
 106  format('Simulated Value')
      return
c
c Error in an Input File Somewhere:
c
 95   stop 'ERROR in calibration table file!'
 96   stop 'ERROR in soft data file!'
 97   stop 'ERROR in table look up file!'
 98   stop 'ERROR in parameter file!'
 99   stop 'ERROR in data file!'
      end



      subroutine sisim(MAXCTX,MAXCTY,MAXCTZ,MAXSBX,MAXSBY,MAXSBZ,
     +                  MAXNOD,MAXVCUT,MAXKR1,MAXROT,MAXCXY,
     +                  MXZCUT)
c-----------------------------------------------------------------------
c
c           Conditional Simulation of a 3-D Rectangular Grid
c           ************************************************
c
c This subroutine generates 3-D conditional simulations of a continuous
c variable with sequential indicator simulation.
c
c
c
c PROGRAM NOTES:
c
c  1. The three dimensional anisotropy parameters of the search ellipse
c     and the variogram ranges are described in section 2.3 of the
c     manual.   The variogram parameters are described in the same place
c
c  2. The conditioning data and previously simulated grid nodes can be
c     searched separately.  There can be a different maximum number of 
c     each and a minimum number of conditioning data can be specified 
c     to restrict simulation beyond the limits of the data.  The 
c     closeness of previously simulated grid nodes is measured according
c     to the variogram structural distance.
c
c  
c
c
c
c Based on the 1990 version of IK3D and the SIS program
c
c-----------------------------------------------------------------------
      use       geostat
      include  'sisim_gs.inc'
      real      ntviol,atviol
      real*8    acorni
      lin       =  1
c
c Set up the rotation/anisotropy matrices that are needed for the
c variogram and search:
c
      write(*,*) 'Setting up rotation matrices for variogram and search'
      do ic=1,ncutz
      do is=1,nst(ic)
            ind = is + (ic-1)*MAXNST
            call setrot(ang1(ind),ang2(ind),ang3(ind),anis1(ind),
     +                  anis2(ind),ind,MAXROT,rotmat)
      end do
      end do
      isrot = MAXNST*MAXVCUT + 1
      call setrot(sang1,sang2,sang3,sanis1,sanis2,isrot,MAXROT,rotmat)
c
c Set up for super block searching:
c
      if(sstrat.eq.0) then
            write(*,*) 'Setting up super block search strategy'
            do i=1,nd
                  actloc(i) = real(i)
                  xold(i) = x(i)
                  yold(i) = y(i)
                  zold(i) = z(i)
            end do
            nsec = 0
            call setsupr(nx,xmn,xsiz,ny,ymn,ysiz,nz,zmn,zsiz,nd,xold,
     +             yold,zold,
     +             actloc,tmp,nsec,sec1,sec2,sec3,MAXSBX,MAXSBY,MAXSBZ,
     +             nisb,nxsup,xmnsup,xsizsup,nysup,ymnsup,ysizsup,nzsup,
     +             zmnsup,zsizsup)
            call picksup(nxsup,xsizsup,nysup,ysizsup,nzsup,zsizsup,
     +             isrot,MAXROT,rotmat,radsqd,nsbtosr,ixsbtosr,
     +             iysbtosr,izsbtosr)
      end if
c
c Set up the covariance table and the spiral search:
c
      call ctable(MAXNOD,MAXCXY,MAXCTX,MAXCTY,MAXCTZ,MAXROT)
c
c MAIN LOOP OVER ALL THE SIMULAUTIONS:
c
      do isim=1,nsim
c
c Work out a random path for this realization:
c
            do ind=1,nxyz
                  sim(ind)   = real(acorni(idum))
                  order(ind) = ind
            end do
c
c The multiple grid search works with multiples of 4 (yes, that is
c somewhat arbitrary):
c
            if(mults.eq.1) then
                  do imult=1,nmult
                        nnz = max(1,nz/(imult*4))
                        nny = max(1,ny/(imult*4))
                        nnx = max(1,nx/(imult*4))
                        jz  = 1
                        jy  = 1
                        jx  = 1
                        do iz=1,nnz
                           if(nnz.gt.1) jz = iz*imult*4
                           do iy=1,nny
                              if(nny.gt.1) jy = iy*imult*4
                              do ix=1,nnx
                                 if(nnx.gt.1) jx = ix*imult*4
                                 index = jx + (jy-1)*nx + (jz-1)*nxy
                                 sim(index) = sim(index) - imult
                              end do
                           end do
                        end do
                  end do
            end if
            call sortem(1,nxyz,sim,1,order,c,d,e,f,g,h)
c
c Initialize the simulation:
c
            do i=1,nxyz
                  sim(i) = UNEST
                  tmp(i) = 0.0
            end do
            write(*,*)
            write(*,*) ' Working on realization number: ',isim
c
c Assign the data to the closest grid node:
c
            TINY = 0.0001
            do id=1,nd
                  call getindx(nx,xmn,xsiz,x(id),ix,testind)
                  call getindx(ny,ymn,ysiz,y(id),iy,testind)
                  call getindx(nz,zmn,zsiz,z(id),iz,testind)
                  ind  = ix + (iy-1)*nx + (iz-1)*nxy
                  xx   = xmn + real(ix-1)*xsiz
                  yy   = ymn + real(iy-1)*ysiz
                  zz   = zmn + real(iz-1)*zsiz
                  test = abs(xx-x(id)) + abs(yy-y(id)) + abs(zz-z(id))
c
c Assign this data to the node (unless there is a closer datum):
c
                  atnode(id) = .false.
                  if(sstrat.eq.1)                  atnode(id) = .true.
                  if(sstrat.eq.0.and.test.le.TINY) atnode(id) = .true.
                  if(atnode(id)) then
                        if(sim(ind).ge.0.0) then
                              id2 = int(sim(ind)+0.5)
                              test2 = abs(xx-x(id2)) + abs(yy-y(id2))
     +                                               + abs(zz-z(id2))
                              if(test.le.test2) sim(ind) = real(id)
                              if(idbg.ge.2) write(ldbg,102) id,id2
                        else
                              sim(ind) = real(id)
                        end if
                  end if
            end do
 102        format(' WARNING data values ',2i5,' are both assigned to ',
     +           /,'         the same node - taking the closest')
c
c Now, enter the hard data values into the "sim" array and keep the
c data number in the "tmp" array (to be reset when a hard value
c is assigned to that node). All other nodes are informed by a 
c soft datum:
c 
            do i=1,nxyz
                  id = int(sim(i)+0.5)
                  if(id.gt.0) then
                        sim(i) = vr(id,MXZCUT)
                  else
                        tmp(i) = real(i)
                        sim(i) = sec(i)
                  end if
            end do
c
c Accumulate the number and magnitude of order relations violations:
c
            nclose = 0
            irepo  = max(1,min((nxyz/10),10000))
            ntviol = 0.0
            atviol = 0.0
            do icut=1,ncutz
                  nviol(icut) =  0
                  aviol(icut) =  0.0
                  xviol(icut) = -1.0
            end do
c
c MAIN LOOP OVER ALL THE NODES:
c
            do in=1,nxyz
                  if((int(in/irepo)*irepo).eq.in) write(*,104) in
 104              format('   currently on node ',i9)
                  index = int(order(in)+0.5)
c
c Do we really need to simulate this grid node location?
c
                  if(tmp(index).lt.0.5) go to 20
c
c Location of the node we are currently working on:
c
                  iz = int((index-1)/nxy) + 1
                  iy = int((index-(iz-1)*nxy-1)/nx) + 1
                  ix = index - (iz-1)*nxy - (iy-1)*nx
                  xx = xmn + real(ix-1)*xsiz
                  yy = ymn + real(iy-1)*ysiz
                  zz = zmn + real(iz-1)*zsiz
                  if(idbg.ge.3)
     +            write(ldbg,*) 'Working on grid index:',ix,iy,iz
c
c Now, we'll simulate the point ix,iy,iz.  First, get the close data
c and make sure that there are enough to actually simulate a value,
c we'll only keep the closest "ndmax" data, and look for previously
c simulated grid nodes:
c
                  if(sstrat.eq.0) then
                        call srchsupr(xx,yy,zz,radsqd,isrot,MAXROT,
     +                       rotmat,nsbtosr,ixsbtosr,iysbtosr,
     +                       izsbtosr,noct,nd,x,y,z,tmpdat,nisb,nxsup,
     +                       xmnsup,xsizsup,nysup,ymnsup,ysizsup,
     +                       nzsup,zmnsup,zsizsup,nclose,close,
     +                       infoct)
                        if(nclose.gt.ndmax) nclose = ndmax
c                       do i=1,nclose
c                             iii = int(close(i)+0.5)
c                             close(i) = real(actloc(iii))
c                       end do
                  endif
                  call srchnd(ix,iy,iz)
                  if(idbg.ge.3) then
                            do iii=1,ncnode
                                  write(ldbg,789) iii,cnodev(iii)
 789                          format('Data ',i3,f8.3)
                            end do
                  end if
c
c What cdf value are we looking for?
c
                  zval   = UNEST
                  cdfval = real(acorni(idum))
c
c Use the global distribution?
c
                  if((nclose+ncnode).le.0) then
                        call beyond(ivtype,ncutz,zcut,cdf,ng,gcut,gcdf,
     +                              zmin,zmax,ltail,ltpar,middle,mpar,
     +                              utail,utpar,zval,cdfval,ierr)
                  else
c
c Estimate the local distribution by indicator kriging:
c
                        do ic=1,ncutz
                              call krige(ix,iy,iz,xx,yy,zz,ic,cdf(ic),
     +                                    ccdf(ic),MAXKR1,MAXCTX,
     +                                    MAXCTY,MAXCTZ,MAXROT)
                        end do
c
c Correct order relations:
c
                        call ordrel(ivtype,ncutz,ccdf,ccdfo,nviol,aviol,
     +                              xviol)
c
c Draw from the local distribution:
c
                        call beyond(ivtype,ncutz,zcut,ccdfo,ng,gcut,
     +                              gcdf,zmin,zmax,ltail,ltpar,middle,
     +                              mpar,utail,utpar,zval,cdfval,ierr)
c
c Write some debugging information:
c
                        if(idbg.ge.3) then
                              do ic=1,ncutz
                              write(ldbg,202) ccdf(ic),ccdfo(ic)
 202                          format('  CDF (original and fixed)',2f7.4)
                              end do
                        endif
                  endif
                  sim(index) = zval
                  if(idbg.ge.3) write(ldbg,*) ' Simulated value = ',
     +                          sim(index)
c
c END MAIN LOOP OVER NODES:
c
 20         continue
            tmp(index) = 0.0
            end do
c
c Write this simulation to the output file:
c
            nxysim = 0
            do ic=1,ncutz
                  ccdf(ic) = 0.0
            end do
            do ind=1,nxyz
                  write(lout,'(g14.8)') sim(ind)
c
c Calculate the cdf of the simulated values (for error checking):
c
                  if(sim(ind).gt.UNEST) then
                        nxysim = nxysim + 1
                        do ic=1,ncutz
                              if(ivtype.eq.0) then
                                    if(sim(ind).eq.zcut(ic))
     +                                ccdf(ic)=ccdf(ic)+1.
                              else
                                    if(sim(ind).le.zcut(ic))
     +                                ccdf(ic)=ccdf(ic)+1.
                              end if
                        end do
                  endif
            end do
c
c Report on the reproduction of the cdf and the number and magnitude
c of order relations violations:
c 
            write(*,203)    isim
            write(ldbg,203) isim
            do icut=1,ncutz
                  ccdf(icut) = ccdf(icut) / max(real(nxysim),1.0)
                  write(*,204)    icut,cdf(icut),ccdf(icut)
                  write(ldbg,204) icut,cdf(icut),ccdf(icut)
            end do
 203        format(/,' Finished simulation ',i2)
 204        format('     threshold ',i3,' input cdf = ',f6.4,
     +                 ' realization cdf = ',f6.4)
            write(*,   300)
            write(ldbg,300)
 300        format(/,' Summary of order relations: ')
            ntot = 0
            atot = 0.0
            do icut=1,ncutz
               ntot = ntot + nviol(icut)
               atot = atot + aviol(icut)
               aviol(icut) = aviol(icut) / real(max(1,nviol(icut)))
               write(*,302) icut,nviol(icut),aviol(icut),xviol(icut)
               write(ldbg,302) icut,nviol(icut),aviol(icut),xviol(icut)
 302           format('     threshold',i2,' number = ',i6,
     +                ' average = ',f8.4,' maximum = ',f8.4)
            end do
            atot = atot / real(max(1,ntot))
            btot =(ntot / real(ncutz*nxysim)) * 100.0
            write(ldbg,303) btot,atot
            write(*,   303) btot,atot
 303        format(/,' total of ',f18.6,'% with average of ',f8.4)
c
c END MAIN LOOP OVER SIMULATIONS:
c
      end do
c
c Return to the main program:
c

      return
      end


      subroutine ctable(MAXNOD,MAXCXY,MAXCTX,MAXCTY,MAXCTZ,MAXROT)
c-----------------------------------------------------------------------
c
c               Establish the Covariance Look up Table
c               **************************************
c
c The idea is to establish a 3-D network that contains the covariance
c value for a range of grid node offsets that should be at as large
c as twice the search radius in each direction.  The reason it has to
c be twice as large as the search radius is because we want to use it
c to compute the data covariance matrix as well as the data-block
c covariance matrix.
c
c Secondly, we want to establish a search for nearby nodes that 
c in order of closeness as defined by the variogram.
c
c
c
c INPUT VARIABLES:
c
c   xsiz,ysiz,zsiz  Definition of the grid being considered
c   MAXCTX,Y,Z      Number of blocks in covariance table
c
c   covariance table parameters
c
c
c
c OUTPUT VARIABLES:  covtab()         Covariance table
c
c EXTERNAL REFERENCES:
c
c   sqdist          Computes 3-D anisotropic squared distance
c   sortem          Sorts multiple arrays in ascending order
c   cova3           Computes the covariance according to a 3-D model
c
c
c
c-----------------------------------------------------------------------
      use       geostat
      parameter(TINY=1.0e-10)
      include  'sisim_gs.inc'
      real*8    sqdist,hsqd
c
c Size of the look-up table:
c
      nctx = (MAXCTX-1)/2
      ncty = (MAXCTY-1)/2
      nctz = min(((MAXCTZ-1)/2),(nz-1))
c
c Initialize the covariance subroutine and cbb at the same time:
c
      call cova3(0.,0.,0.,0.,0.,0.,1,nst,MAXNST,c0,it,cc,aa,1,
     +           MAXROT,rotmat,cmax,cbb)
c
c Now, set up the table and keep track of the node offsets that are
c within the search radius:
c
      ilooku = max((ncutz/2),1)
      nlooku = 0
      do icut=1,ncutz
      irot = 1 + (icut-1)*MAXNST
      do i=-nctx,nctx
      xx = i * xsiz
      ic = nctx + 1 + i
      do j=-ncty,ncty
      yy = j * ysiz
      jc = ncty + 1 + j
      do k=-nctz,nctz
      zz = k * zsiz
      kc = nctz + 1 + k
            call cova3(0.,0.,0.,xx,yy,zz,icut,nst,MAXNST,c0,it,cc,aa,
     +                 irot,MAXROT,rotmat,cmax,cov)
            covtab(ic,jc,kc,icut) = cov
            if(icut.eq.ilooku) then
               hsqd = sqdist(0.0,0.0,0.0,xx,yy,zz,isrot,MAXROT,rotmat)
               if(real(hsqd).le.radsqd) then
                  nlooku = nlooku + 1
c
c We subtract the covariance from a large value so that the ascending
c sort subroutine will accomplish the sort we want.  Furthermore, a
c fraction of the distance is also taken off so that we search by
c anisotropic distance once we are beyond the range:
c
                  tmp(nlooku)   =-(covtab(ic,jc,kc,icut)-TINY*hsqd)
                  order(nlooku) =real((kc-1)*MAXCXY+(jc-1)*MAXCTX+ic)
               endif 
            endif
      end do
      end do
      end do
      end do
c
c Finished setting up the look-up table, now order the nodes such
c that the closest ones, according to variogram distance, are searched
c first. Note: the "loc" array is used because I didn't want to make 
c special allowance for 2 byte integers in the sorting subroutine:
c
      call sortem(1,nlooku,tmp,1,order,c,d,e,f,g,h)
      do il=1,nlooku
            loc = int(order(il))
            iz  = int((loc-1)/MAXCXY) + 1
            iy  = int((loc-(iz-1)*MAXCXY-1)/MAXCTX) + 1
            ix  = loc-(iz-1)*MAXCXY - (iy-1)*MAXCTX
            iznode(il) = iz
            iynode(il) = iy
            ixnode(il) = ix
      end do
      if(nodmax.gt.MAXNOD) then
            write(ldbg,*)
            write(ldbg,*) 'The maximum number of close nodes = ',nodmax
            write(ldbg,*) 'this was reset from your specification due '
            write(ldbg,*) 'to storage limitations.'
            nodmax = MAXNOD
      endif
c
c Debugging output if requested:
c
      if(idbg.le.10) return
      write(ldbg,*)
      write(ldbg,*) 'There are ',nlooku,' nearby nodes that will be '
      write(ldbg,*) 'checked until enough close data are found.'
      write(ldbg,*)
      if(idbg.lt.4) return
      do i=1,nlooku
            xx = (ixnode(i) - nctx - 1) * xsiz
            yy = (iynode(i) - ncty - 1) * ysiz
            zz = (iznode(i) - nctz - 1) * zsiz
            write(ldbg,100) i,xx,yy,zz
      end do
 100  format('Point ',i6,' at ',3f18.6)
c
c All finished:
c
      return
      end



      subroutine srchnd(ix,iy,iz)
c-----------------------------------------------------------------------
c
c               Search for nearby Simulated Grid nodes
c               **************************************
c
c The idea is to spiral away from the node being simulated and note all
c the nearby nodes that have been simulated.
c
c
c
c INPUT VARIABLES:
c
c   ix,iy,iz        index of the point currently being simulated
c   sim(nx,ny,nz)   the simulation so far
c   nodmax          the maximum number of nodes that we want
c   nlooku          the number of nodes in the look up table
c   i[x,y,z]node    the relative indices of those nodes.
c   [x,y,z]mn       the origin of the global grid netwrok
c   [x,y,z]siz      the spacing of the grid nodes.
c
c
c
c OUTPUT VARIABLES:
c
c   ncnode          the number of close nodes
c   icnode()        the number in the look up table
c   cnode[x,y,z]()  the location of the nodes
c   cnodev()        the values at the nodes
c
c
c
c-----------------------------------------------------------------------
      use       geostat
      include  'sisim_gs.inc'
c
c Consider all the nearby nodes until enough have been found:
c
      ncnode = 0
      do il=1,nlooku
            if(ncnode.eq.nodmax) return
            i = ix + (int(ixnode(il))-nctx-1)
            if(i.lt.1.or.i.gt.nx) go to 1
            j = iy + (int(iynode(il))-ncty-1)
            if(j.lt.1.or.j.gt.ny) go to 1
            k = iz + (int(iznode(il))-nctz-1)
            if(k.lt.1.or.k.gt.nz) go to 1
c
c Check this potentially informed grid node:
c
            index = (k-1)*nx*ny + (j-1)*nx + i
            if(sim(index).gt.UNEST.or.tmp(index).gt.0.5) then
                  ncnode         = ncnode + 1
                  icnode(ncnode) = il
                  cnodex(ncnode) = xmn + real(i-1)*xsiz
                  cnodey(ncnode) = ymn + real(j-1)*ysiz
                  cnodez(ncnode) = zmn + real(k-1)*zsiz
                  cnodev(ncnode) = sim(index)
                  cnodet(ncnode) = tmp(index)
            endif
 1          continue
      end do
c
c Return to calling program:
c
      return
      end



      subroutine krige(ix,iy,iz,xx,yy,zz,icut,gmean,cmean,MAXKR1,
     +                  MAXCTX,MAXCTY,MAXCTZ,MAXROT)
c-----------------------------------------------------------------------
c
c            Builds and Solves the SK or OK Kriging System
c            *********************************************
c
c INPUT VARIABLES:
c
c   ix,iy,iz        index of the point currently being simulated
c   xx,yy,zz        location of the point currently being simulated
c   icut            cutoff number to use for either the covariance look
c                     up table or the covariance calculation
c
c
c
c OUTPUT VARIABLES:
c
c   cmean           kriged estimate
c
c
c 
c EXTERNAL REFERENCES: ksol   Gaussian elimination system solution
c
c
c NOTE: 1. the array "aclose" is used to flag those samples which exist
c          at the cutoff currently being kriged.
c
c
c-----------------------------------------------------------------------
      use      geostat
      include 'sisim_gs.inc'
      integer aclose(MAXKR1)
      logical first,krig,somesoft,bothsoft
c
c Size of the kriging system:  Some of the data values may be missing
c which would correspond to a constraint interval.  Note that there
c should not be any missing values if the median approximation is being
c considered.  The variable ``first'' is used as a flag for the
c covariance calculation subroutine.  The variable ``krig'' is used
c to flag whether kriging is to be done or if the previous weights are
c to be used.
c
      somesoft = .false.
      first    = .false.
      krig     = .true.
      if(mik.eq.1.and.icut.gt.1) krig = .false.
      if(krig) then
            mclose = 0
            do i=1,nclose
                  index     =  int(close(i))
                  if(.not.atnode(index).and.vr(index,icut).ge.0.0) then
                        mclose = mclose + 1
                        aclose(mclose) = index
                  endif
            end do
            na  = mclose + ncnode
            neq = na + ktype
      endif
c
c There are no data yet:
c
      irot   = 1 + (icut-1)*MAXNST
c
c Set up kriging matrices:
c
      in = 0
      j1 = 0
      do 1 j=1,na
            softdat(j) = .false.
c
c Sort out the actual location of point "j"
c
            if(j.le.mclose) then
                  index  = aclose(j)
                  vra(j) = vr(index,icut)
                  x1     = x(index)
                  y1     = y(index)
                  z1     = z(index)
                  if(index.gt.nd) softdat(j) = .true.
            else
c
c It is a previously simulated node (keep index for table look-up):
c
                  index  = j-mclose
                  x1     = cnodex(index)
                  y1     = cnodey(index)
                  z1     = cnodez(index)
c
c Is this node informed by a hard datum or a soft datum?
c
                  if(cnodet(index).le.0.5) then
                        if(ivtype.eq.0) then
                           vra(j) = 0.0
                           if(int(cnodev(index)+0.5).eq.
     +                        int(zcut(icut)+0.5)) vra(j) = 1.0
                        else
                           vra(j) = 1.0
                           if(cnodev(index).gt.zcut(icut)) vra(j) = 0.0
                        end if
                        softdat(j) = .false.
                  else
                        secval = cnodev(index)
                        if(secval.le.vcut(1)) then
                           icutv = 1
                        else if(secval.gt.vcut(ncutv)) then
                           icutv = ncutv+1
                        else
                           do 111 i=2,ncutv
                              if(secval.gt.vcut(i-1).
     +                           and.secval.le.vcut(i)) icutv = i
 111                       continue
                        endif
                        vra(j) = calib(icutv,icut)
                        softdat(j) = .true.
                  end if
                  ind    = icnode(index)
                  ix1    = ix + (int(ixnode(ind))-nctx-1)
                  iy1    = iy + (int(iynode(ind))-ncty-1)
                  iz1    = iz + (int(iznode(ind))-nctz-1)
            endif

c
c Only set up the matrix and the RHS if kriging:
c
            if(krig) then
               do 2 i=1,j
c
c Sort out the actual location of point "i"
c
                  if(i.le.mclose) then
                        index  = aclose(i)
                        x2     = x(index)
                        y2     = y(index)
                        z2     = z(index)
                  else
c
c It is a previously simulated node (keep index for table look-up):
c
                        index  = i-mclose
                        x2     = cnodex(index)
                        y2     = cnodey(index)
                        z2     = cnodez(index)
                        ind    = icnode(index)
                        ix2    = ix + (int(ixnode(ind))-nctx-1)
                        iy2    = iy + (int(iynode(ind))-ncty-1)
                        iz2    = iz + (int(iznode(ind))-nctz-1)
                  endif
c
c Now, get the covariance value:
c
                  in = in + 1
c
c Decide whether or not to use the covariance look-up table:
c
                  if(j.le.mclose.or.i.le.mclose) then
                        call cova3(x1,y1,z1,x2,y2,z2,icut,nst,MAXNST,
     +                       c0,it,cc,aa,irot,MAXROT,rotmat,cmax,cov)
                        a(in) = dble(cov)
                  else
c
c Try to use the covariance look-up (if the distance is in range):
c
                        ii = nctx + 1 + (ix1 - ix2)
                        jj = ncty + 1 + (iy1 - iy2)
                        kk = nctz + 1 + (iz1 - iz2)
                        if(ii.lt.1.or.ii.gt.MAXCTX.or.
     +                     jj.lt.1.or.jj.gt.MAXCTY.or.
     +                     kk.lt.1.or.kk.gt.MAXCTZ) then
                              call cova3(x1,y1,z1,x2,y2,z2,icut,nst,
     +                                   MAXNST,c0,it,cc,aa,irot,MAXROT,
     +                                   rotmat,cmax,cov)
                              a(in) = dble(cov)
                        else
                              a(in) = dble(covtab(ii,jj,kk,icut))
                        endif
                  endif
 2          continue
c
c Get the RHS value (possibly with covariance look-up table):
c
            if(j.le.mclose) then
                  call cova3(xx,yy,zz,x1,y1,z1,icut,nst,MAXNST,
     +                       c0,it,cc,aa,irot,MAXROT,rotmat,cmax,cov)
                  r(j) = dble(cov)
            else
c
c Try to use the covariance look-up (if the distance is in range):
c
                  ii = nctx + 1 + (ix - ix1)
                  jj = ncty + 1 + (iy - iy1)
                  kk = nctz + 1 + (iz - iz1)
                  if(ii.lt.1.or.ii.gt.MAXCTX.or.
     +               jj.lt.1.or.jj.gt.MAXCTY.or.
     +               kk.lt.1.or.kk.gt.MAXCTZ) then
                        call cova3(xx,yy,zz,x1,y1,z1,icut,nst,MAXNST,
     +                       c0,it,cc,aa,irot,MAXROT,rotmat,cmax,cov)
                        r(j) = dble(cov)
                  else
                        r(j) = dble(covtab(ii,jj,kk,icut))
                  endif
            endif
            rr(j) = r(j)
c
c End ``if'' block (true if kriging)
c
         endif
c
c End loop over all of the nearby data
c
      if(softdat(j)) somesoft = .true.
 1    continue
c
c If we are doing Markov-Bayes are there are soft data we need to
c correct some of the covariance values in the kriging matrix:
c
      if(somesoft) then
            in = 0
            do j=1,na
                  do i=1,j
                        in = in + 1
                        bothsoft = .false.
                        if(softdat(j).and.softdat(i)) bothsoft = .true.
c
c Correct for soft-soft covariance or soft-hard covariance:
c
                        if(bothsoft) then
                              a(in) = a(in)*dble(beez(icut))
                              if(i.ne.j) a(in) = a(in)*dble(beez(icut))
                        else
                              if(softdat(j).or.softdat(i))
     +                        a(in) = a(in)*dble(beez(icut))
                        end if
                  end do
c
c Correct the right hand side for soft-hard covariance:
c
                  if(softdat(j)) then
                        r(j)  = r(j)*dble(beez(icut))
                        rr(j) = r(j)
                  end if
            end do
      end if
c
c Addition of OK constraint:
c
      if(krig.and.ktype.eq.1) then
            do i=1,na
                  in    = in + 1
                  a(in) = 1.0
            end do
            in      = in + 1
            a(in)   = 0.0
            r(neq)  = 1.0
            rr(neq) = 1.0
      endif
c
c Write out the kriging Matrix if Seriously Debugging:
c
      if(krig.and.idbg.ge.4) then
            write(ldbg,101) ix,iy,iz
            is = 1
            do i=1,neq
                  ie = is + i - 1
                  write(ldbg,102) i,r(i),(a(j),j=is,ie)
                  is = is + i
            end do
 101        format(/,'Kriging Matrices for Node: ',3i4,
     +               ' RHS first')
 102        format('    r(',i2,') =',f7.4,'  a= ',9(10f7.4))
      endif
c
c Solve the Kriging System:
c
      if(krig) then
            if(neq.eq.1.and.ktype.eq.0) then
                  s(1)  = r(1) / a(1)
                  ising = 0
            else
                  call ksol(1,neq,1,a,r,s,ising)
            endif
      endif
c
c Write a warning if the matrix is singular:
c
      if(ising.ne.0) then
            if(idbg.ge.1) then
                  write(ldbg,*) 'WARNING SISIM_GS: singular matrix'
                  write(ldbg,*) '              for block',ix,iy,iz
            endif
            cmean  = 0.0
            return
      endif
c
c Write out the kriging Matrix if Seriously Debugging:
c
      if(krig.and.idbg.ge.4) then
            do i=1,na
                  write(ldbg,140) i,s(i),vra(i)
            end do
 140        format(' Kriging weight for data: ',i4,' = ',
     +              f8.4,f8.4)
      endif
c
c Compute the estimate, the sum of weights, correct for SK, and return:
c
      cmean = 0.0
      sumwt = 0.0
      do i=1,na
            cmean = cmean + real(s(i)) * vra(i)
            sumwt = sumwt + real(s(i))
      end do
      if(ktype.eq.0) cmean = cmean + (1.0-sumwt)*gmean
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
      open(lun,file='sisim_gs.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for SISIM_GS',/,
     +       '                  ***********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('1                              ',
     +       '-1=continuous(cdf), 0=categorical(pdf)')
      write(lun,12)
 12   format('5                              ',
     +       '-number thresholds/categories')
      write(lun,13)
 13   format('0.5   1.0   2.5   5.0   10.0   ',
     +       '-   thresholds / categories')
      write(lun,14)
 14   format('0.12  0.29  0.50  0.74  0.88   ',
     +       '-   global cdf / pdf')
      write(lun,15)
 15   format('cluster.dat                    ',
     +       '-file with data')
      write(lun,16)
 16   format('1   2   0   3                  ',
     +       '-   columns for X,Y,Z, and variable')
      write(lun,17)
 17   format('ydata.dat                      ',
     +       '-file with gridded soft indicator input')
      write(lun,18)
 18   format('1                              ',
     +       '-   columns for secondary variable')
      write(lun,19)
 19   format('bicalib.cal                    ',
     +       '-file with calibration table ')
      write(lun,20)
 20   format('-0.5       1.0e21              ',
     +       '-trimming limits')
      write(lun,21)
 21   format('0.0   30.0                     ',
     +       '-minimum and maximum data value')
      write(lun,22)
 22   format('1      1.0                     ',
     +       '-   lower tail option and parameter')
      write(lun,23)
 23   format('1      1.0                     ',
     +       '-   middle     option and parameter')
      write(lun,24)
 24   format('1      1.0                     ',
     +       '-   upper tail option and parameter')
      write(lun,25)
 25   format('cluster.dat                    ',
     +       '-   file with tabulated values')
      write(lun,26)
 26   format('3   5                          ',
     +       '-      columns for variable, weight')
      write(lun,27)
 27   format('3                              ',
     +       '-debugging level: 0,1,2,3')
      write(lun,28)
 28   format('sisim_gs0.dbg                  ',
     +       '-file for debugging output')
      write(lun,29)
 29   format('sisim_gs.out                   ',
     +       '-file for simulation output')
      write(lun,30)
 30   format('1                              ',
     +       '-number of realizations')
      write(lun,31)
 31   format('50   0.5    1.0                ',
     +       '-nx,xmn,xsiz')
      write(lun,32)
 32   format('50   0.5    1.0                ',
     +       '-ny,ymn,ysiz')
      write(lun,33)
 33   format('1    0.5    1.0                ',
     +       '-nz,zmn,zsiz')
      write(lun,34)
 34   format('69069                          ',
     +       '-random number seed')
      write(lun,35)
 35   format('12                             ',
     +       '-maximum original data  for each kriging')
      write(lun,36)
 36   format('12                             ',
     +       '-maximum previous nodes for each kriging')
      write(lun,37)
 37   format('12                             ',
     +       '-maximum soft indicator nodes for kriging')
      write(lun,38)
 38   format('0                              ',
     +       '-assign data to nodes? (0=no,1=yes)')
      write(lun,39)
 39   format('1     3                        ',
     +       '-multiple grid search? (0=no,1=yes),num')
      write(lun,40)
 40   format('0                              ',
     +       '-maximum per octant    (0=not used)')
      write(lun,41)
 41   format('10.0  10.0  10.0               ',
     +       '-maximum search radii')
      write(lun,42)
 42   format(' 0.0   0.0   0.0               ',
     +       '-angles for search ellipsoid')
      write(lun,43)
 43   format('51    51    11                ',
     +       '-size of covariance lookup table')
      write(lun,44)
 44   format('0    2.5                       ',
     +       '-0=full IK, 1=median approx. (cutoff)')
      write(lun,45)
 45   format('0                              ',
     +       '-0=SK, 1=OK')
      write(lun,46)
 46   format('1    0.15                      ',
     +       '-One   nst, nugget effect')
      write(lun,47)
 47   format('1    0.85 0.0   0.0   0.0      ',
     +       '-      it,cc,ang1,ang2,ang3')
      write(lun,48)
 48   format('         10.0  10.0  10.0      ',
     +       '-      a_hmax, a_hmin, a_vert')
      write(lun,49)
 49   format('1    0.10                      ',
     +       '-Two   nst, nugget effect')
      write(lun,50)
 50   format('1    0.90 0.0   0.0   0.0      ',
     +       '-      it,cc,ang1,ang2,ang3')
      write(lun,51)
 51   format('         10.0  10.0  10.0      ',
     +       '-      a_hmax, a_hmin, a_vert')
      write(lun,52)
 52   format('1    0.10                      ',
     +       '-Three nst, nugget effect')
      write(lun,53)
 53   format('1    0.90 0.0   0.0   0.0      ',
     +       '-      it,cc,ang1,ang2,ang3')
      write(lun,54)
 54   format('         10.0  10.0  10.0      ',
     +       '-      a_hmax, a_hmin, a_vert')
      write(lun,55)
 55   format('1    0.10                      ',
     +       '-Four  nst, nugget effect')
      write(lun,56)
 56   format('1    0.90 0.0   0.0   0.0      ',
     +       '-      it,cc,ang1,ang2,ang3')
      write(lun,57)
 57   format('         10.0  10.0  10.0      ',
     +       '-      a_hmax, a_hmin, a_vert')
      write(lun,58)
 58   format('1    0.15                      ',
     +       '-Five  nst, nugget effect')
      write(lun,59)
 59   format('1    0.85 0.0   0.0   0.0      ',
     +       '-      it,cc,ang1,ang2,ang3')
      write(lun,60)
 60   format('         10.0  10.0  10.0      ',
     +       '-      a_hmax, a_hmin, a_vert')

      close(lun)
      return
      end
