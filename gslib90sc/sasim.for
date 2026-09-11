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
      module    geostat

      integer,allocatable :: ixl(:),iyl(:),izl(:),ndat(:),it(:),
     +          inst(:),iit(:,:)
      real,allocatable    :: secdat(:),pridat(:),wtdat(:),
     +          varavg(:,:),var(:,:,:),secvar(:,:,:),varnew(:),
     +          varmod(:),varact(:),vardiv(:),ivaract(:,:),
     +          ivarnew(:,:),ivarmod(:,:),icut(:),iprop(:),
     +          hzval(:),hqact(:),hqnew(:),histdat(:),histwt(:),
     +          pricut(:,:),seccut(:),refpdf(:,:),actpdf(:,:),
     +          trypdf(:,:),cc(:),aa(:),anis1(:),anis2(:),
     +          ang1(:),ang2(:),ang3(:),ic0(:),icc(:,:),
     +          iaa(:,:),iang1(:,:),iang2(:,:),iang3(:,:),
     +          ianis1(:,:),ianis2(:,:)
      real*8,allocatable  :: rotmat(:,:,:)   
      logical,allocatable ::   cond(:,:,:)

      end module
c
c
c
      program main
c-----------------------------------------------------------------------
c
c                    Annealing-Based Simulation
c                    **************************
c
c Simulated annealing-based simulation of a continuous variable with 
c four different objective functions: (1) histogram, (2) variogram,
c (3) indicator variogram, (4) correlation coefficient, and (4)
c conditional distributions.  Any combination may be considered in a
c total objective function -- although the relative weighting is
c determined automatically, it is often necessary to adjust the weights
c to arrive at solutions where all components go to zero.
c
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use       geostat
      include  'sasim.inc'
c
c Read the Parameter File and the Data:
c
      call readparm
c
c Establish the number of lags to keep
c
      write(*,*) 'Determining the lag vectors to use'
      call getlag
c
c Loop over all the simulations:
c
      write(*,*) 'Looping over nsim realizations'
      do isim=1,nsim
c
c Initialize an image and the statistics:
c
            write(*,*) 'Initializing model'
            call initmod
c
c Call sasim for the simulation:
c
            write(*,*) 'Building realization',isim
            call sasim(isim)
c
c Write the results to the output file:
c
            write(*,*) 'Writing realization to output file',isim
            do k=1,nz
                  do j=1,ny
                        do i=1,nx
                              if(ilog.eq.1) then
                                    write(lout,201) 10**var(i,j,k)
                              else
                                    write(lout,201) var(i,j,k)
                              end if
 201                          format(g14.8)
                        end do
                  end do
            end do
c
c Write the vertically averaged output (if used):
c
            if(vertavg.and.(testcorr.or.testcpdf)) then
                  open(21,file='vertavg.out',status='UNKNOWN')
                  write(21,202)
 202              format('Vert Avg Data',/,'2',/,'sec',/,'vert avg')
                  do j=1,ny
                        do i=1,nx
                              vavg = varavg(i,j) / real(nz)
                              write(21,203) secvar(i,j,1),vavg
 203                          format(f10.3,1x,f10.4)
                        end do
                  end do
                  close(21)
            end if
c
c End loop over all simulations:
c
      end do
c
c Finished:
c
      close(lout)
      close(ldbg)
      write(*,9998) VERSION
 9998 format(/' SASIM Version: ',f5.3, ' Finished'/)
      stop
      end



      subroutine readparm
c-----------------------------------------------------------------------
c
c                  Initialization and Read Parameters
c                  **********************************
c
c The input parameters are read from a file name provided from standard
c input (a default name will be tried if none is keyed in by the user).
c
c Conditioning data is then read in (if available) and assigned to the
c nearest node if within the grid network.
c
c Error checking is performed and the statistics of both the initial
c realization and conditioning data are written to the debugging file.
c
c
c
c-----------------------------------------------------------------------
      use       msflib
      use       geostat
      include  'sasim.inc'
      parameter(MV=20,MAXCEN=20)
      real      val(MV)
      real*8    acorni
      logical   testfl
      character datafl*512,secfl*512,dbgfl*512,condfl*512,histfl*512,
     +          str*512
c
c Unit numbers:
c
      lin  = 1
      lout = 2
      ldbg = 3
c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' SASIM Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'sasim.par           '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'sasim.par           ') then
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
 1    read(lin,'(a4)',end=97) str(1:4)
      if(str(1:4).ne.'STAR') go to 1
c
c Read Input Parameters:
c
      write(*,*) 'Starting to read parameters'
c
c The objective function components:
c
      testhist = .false.
      testvarg = .false.
      testivar = .false.
      testcorr = .false.
      testcpdf = .false.
      read(lin,*,err=97) (val(i),i=1,5)
      read(lin,*,err=97) (userfac(i),i=1,5)
      if(val(1).ge.0.5) testhist = .true.
      if(val(2).ge.0.5) testvarg = .true.
      if(val(3).ge.0.5) testivar = .true.
      if(val(4).ge.0.5) testcorr = .true.
      if(val(5).ge.0.5) testcpdf = .true.
      if(testhist) write(*,*) ' Considering a histogram'
      if(testhist) write(*,*) '   user scaling factor = ',userfac(1)
      if(testvarg) write(*,*) ' Considering a variogram'
      if(testvarg) write(*,*) '   user scaling factor = ',userfac(2)
      if(testivar) write(*,*) ' Considering indicator variogram(s)'
      if(testivar) write(*,*) '   user scaling factor = ',userfac(3)
      if(testcorr) write(*,*) ' Considering a correlation coefficient'
      if(testcorr) write(*,*) '   user scaling factor = ',userfac(4)
      if(testcpdf) write(*,*) ' Considering conditional distributions'
      if(testcpdf) write(*,*) '   user scaling factor = ',userfac(5)

      read(lin,*,err=97) ilog
      write(*,*) ' log transform flag: ',ilog

c
c Number of simulations...
c
      read(lin,*,err=97) nsim
      write(*,*) ' number of simulations: ',nsim

      read(lin,*,err=97) nx,xmn,xsiz
      write(*,*) ' nx, xmn, xsiz: ',nx,xmn,xsiz
      read(lin,*,err=97) ny,ymn,ysiz
      write(*,*) ' ny, ymn, ysiz: ',ny,ymn,ysiz
      read(lin,*,err=97) nz,zmn,zsiz
      write(*,*) ' nz, zmn, zsiz: ',nz,zmn,zsiz

      read(lin,*,err=97) ixv(1)
      write(*,*) ' random number seed: ',ixv(1)
      do i=1,10000
           randnu = real(acorni(idum))
      end do
c
c Debugging level and output information
c
      read(lin,*,err=97) idbg
      write(*,*) ' debug level: ',idbg

      read(lin,'(a512)',err=97) dbgfl
      call chknam(dbgfl,512)
      write(*,*) ' debugging file: ',dbgfl(1:40)

      read(lin,'(a512)',err=97) outfl
      call chknam(outfl,512)
      write(*,*) ' output file: ',outfl(1:40)
c
c Annealing schedule...
c
      read(lin,*,err=97) isas
      write(*,*) ' automatic schedule: ',isas
      read(lin,*,err=97) t0,redfac,kasas,ksas,num,omin
      write(*,*) ' user set schedule: ',t0,redfac,kasas,ksas,num,omin
      omin2  = 0.0
      reltol = 1.0
      read(lin,*,err=97) maxswap,rreport
      write(*,*) ' maximum number of perturbations: ',maxswap
      write(*,*) ' reporting interval: ',rreport
      read(lin,*,err=97) maxnochange
      write(*,*) ' maximum without a change: ',maxnochange
      nxyz = nx*ny*nz
      write(*,101) t0,redfac,kasas*nxyz,ksas*nxyz,maxswap*nxyz
 101  format(/,' You have chosen an initial temperature of ',f6.4,',',/,
     +         ' reducing by',f7.4,' each time ',i6,' successful',/,
     +         ' perturbations or ',i6,' attempted perturbations.',/,
     +         ' The maximum number of perturbations ',i8)

c
c Conditioning Data:
c
      read(lin,*,err=97) icond
      write(*,*) ' consider conditioning: ',icond

      read(lin,'(a512)',err=97) condfl
      call chknam(condfl,512)
      write(*,*) ' conditioning data file: ',condfl(1:40)

      read(lin,*,err=97) ixloc,iyloc,izloc,ivrl
      write(*,*) ' columns: ',ixloc,iyloc,izloc,ivrl
      read(lin,*,err=97) tmin,tmax
      write(*,*) ' trimming limits: ',tmin,tmax

c
c Histogram Data:
c
      read(lin,*,err=97) ihist
      write(*,*) ' consider file with distribution: ',ihist

      read(lin,'(a512)',err=97) histfl
      call chknam(histfl,512)
      write(*,*) ' histogram data file: ',histfl(1:40)

      read(lin,*,err=97) ihvr,ihwt
      write(*,*) ' columns: ',ihvr,ihwt

      read(lin,*,err=97) nhist
      write(*,*) ' number for objective function: ',nhist
      hfactor = real(nx*ny*nz) / real(nhist+1)
c
c Indicator Cutoffs:
c
      read(lin,*,err=97) nicut
      write(*,*) ' number of indicator cutoffs: ',nicut
c
c Find the parameters and allocate the needed memory:
c
      MAXCUT = nicut
c19
      allocate(icut(MAXCUT),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 19: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c20
      allocate(iprop(MAXCUT),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 20: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c21
      allocate(rotmat(MAXROT,3,3),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 21: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c30
      allocate(it(MAXNST),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 30: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c31
      allocate(cc(MAXNST),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 31: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c32
      allocate(aa(MAXNST),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 32: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c33
      allocate(ang1(MAXNST),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 33: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c34
      allocate(ang2(MAXNST),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 34: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c35
      allocate(ang3(MAXNST),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 35: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c36
      allocate(anis1(MAXNST),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 36: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c37
      allocate(anis2(MAXNST),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 37: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c38
      allocate(inst(MAXCUT),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 38: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c39
      allocate(iit(MAXCUT,MAXNST),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 39: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c40
      allocate(ic0(MAXCUT),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 40: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c41
      allocate(icc(MAXCUT,MAXNST),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 41: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c42
      allocate(iaa(MAXCUT,MAXNST),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 42: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c43
      allocate(iang1(MAXCUT,MAXNST),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 43: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c44
      allocate(iang2(MAXCUT,MAXNST),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 44: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c45
      allocate(iang3(MAXCUT,MAXNST),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 45: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c46
      allocate(ianis1(MAXCUT,MAXNST),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 46: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c47
      allocate(ianis2(MAXCUT,MAXNST),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 47: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c
      read(lin,*,err=97) (icut(i),i=1,nicut)
      write(*,*) ' cutoffs: ',(icut(i),i=1,nicut)
      if(ilog.eq.1) then
            do i=1,nicut
                  icut(i) = alog10(max(icut(i),EPSLON))
            end do
      end if
c
c Secondary Data File
c
      read(lin,'(a512)',err=97) secfl
      call chknam(secfl,512)
      write(*,*) ' secondary data file: ',secfl(1:40)
      read(lin,*,err=97) icsecmod
      write(*,*) ' column number = ',icsecmod
      read(lin,*,err=97) i
      write(*,*) ' applies to a vertical average ? ',i
      vertavg = .false.
      if(i.eq.1) vertavg = .true.
c
c Conditional Distributions:
c
      read(lin,*,err=97) corr
      write(*,*) ' correlation: ',corr
      read(lin,'(a512)',err=97) datafl
      call chknam(datafl,512)
      write(*,*) ' data file with paired data. ',datafl(1:40)
      read(lin,*,err=97) icpri,icsec,icwt
      write(*,*) ' columns for prim, sec ',icpri,icsec,icwt
      read(lin,*,err=97) zmin,zmax
      write(*,*) ' minimum and maximum: ',zmin,zmax
      if(ilog.eq.1) then
            zmin = alog10(max(zmin,EPSLON))
            zmax = alog10(max(zmax,EPSLON))
      end if
      read(lin,*,err=97) npricut
      write(*,*) ' number of primary cutoffs ',npricut
      read(lin,*,err=97) nseccut
      write(*,*) ' number of secondary cutoffs ',nseccut
c
c Variogram Information....
c
      read(lin,*,err=97) nlag
      write(*,*) ' nlag: ',nlag
      read(lin,*,err=97) isill
      read(lin,*,err=97) nst(1),c0(1)
      sill = c0(1)
      write(*,100) isill,nst(1),c0(1)
      if(nst(1).le.0) then
            write(*,9997) nst(1)
 9997       format(' nst must be at least 1, it has been set to ',i4,/,
     +             ' The c or a values can be set to zero')
            stop
      endif
      if(nst(1).gt.MAXNST) stop 'nst is too big '
      do i=1,nst(1)
            read(lin,*,err=97) it(i),cc(i),ang1(i),ang2(i),ang3(i)
            read(lin,*,err=97) aa(i),aa1,aa2
            anis1(i) = aa1 / max(aa(i),EPSLON)
            anis2(i) = aa2 / max(aa(i),EPSLON)
            write(*,*) ' it,cc,ang[1,2,3]; ',it(i),cc(i),
     +                   ang1(i),ang2(i),ang3(i)
            write(*,*) ' a1 a2 a3: ',aa(i),aa1,aa2
      end do

c
c Indicator variogram information
c
      if(testivar) then
         do ic=1,nicut
            read(lin,*,err=97) inst(ic),ic0(ic)
            if(inst(ic).gt.MAXNST) stop 'inst is too big '
            do i=1,inst(ic)
               read(lin,*,err=97) iit(ic,i),icc(ic,i),iang1(ic,i),
     +                            iang2(ic,i),iang3(ic,i)
               read(lin,*,err=97) iaa(ic,i),aa1,aa2
               ianis1(ic,i) = aa1 / max(iaa(ic,i),EPSLON)
               ianis2(ic,i) = aa2 / max(iaa(ic,i),EPSLON)
            end do
         end do
      end if
c
c Finished reading parameters:
c
      close(lin)
 100  format(/,' Reset sill: ',i2,/,
     +         '        number of structures = ',i3,/,
     +         '        nugget effect        = ',f8.4)
c
c Reset the annealing schedule if automatic timing is being used:
c
      if(isas.eq.0) then
            t0      = 1.0
            redfac  = 0.1
            kasas   = 75*nx*ny*nz
            ksas    =  8*nx*ny*nz
            num     =  3
            omin    = 0.00001
            maxswap = 100*nx*ny*nz
      else
            kasas   = kasas  *nx*ny*nz
            ksas    = ksas   *nx*ny*nz
            maxswap = maxswap*nx*ny*nz
      endif
      report = int(rreport*nx*ny*nz)
c
c Find the parameters and allocate the needed memory:
c
      MAXX = nx
      MAXY = ny
      MAXZ = nz
      MAXLAG = nlag
      MAXPRI = npricut
      MAXSEC = nseccut
c1
      allocate(ixl(MAXLAG),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 1: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c2
      allocate(iyl(MAXLAG),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 2: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c3
      allocate(izl(MAXLAG),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 3: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c4
      allocate(ndat(0:MAXSEC + 1),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 1: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c8
      allocate(varavg(MAXX,MAXY),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 8: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c9
      allocate(var(MAXX,MAXY,MAXZ),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 9: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c10
      allocate(secvar(MAXX,MAXY,MAXZ),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 10: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c11
      allocate(varnew(MAXLAG),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 11: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c12
      allocate(varmod(MAXLAG),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 12: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c13
      allocate(varact(MAXLAG),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 13: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c14
      allocate(vardiv(MAXLAG),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 14: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c15
      allocate(ivaract(MAXCUT,MAXLAG),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 15: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c16
      allocate(ivarnew(MAXCUT,MAXLAG),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 16: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c17
      allocate(ivarmod(MAXCUT,MAXLAG),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 17: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c25
      allocate(pricut(MAXSEC + 1,MAXPRI + 1),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 25: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c26
      allocate(seccut(MAXSEC + 1),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 25: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c27
      allocate(refpdf(MAXSEC + 1,MAXPRI + 1),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 27: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c28
      allocate(actpdf(MAXSEC + 1,MAXPRI + 1),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 28: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c29
      allocate(trypdf(MAXSEC + 1,MAXPRI + 1),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 29: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c48
      allocate(cond(MAXX,MAXY,MAXZ),stat=test)
            if(test.ne.0)then
                  write(*,*)'ERROR 48: Allocation failed due ',
     +                  'insufficient memory.'
                  stop
            end if
c
c Open the debugging and output files:
c
      open(ldbg,file=dbgfl,status='UNKNOWN')
      write(*,*)
      write(*,*) 'Opening debug file -- will write most messages there'
      write(*,*)
      open(lout,file=outfl,status='UNKNOWN')
      write(lout,105)
 105  format('SASIM Realizations')
      write(lout,106) 1,nx,ny,nz,xmn,ymn,zmn,xsiz,ysiz,zsiz,nsim
 106  format(i2,3(1x,i4),3(1x,g14.8),3(1x,g12.6),i4) 
      write(lout,107)
 107  format('value')
c
c Turn all conditioning flags to false and initialize no secondary
c  variable:
c
      do ix=1,nx
            do iy=1,ny
                  do iz=1,nz
                        cond(ix,iy,iz)   = .false.
                        secvar(ix,iy,iz) = tmin - EPSLON
                  end do
            end do
      end do
c
c Open the paired data file?
c
      if(testcpdf) then
            inquire(file=datafl,exist=testfl)
            if(.not.testfl) then
                  write(*,*) 'ERROR file ',datafl,' does not exist!'
                  write(*,*) '  and yet you have asked for conditional'
                  write(*,*) '  distributions'
                  stop
            endif
            open(lin,file=datafl,status='OLD')
            read(lin,*,err=98)
            read(lin,*,err=98) nvari
            do i=1,nvari
                  read(lin,*)
            end do
            MAXDAT = 0
 55         read(lin,*,end=77,err=98) (val(j),j=1,nvari)
            if(pri.lt.tmin.or.pri.ge.tmax) go to 55
            if(val(icsec).lt.tmin.or.val(icsec).ge.tmax)go to 55
            MAXDAT = MAXDAT + 1
            go to 55
 77         close(lin)
      else
            MAXDAT = 0
      end if
c
c
c Check to see if a file of conditioning data exists, if it does then
c read in the data:
c
      if(icond.eq.1) then
            open(lin,file=condfl,status='OLD')
            read(lin,*,err=99)
            read(lin,*,err=99) nvari
            do i=1,nvari
                  read(lin,*)
            end do
            nd = 0
 191        read(lin,*,end=111,err=99)(val(j),j=1,nvari)
            if(pritry.lt.tmin.or.pritry.ge.tmax)go to 191
            nd  = nd + 1
            go to 191
            MAXCDT = nd
 111        close(lin)
      else
            MAXCDT = 0
      end if
c
c Check to see if a file containing the histogram is needed:
c
      if(vertavg.or.ihist.eq.1) then
            inquire(file=histfl,exist=testfl)
            if(.not.testfl) then
                  write(*,*) 'ERROR file ',histfl,' does not exist!'
                  write(*,*) '  and yet you have asked for a histogram'
                  stop
            endif
            open(lin,file=histfl,status='OLD')
            read(lin,*,err=94)
            read(lin,*,err=94) nvari
            do i=1,nvari
                  read(lin,*,err=99)
            end do
            MAXHST = 0
 201        read(lin,*,end=211,err=94) (val(j),j=1,nvari)
            if(val(ihvr).lt.tmin.or.val(ihvr).ge.tmax) go to 201
            MAXHST = MAXHST + 1
            go to 201
 211        close(lin)
      else
            MAXHST = 0
      end if
c
c Find the amount of storage needed for memory allocation: 
c
      if(MAXHST.gt.MAXDAT)then
            MAXDAT = MAXHST
      end if
      if(MAXCDT.gt.MAXDAT)then
            MAXDAT = MAXCDT
      end if
c
c Allocate the needed memory:
c5
            allocate(secdat(MAXDAT),stat=test)
                  if(test.ne.0)then
                        write(*,*)'ERROR 5: Allocation failed due ',
     +                        'insufficient memory.'
                        stop
                  end if
c6
            allocate(pridat(MAXDAT),stat=test)
                  if(test.ne.0)then
                        write(*,*)'ERROR 6: Allocation failed due ',
     +                        'insufficient memory.'
                        stop
                  end if
c7
            allocate(wtdat(MAXDAT),stat=test)
                  if(test.ne.0)then
                        write(*,*)'ERROR 7: Allocation failed due ',
     +                        'insufficient memory.'
                        stop
                  end if
c18
            allocate(histdat(MAXDAT),stat=test)
                  if(test.ne.0)then
                        write(*,*)'ERROR 18: Allocation failed due ',
     +                        'insufficient memory.'
                        stop
                  end if
c22
            allocate(hzval(MAXDAT),stat=test)
                  if(test.ne.0)then
                        write(*,*)'ERROR 22: Allocation failed due ',
     +                        'insufficient memory.'
                        stop
                  end if
c23
            allocate(hqact(MAXDAT),stat=test)
                  if(test.ne.0)then
                        write(*,*)'ERROR 23: Allocation failed due ',
     +                        'insufficient memory.'
                        stop
                  end if
c24
            allocate(hqnew(MAXDAT),stat=test)
                  if(test.ne.0)then
                        write(*,*)'ERROR 24: Allocation failed due ',
     +                        'insufficient memory.'
                        stop
                  end if
c49
            allocate(histwt(MAXDAT),stat=test)
                  if(test.ne.0)then
                        write(*,*)'ERROR 49: Allocation failed due ',
     +                        'insufficient memory.'
                        stop
                  end if
c
c Open the paired data file?
c
      if(testcpdf) then
            inquire(file=datafl,exist=testfl)
            if(.not.testfl) then
                  write(*,*) 'ERROR file ',datafl,' does not exist!'
                  write(*,*) '  and yet you have asked for conditional'
                  write(*,*) '  distributions'
                  stop
            endif
            open(lin,file=datafl,status='OLD')
            read(lin,*,err=98)
            read(lin,*,err=98) nvari
            do i=1,nvari
                  read(lin,*,err=98)
            end do
c
c Loop over all of the data in the data file:
c
            nd  = 0
 5          read(lin,*,end=6,err=98) (val(j),j=1,nvari)
c
c Is this a legitimate data (pair)?
c
            if(pri.lt.tmin.or.pri.ge.tmax) go to 5
            if(ilog.eq.1) then
                  pri = alog10(max(val(icpri),EPSLON))
            else
                  pri =       (val(icpri))
            end if
            if(icwt.ge.1) then
                  wt = val(icwt)
            else  
                  wt = 1.0
            end if
            sec = val(icsec)
            if(sec.lt.tmin.or.sec.ge.tmax)  go to 5
            nd = nd + 1
 7          continue
c
c Put this data pair in storage:
c
            pridat(nd) = pri + 0.00001*real(acorni(idum))
            secdat(nd) = sec + 0.00001*real(acorni(idum))
            wtdat(nd)  = wt
            go to 5
 6          close(lin)
            ndata = nd
c
c Sort the data by secondary data:
c
            call sortem(1,nd,secdat,2,pridat,wtdat,d,e,f,g,h)
c
c Work out the secondary data cutoffs:
c
            ndat(0) = 0
            do isec=1,nseccut+1
                  itar = (real(isec) / (1.0 + real(nseccut)))*nd
                  seccut(isec) = secdat(itar)
                  ndat(isec)   = itar
                  istart       = ndat(isec-1)+1
                  iend         = ndat(isec)
                  nn           = iend - istart
                  write(ldbg,103) isec,seccut(isec),nn
 103              format(' Secondary threshold ',i3,' = ',f10.4,
     +                   ' number = ',i8)
c
c Work out the primary data cutoffs:
c
                  call sortem(istart,iend,pridat,0,b,c,d,e,f,g,h)
                  do ipri=1,npricut
                        itar = istart + ( real(ipri) / 
     +                             (1.0 + real(npricut)) )*nn
                        pricut(isec,ipri) = pridat(itar)
                        write(ldbg,104) ipri,pricut(isec,ipri)
 104                    format('    primary cutoff ',i3,' = ',f10.4)
                  end do
            end do
c
c Finished with the paired data input:
c
      end if
c
c Check to see if a file of conditioning data exists, if it does then
c read in the data:
c
      if(icond.eq.1) then
            open(lin,file=condfl,status='OLD')
            read(lin,*,err=99)
            read(lin,*,err=99) nvari
            nd = 0
            av = 0.0
            ss = 0.0
            do i=1,nvari
                  read(lin,*,err=99)
            end do
c
c Read all the data until the end of the file:
c
 10         read(lin,*,end=11,err=99) (val(j),j=1,nvari)
            if(ilog.eq.1) then
                  pritry = alog10(max(val(ivrl),EPSLON))
            else
                  pritry = (val(ivrl))
            end if
            if(pritry.lt.tmin.or.pritry.ge.tmax) go to 10
            nd  = nd + 1
            av  = av + pritry
            ss  = ss + pritry*pritry
            ix=min0(max0((int((val(ixloc)-xmn)/xsiz+0.5)+1),1),nx)
            iy=min0(max0((int((val(iyloc)-ymn)/ysiz+0.5)+1),1),ny)
            iz=min0(max0((int((val(izloc)-zmn)/zsiz+0.5)+1),1),nz)
            var(ix,iy,iz)  = pritry
            cond(ix,iy,iz) = .true.
c
c Keep data just in case we need a distribution:
c
            if(nd.le.MAXDAT) then
                  ndhist      = nd
                  histdat(nd) = pritry
                  histwt(nd)  = 1.0
            end if
            go to 10
 11         close(lin)
c
c Compute the averages and variances as an error check for the user:
c
            av = av / max(real(nd),1.0)
            ss =(ss / max(real(nd),1.0)) - av * av
            write(ldbg,*) 'Conditioning Data for SASIM ',ivrl
            write(ldbg,*) '  Number of acceptable data  = ',nd
            write(ldbg,*) '  Equal Weighted Average     = ',av
            write(ldbg,*) '  Equal Weighted Variance    = ',ss
c
c Turn the conditioning data into a sorted CDF:
c
            call sortem(1,ndhist,histdat,1,histwt,c,d,e,f,g,h)
            do i=1,ndhist
                  histwt(i) = ((2.*real(i))-1.)/(2.*real(i))
            end do
      endif
c
c Check to see if a file containing the histogram is needed:
c
      if(vertavg.or.ihist.eq.1) then
            inquire(file=histfl,exist=testfl)
            if(.not.testfl) then
                  write(*,*) 'ERROR file ',histfl,' does not exist!'
                  write(*,*) '  and yet you have asked for a histogram'
                  stop
            endif
            open(lin,file=histfl,status='OLD')
            read(lin,*,err=94)
            read(lin,*,err=94) nvari
            do i=1,nvari
                  read(lin,*,err=99)
            end do
c
c Read all the data until the end of the file:
c
            ndhist  = 0
            av      = 0.0
            ss      = 0.0
            swt     = 0.0
 20         read(lin,*,end=21,err=94) (val(j),j=1,nvari)
            if(val(ihvr).lt.tmin.or.val(ihvr).ge.tmax) go to 20
            ndhist = ndhist + 1
            if(ndhist.gt.MAXDAT) stop 'Too many data in histogram'
            if(ilog.eq.1) then
                  histdat(ndhist) = alog10(max(val(ihvr),EPSLON))
            else
                  histdat(ndhist) = val(ihvr)
            end if
            if(ihwt.gt.0) then
                  histwt(ndhist) = val(ihwt)
            else
                  histwt(ndhist) = 1.0
            end if
            swt = swt + histwt(ndhist)
            av  = av  + histdat(ndhist)*histwt(ndhist)
            ss  = ss  + histdat(ndhist)*histdat(ndhist)*histwt(ndhist)
            go to 20
 21         close(lin)
c
c Compute the averages and variances as an error check for the user:
c
            av = av / max(swt,EPSLON)
            ss =(ss / max(swt,EPSLON)) - av * av
            write(ldbg,*)
            write(ldbg,*) 'Histogram Data: Variable number ',ihvr
            write(ldbg,*) '  Number   = ',ndhist
            write(ldbg,*) '  Average  = ',av
            write(ldbg,*) '  Variance = ',ss
            if(swt.lt.EPSLON) stop ' too few data here '
c
c Turn this data histogram into a sorted CDF:
c
            call sortem(1,ndhist,histdat,1,histwt,c,d,e,f,g,h)
            oldcp = 0.0
            cp    = 0.0
            swt   = 1.0 / swt
            do i=1,ndhist
                  cp        = cp + histwt(i) * swt
                  histwt(i) =(cp + oldcp) * 0.5
                  oldcp     = cp
            end do
c
c Set the "Z" quantiles:
c
            pinc = 1.0 / real(nhist+1)
            pval = 0.0
            do i=1,nhist
                  pval = pval + pinc
                  call locate(histwt,ndhist,1,ndhist,pval,idat)
                  hzval(i) = powint(histwt(idat),histwt(idat+1),
     +                           histdat(idat),histdat(idat+1),pval,1.)
            end do
      endif
c
c Read the 3-D model of the secondary variable:
c
      if(testcorr.or.testcpdf) then
            inquire(file=secfl,exist=testfl)
            if(.not.testfl) then
                  write(*,*) 'ERROR file ',secfl,' does not exist!'
                  write(*,*) '  and yet you have asked for correlation'
                  write(*,*) '  with a secondary variable or for'
                  write(*,*) '  conditional distributions'
                  stop
            endif
            open(lin,file=secfl,status='UNKNOWN')
            read(lin,'()')
            read(lin,*) nvari
            do i=1,nvari
                  read(lin,'()')
            end do
            num = 0
            nnz = nz
            if(vertavg) nnz = 1
            do k=1,nnz
                  do j=1,ny
                        do i=1,nx
                              read(lin,*) (val(l),l=1,nvari)
                              secvar(i,j,k) = val(icsecmod)
                              if(secvar(i,j,k).ge.tmin.and.
     +                           secvar(i,j,k).lt.tmax) num = num + 1
                              if(secvar(i,j,k).ge.tmax)
     +                           secvar(i,j,k) =  tmin - 0.00001
                        end do
                  end do
            end do
            write(ldbg,*) ' Number of Secondary data    = ',num
            close(lin)
c
c Get the mean and variance:
c
            xmen = 0.0
            xnum = 0.0
            do k=1,nnz
                  do j=1,ny
                        do i=1,nx
                              if(secvar(i,j,k).ge.tmin) then
                                    xmen = xmen + secvar(i,j,k)
                                    xnum = xnum + 1.0
                              end if
                        end do
                  end do
            end do
            xmen = xmen / xnum
            xvar = 0.0
            do k=1,nnz
                  do j=1,ny
                        do i=1,nx
                              if(secvar(i,j,k).ge.tmin) xvar = xvar
     +                        + (secvar(i,j,k)-xmen) * 
     +                          (secvar(i,j,k)-xmen)
                        end do
                  end do
            end do
            xvar = sqrt(max((xvar/xnum),0.0))
            write(ldbg,*) ' Number of Secondary data    = ',xnum
            write(ldbg,*) ' Secondary variable mean     = ',xmen
            write(ldbg,*) ' Secondary variable std.dev. = ',xvar
      end if
c
c All finished here:
c
      return
c
c Error in an Input File Somewhere:
c
 94   stop 'ERROR in histogram file!'
 97   stop 'ERROR in parameter file!'
 98   stop 'ERROR in distribution file!'
 99   stop 'ERROR in data file!'
      end



      subroutine sasim(isim)
c-----------------------------------------------------------------------
c
c Main subroutine that does the simulated annealing procedure, i.e.,
c establish the weights for each component objective function, and
c systematically perturbs the grid network to arrive at a low objective
c function realization.
c
c
c
c-----------------------------------------------------------------------
      use      geostat
      parameter(MAXPERT=1000,MAXSTOP=10)
      include 'sasim.inc'
      real*8   acorni
      real     ostop(MAXSTOP)
      real*8   delobj(MAXOBJ),objhist,objvarg,objivar,objcorr,objcpdf,
     +         gobj,gobjt,    obthist,obtvarg,obtivar,obtcorr,obtcpdf
      logical  accept
c
c Should we return already?
c
      if(maxswap.le.0) return
c
c Randomize model if conditional distributions in objective function:
c
      if(testcorr.or.testcpdf) then
            do i=1,nx*ny*nz
 31               ix = int(real(acorni(idum))*nx)+1
                  iy = int(real(acorni(idum))*ny)+1
                  iz = int(real(acorni(idum))*nz)+1
                  if(cond(ix,iy,iz).or.var(ix,iy,iz).le.tmin) go to 31
 32               jx = int(real(acorni(idum))*nx)+1
                  jy = int(real(acorni(idum))*ny)+1
                  jz = int(real(acorni(idum))*nz)+1
                  if(cond(jx,jy,jz).or.var(jx,jy,jz).le.tmin) go to 32
                  vartemp       = var(ix,iy,iz)
                  var(ix,iy,iz) = var(jx,jy,jz)
                  var(jx,jy,jz) = vartemp
            end do
      end if
c
c Initialize ostop() - used as a stopping criterion:
c
      do i=1,MAXSTOP
             ostop(i) = 0.0
      end do
      osfac = 1000.0/real(MAXSTOP*report)
c
c Determine vertical average of primary variable (if needed):
c
      if(vertavg) then
            write(*,*) 'Calculating Vertical Averages'
            do iy=1,ny
            do ix=1,nx
                  varavg(ix,iy) = 0.0
                  do iz=1,nz
                        varavg(ix,iy) = varavg(ix,iy) + var(ix,iy,iz)
                  end do
            end do
            end do
      end if
c
c Initialize the starting objective function components:
c
      objhist = 0.0
      objvarg = 0.0
      objivar = 0.0
      objcorr = 0.0
      objcpdf = 0.0
      write(*,*) 'Initializing objective functions'

      if(testhist) call inithist(objhist)
      if(testvarg) call initvarg(objvarg)
      if(testivar) call initivar(objivar)
      if(testcorr) call initcorr(objcorr)
      if(testcpdf) call initcpdf(objcpdf)

      write(ldbg,101) objhist,objvarg,objivar,objcorr,objcpdf
 101  format(/,'Initial Objective Function Values',/,
     +         '   Histogram             : ',f14.6,/,
     +         '   Variogram             : ',f14.6,/,
     +         '   Indicator Variograms  : ',f14.6,/,
     +         '   Correlation Coeficient: ',f14.6,/,
     +         '   Conditional Dists.    : ',f14.6,/)
c
c Establish weights:
c
      k1 = 1
      k2 = 1
      do i=1,MAXOBJ
            delobj(i) = 0.0
            objscl(i) = 1.0
      end do
      accept  = .false.
      obthist = 0.0
      obtvarg = 0.0
      obtivar = 0.0
      obtcorr = 0.0
      obtcpdf = 0.0
c
c Loop over a large number of random perturbations and keep track of
c how much each objective function component changes:
c
      write(*,*) 'Establishing scaling for objective function comps'
      do i=1,MAXPERT
 300        ix = int(real(acorni(idum))*nx)+1
            iy = int(real(acorni(idum))*ny)+1
            iz = int(real(acorni(idum))*nz)+1
            if(cond(ix,iy,iz).or.var(ix,iy,iz).le.tmin) go to 300
c
c Find a value to try:
c
            call drawcond(secvar(ix,iy,iz),vartry,prilow,prihig)
c
c Update the appropriate component objective functions:
c
            if(testhist) then
                  call updthist(ix,iy,iz,vartry,accept,obthist)
                  if(obthist.le.0.0) obthist = objhist
                  delobj(1) = delobj(1) + abs(objhist-obthist)
            end if
            if(testvarg) then
                  call updtvarg(ix,iy,iz,vartry,accept,obtvarg)
                  if(obtvarg.le.0.0) obtvarg = objvarg
                  delobj(2) = delobj(2) + abs(objvarg-obtvarg)
            end if
            if(testivar) then 
                  call updtivar(ix,iy,iz,vartry,accept,obtivar)
                  if(obtivar.le.0.0) obtivar = objivar
                  delobj(3) = delobj(3) + abs(objivar-obtivar)
            end if
            if(testcorr) then 
                  call updtcorr(ix,iy,iz,vartry,accept,obtcorr)
                  if(obtcorr.le.0.0) obtcorr = objcorr
                  delobj(4) = delobj(4) + abs(objcorr-obtcorr)
            end if
            if(testcpdf) then 
                  call updtcpdf(ix,iy,iz,vartry,accept,obtcpdf)
                  if(obtcpdf.le.0.0) obtcpdf = objcpdf
                  delobj(5) = delobj(5) + abs(objcpdf-obtcpdf)
            end if
      end do
c
c Establish each component scaling and
c
      resc = 0.0
      if(delobj(1).lt.1.0e-10) delobj(1) = 1.0e-10
      if(delobj(2).lt.1.0e-10) delobj(2) = 1.0e-10
      if(delobj(3).lt.1.0e-10) delobj(3) = 1.0e-10
      if(delobj(4).lt.1.0e-10) delobj(4) = 1.0e-10
      if(delobj(5).lt.1.0e-10) delobj(5) = 1.0e-10
      if(testhist) objscl(1) = MAXPERT / delobj(1)
      if(testvarg) objscl(2) = MAXPERT / delobj(2)
      if(testivar) objscl(3) = MAXPERT / delobj(3)
      if(testcorr) objscl(4) = MAXPERT / delobj(4)
      if(testcpdf) objscl(5) = MAXPERT / delobj(5)
      write(ldbg,102) (objscl(i),i=1,5)
 102  format(/,'Scaling of Objective Function Values',/,
     +         '   Histogram             : ',f14.6,/,
     +         '   Variogram             : ',f14.6,/,
     +         '   Indicator Variograms  : ',f14.6,/,
     +         '   Correlation Coeficient: ',f14.6,/,
     +         '   Conditional Dists.    : ',f14.6,/)
c
c Special user scaling of components:
c
      objscl(1) = userfac(1) * objscl(1)
      objscl(2) = userfac(2) * objscl(2)
      objscl(3) = userfac(3) * objscl(3)
      objscl(4) = userfac(4) * objscl(4)
      objscl(5) = userfac(5) * objscl(5)
c
c Continue with scaling:
c
      resc = resc + objscl(1)*objhist + objscl(2)*objvarg
     +            + objscl(3)*objivar + objscl(4)*objcorr
     +            + objscl(5)*objcpdf
      resc = 1.0 / max(resc,EPSLON)
      if(testhist) objscl(1) = objscl(1) * resc
      if(testvarg) objscl(2) = objscl(2) * resc
      if(testivar) objscl(3) = objscl(3) * resc
      if(testcorr) objscl(4) = objscl(4) * resc
      if(testcpdf) objscl(5) = objscl(5) * resc
c
c Scale initial values:
c
      objhist = objhist*objscl(1)
      objvarg = objvarg*objscl(2)
      objivar = objivar*objscl(3)
      objcorr = objcorr*objscl(4)
      objcpdf = objcpdf*objscl(5)
c
c Initial Conditions:
c
      gobj  = objhist + objvarg + objivar + objcorr + objcpdf
      nswap = 0
      iend  = 0
      temp  = t0    
c
c Loop until convergence or the stopping number:
c
      nnochange = 0
      write(*,*) 'Starting the simulation'
 1    naccept = 0
      ntry    = 0
      if(idbg.gt.2) then
            write(*,996)    nswap,temp,isim
            write(*,998)    nswap,objhist,objvarg,objivar,
     +                            objcorr,objcpdf,gobj
            write(ldbg,997) nswap,objhist,objvarg,objivar,
     +                            objcorr,objcpdf
 996        format(' Changes: ',i12,' Temp:',f13.10,' (#',i3,')')
 997        format('0 ',i12,10(1x,f8.6))
      endif
c
c Keep attempting to swap values until some limit is exceeded:
c
 2    ntry      = ntry  + 1
      nswap     = nswap + 1
      nnochange = nnochange + 1
      if(idbg.gt.2) then
            if((int(nswap/report)*report).eq.nswap) then
c
c Establish the min and max in the last MAXSTOP reports:
c
                  do i=MAXSTOP,2,-1
                        ostop(i) = ostop(i-1)
                  end do
                  osmin = 99.0
                  osmax =  0.0
                  ostop(1) = real(gobj)
                  do i=1,MAXSTOP
                        if(ostop(i).lt.osmin) osmin = ostop(i)
                        if(ostop(i).gt.osmax) osmax = ostop(i)
                  end do

                  write(*,998)    nswap,objhist,objvarg,objivar,
     +                                  objcorr,objcpdf,gobj
                  write(ldbg,999) nswap,objhist,objvarg,objivar,
     +                                  objcorr,objcpdf
            endif
 998       format(' Changes: ',i12,5(1x,f8.6),' : ',f8.6)
 999       format('1 ',i12,10(1x,f8.6))
      endif
c
c Pick a non-conditioning value to perturb:
c
 3    ix = int(real(acorni(idum))*nx)+1
      iy = int(real(acorni(idum))*ny)+1
      iz = int(real(acorni(idum))*nz)+1
      if(cond(ix,iy,iz).or.var(ix,iy,iz).le.tmin) go to 3
c
c Find a value to try:
c
      nattemp = 0
 4    nattemp = nattemp + 1
      call drawcond(secvar(ix,iy,iz),vartry,prilow,prihig)
c
c Calculate Objective Function:
c
      if(testhist) call updthist(ix,iy,iz,vartry,accept,obthist)
      if(testvarg) call updtvarg(ix,iy,iz,vartry,accept,obtvarg)
      if(testivar) call updtivar(ix,iy,iz,vartry,accept,obtivar)
      if(testcorr) call updtcorr(ix,iy,iz,vartry,accept,obtcorr)
      if(testcpdf) call updtcpdf(ix,iy,iz,vartry,accept,obtcpdf)
      if(obthist.lt.0.0) obthist = objhist
      if(obtvarg.lt.0.0) obtvarg = objvarg
      if(obtivar.lt.0.0) obtivar = objivar
      if(obtcorr.lt.0.0) obtcorr = objcorr
      if(obtcpdf.lt.0.0) obtcpdf = objcpdf
      gobjt = obthist + obtvarg + obtivar + obtcorr + obtcpdf
c
c Accept the swap if the objective has gone down and with a certain
c probability if the objective has gone up:
c
      accept = .false.
      if(gobjt.ge.gobj) then
            if(temp.gt.0.0) then
                  unif = max(EPSLON,real(acorni(idum)))
                  if(gobjt.lt.(gobj-dble(temp*alog(unif))))
     +            accept = .true.
            else
                  if(nattemp.le.9) go to 4
            end if
      else
            accept = .true.
      endif
c
c If we are keeping it then update certain variables:
c
      if(accept) then
            if(dabs(gobj-gobjt).gt.1.0e-21) nnochange = 0
            if(vertavg)
     +      varavg(ix,iy) = varavg(ix,iy) - var(ix,iy,iz) + vartry
            var(ix,iy,iz) = vartry
            naccept       = naccept + 1
            gobj          = gobjt
            objhist       = obthist
            objvarg       = obtvarg
            objivar       = obtivar
            objcorr       = obtcorr
            objcpdf       = obtcpdf
      endif
c
c CHECK the stopping criteria:
c
      if(real(gobj).gt.omin2) osmax = 9999.
      if(real(gobj).le.omin.or.iend.ge.num.or.nswap.eq.maxswap.or.
     +   (osmax-osmin)*osfac.lt.reltol.or.
     +   nnochange.gt.maxnochange) then
            if(gobj.le.omin)                  write(*,401)
            if(iend.ge.num)                   write(*,402)
            if(nswap.eq.maxswap)              write(*,403)
            if((osmax-osmin)*osfac.lt.reltol) write(*,404)
            if(nnochange.gt.maxnochange)      write(*,405)
            if(gobj.le.omin)                  write(ldbg,401)
            if(iend.ge.num)                   write(ldbg,402)
            if(nswap.eq.maxswap)              write(ldbg,403)
            if((osmax-osmin)*osfac.lt.reltol) write(ldbg,404)
            if(nnochange.gt.maxnochange)      write(ldbg,405)
 401        format(' Stopped because of obj lt omin')
 402        format(' Stopped because of iend gt num')
 403        format(' Stopped because of nswap gt maxswap')
 404        format(' Stopped because of O changing less than rate')
 405        format(' Stopped because of number of iterations without'
     +            ,' a change')

            if(testhist) call inithist(objhist)
            if(testvarg) call initvarg(objvarg)
            if(testivar) call initivar(objivar)
            if(testcorr) call initcorr(objcorr)
            if(testcpdf) call initcpdf(objcpdf)
            return
      endif
c
c Tried too many at this "temperature"?
c
      if(ntry.gt.kasas) then
            iend = iend + 1
            temp = redfac * temp
            go to 1
      endif
c
c Accepted enough at this "temperature"?
c
      if(naccept.ge.ksas) then
            temp = redfac * temp
            iend = 0
            go to 1
      endif
c
c Go back for another attempted swap:
c
      go to 2
      end



      subroutine getlag
c-----------------------------------------------------------------------
c       Establish the number and location of the lags to consider
c       *********************************************************
c
c
c
c-----------------------------------------------------------------------
      use        geostat
      include   'sasim.inc'
      real       maxcov
c
c Compute maximum covariance:
c
      do is=1,nst(1)
            call setrot(ang1(is),ang2(is),ang3(is),anis1(is),anis2(is),
     +                  is,MAXROT,rotmat)
      end do
      call cova3(0.0,0.0,0.0,0.0,0.0,0.0,1,nst,MAXNST,c0,it,cc,aa,
     +           1,MAXROT,rotmat,cmax,maxcov)
c
c Initialize the variogram and lag arrays:
c
      do i=1,nlag
            varmod(i) = 1.0e+20
            ixl(i)    = 0
            iyl(i)    = 0
            izl(i)    = 0
      end do
c
c Calculate the Experimental Variogram:
c
      na  = 0
      nxl = nx/2
      nyl = ny/2
      nzl = nz/2
      do ix=   0,nxl
      do iy=-nyl,nyl
      do iz=-nzl,nzl
            if(ix.eq.0.and.iy.le.0.and.iz.le.0) go to 2
            dx = real(ix) * xsiz
            dy = real(iy) * ysiz
            dz = real(iz) * zsiz
            call cova3(0.0,0.0,0.0,dx,dy,dz,1,nst,MAXNST,c0,
     +                 it,cc,aa,1,MAXROT,rotmat,cmax,cova)
            vario = maxcov - cova
            if(na.eq.nlag.and.vario.gt.varmod(na)) go to 2
c
c Consider this sample (it will be added in the correct location):
c
            if(na.lt.nlag) na = na + 1
            ixl(na)    = ix
            iyl(na)    = iy
            izl(na)    = iz
            varmod(na) = vario
            if(na.eq.1) go to 2
c
c Sort samples found thus far in increasing order of distance:
c
            n1 = na-1
            do ii=1,n1
                  k=ii
                  if(vario.lt.varmod(ii)) then
                        jk = 0
                        do jj=k,n1
                              j  = n1-jk
                              jk = jk+1
                              j1 = j+1
                              varmod(j1) = varmod(j)
                              ixl(j1)    = ixl(j)
                              iyl(j1)    = iyl(j)
                              izl(j1)    = izl(j)
                        end do
                        varmod(k) = vario
                        ixl(k)    = ix
                        iyl(k)    = iy
                        izl(k)    = iz
                        go to 2
                  endif
            end do
 2    continue
      end do
      end do
      end do
c
c Debugging information:
c
      if(idbg.gt.3) then
            write(ldbg,100) nlag
 100        format(/'Closest ',i3,' lags:  Lag number  variogram   ',
     +              'offsets')
            do i=1,nlag
                  write(ldbg,101)i,varmod(i),ixl(i),iyl(i),izl(i)
 101              format(i2,1x,f12.4,3i3)
            end do
      end if
c
c Return with the closest lags:
c
      return
      end



      subroutine initmod
c-----------------------------------------------------------------------
c
c                       Initialization of Grid
c                       **********************
c
c
c-----------------------------------------------------------------------
      use       geostat
      include  'sasim.inc'
c
c Initialize all nodes to some random quantile:
c
      do k=1,nz
      do j=1,ny
      do i=1,nx
c
c Handle the case when we are considering a vertical average:
c
      kk = k
      if(vertavg) kk = 1
c
c Only initialize if not a conditioning datum:
c
      if(.not.cond(i,j,k)) then
         if(testcorr.or.testcpdf) then
            if(secvar(i,j,kk).le.tmin)then
                  var(i,j,k) = tmin - EPSLON
            else
                  call drawcond(secvar(i,j,kk),var(i,j,k),prilow,prihig)
            end if
         else
                  call drawcond(secvar(i,j,kk),var(i,j,k),prilow,prihig)
         end if
      endif
c
c End loop over initialization:
c
      end do
      end do
      end do
c
c Renormalize the variogram parameters to the variance of the
c realization if requested:
c
      if(isill.eq.1) then
c
c Get current sill of variogram:
c
            sill = c0(1)
            do i=1,nst(1)
                  sill = sill + cc(i)
            end do
c
c Get variance of realization:
c
            xmen = 0.0
            xnum = 0.0
            xmin = 1.0e20
            xmax =-1.0e20
            do k=1,nz
                  do j=1,ny
                        do i=1,nx
                           if(var(i,j,k).gt.tmin) then
                              xnum = xnum + 1.0
                              xmen = xmen + var(i,j,k)
                              if(var(i,j,k).lt.xmin) xmin = var(i,j,k)
                              if(var(i,j,k).gt.xmax) xmax = var(i,j,k)
                           end if
                        end do
                  end do
            end do
            xmen = xmen / xnum
            xvar = 0.0
            do k=1,nz
                  do j=1,ny
                        do i=1,nx
                           if(var(i,j,k).gt.tmin) then
                              xvar = xvar + (var(i,j,k)-xmen) * 
     +                                      (var(i,j,k)-xmen)
                           end if
                        end do
                  end do
            end do
            xvar = xvar/xnum
            xstd = sqrt(max(xvar,0.0))
            write(ldbg,101) int(xnum),xmen,xstd,xmin,xmax
 101        format(/,'Initial Model',/,
     +               '   Number    = ',i8,/,
     +               '   mean      = ',f14.6,/,
     +               '   std.dev.  = ',f14.6,/,
     +               '   minimum   = ',f14.6,/,
     +               '   maximum   = ',f14.6,/)
c
c Now, scale the variogram parameters:
c
            fac   = xvar/sill
            c0(1) = c0(1) * fac
            do i=1,nst(1)
                  cc(i)  = cc(i) * fac
            end do
      endif
c
c Get the indicator proportions if needed:
c
      if(testivar) then
            do ic=1,nicut
                  iprop(ic) = 0.0
            end do
            do iz=1,nz
                  do iy=1,ny
                        do ix=1,nx
                           if(var(ix,iy,iz).gt.tmin) then
                              do ic=1,nicut
                                    if(var(ix,iy,iz).le.icut(ic))
     +                              iprop(ic) = iprop(ic) + 1.0
                              end do
                           end if
                        end do
                  end do
            end do
            do ic=1,nicut
                  iprop(ic) = iprop(ic) / real(nx*ny*nz)
            end do
      end if
c
c Finished getting initial image:
c
      return
      end



      subroutine drawcond(secval,prival,prilow,prihig)
c-----------------------------------------------------------------------
c
c                 Draw from Conditional Distribution
c                 **********************************
c
c
c-----------------------------------------------------------------------
      use       geostat
      include  'sasim.inc'
      real*8    acorni
c
c If the value of secval is "missing":
c
      if(vertavg.or.secval.le.tmin.or.secval.gt.tmax) then
            prival = tmin
            prilow = tmin
            prihig = tmin
            if(ndhist.gt.0) then
                 randnu = real(acorni(idum))
                 call locate(histwt,ndhist,1,ndhist,randnu,idat)
                 randnu = real(acorni(idum))
                 idat = min(max(idat,1),(ndhist-1))
                 prival = powint(0.0,1.0,histdat(idat),histdat(idat+1),
     +                           randnu,1.0)
                 prilow = prival - EPSLON
                 prihig = prival + EPSLON
            end if
            return
      end if
c
c Determine which class:
c
      if(nseccut.gt.0) then
            seclow = -1.0e20
            do l=1,nseccut
                  if(secval.ge.seclow.and.secval.lt.seccut(l)) isec = l
                  seclow = seccut(l)
            end do
            if(secval.ge.seclow)  isec = nseccut+1
      else
            isec = 1
      end if
c
c Bounds of data:
c
      ilow = ndat(isec-1) + 1
      ihig = ndat(isec)
      prilow = pridat(ilow)
      prihig = pridat(ihig)
c
c Randomly select a data value between ilow and ihig-1:
c
      ntry   = 0
      randnu = real(acorni(idum))
      idat   = ilow + int( randnu*(ihig-ilow) )
c
c Draw uniformly between this data and the one above it:
c
      randnu = real(acorni(idum))
      prival = powint(0.0,1.0,pridat(idat),pridat(idat+1),randnu,1.0)
c
c Finished drawing from conditional distribution:
c
      return
      end



      subroutine inithist(obj)
c-----------------------------------------------------------------------
c              Compute the Initial Objective Function
c                           Histogram
c              **************************************
c
c
c
c
c-----------------------------------------------------------------------
      use        geostat
      include   'sasim.inc'
      real*8     obj
c
c Initialize fraction less than thresholds:
c
      do i=1,nhist
            hqact(i) = 0.0
      end do
c
c Compute the fraction less than each threshold:
c
      thnum = 0.0
      do iz=1,nz
            do iy=1,ny
                  do ix=1,nx
                     if(var(ix,iy,iz).ge.tmin) then
                        thnum  = thnum  + 1.0
                        do i=1,nhist
                           if(var(ix,iy,iz).le.hzval(i))
     +                        hqact(i) = hqact(i) + 1
                        end do
                     end if
                  end do
            end do
      end do
c
c Objective function:
c
      obj = 0.0
      write(ldbg,100)
 100  format(/'Histogram Reproduction at this time:')
      do i=1,nhist
            ptar = real(i)/real(nhist+1)
            pwgt = 0.5 - ptar*(1.0-ptar)
            obj = obj + dble(pwgt*abs(hqact(i)/thnum-ptar))
            write(ldbg,101) i,hzval(i),real(i)/real(nhist+1),
     +                      hqact(i)/thnum
 101        format(' Quantile ',i3,' z value ',f12.4,' target ',f7.4,
     +                                               ' actual ',f7.4)
      end do
c
c Return with the component objective function:
c
      return
      end
 
 
 
      subroutine updthist(ix,iy,iz,vartry,accept,objt)
c-----------------------------------------------------------------------
c
c                   Update Histogram for a Change
c                   *****************************
c
c
c
c-----------------------------------------------------------------------
      use        geostat
      include   'sasim.inc'
      logical    accept
      real*8     objt
c
c Set the correct histogram:
c
      varcur = var(ix,iy,iz)
      if(accept) then
            do i=1,nhist
                  hqact(i) = hqnew(i)
            end do
      else
            do i=1,nhist
                  hqnew(i) = hqact(i)
            end do
      end if
c
c Update:
c
      objt = 0.0
      do i=1,nhist
            if(varcur.le.hzval(i)) hqnew(i) = hqnew(i) - 1
            if(vartry.le.hzval(i)) hqnew(i) = hqnew(i) + 1
            ptar = real(i)/real(nhist+1)
            pwgt = 0.5 - ptar*(1.0-ptar)
            objt = objt + dble(pwgt*abs(hqnew(i)/thnum-ptar))
      end do
      objt = objt * objscl(1)
c
c Return with updated value:
c
      return
      end



      subroutine initvarg(obj)
c-----------------------------------------------------------------------
c              Compute the Initial Objective Function
c                       Variogram Component
c              **************************************
c
c The objective function is the squared difference from the model vario-
c gram and the experimental variogram. The lag separation distances
c and the number of lags that contribute to the objective function are
c computed outside of this subroutine.
c
c
c
c-----------------------------------------------------------------------
      use        geostat
      include   'sasim.inc'
      real       maxcov
      real*8     obj
c
c Initialize:
c
      do j=1,nlag
            vardiv(j) = 0.0
            varact(j) = 0.0
      end do
c
c Compute maximum covariance:
c
      do is=1,nst(1)
            call setrot(ang1(is),ang2(is),ang3(is),anis1(is),anis2(is),
     +                  is,MAXROT,rotmat)
      end do
      call cova3(0.0,0.0,0.0,0.0,0.0,0.0,1,nst,MAXNST,c0,it,cc,aa,
     +           1,MAXROT,rotmat,cmax,maxcov)
c
c Note that the variogram parameters have already been scaled by the
c variance in the getlag subroutine:
c
      do i=1,nlag
            dx = real(ixl(i)) * xsiz
            dy = real(iyl(i)) * ysiz
            dz = real(izl(i)) * zsiz
            call cova3(0.0,0.0,0.0,dx,dy,dz,1,nst,MAXNST,c0,it,cc,aa,
     +                 1,MAXROT,rotmat,cmax,cova)
            varmod(i) = maxcov - cova
      end do
c
c Loop over all of the grid nodes:
c
      do iz=1,nz
            do iy=1,ny
                  do ix=1,nx
                        v1 = var(ix,iy,iz)
                        if(v1.gt.tmin) then
                           do il=1,nlag
                              jx = ix + ixl(il)
                              jy = iy + iyl(il)
                              jz = iz + izl(il)
                              if(jx.gt.0.and.jx.le.nx.and.
     +                           jy.gt.0.and.jy.le.ny.and.
     +                           jz.gt.0.and.jz.le.nz) then
                                 if(var(jx,jy,jz).gt.tmin) then
                                    varact(il) = varact(il) +
     +                                 (v1-var(jx,jy,jz))*
     +                                 (v1-var(jx,jy,jz))
                                    vardiv(il) = vardiv(il) + 2.
                                 end if
                              end if
                           end do
                        end if
                  end do
            end do
      end do
c
c Compute the objective function:
c
      obj = 0.0
      write(ldbg,100)
 100  format(/'Variogram Reproduction at this time:')
      do il=1,nlag
            if(vardiv(il).le.0.0) then
                  write(*,*) 'ERROR: lag ',il,'there are no pairs!!'
                  stop
            endif
            act = varact(il)/vardiv(il)
            obj = obj + dble( (varmod(il)-act)*(varmod(il)-act) /
     +                             (varmod(il)*varmod(il)) )
            write(ldbg,101) il,varmod(il),act
 101        format('    Lag: ',i3,' model ',f12.4,' actual ',f12.4)
      end do
c
c Return with the component objective function:
c
      return
      end
 
 
 
      subroutine updtvarg(ix,iy,iz,vartry,accept,objt)
c-----------------------------------------------------------------------
c
c                     Update Variogram for a Change
c                     ***************************
c
c Update the Experimental Variogram and then compute the objective
c function as the squared difference between the actual and the model
c variogram.
c
c
c
c
c-----------------------------------------------------------------------
      use        geostat
      logical    accept
      include   'sasim.inc'
      real*8     objt
c
c Set the correct variogram:
c
      varcur = var(ix,iy,iz)
      if(accept) then
            do il=1,nlag
                  varact(il) = varnew(il)
            end do
      else
            do il=1,nlag
                  varnew(il) = varact(il)
            end do
      end if
      objt = -1.0
      if(vartry.le.tmin.or.varcur.le.tmin) return
c
c MAIN LOOP to consider the change to all lags:
c
      do il=1,nlag
c
c Update the variogram for the positive lag:
c
            jx = ix + ixl(il)
            jy = iy + iyl(il)
            jz = iz + izl(il)
            if(jx.ge.1.and.jx.le.nx.and.
     +         jy.ge.1.and.jy.le.ny.and.
     +         jz.ge.1.and.jz.le.nz) then
                  v0 = var(jx,jy,jz)
                  if(v0.gt.tmin) 
     +            varnew(il) = varnew(il) 
     +                       - (varcur-v0)*(varcur-v0)
     +                       + (vartry-v0)*(vartry-v0)
            end if
c
c Update the variogram for the negative lag:
c
            jx = ix - ixl(il)
            jy = iy - iyl(il)
            jz = iz - izl(il)
            if(jx.ge.1.and.jx.le.nx.and.
     +         jy.ge.1.and.jy.le.ny.and.
     +         jz.ge.1.and.jz.le.nz) then
                  v0 = var(jx,jy,jz)
                  if(v0.gt.tmin) 
     +            varnew(il) = varnew(il) 
     +                       - (varcur-v0)*(varcur-v0)
     +                       + (vartry-v0)*(vartry-v0)
            end if
      end do
c
c Compute the objective function and return:
c
      objt = 0.0
      do il=1,nlag
            act  = varnew(il)/vardiv(il)
            objt = objt + dble( (varmod(il)-act)*(varmod(il)-act) /
     +                               (varmod(il)*varmod(il)) )
      end do
      objt = objt * objscl(2)
c
c Return with tentative objective function:
c
      return
      end



      subroutine initivar(obj)
c-----------------------------------------------------------------------
c              Compute the Initial Objective Function
c                   Indicator Variogram Component
c              **************************************
c
c The objective function is the squared difference from the model vario-
c gram and the experimental variogram. The lag separation distances
c and the number of lags that contribute to the objective function are
c computed outside of this subroutine.
c
c
c
c-----------------------------------------------------------------------
      use      geostat
      include 'sasim.inc'
      integer  nsti(1),iti(MAXNST)
      real     c0i(1),cci(MAXNST),aai(MAXNST),ang1i(MAXNST),
     +         ang2i(MAXNST),ang3i(MAXNST),anis1i(MAXNST),
     +         anis2i(MAXNST),maxcov
      real*8   obj
c
c Initialize:
c
      do ic=1,nicut
            do il=1,nlag
                  vardiv(il)     = 0.0
                  ivaract(ic,il) = 0.0
            end do
      end do
c
c Get the model variogram:
c
      do ic=1,nicut
c
c Load the right model:
c
            nsti(1) = inst(ic)
            c0i(1)  = ic0(ic)
            sill    = c0i(1)
            do i=1,nsti(1)
                  iti(i)    = iit(ic,i)
                  aai(i)    = iaa(ic,i)
                  cci(i)    = icc(ic,i)
                  sill      = sill + cci(i)
                  ang1i(i)  = iang1(ic,i)
                  ang2i(i)  = iang2(ic,i)
                  ang3i(i)  = iang3(ic,i)
                  anis1i(i) = ianis1(ic,i)
                  anis2i(i) = ianis2(ic,i)
            end do
c
c Rescale sill parameters:
c
            fac    = (iprop(ic)*(1.0-iprop(ic))) / sill
            c0i(1) = c0i(1) * fac
            do i=1,nsti(1)
                  cci(i) = cci(i) * fac
            end do
c
c Compute maximum covariance:
c
            do is=1,nst(1)
                  call setrot(ang1i(is),ang2i(is),ang3i(is),anis1i(is),
     +                        anis2i(is),is,MAXROT,rotmat)
            end do
            call cova3(0.0,0.0,0.0,0.0,0.0,0.0,1,nsti,MAXNST,c0i,iti,
     +                 cci,aai,1,MAXROT,rotmat,cmax,maxcov)
c
c Note that the variogram parameters have already been scaled by the
c variance:
c
            do i=1,nlag
                  dx = real(ixl(i)) * xsiz
                  dy = real(iyl(i)) * ysiz
                  dz = real(izl(i)) * zsiz
                  call cova3(0.0,0.0,0.0,dx,dy,dz,1,nsti,MAXNST,c0i,
     +                       iti,cci,aai,1,MAXROT,rotmat,cmax,cova)
                  ivarmod(ic,i) = maxcov - cova
            end do
      end do
c
c Finished computing all of the model indicator variograms, now loop
c over all of the grid nodes computing the experimental indicator
c variograms:
c
      do ix=1,nx
         do iy=1,ny
            do iz=1,nz
c
c Consider the first value in the pair and all directions and lags:
c
            v1 = var(ix,iy,iz)
            if(v1.gt.tmin) then
               do il=1,nlag
                  jx = ix + ixl(il)
                  jy = iy + iyl(il)
                  jz = iz + izl(il)
                  if(jx.gt.0.and.jx.le.nx.and.
     +               jy.gt.0.and.jy.le.ny.and.
     +               jz.gt.0.and.jz.le.nz) then
                        v2 = var(jx,jy,jz)
                        vardiv(il) = vardiv(il) + 2.0
                        do ic=1,nicut
                           if(v1.le.icut(ic).and.v2.gt.icut(ic))
     +                     ivaract(ic,il) = ivaract(ic,il) + 1.0
                           if(v1.gt.icut(ic).and.v2.le.icut(ic))
     +                     ivaract(ic,il) = ivaract(ic,il) + 1.0
                        end do
                  end if
               end do
            end if
c
c End looping over all seed points:
c
            end do
         end do
      end do
c
c Compute the experimental variogram:
c
      obj = 0.0
      write(ldbg,100)
 100  format(/'Indicator Variogram Reproduction at this time:')
      do il=1,nlag
            if(vardiv(il).le.0.0) then
                  write(*,*) 'ERROR: lag ',il,'there are no pairs!!'
                  stop
            endif
            do ic=1,nicut
                  act = ivaract(ic,il)/vardiv(il)
                  obj = obj + (ivarmod(ic,il)-act)*(ivarmod(ic,il)-act)
                  write(ldbg,101) il,ic,ivarmod(ic,il),act
 101              format('    Lag ',i3,' threshold ',i3,' model ',f12.4,
     +                                                 ' actual ',f12.4)
            end do
      end do
c
c Return with the component objective function:
c
      return
      end
 
 
 
      subroutine updtivar(ix,iy,iz,vartry,accept,objt)
c-----------------------------------------------------------------------
c
c                Update Indicator Variograms for a Change
c                **************************************
c
c Update the Experimental Variogram and then compute the objective
c function as the squared difference between the actual and the model
c variogram.
c
c
c
c
c-----------------------------------------------------------------------
      use        geostat
      logical    accept
      include   'sasim.inc'
      real*8     objt
c
c The points being swapped and the correct:
c
      varcur = var(ix,iy,iz)
      if(accept) then
            do ic=1,nicut
                  do il=1,nlag
                        ivaract(ic,il) = ivarnew(ic,il)
                  end do
            end do
      else
            do ic=1,nicut
                  do il=1,nlag
                        ivarnew(ic,il) = ivaract(ic,il)
                  end do
            end do
      end if
      objt = -1.0
      if(varcur.le.tmin.or.vartry.le.tmin) return
c
c Loop over all lag vectors:
c
      do il=1,nlag
c
c Update the variogram for the positive lag:
c
            jx = ix + ixl(il)
            jy = iy + iyl(il)
            jz = iz + izl(il)
            if(jx.ge.1.and.jx.le.nx.and.
     +         jy.ge.1.and.jy.le.ny.and.
     +         jz.ge.1.and.jz.le.nz) then
                  v0 = var(jx,jy,jz)
                  do ic=1,nicut
                        if(varcur.le.icut(ic).and.v0.gt.icut(ic))
     +                  ivarnew(ic,il) = ivarnew(ic,il) - 1.0
                        if(varcur.gt.icut(ic).and.v0.le.icut(ic))
     +                  ivarnew(ic,il) = ivarnew(ic,il) - 1.0
                        if(vartry.le.icut(ic).and.v0.gt.icut(ic))
     +                  ivarnew(ic,il) = ivarnew(ic,il) + 1.0
                        if(vartry.gt.icut(ic).and.v0.le.icut(ic))
     +                  ivarnew(ic,il) = ivarnew(ic,il) + 1.0
                  end do
            end if
c
c Update the variogram for the negative lag:
c
            jx = ix - ixl(il)
            jy = iy - iyl(il)
            jz = iz - izl(il)
            if(jx.ge.1.and.jx.le.nx.and.
     +         jy.ge.1.and.jy.le.ny.and.
     +         jz.ge.1.and.jz.le.nz) then
                  v0 = var(jx,jy,jz)
                  do ic=1,nicut
                        if(varcur.le.icut(ic).and.v0.gt.icut(ic))
     +                  ivarnew(ic,il) = ivarnew(ic,il) - 1.0
                        if(varcur.gt.icut(ic).and.v0.le.icut(ic))
     +                  ivarnew(ic,il) = ivarnew(ic,il) - 1.0
                        if(vartry.le.icut(ic).and.v0.gt.icut(ic))
     +                  ivarnew(ic,il) = ivarnew(ic,il) + 1.0
                        if(vartry.gt.icut(ic).and.v0.le.icut(ic))
     +                  ivarnew(ic,il) = ivarnew(ic,il) + 1.0
                  end do
            end if
c
c End loop over lags:
c
      end do
c
c Compute the objective function and return:
c
      objt = 0.0
      do ic=1,nicut
            do il=1,nlag
                  act = ivarnew(ic,il)/vardiv(il)
                  objt = objt+(ivarmod(ic,il)-act)*(ivarmod(ic,il)-act)
            end do
      end do
      objt = objt * objscl(3)
c
c Return with tentative objective function:
c
      return
      end



      subroutine initcorr(obj)
c-----------------------------------------------------------------------
c              Compute the Initial Objective Function
c                     Correlation Coefficient
c              **************************************
c
c The objective function is the squared difference from the model
c and experimental correlation coefficient.
c
c
c
c-----------------------------------------------------------------------
      use        geostat
      include   'sasim.inc'
      real*8     obj
      logical    first
      data       first/.true./
c
c Compute the correlation coefficeint:
c
      tnum   = 0.0
      sumsx  = 0.0
      sumsy  = 0.0
      sumsxx = 0.0
      sumsyy = 0.0
      sumsxy = 0.0
      nnz = nz
      if(vertavg) nnz = 1
      do iz=1,nnz
         do iy=1,ny
            do ix=1,nx
               varcur = var(ix,iy,iz)
               if(vertavg) varcur = varavg(ix,iy) / real(nz)
               if(varcur.ge.tmin.and.secvar(ix,iy,iz).ge.tmin) then
                  tnum   =tnum  +1.0
                  sumsx  =sumsx +dble(varcur)
                  sumsy  =sumsy +dble(secvar(ix,iy,iz))
                  sumsxx =sumsxx+dble(varcur*varcur)
                  sumsyy =sumsyy+dble(secvar(ix,iy,iz)*secvar(ix,iy,iz))
                  sumsxy =sumsxy+dble(varcur*secvar(ix,iy,iz))
               end if
            end do
         end do
      end do
      sumsx  = sumsx  / dble(tnum)
      sumsy  = sumsy  / dble(tnum)
      sumsxx = sumsxx / dble(tnum)
      sumsyy = sumsyy / dble(tnum)
      sumsxy = sumsxy / dble(tnum)
      corrtry  =        real (  (sumsxy-sumsx*sumsy) / 
     +  sqrt( ((sumsxx-sumsx*sumsx)*(sumsyy-sumsy*sumsy))))
      obj      = (corr-corrtry)**2
c
c Write some debugging information:
c
      write(ldbg,102) obj,corrtry
 102  format('Objective Function for correlation coeficient: ',f12.6,/,
     +       '                       correlation coeficient: ',f12.6)
      if(.not.first) write(ldbg,103) obj*objscl(4)
 103  format('                                       Scaled: ',f12.6)
      first = .false.
c
c Return with the component objective functions:
c
      return
      end
 
 
 
      subroutine updtcorr(ix,iy,iz,vartry,accept,objt)
c-----------------------------------------------------------------------
c
c              Update Correlation Coefficient for a Change
c              *****************************************
c
c Considering perturbing point (ix,iy,iz) to vartry
c
c Update the Experimental correlation and then compute the objective
c function as the squared difference between the actual and the model
c correlation.
c
c
c
c
c-----------------------------------------------------------------------
      use        geostat
      include   'sasim.inc'
      logical    accept
      real*8     objt
c
c Make sure things are up-to-date:
c
      if(accept) then
            sumsx  = sumtx
            sumsxx = sumtxx
            sumsxy = sumtxy
      else
            sumtx  = sumsx
            sumtxx = sumsxx
            sumtxy = sumsxy
      end if
      objt = -1.0
      if(vertavg) then
            secc = secvar(ix,iy,1)
            varc = varavg(ix,iy) / real(nz)
            vart = varavg(ix,iy) - var(ix,iy,iz)
            vart = vart          + vartry
            vart = vart / real(nz)
      else
            secc = secvar(ix,iy,iz)
            varc = var(ix,iy,iz)
            vart = vartry
      end if
      if(varc.le.tmin.or.vart.le.tmin.or.secc.le.tmin) return
c
c Update correlation coefficient:
c
      sumtx  = sumtx  * dble(tnum)
      sumtxx = sumtxx * dble(tnum)
      sumtxy = sumtxy * dble(tnum)
      sumtx  = sumtx  - dble(varc)
      sumtx  = sumtx  + dble(vart)
      sumtxx = sumtxx - dble(varc*varc)
      sumtxx = sumtxx + dble(vart*vart)
      sumtxy = sumtxy - dble(varc*secc)
      sumtxy = sumtxy + dble(vart*secc)
      sumtx  = sumtx  / dble(tnum)
      sumtxx = sumtxx / dble(tnum)
      sumtxy = sumtxy / dble(tnum)
      corrtry  =        real(   (sumtxy-sumtx*sumsy) / 
     +   sqrt( ((sumtxx-sumtx*sumtx)*(sumsyy-sumsy*sumsy))))
      objt     = ((corr-corrtry)**2) * objscl(4)
c
c Return with updated value:
c
      return
      end



      subroutine initcpdf(obj)
c-----------------------------------------------------------------------
c              Compute the Initial Objective Function
c                    Conditional Distributions
c              **************************************
c
c The objective function is the squared difference from the model
c and experimental conditional distributions.
c
c
c
c-----------------------------------------------------------------------
      use        geostat
      include   'sasim.inc'
      real*8  obj
      logical first
      data    first/.true./
c
c Initialize
c
      do isec=1,nseccut+1
            do ipri=1,npricut+1
                  actpdf(isec,ipri) = 0.0
            end do
      end do
c
c Loop through all of the nodes:
c
      nnz = nz
      if(vertavg) nnz = 1
      do iz=1,nnz
      do iy=1,ny
      do ix=1,nx
        sec    = secvar(ix,iy,iz)
        if(sec.ge.tmin) then
          seclow = -BIGNUM
          do i=1,nseccut
                if(sec.ge.seclow.and.sec.lt.seccut(i)) isec = i
                seclow = seccut(i)
          end do
          if(sec.ge.seclow)  isec = nseccut+1
          pri = var(ix,iy,iz)
          if(vertavg) pri = varavg(ix,iy) / real(nz)
          if(pri.ge.tmin) then
            prilow = -BIGNUM
            do i=1,npricut
                  if(pri.ge.prilow.and.pri.lt.pricut(isec,i)) ipri = i
                  prilow = pricut(isec,i)
            end do
            if(pri.ge.prilow) ipri = npricut+1
            actpdf(isec,ipri) = actpdf(isec,ipri)+1.
          end if
        end if
      end do
      end do
      end do
c
c Now, set the reference conditional distributions to the number
c of samples in the present case:
c
      if(first) then
            do isec=1,nseccut+1
                  sumact = 0.0
                  do ipri=1,npricut+1
                        sumact = sumact + actpdf(isec,ipri)
                  end do
                  do ipri=1,npricut+1
                        refpdf(isec,ipri) = sumact / real(npricut+1)
                  end do
            end do
      end if
c
c Compute the conditional distribution objective function:
c
      obj = 0.0
      write(ldbg,100)
 100  format(/'Conditional Distribution Reproduction at this time:')
      do isec=1,nseccut+1
            do ipri=1,npricut+1
                  obj = obj + 
     +                  dble(abs(refpdf(isec,ipri)-actpdf(isec,ipri)))
            write(ldbg,101)ipri,isec,refpdf(isec,ipri),actpdf(isec,ipri)
 101        format('    ipri,isec:',2i3,' model ',f9.0,' actual ',f9.0)
            end do
      end do
      write(ldbg,102) obj
 102  format('Objective Function: ',f12.6)
      if(.not.first) write(ldbg,103) obj*objscl(5)
 103  format('            Scaled: ',f12.6)
c
c Return with the component objective function:
c
      first = .false.
      return
      end
 
 
 
      subroutine updtcpdf(ix,iy,iz,vartry,accept,objt)
c-----------------------------------------------------------------------
c
c             Update Conditional Distributions for a Change
c             *********************************************
c
c Update the experimental conditional distributions and then compute the
c objective function as the squared difference between the actual and
c the model.
c
c
c
c
c-----------------------------------------------------------------------
      use        geostat
      include   'sasim.inc'
      logical    accept
      real*8     objt
c
c Reset either actpdf or trypdf:
c
      if(accept) then
            do isec=1,nseccut+1
                  do ipri=1,npricut+1
                        actpdf(isec,ipri) = trypdf(isec,ipri)
                  end do
            end do
      else
            do isec=1,nseccut+1
                  do ipri=1,npricut+1
                        trypdf(isec,ipri) = actpdf(isec,ipri)
                  end do
            end do
      end if
c
c Missing secondary variable?
c
      objt = -1.0
      iiz  = iz
      if(vertavg) iiz = 1
      if(secvar(ix,iy,iiz).le.tmin.or.var(ix,iy,iz).le.tmin) return
c
c Secondary variable bin:
c
      sec    = secvar(ix,iy,iiz)
      seclow = -BIGNUM
      do i=1,nseccut
            if(sec.ge.seclow.and.sec.lt.seccut(i)) isec = i
            seclow = seccut(i)
      end do
      if(sec.ge.seclow)  isec = nseccut+1
c
c Old primary bin:
c
      pri = var(ix,iy,iz)
      if(vertavg) pri = varavg(ix,iy) / real(nz)
      prilow = -BIGNUM
      do i=1,npricut
            if(pri.ge.prilow.and.pri.lt.pricut(isec,i)) ip1 = i
            prilow = pricut(isec,i)
      end do
      if(pri.ge.prilow) ip1 = npricut+1
c
c New primary bin:
c
      pri = vartry
      if(vertavg) then
            pri = varavg(ix,iy) - var(ix,iy,iz)
            pri = pri           + vartry
            pri = pri / real(nz)
      end if
      prilow = -BIGNUM
      do i=1,npricut
            if(pri.ge.prilow.and.pri.lt.pricut(isec,i)) ip2 = i
            prilow = pricut(isec,i)
      end do
      if(pri.ge.prilow) ip2 = npricut+1
c
c Subtract old contribution and add new one:
c
      trypdf(isec,ip1) = trypdf(isec,ip1) - 1.0
      trypdf(isec,ip2) = trypdf(isec,ip2) + 1.0
c
c Compute the new objective function:
c
      objt = 0.0
      do isec=1,nseccut+1
            do ipri=1,npricut+1
                  objt = objt + 
     +                   dble(abs(refpdf(isec,ipri)-trypdf(isec,ipri)))
            end do
      end do
      objt = objt * objscl(5)
c
c Return with updated value:
c
 99   continue
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
      open(lun,file='sasim.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for SASIM',/,
     +       '                  ********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('1  1  1  0  0                    ',
     +       '-components: hist,varg,ivar,corr,cpdf')
      write(lun,12)
 12   format('1  1  1  1  1                    ',
     +       '-weight:     hist,varg,ivar,corr,cpdf')
      write(lun,13)
 13   format('1                                ',
     +       '-0=no transform, 1=log transform')
      write(lun,14)
 14   format('1                                ',
     +       '-number of realizations')
      write(lun,15)
 15   format('50      0.5    1.0               ',
     +       '-grid definition: nx,xmn,xsiz')
      write(lun,16)
 16   format('50      0.5    1.0               ',
     +       '-                 ny,ymn,ysiz')
      write(lun,17)
 17   format(' 1      0.5    1.0               ',
     +       '-                 nz,zmn,zsiz')
      write(lun,18)
 18   format('69069                            ',
     +       '-random number seed')
      write(lun,19)
 19   format('4                                ',
     +       '-debugging level')
      write(lun,20)
 20   format('sasim.dbg                        ',
     +       '-file for debugging output')
      write(lun,21)
 21   format('sasim.out                        ',
     +       '-file for simulation output')
      write(lun,22)
 22   format('1                                ',
     +       '-schedule (0=automatic,1=set below)')
      write(lun,23)
 23   format('0.0   0.05  10   3  5  0.001     ',
     +       '-   schedule: t0,redfac,ka,k,num,Omin')
      write(lun,24)
 24   format('10   0.1                        ',
     +       '-   maximum perturbations, reporting')
      write(lun,25)
 25   format('100                              ',
     +       '-   maximum number without a change')
      write(lun,26)
 26   format('0                                ',
     +       '-conditioning data:(0=no, 1=yes)')
      write(lun,27)
 27   format('../data/cluster.dat              ',
     +       '-   file with data')
      write(lun,28)
 28   format('1   2   0   3                    ',
     +       '-   columns: x,y,z,attribute')
      write(lun,29)
 29   format('-1.0e21    1.0e21                ',
     +       '-   trimming limits')
      write(lun,30)
 30   format('1                                ',
     +       '-file with histogram:(0=no, 1=yes)')
      write(lun,31)
 31   format('../data/cluster.dat              ',
     +       '-   file with histogram')
      write(lun,32)
 32   format('3   5                            ',
     +       '-   column for value and weight')
      write(lun,33)
 33   format('99                               ',
     +       '-   number of quantiles for obj. func.')
      write(lun,34)
 34   format('1                                ',
     +       '-number of indicator variograms')
      write(lun,35)
 35   format('2.78                             ',
     +       '-   indicator thresholds')
      write(lun,36)
 36   format('../data/seisdat.dat              ',
     +       '-file with gridded secondary data')
      write(lun,37)
 37   format('1                                ',
     +       '-   column number')
      write(lun,38)
 38   format('1                                ',
     +       '-   vertical average (0=no, 1=yes)')
      write(lun,39)
 39   format('0.60                             ',
     +       '-correlation coefficient')
      write(lun,40)
 40   format('../data/cal.dat                  ',
     +       '-file with paired data')
      write(lun,41)
 41   format('2    1    0                      ',
     +       '-   columns for primary, secondary, wt')
      write(lun,42)
 42   format('-0.5    100.0                    ',
     +       '-   minimum and maximum')
      write(lun,43)
 43   format('5                                ',
     +       '-   number of primary thresholds')
      write(lun,44)
 44   format('5                                ',
     +       '-   number of secondary thresholds')
      write(lun,45)
 45   format('51                               ',
     +       '-Variograms: number of lags')
      write(lun,46)
 46   format('1                                ',
     +       '-   standardize sill (0=no,1=yes)')
      write(lun,47)
 47   format('1    0.1                         ',
     +       '-   nst, nugget effect')
      write(lun,48)
 48   format('1    0.9  0.0   0.0   0.0        ',
     +       '-   it,cc,ang1,ang2,ang3')
      write(lun,49)
 49   format('         10.0  10.0  10.0        ',
     +       '-   a_hmax, a_hmin, a_vert')
      write(lun,50)
 50   format('1    0.1                         ',
     +       '-   nst, nugget effect')
      write(lun,51)
 51   format('1    0.9  0.0   0.0   0.0        ',
     +       '-   it,cc,ang1,ang2,ang3')
      write(lun,52)
 52   format('         10.0  10.0  10.0        ',
     +       '-   a_hmax, a_hmin, a_vert')

      close(lun)
      return
      end
