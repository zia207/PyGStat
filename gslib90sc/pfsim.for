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
      program main
c-----------------------------------------------------------------------
c
c                 Probability Field Simulation
c                 ****************************
c
c The program is executed with no command line arguments.  The user
c will be prompted for the name of a parameter file.  The parameter
c file is described in the documentation (see the example gtsim.par)
c
c The output file will be a GEOEAS file containing the simulated values
c The file is ordered by x,y,z, and then simulation (i.e., x cycles
c fastest, then y, then z, then simulation number).
c
c
c PARAMETERS:
c
c     MAXSIZ is the maximum nx*ny*nz that can be handled by the program
c     MAXCUT must be at least 2 and the number of categories/cutoffs
c
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use       msflib
c
c Fixed parameters:
c
      parameter(EPSLON=1.0e-20,VERSION=3.000)
c
c Variable declaration:
c
      real      var(50),ltpar,mpar,utpar,val(100)
      integer   utail,test
      character datafl*512,cdffl*512,pffl*512,outfl*512,str*512
      logical   testfl
c
c Declare dynamic arrays:
c
      real,allocatable    :: ccdf(:,:),thres(:),ccl(:),cut(:),cdf(:)
      integer,allocatable :: icols(:)
      
c
c Input/Output units used:
c
      lin  = 1
      lout = 2
c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' PFSIM Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'pfsim.par           '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'pfsim.par           ') then
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
      write(*,*) ' variable type (1=continuous, 0=categorical) =',ivtype
      
      read(lin,*,err=98) indic
      write(*,*) ' cdf type (1=indicator, 0=Gaussian) = ',indic
      if(ivtype.eq.0) indic = 1
      
      if(indic.ne.1) then
            read(lin,*,err=98)
c
c Allocate the needed memory:
c
            MAXCUT = 2
            allocate(thres(MAXCUT),stat = test)
                  if(test.ne.0)then
                        write(*,*)'ERROR: Allocation failed due ',
     +                              'to insufficient memory.'
                        stop
                  end if
c
            read(lin,*,err=98)
            ncols = 2
      else 
            read(lin,*,err=98) ncat
            write(*,*) ' number of thresholds/categories = ',ncat
            MAXCUT = ncat
c
            allocate(thres(MAXCUT),stat = test)
                  if(test.ne.0)then
                        write(*,*)'ERROR: Allocation failed due ',
     +                              'to insufficient memory.'
                        stop
                  end if
c
            read(lin,*,err=98) (thres(i),i=1,ncat)
            write(*,*) ' thresholds/categories = ',(thres(i),i=1,ncat)
            ncols = ncat
      end if 
      

c
      allocate(icols(MAXCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due ',
     +                        'to insufficient memory.'
                  stop
            end if
c
      allocate(ccl(MAXCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due ',
     +                        'to insufficient memory.'
                  stop
            end if
c
      read(lin,'(a512)',err=97) datafl
      call chknam(datafl,512)
      write(*,*) ' tabulated quantiles = ',datafl(1:40)

      read(lin,*,err=97) ivr,iwt
      write(*,*) ' input columns: ',ivr,iwt

      read(lin,*,err=97) zmin,zmax
      write(*,*) ' minimum and maximum = ',zmin,zmax

      read(lin,*,err=97) ltail,ltpar
      write(*,*) ' ltail, ltpar = ',ltail,ltpar

      read(lin,*,err=97) middle,mpar
      write(*,*) ' middle, mpar = ',middle,mpar

      read(lin,*,err=97) utail,utpar
      write(*,*) ' utail, utpar = ',utail, utpar

      read(lin,'(a512)',err=98) cdffl
      call chknam(cdffl,512)
      write(*,*) ' data file = ',cdffl(1:40)
      inquire(file=cdffl,exist=testfl)
      if(.not.testfl) stop ' This file does not exist!'

      read(lin,*,err=98) (icols(i),i=1,ncols)
      write(*,*) ' column numbers = ',(icols(i),i=1,ncols)

      read(lin,*,err=98) tmin,tmax
      write(*,*) ' trimming limits = ',tmin,tmax

      read(lin,'(a512)',err=98) pffl
      call chknam(pffl,512)
      write(*,*) ' input p-field file = ',pffl(1:40)
      inquire(file=pffl,exist=testfl)
      if(.not.testfl) stop ' This file does not exist!'

      read(lin,*,err=98) ipcol
      write(*,*) ' column for p-value = ',ipcol

      read(lin,*,err=98) igaus
      write(*,*) ' 0=Gaussian, 1=unfirm = ',igaus

      read(lin,'(a512)',err=98) outfl
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,*,err=98) nsim
      write(*,*) ' number of realizations = ',nsim

      read(lin,*,err=98) nx,ny,nz
      write(*,*) ' nx,ny,nz = ',nx,ny,nz

      close(lin)
c
c Set parameters needed for dynamic array allocation:
c
      MAXSIZ = nx * ny * nz
c
c Do we have a global distribution and do we need it anyway?
c
c Allocate the needed memory:
c
      allocate(ccdf(MAXSIZ,MAXCUT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due ',
     +                        'to insufficient memory.'
                  stop
            end if
c
c
      inquire(file=datafl,exist=testfl)
      if(ltail.ne.3.and.middle.ne.3.and.utail.ne.3) testfl = .false.
c
c Read in the global cdf ("cut" and "cdf" arrays):
c
      if(testfl) then
            open(lin,file=datafl,status='OLD')
            read(lin,*,err=98)
            read(lin,*,err=98) nvari
            do i=1,nvari
                  read(lin,*)
            end do
            MAXDAT = 0
 22         read(lin,*,end=44,err=99)(val(j),j=1,nvari)
            if(val(ivr).lt.tmin.or.val(ivr).ge.tmax)go to 22
            MAXDAT = MAXDAT + 1
            go to 22
 44         continue
            write(*,*)'MAXDAT = ',MAXDAT
c
c Allocate the needed memory:
c
            allocate(cut(MAXDAT),stat = test)
                  if(test.ne.0)then
                        write(*,*)'ERROR: Allocation failed due ',
     +                              'to insufficient memory.'
                        stop
                  end if
c
            allocate(cdf(MAXDAT),stat = test)
                  if(test.ne.0)then
                        write(*,*)'ERROR: Allocation failed due ',
     +                              'to insufficient memory.'
                        stop
                  end if
c
            rewind(lin)
            tcdf = 0.0
            read(lin,*,err=98)
            read(lin,*,err=98) nvari
            do i=1,nvari
                  read(lin,*,err=98)
            end do
            gmean = 0.0
            ncut  = 0
 5          read(lin,*,end=6,err=98) (val(j),j=1,nvari)
            if(val(ivr).lt.tmin.or.val(ivr).ge.tmax) go to 5
            if(iwt.ge.1) then
                  if(val(iwt).le.0.0) go to 5
            endif
            ncut = ncut + 1
            cut(ncut) = val(ivr)
            gmean     = gmean + val(ivr)
            if(iwt.le.0) then
                  cdf(ncut) = 1.0
            else
                  cdf(ncut) = val(iwt)
            endif
            tcdf = tcdf + cdf(ncut)
            go to 5
 6          close(lin)
            if(tcdf.le.0) stop ' total global CDF <= 0'
            gmean = gmean / tcdf
c
c Turn the (possibly weighted) distribution into a cdf that is useful:
c
            call sortem(1,ncut,cut,1,cdf,c,d,e,f,g,h)
            oldcp = 0.0
            cp    = 0.0
            tcdf  = 1.0 / tcdf
            do i=1,ncut
                  cp     = cp + cdf(i) * tcdf
                  cdf(i) =(cp + oldcp) * 0.5
                  oldcp  = cp
            end do
c
c Get median and write some info to the screen:
c
            call locate(cdf,ncut,1,ncut,0.5,j)
            gmedian = powint(cdf(j),cdf(j+1),cut(j),cut(j+1),0.5,1.)
            write(*,*) 'Global cdf from file: ',datafl
            write(*,*) '   number of data: ',ncut
            write(*,*) '   global mean:    ',gmean
            write(*,*) '   global median:  ',gmedian
            write(*,*)
      end if
c
c Read the input distributions:
c
      open(lin,file=cdffl,status='OLD')
      read(lin,'(a40)',err=99) str
      read(lin,*,err=99) nvari
      do i=1,nvari
            read(lin,*,err=99)
      end do
      if(nvari.lt.ncols) stop ' Too few columns in input data file'
      index = 0
      do iz=1,nz
      do iy=1,ny
      do ix=1,nx
            index = index + 1
            read(lin,*,err=99) (var(i),i=1,nvari)
            if(indic.ne.1) then
                  ccdf(index,1) = var(icols(1))
                  ccdf(index,2) = var(icols(2))
                  if(ccdf(index,2).gt.0.0) ccdf(index,2) =
     +                                     sqrt(ccdf(index,2))
            else 
                  do icut=1,ncat
                        ccdf(index,icut) = var(icols(icut))
                        if(ivtype.eq.0.and.icut.gt.1)
     +                  ccdf(index,icut) = ccdf(index,icut) + 
     +                                     ccdf(index,icut-1)
                  end do
            end if 
      end do
      end do
      end do
      close(lin)
c
c Prepare to read the p-fields and write the output file:
c
      open(lin,file=pffl,status='OLD')
      read(lin,'(a40)',err=97) str
      read(lin,*,err=97) nvari
      if(ipcol.gt.nvari) stop ' Too few columns in p-field file'
      do i=1,nvari
            read(lin,*,err=97)
      end do
      open(lout,file=outfl,status='UNKNOWN')
      write(lout,'(a40)') str
      xmn   = 0.
      ymn   = 0.
      zmn   = 0.
      xsiz  = 1.
      ysiz  = 1.
      zsiz  = 1.  
      write(lout,201) 1,nx,ny,nz,xmn,ymn,zmn,xsiz,ysiz,zsiz,nsim
 201  format(i2,3(1x,i4),3(1x,g14.8),3(1x,g12.6),i4)
      write(lout,202)
 202  format('value')
c
c Loop over all of the realizations and locations:
c
      do isim=1,nsim
      index = 0
      do iz=1,nz
      do iy=1,ny
      do ix=1,nx
            index = index + 1
            read(lin,*,err=99) (var(i),i=1,nvari)
            pval  = var(ipcol)
            if(igaus.eq.0.and.indic.eq.1) pval = gcum(pval)
c
c Draw from Gaussian or Indicator?
c
            if(indic.ne.1) then
                  pdbl = dble(pval)
                  call gauinv(pdbl,xp,ierr)
                  sim = xp * ccdf(index,2) + ccdf(index,1)
            else 
                  if(ivtype.eq.0) then
                        do icut=1,ncat
                              jcut = icut
                              if(pval.le.ccdf(index,icut)) then
                                    jcut = icut
                                    go to 3
                              end if
                        end do
 3                      sim = thres(jcut)
                  else
                        sim = -1.0
                        do i=1,ncat
                              ccl(i) = ccdf(index,i)
                        end do
                        call beyond(1,ncat,thres,ccl,ncut,cut,cdf,zmin,
     +                              zmax,ltail,ltpar,middle,mpar,utail,
     +                              utpar,sim,pval,ierr)
                  end if
            end if 
            write(lout,'(g14.8)') sim
      end do
      end do
      end do
      end do
c
c Finished:
c
      write(*,9998) VERSION
 9998 format(/' PFSIM Version: ',f5.3, ' Finished'/)
      stop
 97   stop 'ERROR in p field file!'
 98   stop 'ERROR in parameter file!'
 99   stop 'ERROR in data file!'
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
      open(lun,file='pfsim.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for PFSIM',/,
     +       '                  ********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('1                           ',
     +       '-1=continuous(cdf), 0=categorical(pdf)')
      write(lun,12)
 12   format('1                           ',
     +       '-1=indicator ccdfs, 0=Gaussian (mean,var)')
      write(lun,13)
 13   format('5                           ',
     +       '-   number thresholds/categories')
      write(lun,14)
 14   format('0.5  1.0  2.5  5.0  10.0    ',
     +       '-   thresholds / categories')
      write(lun,15)
 15   format('cluster.dat                 ',
     +       '-Within-class details: file with global dist')
      write(lun,16)
 16   format('3   0                       ',
     +       '-   ivr,  iwt')
      write(lun,17)
 17   format('0.0    30.0                 ',
     +       '-   minimum and maximum Z value')
      write(lun,18)
 18   format('1   1.0                     ',
     +       '-   lower tail: option, parameter')
      write(lun,19)
 19   format('1   1.0                     ',
     +       '-   middle    : option, parameter')
      write(lun,20)
 20   format('1   2.0                     ',
     +       '-   upper tail: option, parameter')
      write(lun,21)
 21   format('kt3d.out                    ',
     +       '-file with input conditional distributions')
      write(lun,22)
 22   format('1   2                       ',
     +       '-  columns for mean, var, or ccdf values')
      write(lun,23)
 23   format('-1.0e21    1.0e21           ',
     +       '-  trimming limits')
      write(lun,24)
 24   format('sgsim.out                   ',
     +       '-file with input p-field realizations')
      write(lun,25)
 25   format('1                           ',
     +       '-  column number in p-field file')
      write(lun,26)
 26   format('0                           ',
     +       '-  0=Gaussian, 1=uniform [0,1]')
      write(lun,27)
 27   format('pfsim.out                   ',
     +       '-file for output realizations')
      write(lun,28)
 28   format('1                           ',
     +       '-number of realizations')
      write(lun,29)
 29   format('50   50   1                 ',
     +       '-nx, ny, nz')

      close(lun)
      return
      end
