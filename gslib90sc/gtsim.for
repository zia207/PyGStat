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
c                 Gaussian Truncated Simulation
c                 *****************************
c
c The program is executed with no command line arguments.  The user
c will be prompted for the name of a parameter file.  The parameter
c file is described in the documentation (see the example gtsim.par)
c
c The output file will be a GEOEAS file containing the simulated values
c The file is ordered by x,y,z, and then simulation (i.e., x cycles
c fastest, then y, then z, then simulation number).  The values will be
c the categorical values from the truncated Gaussian field
c
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use       msflib
c
c Parameters:
c
      parameter(VERSION=3.000)
c
c Variable declaration:
c
      real      var(500)
      integer   test
      character inpfl*512,outfl*512,str*512
      logical   testfl
c
c Declare dynamic arrays:
c
      real,allocatable          :: cat(:),pdf(:),thres(:)
      integer,allocatable       :: ipcol(:),nvarip(:)
      character*512,allocatable :: propfl(:)
c
c Input/Output units used:
c
      lin  = 1
      lout = 2
c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' GTSIM Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'gtsim.par           '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'gtsim.par           ') then
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
      read(lin,'(a512)',err=98) inpfl
      call chknam(inpfl,512)
      write(*,*) ' data file = ',inpfl(1:40)

      inquire(file=inpfl,exist=testfl)
      if(.not.testfl) stop ' This file does not exist!'

      read(lin,'(a512)',err=98) outfl
      call chknam(outfl,512)
      write(*,*) ' output file ',outfl(1:40)

      read(lin,*,err=98) nsim
      write(*,*) ' number of realizations = ',nsim

      read(lin,*,err=98) nx,ny,nz
      write(*,*) ' nx,ny,nz = ',nx,ny,nz

      read(lin,*,err=98) ncat
      write(*,*) ' number of categories = ',ncat
c
c Allocate the needed memory:
c
      allocate(cat(ncat),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c
      allocate(pdf(ncat),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c
      allocate(thres(ncat),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c
      allocate(ipcol(ncat),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c
      allocate(nvarip(ncat),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c
      allocate(propfl(ncat),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c

      cp = 0.0
      do i=1,ncat
            read(lin,*,err=98) cat(i),pdf(i)
            write(*,*) ' category and pdf: ',cat(i),pdf(i)
            cp = cp + pdf(i)
      end do
      if(abs(cp-1.0).gt.0.01) stop 'Sum of proportions should be 1.0'

      read(lin,*,err=98) iprop
      write(*,*) ' proportion curves (0=no, 1=yes) = ',iprop

      if(iprop.eq.1) then
            do i=1,ncat-1
                  read(lin,'(a512)',err=98) propfl(i)
                  call chknam(propfl(i),512)
                  write(*,*) ' file   for proportion = ',propfl(i)(1:40)
                  read(lin,*,err=98) ipcol(i)
                  write(*,*) ' column for proportion = ',ipcol(i)
                  inquire(file=propfl(i),exist=testfl)
                  if(.not.testfl) stop ' This file does not exist!'
                  lun = 10 + i
                  open(lun,file=propfl(i),status='OLD')
            end do
            write(*,*) ' NOTE: only ncat-1 files is required'
            write(*,*) '       last proportion is 1.0-sum of the rest'
      end if

      close(lin)
c
c Finished reading parameters -- prepare the input and output files:
c
      open(lin,file=inpfl,status='OLD')
      read(lin,'(a40)',err=99) str
      read(lin,*,err=99) nvari
      if(nvari.ne.1) then
            write(*,*)
            write(*,*) 'WARNING: GTSIM expects there to be only one '
            write(*,*) '         column in input SGSIM file!'
            write(*,*)
            write(*,*) '         using the first column'
            write(*,*)
      end if
      do i=1,nvari
            read(lin,*,err=99)
      end do
      open(lout,file=outfl,status='UNKNOWN')
      write(lout,200) str
 200  format('GTSIM output',a40)
      xmn   = 0.
      ymn   = 0.
      zmn   = 0.
      xsiz  = 1.
      ysiz  = 1.
      zsiz  = 1. 
      write(lout,201) 1,nx,ny,nz,xmn,ymn,zmn,xsiz,ysiz,zsiz,nsim
 201  format(i2,3(1x,i4),3(1x,g14.8),3(1x,g12.6),i4) 
      write(lout,202)
 202  format('category')
c
c Global thresholds?
c
      cp = 0.0
      do i=1,ncat-1
            cp = cp + pdf(i)
            call gauinv(dble(cp),xp,ierr)
            thres(i) = xp
      end do
c
c MAIN LOOP Over all realizations:
c
      do isim=1,nsim
            write(*,*)
            write(*,*) 'Working on realization number ',isim
c
c Do we need to rewind the local proportion curve files:
c
            if(iprop.eq.1) then
                  do i=1,ncat-1
                        lun = 10 + i
                        rewind(lun)
                        read(lun,*,err=97)
                        read(lun,*,err=97) nvarip(i)
                        do j=1,nvarip(i)
                              read(lun,*,err=97)
                        end do
                  end do
            end if
c
c Loop over all grid nodes in this realization:
c
            do ixyz=1,nx*ny*nz
c
c Get normal score value:
c
                  read(lin,*,err=99) yval
c
c Get local thresholds?
c
                  if(iprop.eq.1) then
                        cp = 0.0
                        do i=1,ncat-1
                              lun = 10 + i
                              read(lun,*,err=97) (var(j),j=1,nvarip(i))
                              icol = ipcol(i)
                              if(var(icol).lt.0.0.or.var(icol).gt.1.0)
     +                                      stop 'INVALID local pdf'
                              cp = cp + var(icol)
                              if(cp.gt.1.0) stop 'INVALID local pdf'
                              call gauinv(dble(cp),xp,ierr)
                              thres(i) = xp
                        end do
                  end if
c
c Get the categorical value for this location:
c
                  do i=1,ncat-1
                        if(yval.lt.thres(i)) then
                              icat = i
                              go to 7
                        end if
                  end do
                  icat = ncat
 7                continue
                  write(lout,'(f4.2)') cat(icat)
c
c End loop over all nodes:
c
            end do
c
c End loop over all realizations:
c
      end do
      close(lin)
      close(lout)
      if(iprop.eq.1) then
            do i=1,ncat-1
                  lun = 10 + i
                  close(lun)
            end do
      end if
c
c Finished:
c
      write(*,9998) VERSION
 9998 format(/' GTSIM Version: ',f5.3, ' Finished'/)
      stop
 97   stop 'ERROR in proportion file!'
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
      open(lun,file='gtsim.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for GTSIM',/,
     +       '                  ********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('sgsim.out                    ',
     +       '-file with input Gaussian realizations')
      write(lun,12)
 12   format('gtsim.out                    ',
     +       '-file for output categorical realizations')
      write(lun,13)
 13   format('1                            ',
     +       '-number of realizations')
      write(lun,14)
 14   format('50   50   1                  ',
     +       '-nx,ny,nz')
      write(lun,15)
 15   format('3                            ',
     +       '-number of categories')
      write(lun,16)
 16   format('1    0.25                    ',
     +       '-   cat(1)  global proportion(1)')
      write(lun,17)
 17   format('2    0.25                    ',
     +       '-   cat(2)  global proportion(2)')
      write(lun,18)
 18   format('3    0.50                    ',
     +       '-   cat(3)  global proportion(3)')
      write(lun,19)
 19   format('0                            ',
     +       '-proportion curves (0=no, 1=yes)')
      write(lun,20)
 20   format('propc01.dat                  ',
     +       '-   file with local proportion (1)')
      write(lun,21)
 21   format('1                            ',
     +       '-   column number for proportion')
      write(lun,22)
 22   format('propc02.dat                  ',
     +       '-   file with local proportion (2)')
      write(lun,23)
 23   format('1                            ',
     +       '-   column number for proportion')
      write(lun,24)
 24   format('propc03.dat                  ',
     +       '-   file with local proportion (3)')
      write(lun,25)
 25   format('1                            ',
     +       '-   column number for proportion')

      close(lun)
      return
      end
