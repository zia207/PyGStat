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
c       Simple Stochastic Simulation (Random Drawing) Program
c       *****************************************************
c
c Monte Carlo simulation with replacement of observations in an input
c data file - could be easily modified to include a "bootstrap"
c statistic
c
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use       msflib
c
c Parameters:
c
      parameter (EPSLON=1.0e-10,VERSION=3.000,
     +           KORDEI=12,MAXOP1=KORDEI+1,MAXINT=2**30)
c
c Variable Declaration:
c
      real      tmp(500),var(500)
      integer   colprob,test
      character datafl*512,outfl*512,str*512
      logical   testfl
c
c Declaration of dynamic arrays:
c 
      real,allocatable :: prob(:),val(:,:)
      integer,allocatable :: colkeep(:)
      character*12,allocatable :: labels(:)
c
c For random number generator:
c
      real*8  acorni
      common /iaco/ ixv(MAXOP1)
      data   ixv/MAXOP1*0.0/
c
c Note VERSION number:
c
      lin  = 1
      lout = 2
      ldep = 3
      write(*,9999) VERSION
 9999 format(/' DRAW Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'draw.par            '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'draw.par            ') then
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
      read(lin,'(a512)',err=97) datafl
      call chknam(datafl,512)
      write(*,*) ' input distribution file = ',datafl(1:40)

      read(lin,*,err=97) ncolkeep
      write(*,*) ' number of variables to keep = ',ncolkeep
c
      allocate (colkeep(ncolkeep),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
c
      read(lin,*,err=97) (colkeep(i),i=1,ncolkeep)
      write(*,*) ' columns to keep = ',(colkeep(i),i=1,ncolkeep)

      read(lin,*,err=97) colprob
      write(*,*) ' column for probability = ',colprob

      read(lin,*,err=97) tmin,tmax
      write(*,*) ' trimming limits = ',tmin,tmax

      read(lin,*,err=97) ixv(1),ndraw
      write(*,*) ' random number seed and number to draw = ',ixv(1)
      do i=1,10000
            zz = real(acorni(idum))
      end do

      read(lin,'(a512)',err=97) outfl 
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      close(lin)
c
c Read in the header information and the size of the
c data to find MAXDIS:
c
      inquire(file=datafl,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR: no distribution file'
            stop
      endif
      open(lin,file=datafl,status='OLD')
      read(lin,*,err=98)
      read(lin,*,err=98)     nvari
      do i=1,nvari
            read(lin,*)
      end do
      maxdis = 0
 22   read(lin,*,end=44,err=98)(var(j),j=1,nvari)
      maxdis = maxdis + 1
      go to 22
 44   continue
c
c Allocate the needed memory:
c
      allocate (labels(ncolkeep),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
c
      allocate (prob(maxdis),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
c
      allocate (val(ncolkeep,maxdis),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
c
c
      rewind(lin)
      read(lin,'(a)',err=98) str
      read(lin,*,err=98)     nvari
      do i=1,nvari
            read(lin,'(a12)',err=98) str(1:12)
            do j=1,ncolkeep
                  if(i.eq.colkeep(j)) labels(j) = str(1:12)
            end do
      end do
c
c Read through the input file:
c
      nd     = 0
      sumwts = 0.0
 2    read(lin,*,err=98,end=3) (tmp(i),i=1,nvari)
      if(colprob.gt.0.and.colprob.le.nvari.and.
     +  (tmp(colprob).lt.tmin.or.tmp(colprob).ge.tmax)) go to 2
      nd = nd + 1
      if(nd.gt.MAXDIS) then
            write(*,*) 'ERROR: only have ',MAXDIS,' room for data'
            stop
      end if
      prob(nd) = 1.0
      if(colprob.gt.0.and.colprob.le.nvari) prob(nd) = tmp(colprob)
      sumwts   = sumwts + prob(nd)
      do i=1,ncolkeep
            ivar = colkeep(i)
            val(i,nd) = tmp(ivar)
      end do
      go to 2
 3    continue
      close(lin)
      write(*,*)
      write(*,*) ' There were ',nd,' data.  Sum of weights = ',sumwts
      write(*,*)
      if(nd.lt.2.or.sumwts.lt.EPSLON) then 
            write(*,*) ' too few data ',nd
            write(*,*) ' too low sum of weights ',sumwts
            stop
      end if
c
c Turn the data distribution into a CDF:
c
      oldcp   = 0.0
      cp      = 0.0
      sumwts  = 1.0 / sumwts
      do i=1,nd
            cp      = cp + prob(i) * sumwts
            prob(i) =(cp + oldcp)  * 0.5
            oldcp   = cp
      end do
c
c Get ready to do make the drawings:
c
      open(lout,file=outfl,status='UNKNOWN')
      write(lout,101) ncolkeep
 101  format('DRAW output',/,i3)
      do i=1,ncolkeep
            write(lout,'(a12)') labels(i)
      end do
c
c Make the drawings and write them to the output file:
c
      write(*,*) 'Working on realizations'
      irepo = max(1,min((ndraw/10),10000))
      do i=1,ndraw
            if((int(i/irepo)*irepo).eq.i) write(*,103) i
 103        format('   currently on node ',i9)

            cdf = real(acorni(idum))
            call locate(prob,nd,1,nd,cdf,j)
            j = max(min(j,nd),1)
            do ivar = 1,ncolkeep
                  call numtext(val(ivar,j),labels(ivar))
            end do

            write(lout,102) (labels(ivar),ivar=1,ncolkeep)
 102        format(48(a12,1x))
      end do
      close(lout)
c
c Finished:
c
      write(*,9998) VERSION
 9998 format(/' DRAW Version: ',f5.3, ' Finished'/)
      stop
 97   stop 'ERROR in parameter file'
 98   stop 'ERROR in data file'
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
      open(lun,file='draw.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for DRAW',/,
     +       '                  *******************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('../data/cluster.dat              ',
     +       '-file with data')
      write(lun,12)
 12   format('3                                ',
     +       '-   number of variables')
      write(lun,13)
 13   format('1   2   3                        ',
     +       '-   columns for variables')
      write(lun,14)
 14   format('5                                ',
     +       '-   column for probabilities (0=equal)')
      write(lun,15)
 15   format('-1.0e21   1.0e21                 ',
     +       '-   trimming limits')
      write(lun,16)
 16   format('69069    100                     ',
     +       '-random number seed, number to draw')
      write(lun,17)
 17   format('draw.out                         ',
     +       '-file for realizations')

      close(lun)
      return
      end
