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
c                     Gaussian Back Transformation
c                     ****************************
c
c PROGRAM NOTES:
C
c  1. ltail, utail options: 1=linear interpolation, 2=power model
c     interpolation, and 4=hyperbolic model interpolation (only for
c     upper tail)
c
c
c
c EXTERNAL REFERENCES:
c
c   gcum     Inverse of Gaussian cumulative distribution function
c   locate   locate a position in an array
c   powint   power law interpolation
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use       msflib
      parameter(MAXLEN=512,EPSLON=0.00001,VERSION=3.000)

      real      var(500),ltpar,utpar
      real*8    p
      character datafl*512,outfl*512,transfl*512,str*512
      integer   ltail,utail,test
      logical   testfl,getrank
      data      lin/1/,lout/2/
c
c Declare dynamic arrays:
c
      real, allocatable :: vr(:), vrg(:)
c
c Note VERSION number before anything else:
c
      write(*,9999) VERSION
 9999 format(/' BACKTR Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'backtr.par          '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'backtr.par          ') then
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
 1    read(lin,'(a40)',end=98) str
      if(str(1:4).ne.'STAR') go to 1
c
c Read Input Parameters:
c
      read(lin,'(a512)',err=98) datafl
      call chknam(datafl,512)
      write(*,*) ' data file = ',datafl(1:40)

      getrank = .false.
      read(lin,*,err=98) ivr
      if(ivr.lt.0) then
            getrank = .true.
            ivr     = -ivr
      end if
      write(*,*) ' column of normal = ',ivr

      read(lin,*,err=98) tmin,tmax
      write(*,*) ' trimming limits = ',tmin,tmax

      read(lin,'(a512)',err=98) outfl
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,'(a512)',err=98) transfl
      call chknam(transfl,512)
      write(*,*) ' transformation file = ',transfl(1:40)

      read(lin,*,err=98) zmin,zmax  
      write(*,*) ' data limits = ',zmin,zmax

      read(lin,*,err=98) ltail,ltpar
      write(*,*) ' lower tail option = ',ltail,ltpar

      read(lin,*,err=98) utail,utpar
      write(*,*) ' upper tail option = ',utail,utpar

      close(lin)
c
c Check for error situation:
c     
      if(ltail.ne.1.and.ltail.ne.2) then
            write(*,*) 'ERROR invalid lower tail option ',ltail
            write(*,*) '      only allow 1 or 2 - see manual '
            stop
      endif
      if(utail.ne.1.and.utail.ne.2.and.utail.ne.4) then
            write(*,*) 'ERROR invalid upper tail option ',ltail
            write(*,*) '      only allow 1,2 or 4 - see manual '
            stop
      endif
      if(utail.eq.4.and.utpar.lt.1.0) then
            write(*,*) 'ERROR invalid power for hyperbolic tail',utpar
            write(*,*) '      must be greater than 1.0!'
            stop
      endif
c
c Read in the transformation table:
c
      inquire(file=transfl,exist=testfl)
      if(.not.testfl)then
            write(*,*) 'ERROR transformation file does not exist'
            stop
c
c The data file exists so open the file and read in the header
c information. Initialize the storage that will be used to summarize
c the data found in the file:
c

      else
            open(lin,file=transfl,status='OLD')
            maxdat = 0
 20         read(lin,*,end=40,err=99)(var(j),j=1,2)
            if(var(1).le.tmin.or.var(1).ge.tmax)go to 40
            maxdat = maxdat + 1
            go to 20
 40         continue
c 
c Allocate arrays needed for transformation table:     
c
            allocate (vr(maxdat),stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error 1: Allocation failed due to ',
     +                       'insufficient memory!', test
                  stop
            end if
            allocate (vrg(maxdat),stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error 1: Allocation failed due to ',
     +                       'insufficient memory!', test
                  stop
            end if
c     
c Now, rewind the data file and store the transformation table:
c
            rewind(lin)                
            nt = 0
 2          read(lin,*,end=3) (var(i),i=1,2)
            nt = nt + 1
            vr(nt)  = var(1)
            vrg(nt) = var(2)
            if(nt.gt.1) then
            if(vr(nt) .lt.vr(nt-1) .or.
     +         vrg(nt).lt.vrg(nt-1)) then
                  write(*,*) 'ERROR transformation table must be '
                  write(*,*) '      monotonic increasing! '
                  write(*,*) '      Inconsistency at line ',nt
                  stop
            endif
            endif
            go to 2
 3          continue
      close(lin)
      end if
c
c Check for error situation:
c
      if(utail.eq.4.and.vr(nt).le.0.0) then
            write(*,*) 'ERROR can not use hyperbolic tail with '
            write(*,*) '      negative values! - see manual '
            stop
      endif
      if(zmin.gt.vr(1)) then
            write(*,*) 'ERROR zmin should be no larger than the first'
            write(*,*) '      entry in the transformation table '
            write(*,*) '      zmin = ',zmin,' vr1 ',vr(1)
            stop
      endif
      if(zmax.lt.vr(nt)) then
            write(*,*) 'ERROR zmax should be no less than the last'
            write(*,*) '      entry in the transformation table '
            write(*,*) '      zmax = ',zmax,' vrnt ',vr(nt)
            stop
      endif
c
c Now read through the data:
c
      inquire(file=datafl,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR data file ',datafl,' does not exist!'
            stop
      endif
c
c The data file exists so open the file and read in the header and
c write a header on the output file:
c
      open(lin,file=datafl, status='OLD')
      open(lout,file=outfl,status='UNKNOWN')
      read(lin,'(a40)',err=99) str(1:40)
      write(lout,100)          str(1:40)
      read(lin,*,err=99)       nvari
      write(lout,'(i2)')       nvari+1
      do i=1,nvari
            read(lin,'(a40)',err=99) str(1:40)
            write(lout,'(a40)')      str(1:40)
      end do
      write(lout,101)
 100  format('Back Transform:',a40)
 101  format('Back Transform')
c
c Read the value to be back transformed
c
 7    read(lin,*,end=8,err=99) (var(i),i=1,nvari)
      if(getrank) then
            p = dble(var(ivr))
            call gauinv(p,var(ivr),ierr)
      end if
      bac = backtr(var(ivr),nt,vr,vrg,zmin,zmax,ltail,ltpar,utail,utpar)
      if(bac.lt.zmin) bac = zmin
      if(bac.gt.zmax) bac = zmax
c
c Write out the results:
c
      backspace lin
      read(lin,'(a)') str
      call strlen(str,MAXLEN,lostr)
      write(lout,'(a,1x,g14.8)') str(1:lostr),bac
      go to 7
 8    continue
c
c Finished:
      close(lin)
      close(lout)
      write(*,9998) VERSION
 9998 format(/' BACKTR Version: ',f5.3, ' Finished'/)
      stop
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
      open(lun,file='backtr.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for BACKTR',/,
     +       '                  *********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('nscore.out                    ',
     +       '-file with data')
      write(lun,12)
 12   format('6                             ',
     +       '-  column with Gaussian variable')
      write(lun,13)
 13   format('-1.0e21   1.0e21              ',
     +       '-  trimming limits')
      write(lun,14)
 14   format('backtr.out                    ',
     +       '-file for output')
      write(lun,15)
 15   format('nscore.trn                    ',
     +       '-file with input transformation table')
      write(lun,16)
 16   format('0.0 60.0                      ',
     +       '-minimum and maximum data value')
      write(lun,17)
 17   format('1    0.0                      ',
     +       '-lower tail option and parameter')
      write(lun,18)
 18   format('1   60.0                      ',
     +       '-upper tail option and parameter')

      close(lun)
      return
      end
