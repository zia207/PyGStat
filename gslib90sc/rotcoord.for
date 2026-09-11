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
c                     2-D Coordinate Rotation
c                     ***********************
c
c INPUT/OUTPUT Parameters:
c
c   datafl         the input data file
c   ix,iy          column numbers for X, Y, and value
c   outfl          output file with rotated coordinates appended
c   xorig,yorig    Origin in X and Y
c   angle          rotation angle (degrees clockwise)
c   ireverse       reverse coordinate rotation (1=yes)
c
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use       msflib
      parameter(DEG2RAD=3.14159265/180.0,EPSLON=1.0e-6,MAXLEN=132,
     +          VERSION=3.000)
c
c Dimensioning:
c
      real      var(100)
      character datafl*512,outfl*512,str*512
      logical   testfl
      data      lin/1/,lout/2/
c
c Note VERSION number before anything else:
c
      write(*,9999) VERSION
 9999 format(/' ROTCOORD Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'rotcoord.par        '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'rotcoord.par        ') then
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

      read(lin,*,err=98) icolx,icoly
      write(*,*) ' columns = ',icolx,icoly

      read(lin,'(a512)',err=98) outfl
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,*,err=98) xorig,yorig
      write(*,*) ' xorig,yorig = ',xorig,yorig

      read(lin,*,err=98) angle
      write(*,*) ' angle = ',angle
      angle = -angle * DEG2RAD

      read(lin,*,err=98) ireverse
      write(*,*) ' ireverse',ireverse

      close(lin)
c
c Make sure that we don't get into an infinite loop:
c
      if(datafl.eq.outfl) then
            write(*,*) 'ERROR: same input and output file specified'
            stop
      end if
c
c Make sure that we have a data file:
c
      inquire(file=datafl,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR: ',datafl,' does not exist'
            write(*,*) '       you need a data file! '
            stop
      endif
c
c Open up the input and output files:
c
      open(lin,file=datafl, status='OLD')
      open(lout,file=outfl, status='UNKNOWN')
      read(lin,'(a40)',err=99) str(1:40)
      write(lout,100)          str(1:40)
      read(lin,*,err=99)       nvari
      write(lout,'(i2)')       nvari+2
      do i=1,nvari
            read(lin,'(a40)',err=99) str(1:40)
            write(lout,'(a40)')      str(1:40)
      end do
      write(lout,101)
 100  format('With Rotated Coordinates:',a40)
 101  format('Rotated X',/,'Rotated Y')
      if(icolx.gt.nvari) stop 'icolx too big'
      if(icoly.gt.nvari) stop 'icoly too big'
c
c Read and write all the data until the end of the file:
c
      id = 0
 7    read(lin,*,end=9) (var(i),i=1,nvari)
c
c Coordinate Transformation:
c
      xx = var(icolx)
      rx = xx
      yy = var(icoly)
      ry = yy
      call rotc(xorig,yorig,angle,ireverse,xx,yy,rx,ry)
      if(ireverse.eq.1) then
            rx = xx
            ry = yy
      end if
c
c Write out the results:
c
      backspace lin
      read(lin,'(a)') str
      call strlen(str,MAXLEN,lostr)
      write(lout,'(a,1x,g14.8,1x,g14.8)') str(1:lostr),rx,ry
      go to 7
 9    continue
c
c Finished:
c
      close(lin)
      close(lout)
      write(*,9998) VERSION
 9998 format(/' ROTCOORD Version: ',f5.3, ' Finished'/)
      stop
 98   stop ' ERROR in parameter file'
 99   stop ' ERROR in data file'
      end



      subroutine rotc(xorig,yorig,angle,ireverse,xx,yy,rx,ry)
c-----------------------------------------------------------------------
c
c xorig,yorig   origin of rotated system in original coordinates
c angle         angle of rotation (radians counter clockwise)
c ireverse      take rx,ry and convert to xx,yy
c
c xx,yy         location in original coordinates
c rx,ry         location in rotated  coordinates
c
c
c
c-----------------------------------------------------------------------
      real    xorig,yorig,angle,xx,yy,rx,ry
      integer ireverse
c
c change from rotated to original coordinate system:
c
      if(ireverse.eq.1) then
            yy = yorig + ry*cos(angle) + rx*sin(angle)
            xx = xorig - ry*sin(angle) + rx*cos(angle)
      else
c
c change from original to rotated coordinate system:
c
            ry = (yy-yorig)*cos(angle)-(xx-xorig)*sin(angle)
            rx = (yy-yorig)*sin(angle)+(xx-xorig)*cos(angle)
      endif
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
      open(lun,file='rotcoord.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for ROTCOORD',/,
     +       '                  ***********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('../data/cluster.dat   ',
     +       '-file with data')
      write(lun,12)
 12   format('1   2                 ',
     +       '-  columns for original X and Y coordinates')
      write(lun,13)
 13   format('rotcoord.out          ',
     +       '-file for output with new coordinates')
      write(lun,14)
 14   format('0.0       0.0         ',
     +       '-origin of rotated system in original coordinates')
      write(lun,15)
 15   format('30.0                  ',
     +       '-rotation angle (in degrees clockwise)')
      write(lun,16)
 16   format('0                     ',
     +       '-0=convert to rotated coordinate system')
      write(lun,17)
 17   format('                      ',
     +       '-1=convert from rotated system to original system')

      close(lun)
      return
      end
