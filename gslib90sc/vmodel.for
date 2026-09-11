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
c                  Variogram Output from a Model
c                  *****************************
c
c This program calculates the semivariogram model values at certain lags
c and writes the result in a format compatible with "vargplt".
c
c INPUT/OUTPUT Parameters:
c
c   outfl            output file for the variograms
c   ndir,nlag        number of directions and lags (same for all ndir)
c   azm,dip,xlag     direction and lags for each variogram
c   nst,c0           number of nested structures and the nugget effect
c   it               type of nested structure (1=spherical, 
c                    2=exponential, 3=Gaussian, 4=power model)
c   aa               range parameters in each direction
c   cc               sill parameter
c   ang1,ang2,ang3   angle definition
c
c
c
c PROGRAM NOTES:
c
c   1. The MAXNST parameter controls the maximum number of nested
c      structures that can be considered at one time.  The MDIR
c      parameter controls the maximum number of directions that may be
c      considered for a particular cell size at one time.
c
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use       msflib
      parameter(MAXNST=4, MDIR=100, DEG2RAD=3.14159265/180.0, 
     +          MAXROT=MAXNST+1, EPSLON = 1.0e-20, VERSION=3.000)

      real*8     rotmat(MAXROT,3,3)
      real       aa(MAXNST),c0(1),cc(MAXNST),ang1(MAXNST),ang2(MAXNST),
     +           ang3(MAXNST),anis1(MAXNST),anis2(MAXNST),
     +           xoff(MDIR),yoff(MDIR),zoff(MDIR),maxcov
      integer    nst(1),it(MAXNST)
      character  outfl*512,str*512
      logical    testfl
      data       lin/1/,lout/2/
c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' VMODEL Version: ',f5.3/)
c
c Get the name of the parameter file - try the default name if no input:
c
      str(1:1) =' '
      call getarg(1,str)
      if(str(1:1).eq.' ')then
            write(*,*) 'Which parameter file do you want to use?'
            read (*,'(a20)') str(1:20)
      end if
      if(str(1:1).eq.' ') str(1:20) = 'vmodel.par          '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'vmodel.par          ') then
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
      read(lin,'(a512)',err=98) outfl
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl

      read(lin,*,err=98) ndir,nlag
      write(*,*) ' ndir,nlag = ',ndir,nlag

      do i=1,ndir
            read(lin,*,err=98) azm,dip,xlag
            write(*,*) ' azm, dip, lag = ',azm,dip,xlag
            xoff(i) = sin(DEG2RAD*azm)*cos(DEG2RAD*dip)*xlag
            yoff(i) = cos(DEG2RAD*azm)*cos(DEG2RAD*dip)*xlag
            zoff(i) = sin(DEG2RAD*dip)*xlag
            write(*,*) ' x,y,z offsets = ',xoff(i),yoff(i),zoff(i)
            write(*,*)
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
            if(it(i).eq.4) then
                  if(aa(i).lt.0.0) stop ' INVALID power variogram'
                  if(aa(i).gt.2.0) stop ' INVALID power variogram'
            end if
            write(*,*) ' it,cc,ang[1,2,3] =  ',it(i),cc(i),
     +                   ang1(i),ang2(i),ang3(i)
            write(*,*) ' a1 a2 a3 = ',aa(i),aa1,aa2
            write(*,*) ' anis 1 2 = ',anis1(i),anis2(i)
      end do
      close(lin)
c
c Get ready to go:
c
      do is=1,nst(1)
            call setrot(ang1(is),ang2(is),ang3(is),anis1(is),anis2(is),
     +                  is,MAXROT,rotmat)
      end do
      call cova3(0.0,0.0,0.0,0.0,0.0,0.0,1,nst,MAXNST,c0,it,cc,aa,
     +                 1,MAXROT,rotmat,cmax,maxcov)
      open(lout,file=outfl,status='UNKNOWN')
c
c Set the variogram data, direction by direction up to ndir directions:
c
      do id=1,ndir
      xx  = -xoff(id)
      yy  = -yoff(id)
      zz  = -zoff(id)
      write(lout,100) id
 100  format('Model Variogram for Direction: ',i2)
      do il=1,nlag+1
            xx  = xx + xoff(id)
            yy  = yy + yoff(id)
            zz  = zz + zoff(id)
            call cova3(0.0,0.0,0.0,xx,yy,zz,1,nst,MAXNST,c0,it,cc,aa,
     +                 1,MAXROT,rotmat,cmax,cov)
            gam = maxcov - cov
            ro  = cov/maxcov
            h   = sqrt(max((xx*xx+yy*yy+zz*zz),0.0))
            write(lout,101) il,h,gam,ndir,cov,ro
            if(il.eq.1) then
                  x1 = xx + 0.0001*xoff(id)
                  y1 = yy + 0.0001*yoff(id)
                  z1 = zz + 0.0001*zoff(id)
                  call cova3(0.0,0.0,0.0,x1,y1,z1,1,nst,MAXNST,c0,it,cc,
     +                       aa,1,MAXROT,rotmat,cmax,cov)
                  gam = maxcov - cov
                  ro  = cov/maxcov
                  h   = sqrt(max((xx*xx+yy*yy+zz*zz),0.0))
                  write(lout,101) il,h,gam,ndir,cov,ro
            endif
 101        format(1x,i3,1x,f9.3,1x,f12.5,1x,i6,4(1x,f12.5))
      end do
      end do
c
c Finished:
c
      close(lout)
      write(*,9998) VERSION
 9998 format(/' VMODEL Version: ',f5.3, ' Finished'/)
      stop
 98   stop 'Error in parameter file somewhere'
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
      open(lun,file='vmodel.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for VMODEL',/,
     +       '                  *********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('vmodel.var                   ',
     +       '-file for variogram output')
      write(lun,12)
 12   format('2   40                       ',
     +       '-number of directions and lags')
      write(lun,13)
 13   format(' 0.0   0.0    0.5            ',
     +       '-azm, dip, lag distance')
      write(lun,14)
 14   format('90.0   0.0    0.5            ',
     +       '-azm, dip, lag distance')
      write(lun,15)
 15   format('2    0.2                     ',
     +       '-nst, nugget effect')
      write(lun,16)
 16   format('1    0.4  0.0   0.0   0.0    ',
     +       '-it,cc,ang1,ang2,ang3')
      write(lun,17)
 17   format('         10.0   5.0  10.0    ',
     +       '-a_hmax, a_hmin, a_vert')
      write(lun,18)
 18   format('1    0.4  0.0   0.0   0.0    ',
     +       '-it,cc,ang1,ang2,ang3')
      write(lun,19)
 19   format('         10.0   5.0  10.0    ',
     +       '-a_hmax, a_hmin, a_vert')

      close(lun)
      return
      end
