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
c                 Boolean Simulation of Ellipsoids
c                 ********************************
c
c The program is executed with no command line arguments.  The user
c will be prompted for the name of a parameter file.  The parameter
c file is described in the documentation (see the example ellipsim.par)
c and should contain the following information:
c
c The output file will be a GEOEAS file containing the simulated integer
c code ordered by x, y, and then simulation.
c
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use       msflib
      parameter(MAXELP=25)
      parameter(KORDEI=12,MAXOP1=KORDEI+1,MAXINT=2**30,MAXROT=MAXELP,
     +          EPSLON=1.0e-10,VERSION=3.000)
c
c Variable declaration:
c
      real      aa(MAXELP),anis1(MAXELP),anis2(MAXELP),ang1(MAXELP),
     +          ang2(MAXELP),ang3(MAXELP),wt(MAXELP),tmp(50)
      real*8    rotmat(MAXROT,3,3),sdist,sqdist
      integer   test
      character outfl*512,str*512
      logical   testfl
c
c Declare dynamic arrays:
c
      integer,allocatable :: sim(:)
c
c For random number generator:
c
      real*8  acorni
      common /iaco/ ixv(MAXOP1)
      data   ixv/MAXOP1*0.0/
      data   lin/1/,lout/2/
c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' ELLIPSIM Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'ellipsim.par        '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'ellipsim.par        ') then
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
 1    read(lin,'(a4)',end=99) str(1:4)
      if(str(1:4).ne.'STAR') go to 1
c
c Read Input Parameters:
c
      write(*,*) ' Starting to read input parameters'

      read(lin,'(a512)',err=99) outfl
      call chknam(outfl,512)
      write(*,*) ' output file: ',outfl(1:40)

      read(lin,*,err=99) nsim
      write(*,*) ' number of realizations = ',nsim

      read(lin,*,err=99) nx,xmn,xsiz
      write(*,*) ' nx, xmn, and xsiz = ',nx,xmn,xsiz

      read(lin,*,err=99) ny,ymn,ysiz
      write(*,*) ' ny, ymn, and ysiz = ',ny,ymn,ysiz

      read(lin,*,err=99) nz,zmn,zsiz
      write(*,*) ' nz, zmn, and zsiz = ',nz,zmn,zsiz

      read(lin,*,err=99) ixv(1)
      write(*,*) ' random number seed = ',ixv(1)
      do i=1,10000
            zz = real(acorni(idum))
      end do

      read(lin,*,err=99) targprop
      write(*,*) ' target proportion = ',targprop
      targprop = targprop * nx * ny * nz

      nelp  = 0
      sumwt = 0.0
 2    read(lin,*,end=3,err=98) (tmp(i),i=1,7)
      nelp = nelp + 1
      aa(nelp)    = tmp(1)
      anis1(nelp) = tmp(2) / max(aa(nelp),EPSLON)
      anis2(nelp) = tmp(3) / max(aa(nelp),EPSLON)
      ang1(nelp)  = tmp(4)
      ang2(nelp)  = tmp(5)
      ang3(nelp)  = tmp(6)
      wt(nelp)    = tmp(7)
      sumwt       = sumwt + wt(nelp)
      dmax = max(dmax,aa(nelp))
      go to 2

 3    close(lin)
      if(nelp.lt.1.or.sumwt.lt.EPSLON) then
            write(*,*) 'ERROR: no ellipsoid geometries '
            stop
      end if
c
c Get CDF of ellipsoid specifications:
c
      do i=1,nelp
            wt(i) = wt(i) / sumwt
      end do
      wt(1) = wt(1) / sumwt
      do i=2,nelp
            wt(i) = wt(i-1) + wt(i)/sumwt
      end do
c
c Border zone:
c
      ixa = 1 + int(dmax/max(xsiz,EPSLON))
      iya = 1 + int(dmax/max(ysiz,EPSLON))
      iza = 1 + int(dmax/max(zsiz,EPSLON))
      write(*,*)
      write(*,*) 'Read in ',nelp,' possible ellipsoid geometries'
      write(*,*)
c
c Perform some quick error checking:
c
      nxyz = nx * ny * nz
c
c Set up the rotation/anisotropy matrices for each ellipsoid:
c
      write(*,*) 'Setting up rotation matrices for ellipsoids'
      write(*,*)
      do ielp=1,nelp
            call setrot(ang1(ielp),ang2(ielp),ang3(ielp),anis1(ielp),
     +                  anis2(ielp),ielp,MAXROT,rotmat)
      end do
      open(lout,file=outfl,status='UNKNOWN')
      write(lout,200)
 200  format('Output from ELLIPSIM')
      write(lout,201) 1,nx,ny,nz,xmn,ymn,zmn,xsiz,ysiz,zsiz,nsim
 201  format(i2,3(1x,i4),3(1x,g14.8),3(1x,g12.6),i4) 
      write(lout,202)
 202  format('binary code')
c
c Allocate the needed memory:
c
      allocate(sim(nxyz),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                  'insufficient memory.'
                  stop
            end if
c
c Loop over all simulations:
c
      do isim=1,nsim
            write(*,*) ' Starting simulation ',isim
            do i=1,nxyz
                  sim(i) = 0
            end do
            prop = 0.0
c
c Keep looping until target proportion is reached:
c
 20         if(prop.ge.targprop) go to 21
c
c Randomly choose the parameters for the ellipsoid:
c
            rnum = real(acorni(idum))
            do i=1,nelp
                  if(rnum.lt.wt(i)) then
                        ielp = i
                        go to 10
                  end if
            end do
 10         continue
            radsqd = aa(ielp) * aa(ielp)
            nxc = max(min((1+int(aa(ielp)/max(xsiz,EPSLON))),nx),1)
            nyc = max(min((1+int(aa(ielp)/max(ysiz,EPSLON))),ny),1)
            nzc = max(min((1+int(aa(ielp)/max(zsiz,EPSLON))),nz),1)
c
c Centroid of the ellipsoid:
c
            ixc = -ixa + int(real(acorni(idum)) * (nx+2*ixa))
            iyc = -iya + int(real(acorni(idum)) * (ny+2*iya))
            izc = -iza + int(real(acorni(idum)) * (nz+2*iza))
            if(nx.eq.1) ixc = 1
            if(ny.eq.1) iyc = 1
            if(nz.eq.1) izc = 1
c
c Fill the ellipsoid:
c
            do 40 ix=-nxc,nxc
                  i = ixc + ix
                  if(i.le.0.or.i.gt.nx) go to 40
                  do 41 iy=-nyc,nyc
                        j = iyc + iy
                        if(j.le.0.or.j.gt.ny) go to 41
                        do 42 iz=-nzc,nzc
                              k = izc + iz
                              if(k.le.0.or.k.gt.nz) go to 42

                              xx = real(ix) * xsiz
                              yy = real(iy) * ysiz
                              zz = real(iz) * zsiz

                              sdist = sqdist(0.,0.,0.,xx,yy,zz,
     +                                       ielp,MAXROT,rotmat)
                              if(real(sdist).lt.radsqd) then
                                    ind = i+(j-1)*nx+(k-1)*nx*ny
                                    if(sim(ind).eq.0) then
                                          sim(ind) = 1
                                          prop = prop + 1.
                                    endif
                              endif

 42                     continue
 41               continue
 40         continue
c
c Go back for another ellipse:
c
            go to 20
 21         continue
c
c Write out simulation and go back for another:
c
            do i=1,nxyz
                  write(lout,'(i1)') sim(i)
            end do
      end do
c
c Finished:
c
      close(lout)
      write(*,9998) VERSION
 9998 format(/' ELLIPSIM Version: ',f5.3, ' Finished'/)
      stop
 98   stop 'ERROR in data file'
 99   stop 'ERROR in parameter file'
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
      open(lun,file='ellipsim.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for ELLIPSIM',/,
     +       '                  ***********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('ellipsim.out              ',
     +       '-file for output realizations')
      write(lun,12)
 12   format('1                         ',
     +       '-number of realizations')
      write(lun,13)
 13   format('100  0. 1.0               ',
     +       '-nx,xmn,xsiz')
      write(lun,14)
 14   format('100  0. 1.0               ',
     +       '-ny,ymn,ysiz')
      write(lun,15)
 15   format('1    0. 1.0               ',
     +       '-nz,zmn,zsiz')
      write(lun,16)
 16   format('69069                     ',
     +       '-random number seed')
      write(lun,17)
 17   format('0.25                      ',
     +       '-target proportion (in ellipsoids)')
      write(lun,18)
 18   format(' 10.0  10.0   4.0  0.0 0.0 0.0   1.0 ',
     +       '-radius[1,2,3],angle[1,2,3],weight')
      write(lun,19)
 19   format('  2.0   2.0   2.0  0.0 0.0 0.0   1.0 ',
     +       '-radius[1,2,3],angle[1,2,3],weight')
      write(lun,20)
 20   format('  1.0   1.0   1.0  0.0 0.0 0.0   1.0 ',
     +       '-radius[1,2,3],angle[1,2,3],weight')

      close(lun)
      return
      end
