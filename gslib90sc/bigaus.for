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
c          Indicator Variograms for BiGaussian Distribution
c          ************************************************
c
c
c This program returns the values of the theoretical indicator
c semivariograms corresponding to an input normal scores semivariogram
c
c INPUT/OUTPUT PARAMETERS:
c
c   outfl                  the output file for vargplt
c   ncut                   number of cutoffs
c   cut()                  the cutoffs (in cdf units) 
c   ndir,nlag              number of directions, number of lags
c   xoff,yoff,zoff         for each direction: the specification
c   nst,c0                 Normal scores variogram: nst, nugget
c   it,aa,cc               type, a parameter, c parameter
c   ang1,ang2,...,anis2    anisotropy definition
c
c
c
c PROGRAM NOTES:
c
c   1. Setting the number of cutoffs to a -1 times the number of
c      cutoffs will cause the variograms to be standardized to a sill 
c      of one.
c
c
c
c EXTERNAL REFERENCES:
c
c       f        to calculate exp(-zc**2/(1+x))/sqrt(1-x**2)
c       g        to calculate exp(-x**2/2)
c       simpson  to do the numerical calculation
c       gauinv   to get standard normal deviate
c
c
c
c Original: H. Xiao                                                 1975
c Modified: Clayton V. Deutsch                                 1989-1999
c-----------------------------------------------------------------------
      use       msflib
      external   f
      external   g
      parameter  (PI=3.14159265, DEG2RAD=PI/180.0, EPSLON = 1.0e-20,
     +            MLAG=5000, MDIR=10, MAXNST=4, MAXROT=4, VERSION=3.000)
     
      real       b, p, h(MLAG*MDIR), ri(MLAG*MDIR), ci(MLAG*MDIR),
     +           ro(MLAG*MDIR),gam(MLAG*MDIR),rop(MLAG*MDIR),maxcov,
     +           xoff(MDIR),yoff(MDIR),zoff(MDIR)
      real       c0(1),aa(MAXNST),cc(MAXNST),ang1(MAXNST),ang2(MAXNST),
     +           ang3(MAXNST),anis1(MAXNST),anis2(MAXNST)
      real*8     rotmat(MAXROT,3,3)
      integer    nst(1),it(MAXNST)
      character  outfl*512,str*512
      logical    testfl,corr
      common     i, zc(100)
      data       lin/1/,lout/2/,corr/.false./
c
c Note VERSION number before anything else:
c
      write(*,9999) VERSION
 9999 format(/' BIGAUS Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'bigaus.par          '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'bigaus.par          ') then
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
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,*,err=98) ncut
      write(*,*) ' number of cutoffs = ',ncut
      if(ncut.lt.0) then
            ncut = -ncut
            corr = .true.
      endif

      read(lin,*,err=98) (zc(i),i=1,ncut)
      write(*,*) ' cutoffs = ',(zc(i),i=1,ncut)

      read(lin,*,err=98) ndir,nlag
      write(*,*) ' ndir,nlag = ',ndir,nlag

      do i=1,ndir
            read(lin,*,err=98) azm,dip,xlag
            write(*,*) ' azm, dip, lag = ',azm,dip,xlag
            xoff(i) = sin(DEG2RAD*azm)*xlag*cos(DEG2RAD*dip)
            yoff(i) = cos(DEG2RAD*azm)*xlag*cos(DEG2RAD*dip)
            zoff(i) = sin(DEG2RAD*dip)*xlag
            write(*,*) ' x,y,z offsets = ',xoff(i),yoff(i),zoff(i)
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
      end do

      close(lin)
c
c Set up the rotation matrices:
c
      do is=1,nst(1)
            call setrot(ang1(is),ang2(is),ang3(is),anis1(is),anis2(is),
     +                 is,MAXROT,rotmat)
      end do
c
c Convert cutoffs cumulative probabilities to standard normal cutoffs:
c
      do icut=1,ncut
            call gauinv(dble(zc(icut)),xp,ierr)
            zc(icut) = xp
      end do
c
c Get ready to go:
c
      call cova3(0.0,0.0,0.0,0.0,0.0,0.0,1,nst,MAXNST,c0,it,cc,aa,
     +           1,MAXROT,rotmat,cmax,maxcov)
      open(lout,file=outfl,status='UNKNOWN')
c
c Set the variogram data, direction by direction up to ndir directions:
c
      i = 0
      do id=1,ndir
            xx  = -xoff(id)
            yy  = -yoff(id)
            zz  = -zoff(id)
            do il=1,nlag
                  xx  = xx + xoff(id)
                  yy  = yy + yoff(id)
                  zz  = zz + zoff(id)
                  call cova3(0.0,0.0,0.0,xx,yy,zz,1,nst,MAXNST,c0,it,
     +                       cc,aa,1,MAXROT,rotmat,cmax,cov)
                  i      = i + 1
                  gam(i) = maxcov - cov
                  ro(i)  = cov/maxcov
                  h(i)   = sqrt(xx*xx+yy*yy+zz*zz)
            end do
      end do
c
c Now, loop over all the cutoffs:
c
      do i=1,ncut
            ci0 = simpson(0.0, pi/2.0, 1.e-6, 40, f)
            p   = simpson(0.0,zc(i),1.e-6,40,g)
            p   = 0.5 + p/sqrt(2*PI)
            write(*,*) ' zc= ',zc(i),'p= ',p
            do l=1,ndir
                  write(lout,100) zc(i)
 100              format('Model Indicator Variogram: cutoff = ',f8.3,
     +                   ' Direction ',i2)
                  do j=1,nlag
                        ii = (l-1)*nlag + j
                        b = ro(ii)
                        b = asin(b)
                        ci(ii) = simpson (0.0, b, 1.e-6, 40, f)
                        ri(ii) = ( ci0-ci(ii)) / (2*pi)
                        rop(ii) = ri(ii)/(p*(1-p))
                        if(corr) then
                              write(lout,101) j,h(ii),rop(ii),l,
     +                                                ci(ii),ri(ii)
                        else
                              write(lout,101) j,h(ii),ri(ii),l,
     +                                                ci(ii),rop(ii)
                        endif
 101                    format(x,i3,1x,f8.3,1x,f12.3,1x,i6,4(1x,f12.5))
                  end do
            end do
      end do
c
c Finished:
c
      close(lout)
      write(*,9998) VERSION
 9998 format(/' BIGAUS Version: ',f5.3, ' Finished'/)
      stop
 98   stop 'Error in parameter file somewhere'
      end 





      real function f(x)
c-----------------------------------------------------------------------
c
c This function calculates the values of the function 
c       f( zc, xi)= exp ( -zc**2/ (1+sin(x))) for zc and x
c
c-----------------------------------------------------------------------
      real       x
      common    i, zc(100)
      f = exp ( -zc(i)**2 / (1+sin(x)))
      return
      end


      real function g(x)
      real x 
      g= exp ( - x**2/2.0)
      return
      end



        function simpson(alim,blim,tol,nitn,f)
c-----------------------------------------------------------------------
c
c       simpson performs numerical integration over the interval
c       a,b of the function f. the method employed is modified
c       simpson's rule. the procedure was adapted from algorithm
c       182 of "collected algorithms from acm" vol. 1.
c       the algorithm has been tested and found particularly useful
c       for integration of a strongly peaked function. the function
c       used for testing was not able to be suitably evaluated using
c       gauss quadrature or romberg integration.
c
c PARAMETERS:
c
c       a    -    lower limit of integration      (real)
c       b    -    upper limit of integration      (real)
c       eps  -    required tolerance            (real)
c       nitn -    maximum level of subdivision (integer)
c
c
c-----------------------------------------------------------------------
      external f
      double precision absarea
      parameter (num1 = 100)

      dimension dx(num1) , x2(num1) , x3(num1) , f2(num1) ,
     <            f3(num1) , epsp(num1) , f4(num1) , fmp(num1) ,
     <            fbp(num1) , est2(num1) , est3(num1) ,
     <            pval(num1,3) , irtrn(num1)
c
c       initialise the return level array.
c
      do 30 i = 1,num1
      irtrn(i) = 0
30    continue
      if(nitn.gt.50) stop 'Sorry, I only do 50 iterations!'

      a = alim
      b = blim
      eps = tol
      lvl = 0
      est = 1.0
      absarea = 1.0
      da = b - a
      fm = 4.0 * f((a+b)/2.0)
      fa = f(a)
      fb = f(b)

10    continue
      lvl = lvl + 1
      dx(lvl) = da / 3.0
      sx = dx(lvl) / 6.0
      f1 = 4.0 * f(a+dx(lvl)/2.0)
      x2(lvl) = a + dx(lvl)
      f2(lvl) = f(x2(lvl))
      x3(lvl) = x2(lvl) + dx(lvl)
      f3(lvl) = f(x3(lvl))
      epsp(lvl) = eps
      f4(lvl) = 4.0 * f(x3(lvl) + dx(lvl)/2.0)
      fmp(lvl) = fm
      est1 = ( fa + f1 + f2(lvl)) * sx
      fbp(lvl) = fb
      est2(lvl) = (f2(lvl) + f3(lvl) + fm) * sx
      est3(lvl) = (f3(lvl) + f4(lvl) + fb) * sx
      sum = est1 + est2(lvl) + est3(lvl)
      absarea = absarea - abs(est) + abs(est1) +
     <            abs(est2(lvl)) + abs(est3(lvl))

      if ((abs(est-sum) .le. epsp(lvl) * absarea
     <        .and. est .ne. 1.0)
     <        .or. lvl .ge. nitn) then
      lvl = lvl - 1
      pval(lvl,irtrn(lvl)) = sum
      ipoint = irtrn(lvl) + 1
      goto (1,2,3,4) ipoint
      endif

1     da = dx(lvl)
      fm = f1
      fb = f2(lvl)
      est = est1
      eps = epsp(lvl) / 1.7
      irtrn(lvl) = 1

      goto 10

2     da = dx(lvl)
      fa = f2(lvl)
      fm = fmp(lvl)
      fb = f3(lvl)
      eps = epsp(lvl) / 1.7
      est = est2(lvl)
      a = x2(lvl)
      irtrn(lvl) = 2

      goto 10

3     da = dx(lvl)
      fa = f3(lvl)
      fm = f4(lvl)
      fb = fbp(lvl)
      eps = epsp(lvl) / 1.7
      est = est3(lvl)
      a = x3(lvl)
      irtrn(lvl) = 3

      goto 10

4     sum = pval(lvl,1) + pval(lvl,2) + pval(lvl,3)

      if ( lvl .gt. 1 ) then
      lvl = lvl - 1
      pval(lvl,irtrn(lvl)) = sum
      ipoint = irtrn(lvl) + 1
      goto (1,2,3,4) ipoint
      endif

      simpson = sum
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
      open(lun,file='bigaus.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for BIGAUS',/,
     +       '                  *********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('bigaus.out                        ',
     +       '-file for output of variograms')
      write(lun,12)
 12   format('3                                 ',
     +       '-number of thresholds')
      write(lun,13)
 13   format('0.25   0.50    0.75               ',
     +       '-threshold cdf values')
      write(lun,14)
 14   format('1   20                            ',
     +       '-number of directions and lags')
      write(lun,15)
 15   format('0.0   0.0    1.0                  ',
     +       '-azm(1), dip(1), lag(1)')
      write(lun,16)
 16   format('2    0.2                          ',
     +       '-nst, nugget effect')
      write(lun,17)
 17   format('1    0.4  0.0   0.0   0.0         ',
     +       '-it,cc,ang1,ang2,ang3')
      write(lun,18)
 18   format('         10.0   5.0  10.0         ',
     +       '-a_hmax, a_hmin, a_vert')
      write(lun,19)
 19   format('1    0.4  0.0   0.0   0.0         ',
     +       '-it,cc,ang1,ang2,ang3')
      write(lun,20)
 20   format('         10.0   5.0  10.0         ',
     +       '-a_hmax, a_hmin, a_vert')

      close(lun)
      return
      end
