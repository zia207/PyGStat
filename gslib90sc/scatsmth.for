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
      module dec_dy
      
      real, allocatable :: xval(:),yval(:),pval(:)
      
      end module
c
c
c
      program main
c-----------------------------------------------------------------------
c
c        Smooth a Bivariate Distribution with Simulated Annealing
c        ********************************************************
c
c INPUT/OUTPUT Parameters:
c
c   datafl      file with input data
c   ix,iy,iw    columns for X , Y , weight
c   smxfl       file with smoothed X distribution
c   ivr,iwt     columns for variable, weight
c   smyfl       file with smoothed Y distribution
c   ivr,iwt     columns for variable, weight
c   ilogx,ilogy log scaling for x and y (0=no,1=yes)
c   dbgfl       file for debug information
c   outfl       file for output
c   options:    maxpert, report, omin, seed
c   flags:      1=y,0=n: mar,cor,smt,bhist
c   weights:    weights: mar,cor,smt,bhist
c   nsmooth     smoothing Window Size (X and Y)
c   corr        correlation (-999 take from data)
c   nxq,nyq     number of X and Y Quan/BHist
c   npts.       number of points defining envelope (0=none)
c   x(),y(i)    points defining envelope
c
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use      dec_dy
      include 'scatsmth.inc'
c
c Read the parameters and reference data distribution:
c
      call readparm
c
c Determine which lag vectors in smoothing window:
c
      call getlag
c
c Smooth the cross plot:
c
      call smooth
c
c Compute the final correlation coefficient and write the final values:
c
      sx   = 0.0
      sy   = 0.0
      sxx  = 0.0
      syy  = 0.0
      sxy  = 0.0
      tcdf = 0.0
      do j=1,nclsy
            yy = ymin + real(j-1)*yinc
            do i=1,nclsx
                  xx = xmin + real(i-1)*xinc
                  if(ilogx.eq.1.and.ilogy.eq.1)
     +               write(lout,202) xx,yy,10**xx,10**yy,val(i,j)
                  if(ilogx.eq.1.and.ilogy.ne.1)
     +               write(lout,201) xx,yy,10**xx,val(i,j)
                  if(ilogx.ne.1.and.ilogy.eq.1)
     +               write(lout,201) xx,yy,10**yy,val(i,j)
                  if(ilogx.ne.1.and.ilogy.ne.1)
     +               write(lout,200) xx,yy,val(i,j)
 200              format(f10.4,1x,f10.4,1x,f9.6)
 201              format(f10.4,1x,f10.4,1x,f10.4,1x,f9.6)
 202              format(f10.4,1x,f10.4,1x,f10.4,1x,f10.4,1x,f9.6)
                  sx   = sx   + xx * val(i,j)
                  sy   = sy   + yy * val(i,j)
                  sxx  = sxx  + xx * xx * val(i,j)
                  syy  = syy  + yy * yy * val(i,j)
                  sxy  = sxy  + xx * yy * val(i,j)
                  tcdf = tcdf + val(i,j)
            end do
      end do
      close(lout)
c
c Report final statistics to the screen and the debug file:
c
      sx   = sx  / tcdf
      sy   = sy  / tcdf
      sxx  = sxx / tcdf - sx*sx
      syy  = syy / tcdf - sy*sy
      sxy  = sxy / tcdf
      corr = (sxy-sx*sy)/sqrt(max((sxx*syy),0.0))
      write(ldbg,205) sx,sxx,sy,syy,corr
      write(*,205)    sx,sxx,sy,syy,corr
 205  format(/,'   Final Statistics from SCATSMTH',/,
     +         '   X mean and variance   : ',2f12.4,
     +         '   Y mean and variance   : ',2f12.4,
     +         '   correlation coefficint: ',f12.4)
c
c Write the resulting marginals to a file for comparison/debugging:
c
      open(lout,file=xoutfl,status='UNKNOWN')
      write(lout,101)
 101  format('Target and Actual X Distributions from SCATSMTH',/,'3',
     +       /,'Value',/,'Target Probability',/,'Actual Probability')
      do i=1,nclsx
            xx = xmin + real(i-1)*xinc
            write(lout,'(f10.4,1x,f9.6,1x,f9.6)') xx,xuni(i),xunir(i)
      end do
      close(lout)
      open(lout,file=youtfl,status='UNKNOWN')
      write(lout,102)
 102  format('Target and Actual Y Distributions from SCATSMTH',/,'3',
     +       /,'Value',/,'Target Probability',/,'Actual Probability')
      do i=1,nclsy
            yy = ymin + real(i-1)*yinc
            write(lout,'(f10.4,1x,f9.6,1x,f9.6)') yy,yuni(i),yunir(i)
      end do
      close(lout)
c
c END Main loop:
c
      stop
      end



      subroutine readparm
c-----------------------------------------------------------------------
c
c
c
c
c
c
c-----------------------------------------------------------------------
      use      msflib
      use      dec_dy
      include 'scatsmth.inc'

      parameter (KORDEI=12,MAXOP1=KORDEI+1,MAXINT=2**30)

      real*8 acorni
      common/iaco/ ixv(MAXOP1)

      integer   swath
      common    x(MAXLIM),y(MAXLIM),yintvl(MAXLIM),intvls,
     +          swath(MAXLIM,50),rslope(MAXLIM)

      character xsmthfl*512,ysmthfl*512,datafl*512,outfl*512,
     +          dbgfl*512,str*512
      real      tmpval(50)
      logical   testfl
c
c Note VERSION number before anything else:
c
      lin  = 1
      lout = 2
      ldbg = 3
      lps  = 4
      write(*,9999) VERSION
 9999 format(/' SCATSMTH Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'scatsmth.par        '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'scatsmth.par        ') then
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
      write(*,*) ' reference data file = ',datafl(1:40)

      read(lin,*,err=97) ivrx,ivry,iwt
      write(*,*) ' columns = ',ivrx,ivry,iwt

      read(lin,'(a512)',err=97) xsmthfl 
      call chknam(xsmthfl,512)
      write(*,*) ' X smoothed dist. file = ',xsmthfl(1:40)

      read(lin,*,err=97) ixvr,ixwt
      write(*,*) ' columns = ',ixvr,ixwt

      read(lin,'(a512)',err=97) ysmthfl 
      call chknam(ysmthfl,512)
      write(*,*) ' Y smoothed dist. file = ',ysmthfl(1:40)

      read(lin,*,err=97) iyvr,iywt
      write(*,*) ' columns = ',iyvr,iywt

      read(lin,*,err=97) ilogx,ilogy
      write(*,*) ' log scaling flags = ',ilogx,ilogy

      read(lin,'(a512)',err=97) dbgfl 
      call chknam(dbgfl,512)
      write(*,*) ' debugging file = ',dbgfl(1:40)
      open(ldbg,file=dbgfl,status='UNKNOWN')

      read(lin,'(a512)',err=97) xoutfl 
      call chknam(xoutfl,512)
      write(*,*) ' X output file = ',xoutfl(1:40)

      read(lin,'(a512)',err=97) youtfl 
      call chknam(youtfl,512)
      write(*,*) ' Y output file = ',youtfl(1:40)

      read(lin,'(a512)',err=97) outfl 
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,*,err=97) xm,xr,omin,ixv(1)
      write(*,*) ' maxpert,report,omin = ',xm,xr,omin,ixv(1)
      do i=1,int(100.0*xm)
            zz = real(acorni(idum))
      end do

c
c Use the bivariate histogram option instead of quantiles
c
      lquan   = .false.
      sclquan = 1.0
      read(lin,*,err=97) i1,i2,i3,i4
      write(*,*) ' Components in objective func = ',i1,i2,i3,i4
                  lmarg = .false.
                  lcorr = .false.
                  lsmth = .false.
                  lbhst = .false.
      if(i1.ge.1) lmarg = .true.
      if(i2.ge.1) lcorr = .true.
      if(i3.ge.1) lsmth = .true.
      if(i4.ge.1) lbhst = .true.
      if(lquan.and.lbhst) then
            write(*,*) ' WARNING: can not simultaneously have quantiles'
            write(*,*) '          and bivariate histogram'
            write(*,*) ' PROCEEDING with bivariate histogram only'
            lbhst = .false.
      end if

      read(lin,*,err=97) sclmarg,sclcorr,sclsmth,sclbhst
      write(*,*) ' Scaling for objective func = ',sclmarg,sclcorr,
     +                                           sclsmth,sclbhst

      read(lin,*,err=97) nsmooth
      write(*,*) ' Size of smoothing half window = ',nsmooth
      if(nsmooth.gt.MAXSMT) stop 'nsmooth is too big'

      read(lin,*,err=97) corr
      write(*,*) ' Target correlation coefficient = ',corr

      read(lin,*,err=97) nqx,nqy
      write(*,*) ' number of data quantiles = ',nqx,nqy
      if(lquan.and.(nqx.gt.MAXQUA)) stop 'nqx too big'
      if(lquan.and.(nqy.gt.MAXQUA)) stop 'nqy too big'
      if(lbhst.and.(nqx.gt.(MAXQUA-1))) stop 'nqx too big'
      if(lbhst.and.(nqy.gt.(MAXQUA-1))) stop 'nqy too big'

      read(lin,*,err=97) nlim
      write(*,*) ' number of points in limits',nlim
      do i=1,nlim
            read(lin,*,err=97) x(i),y(i)
      end do
      if(nlim.le.0) then
            nlim = 4
            x(1) = -1.0e20
            y(1) = -1.0e20
            x(2) = -1.0e20
            y(2) =  1.0e20
            x(3) =  1.0e20
            y(3) =  1.0e20
            x(4) =  1.0e20
            y(4) = -1.0e20
      end if

      close(lin)
c
c Read in the smoothed X distribution:
c
      inquire(file=xsmthfl,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR: No smoothed X distribution file'
            stop
      endif
      open(lin,file=xsmthfl,status='UNKNOWN')
      read(lin,'(a)',err=98) str
      read(lin,*,err=98)     nvari
      do i=1,nvari
            read(lin,'()',err=98)
      end do
      nclsx = 0
 3    read(lin,*,end=4,err=98) (tmpval(j),j=1,nvari)
      nclsx       = nclsx + 1
      xx          = tmpval(ixvr)
      xuni(nclsx) = tmpval(ixwt)
      if(nclsx.eq.1) xmin = xx
      if(nclsx.eq.2) xinc = xx - xold
      xold = xx
      go to 3
 4    xmax = xold
      close(lin)
      write(*,101)    nclsx
      write(ldbg,101) nclsx
 101  format(//,' SCATSMTH: smoothed values in x: ',i5)
c
c Read in the smoothed X distribution:
c
      inquire(file=ysmthfl,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR: No smoothed Y distribution file'
            stop
      endif
      open(lin,file=ysmthfl,status='UNKNOWN')
      read(lin,'(a)',err=98) str
      read(lin,*,err=98)     nvari
      do i=1,nvari
            read(lin,'()',err=98)
      end do
      nclsy = 0
 5    read(lin,*,end=6,err=98) (tmpval(j),j=1,nvari)
      nclsy       = nclsy + 1
      yy          = tmpval(iyvr)
      yuni(nclsy) = tmpval(iywt)
      if(nclsy.eq.1) ymin = yy
      if(nclsy.eq.2) yinc = yy - yold
      yold = yy
      go to 5
 6    ymax = yold
      close(lin)
      write(*,102)    nclsy
      write(ldbg,102) nclsy
 102  format( /,' SCATSMTH: smoothed values in y: ',i5)
c
c Maximum perturbations....
c
      maxpert = int(xm*real(nclsx*nclsy))
      report  = int(xr*real(nclsx*nclsy))
c
c Read in the reference data distribution:
c
      inquire(file=datafl,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR: No data file'
            stop
      endif
c
c The first data file exists so open the file and read in the header
c information. Initialize the storage that will be used to summarize
c the data found in the file:
c
      open(lin,file=datafl,status='UNKNOWN')
      read(lin,*,err=98)
      read(lin,*,err=98)     nvari
      do i=1,nvari
            read(lin,*)
      end do
      maxdat = 0
 22   read(lin,*,end=33,err=98)(var(j),j=1,nvari)
      maxdat = maxdat + 1
      go to 22
 33   continue
c
c  Now allocate the needed memory:
c
      allocate (xval(maxdat),stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c
      allocate (yval(maxdat),stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c
      allocate (pval(maxdat),stat = test)
            if (test.ne.0) then
                  write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
                  stop
            end if
c
c  Now read the data for real:
c
      rewind(lin)
      read(lin,'(a)',err=98) str
      read(lin,*,err=98)     nvari
      do i=1,nvari
            read(lin,'()',err=98)
      end do
c
c Read as much data as possible:
c
      nd   = 0
      nt   = 0
      tcdf = 0
      sx   = 0.0
      sy   = 0.0
      sxx  = 0.0
      syy  = 0.0
      sxy  = 0.0
 7    read(lin,*,end=8,err=98) (tmpval(j),j=1,nvari)
c
c Trim this data?
c
      if(tmpval(ivrx).lt.xmin.or.tmpval(ivrx).ge.xmax.or.
     +   tmpval(ivry).lt.ymin.or.tmpval(ivry).ge.ymax)     then
            nt = nt + 1
            go to 7
      endif
      if(iwt.ge.1) then
            if(tmpval(iwt).le.EPSLON) then
                  nt = nt + 1
                  go to 7
            endif
      endif
c
c Accept this data:
c
      nd = nd + 1
      xval(nd) = tmpval(ivrx)
      yval(nd) = tmpval(ivry)
      if(iwt.ge.1) then
            pval(nd) = tmpval(iwt)
      else
            pval(nd) = 1.0
      endif
      tcdf = tcdf + pval(nd)
      sx   = sx  + xval(nd) * pval(nd)
      sy   = sy  + yval(nd) * pval(nd)
      sxx  = sxx + xval(nd) * xval(nd) * pval(nd)
      syy  = syy + yval(nd) * yval(nd) * pval(nd)
      sxy  = sxy + xval(nd) * yval(nd) * pval(nd)
      go to 7
 8    close(lin)
      xslp = (sx*sy - tcdf*sxy) / (sx*sx - tcdf*sxx)
      xint = (sy-sx*xslp)/tcdf
      write(*,104)    nd,xslp,xint
      write(ldbg,104) nd,xslp,xint
 104  format( /,' SCATSMTH: data values:          ',i5,
     +        /,'           slope:                ',f12.7,
     +        /,'           intercept:            ',f12.7)
c
c Get the slope through the cloud of points:
c
      sx   = sx / tcdf
      sy   = sy / tcdf
      sxm  = 0.0
      sym  = 0.0
      do i=1,nd
            sxm = sxm + (pval(nd) * (xval(nd)-sx)**2)
            sym = sym + (pval(nd) * (yval(nd)-sy)**2)
      end do
      xslp = sym / sxm
      write(*,105)    xslp
      write(ldbg,105) xslp
 105  format( /,' slope for smoothing window: ',f12.7,/)
c
c Set correlation coeficient if user set to missing:
c
      if(corr.lt.-1.0) then
            corr = (sxy/tcdf-sx/tcdf*sy/tcdf) / 
     +      sqrt(  max( ((sxx/tcdf-sx/tcdf*sx/tcdf)
     +            *(syy/tcdf-sy/tcdf*sy/tcdf)),0.0)  )
            write(*,*) 'Setting correlation coefficient to: ',corr
      end if

c
c Establish the quantiles:
c
c
c Quantile limits for the X variable:
c
      qinc = 1.0 / real(nqx+1)
      do iq=1,nqx
            qq = real(iq) / real(nqx+1)
            sp = 0.0
            do i=1,nclsx
                  sp = sp + xuni(i)  
                  if(sp.ge.qq) then
                        ind = max(min(i,nclsx),1)
                        qvalx(iq) = xmin + real(ind-1)*xinc
                        indqx(iq) = ind
                        go to 10
                  end if
            end do
 10         continue
      end do
c
c Quantile limits for the Y variable:
c
      qinc = 1.0 / real(nqy+1)
      do iq=1,nqy
            qq = real(iq) / real(nqy+1)
            sp = 0.0
            do i=1,nclsy
                  sp = sp + yuni(i)  
                  if(sp.ge.qq) then
                        ind = max(min(i,nclsy),1)
                        qvaly(iq) = ymin + real(ind-1)*yinc
                        indqy(iq) = ind
                        go to 11
                  end if
            end do
 11         continue
      end do
c
c Quantiles from data:
c
      if(lquan) then
            do iqx=1,nqx
            do iqy=1,nqy
                  qval(iqx,iqy) = 0.0
                  do i=1,nd
                     if(xval(i).le.qvalx(iqx).and.yval(i).le.qvaly(iqy))
     +                  qval(iqx,iqy) = qval(iqx,iqy) + pval(i)
                  end do
            end do
            end do
            do iqx=1,nqx
                  do iqy=1,nqy
                        qval(iqx,iqy) = qval(iqx,iqy) / tcdf
                  end do
            end do
      end if
c
c Bivariate histogram from data:
c
      if(lbhst) then
            qvalx(0)     = -1.0e21 
            qvalx(nqx+1) =  1.0e21
            qvaly(0)     = -1.0e21 
            qvaly(nqy+1) =  1.0e21
            do iqx=1,nqx+1
                  do iqy=1,nqy+1
                        qval(iqx,iqy) = 0.0
                        do i=1,nd
                              if( xval(i).gt.qvalx(iqx-1).and.
     +                            xval(i).le.qvalx(iqx)  .and.
     +                            yval(i).gt.qvaly(iqy-1).and.
     +                            yval(i).le.qvaly(iqy)         )
     +                        qval(iqx,iqy) = qval(iqx,iqy) + pval(i)
                  end do
               end do
            end do
            do iqx=1,nqx+1
                  do iqy=1,nqy+1
                        qval(iqx,iqy) = qval(iqx,iqy) / tcdf
                  end do
            end do
      end if
c
c Get the output file ready:
c
      open(lout,file=outfl,status='UNKNOWN')
      write(lout,200)
 200  format('Smooth Results')
      ncols = 3
      if(ilogx.eq.1) ncols = ncols + 1
      if(ilogy.eq.1) ncols = ncols + 1
      write(lout,201) ncols,nclsy,nclsx,1
 201  format(4(1x,i4))
      write(lout,202)
 202  format('X-value',/,'Y-value')
      if(ilogx.eq.1) write(lout,111)
      if(ilogy.eq.1) write(lout,112)
      write(lout,113)
 111  format('10**X-value')
 112  format('10**Y-value')
 113  format('P-value')
c
c END Main loop:
c
      return
 97   stop 'ERROR in parameter file!'
 98   stop 'ERROR in a data file!'
      end



      subroutine smooth
c-----------------------------------------------------------------------
c
c        Main subroutine that performs the Annealing Smoothing
c        *****************************************************
c
c
c
c
c
c-----------------------------------------------------------------------
      use       dec_dy
      include  'scatsmth.inc'
      real*8    acorni
      logical   inout
      integer   swath
c
c Common block for point-in-polygon routines:
c
      common x(MAXLIM),y(MAXLIM),yintvl(MAXLIM),intvls,
     +       swath(MAXLIM,50),rslope(MAXLIM)
c
c Initialize the envelope of non-zero probabilities:
c
      call preply(x,y,nlim,yintvl,intvls,swath,rslope)
c
c Initialize the p values:
c
      nposs = 0
      nin   = 0
      do iy=1-MAXSML,nclsy+MAXSML
            yy = ymin + real(iy-1)*yinc
            do ix=1-MAXSML,nclsx+MAXSML
                  xx = xmin + real(ix-1)*xinc
                  val(ix,iy) = 0.0
                  if(ix.ge.1.and.ix.le.nclsx.and.
     +               iy.ge.1.and.iy.le.nclsy) then
                        nposs = nposs + 1
                        if(inout(xx,yy)) then
                              nin = nin + 1
                              val(ix,iy) = real(acorni(idum))
                        end if
                  end if
            end do
      end do
      write(*,101)    nin,nposs
      write(ldbg,101) nin,nposs
 101  format( /,' Number of non-zero probabilities',i8,/,
     +          ' Number of possibilities         ',i8)
c
c Check sum of P values and renormalize:
c
      sump = 0.0
      do iy=1,nclsy
            do ix=1,nclsx
                  sump = sump + val(ix,iy)
            end do
      end do
      do iy=1,nclsy
            do ix=1,nclsx
                  val(ix,iy) = val(ix,iy) * 1.0 / sump
            end do
      end do
c
c Initialize:
c
      objmarg = 0.0
      objcorr = 0.0
      objsmth = 0.0
      objquan = 0.0
      objbhst = 0.0

      obtmarg = 0.0
      obtcorr = 0.0
      obtsmth = 0.0
      obtquan = 0.0
      obtbhst = 0.0

      snomarg = 0.0
      snocorr = 0.0
      snosmth = 0.0
      snoquan = 0.0
      snobhst = 0.0
c
c Establish initial objective function values:
c
      call initobj
c
c Scale factors:
c
      call getsclf
c
c Scale factors for reporting:
c
      if(lmarg) snomarg = 1.0 / objmarg
      if(lcorr) snocorr = 1.0 / objcorr
      if(lsmth) snosmth = 1.0 / objsmth
      if(lquan) snoquan = 1.0 / objquan
      if(lbhst) snobhst = 1.0 / objbhst
      write(*,103)    maxpert,report,omin
      write(ldbg,103) maxpert,report,omin
 103  format( /,' SCATSMTH: maxpert, report, and omin: ',2i8,f9.6)
c
c Loop until convergence or the stopping number:
c
      naccept = 0
      do iswap=1,maxpert
c
c Report?
c
            if(iswap.eq.1.or.(int(iswap/report)*report).eq.iswap) then
                  write(*,996)    iswap,real(objmarg) * snomarg,
     +                                  real(objcorr) * snocorr,
     +                                  real(objsmth) * snosmth,
     +                                  real(objbhst) * snobhst
                  write(ldbg,996) iswap,real(objmarg) * snomarg,
     +                                  real(objcorr) * snocorr,
     +                                  real(objsmth) * snosmth,
     +                                  real(objbhst) * snobhst
 996              format(' Status at',i10:,' m=',f8.5,' c=',f8.5,
     +                                     ' s=',f8.5,' b=',f8.5)
            end if
c
c Perturb:
c
            call getpert(ipertx,iperty,ptry)
c
c Update:
c
            obj    = objmarg * sclmarg 
     +             + objcorr * sclcorr
     +             + objsmth * sclsmth
     +             + objquan * sclquan
     +             + objbhst * sclbhst

            call object (ipertx,iperty,ptry)

            objtry = obtmarg * sclmarg 
     +             + obtcorr * sclcorr
     +             + obtsmth * sclsmth
     +             + obtquan * sclquan
     +             + obtbhst * sclbhst
c
c Decide whether or not to keep:
c
            if(objtry.lt.obj) then
                  del = ptry - val(ipertx,iperty)
                  val(ipertx,iperty) = ptry
                  objmarg = obtmarg
                  objcorr = obtcorr
                  objsmth = obtsmth 
                  objquan = obtquan
                  objbhst = obtbhst

                  if(lmarg) then
                        xunir(ipertx) = xunir(ipertx) + del
                        yunir(iperty) = yunir(iperty) + del
                  end if

                  if(lcorr) then
                        asumx   = tsumx
                        asumy   = tsumy
                        asumxx  = tsumxx
                        asumyy  = tsumyy
                        asumxy  = tsumxy
                  end if

                  if(lsmth) then
                        do is=1,nsmooth
                              ix = ipertx + ixs(is)
                              iy = iperty + iys(is)
                              if(ix.ge.1.and.ix.le.nclsx.and.
     +                           iy.ge.1.and.iy.le.nclsy) 
     +                        sval(ix,iy) = svali(is)
                        end do
                  end if

                  if(lquan) then
                        do iqx=1,nqx
                              do iqy=1,nqy
                                    qact(iqx,iqy) = qtry(iqx,iqy)
                              end do
                        end do
                  end if

                  if(lbhst) then
                        do iqx=1,nqx+1
                              do iqy=1,nqy+1
                                    qact(iqx,iqy) = qtry(iqx,iqy)
                              end do
                        end do
                  end if

            endif
c
c Go back for another perturbation?
c
            if(real(obj).lt.omin) go to 30
      end do
c
c Finished:
c
 30   continue
      write(*,996)    iswap,real(objmarg) * snomarg,
     +                      real(objcorr) * snocorr,
     +                      real(objsmth) * snosmth,
     +                      real(objbhst) * snobhst
      write(ldbg,996) iswap,real(objmarg) * snomarg,
     +                      real(objcorr) * snocorr,
     +                      real(objsmth) * snosmth,
     +                      real(objbhst) * snobhst
      call initobj
      write(*,996)    iswap,real(objmarg) * snomarg,
     +                      real(objcorr) * snocorr,
     +                      real(objsmth) * snosmth,
     +                      real(objbhst) * snobhst
      write(ldbg,996) iswap,real(objmarg) * snomarg,
     +                      real(objcorr) * snocorr,
     +                      real(objsmth) * snosmth,
     +                      real(objbhst) * snobhst
      return
      end



      subroutine getpert(ipertx,iperty,ptry)
c-----------------------------------------------------------------------
c
c                      Perturbation Mechanism
c                      **********************
c
c
c
c
c-----------------------------------------------------------------------
      use       dec_dy
      include  'scatsmth.inc'
      real*8    acorni
      logical   inout
      integer   swath
c
c Common block for point-in-polygon routines:
c
      common x(MAXLIM),y(MAXLIM),yintvl(MAXLIM),intvls,
     +       swath(MAXLIM,50),rslope(MAXLIM)
c
c Get a possible location:
c
 2    ipertx = 1 + int(real(acorni(idum))*nclsx)
      iperty = 1 + int(real(acorni(idum))*nclsy)
c
c Point in the envelope?
c
      xx = xmin + real(ipertx-1)*xinc
      yy = ymin + real(iperty-1)*yinc
      if(.not.inout(xx,yy)) go to 2
c
c decide how much to perturb:
c
      plowr = max(0.0,(0.8*val(ipertx,iperty)))
      puppr =          1.2*val(ipertx,iperty)
      ptry  = plowr + real(acorni(idum)) * (puppr-plowr)
c
c End Perturbation mechanism:
c
      return
      end



      subroutine getsclf
c-----------------------------------------------------------------------
c
c                             Scale Factors
c                             *************
c
c
c
c
c-----------------------------------------------------------------------
      use      dec_dy
      include 'scatsmth.inc'

      delmarg = 0.0
      delcorr = 0.0
      delsmth = 0.0
      delquan = 0.0
      delbhst = 0.0
c
c Establish weights by looping over a large number of random
c perturbations and keep track of how much each objective function
c component changes:
c
      do i=1,NUMSCL
            call getpert(ipertx,iperty,ptry)
            call object (ipertx,iperty,ptry)
            if(lmarg) delmarg  = delmarg  + abs(objmarg-obtmarg)
            if(lcorr) delcorr  = delcorr  + abs(objcorr-obtcorr)
            if(lsmth) delsmth  = delsmth  + abs(objsmth-obtsmth)
            if(lquan) delquan  = delquan  + abs(objquan-obtquan)
            if(lbhst) delbhst  = delbhst  + abs(objbhst-obtbhst)
      end do
c
c Establish each component scaling
c
      if(lmarg.and.delmarg.gt.0.0) then
            sclmarg = sclmarg / delmarg
      else
            sclmarg = 0.0
      end if
      if(lcorr.and.delcorr.gt.0.0) then
            sclcorr = sclcorr / delcorr
      else
            sclcorr = 0.0
      end if
      if(lsmth.and.delsmth.gt.0.0) then
            sclsmth = sclsmth / delsmth
      else
            sclsmth = 0.0
      end if
      if(lquan.and.delquan.gt.0.0) then
            sclquan = sclquan / delquan
      else
            sclquan = 0.0
      end if
      if(lbhst.and.delbhst.gt.0.0) then
            sclbhst = sclbhst / delbhst
      else
            sclbhst = 0.0
      end if
      resc    = 1.0 / (  sclmarg*objmarg
     +                 + sclcorr*objcorr
     +                 + sclsmth*objsmth
     +                 + sclquan*objquan
     +                 + sclbhst*objbhst )
      if(lmarg) sclmarg = resc * sclmarg
      if(lcorr) sclcorr = resc * sclcorr
      if(lsmth) sclsmth = resc * sclsmth
      if(lquan) sclquan = resc * sclquan
      if(lbhst) sclbhst = resc * sclbhst
c
c Return to annealing program:
c
      return
      end



      subroutine initobj
c-----------------------------------------------------------------------
c
c                  Objective Function Values
c                  *************************
c
c
c
c
c-----------------------------------------------------------------------
      use      dec_dy
      include 'scatsmth.inc'
      real*8   corrcalc
c
c Compute Smooth Sums:
c
      do iy=1,nclsy
      do ix=1,nclsx
            sval(ix,iy) = 0.0
            do is=1,nsmooth
                  iix = ix + ixs(is)
                  iiy = iy + iys(is)
                  sval(ix,iy) = sval(ix,iy) + val(iix,iiy)
            end do
      end do
      end do
      sfac = 1.0 / real(nsmooth)
      write(*,*)
      write(*,*) 'Smoothing Factor ',sfac
c
c Objective Function Component:        MARGINALS
c
      if(lmarg) then
            do ix=1,nclsx
                  xunir(ix) = 0.0
                  do iy=1,nclsy
                        xunir(ix) = xunir(ix) + val(ix,iy)
                  end do
            end do
            do iy=1,nclsy
                  yunir(iy) = 0.0
                  do ix=1,nclsx
                        yunir(iy) = yunir(iy) + val(ix,iy)
                  end do
            end do
            objmarg = 0.0
            do ix=1,nclsx
                  objmarg = objmarg + (xunir(ix)-xuni(ix))
     +                              * (xunir(ix)-xuni(ix))
            end do
            do iy=1,nclsy
                  objmarg = objmarg + (yunir(iy)-yuni(iy))
     +                              * (yunir(iy)-yuni(iy))
            end do
            write(*   ,101) objmarg
            write(ldbg,101) objmarg
 101        format(/' Marginal mismatch : ',f12.6)
      end if
c
c Objective Function Component:        CORRELATION COEFFICIENT:
c
      if(lcorr) then
            asumx  = 0.0
            asumy  = 0.0
            asumxx = 0.0
            asumyy = 0.0
            asumxy = 0.0
            do iy=1,nclsy
                  yy = ymin + real(iy-1)*yinc
                  do ix=1,nclsx
                        xx = xmin + real(ix-1)*xinc
                        asumx  = asumx  + dble(xx * val(ix,iy))
                        asumy  = asumy  + dble(yy * val(ix,iy))
                        asumxx = asumxx + dble(xx * xx * val(ix,iy))
                        asumyy = asumyy + dble(yy * yy * val(ix,iy))
                        asumxy = asumxy + dble(xx * yy * val(ix,iy))
                  end do
            end do
            corract = corrcalc(asumx,asumy,asumxx,asumyy,asumxy)
            objcorr  = real((corr-corract)**2)
            write(*   ,102) corract,corr
            write(ldbg,102) corract,corr
 102        format(/' Actual correlation: ',f12.6,' target is ',f12.6)
      end if
c
c Objective Function Component:        SMOOTHNESS
c
      if(lsmth) then
            objsmth = 0.0
            do iy=1,nclsy
            do ix=1,nclsx
                  objsmth = objsmth + (val(ix,iy)-sval(ix,iy)*sfac)
     +                              * (val(ix,iy)-sval(ix,iy)*sfac)
            end do
            end do
            write(*   ,104) objsmth
            write(ldbg,104) objsmth
 104        format(/' Measure of smoothness = ',f12.9)
      end if
c
c Objective Function Component:        QUANTILES
c
      if(lquan) then
            objquan = 0.0
            do iqx=1,nqx
            do iqy=1,nqy
                  qact(iqx,iqy) = 0.0
                  do iy=1,indqy(iqy)
                  do ix=1,indqx(iqx)
                        qact(iqx,iqy) = qact(iqx,iqy) + val(ix,iy)
                  end do
                  end do
                  objquan = objquan + (  (qact(iqx,iqy)-qval(iqx,iqy))
     +                                  *(qact(iqx,iqy)-qval(iqx,iqy)) )
            end do
            end do
            write(*   ,105) objquan/real(nqx*nqy)
            write(ldbg,105) objquan/real(nqx*nqy)
 105        format(/' Measure of mismatch for quantiles = ',f12.4,//)
      end if
c
c Objective Function Component:        BIVARIATE HISTOGRAM
c
      if(lbhst) then
            indqx(0)     = 0
            indqx(nqx+1) = nclsx
            indqy(0)     = 0
            indqy(nqy+1) = nclsy
            objbhst = 0.0
            do iqy=1,nqy+1
            do iqx=1,nqx+1
                  qact(iqx,iqy) = 0.0
                  do iy=(indqy(iqy-1)+1),indqy(iqy)
                  do ix=(indqx(iqx-1)+1),indqx(iqx)
                        qact(iqx,iqy) = qact(iqx,iqy) + val(ix,iy)
                  end do
                  end do
                  objbhst = objbhst + (  (qact(iqx,iqy)-qval(iqx,iqy))
     +                                  *(qact(iqx,iqy)-qval(iqx,iqy)) )
            end do
            end do
            write(*   ,106) objbhst/real((nqx+1)*(nqy+1))
            write(ldbg,106) objbhst/real((nqx+1)*(nqy+1))
 106        format(/' Measure of mismatch for biv histo = ',f12.4,//)
      end if
c
c Finished:
c
      write(*   ,'()')
      write(ldbg,'()')
      return
      end



      subroutine object(ipertx,iperty,valtry)
c-----------------------------------------------------------------------
c
c                  Update Objective Function Values
c                  ********************************
c
c
c
c
c-----------------------------------------------------------------------
      use      dec_dy
      include 'scatsmth.inc'
      real*8   corrcalc
c
c New tries:
c
      vali = val(ipertx,iperty)
      del  = valtry - vali
      xx   = xmin + real(ipertx-1)*xinc
      yy   = ymin + real(iperty-1)*yinc
c
c Objective Function Component:        MARGINALS
c
      if(lmarg) then
            obtmarg  = objmarg
            unitry   = xunir(ipertx)+del
            obtmarg  = obtmarg - (xunir(ipertx)-xuni(ipertx))
     +                         * (xunir(ipertx)-xuni(ipertx))
     +                         + (unitry       -xuni(ipertx))
     +                         * (unitry       -xuni(ipertx))
            unitry   = yunir(iperty)+del
            obtmarg  = obtmarg - (yunir(iperty)-yuni(iperty))
     +                         * (yunir(iperty)-yuni(iperty))
     +                         + (unitry       -yuni(iperty))
     +                         * (unitry       -yuni(iperty))
      end if
c
c Objective Function Component:        CORRELATION COEFFICIENT
c
      if(lcorr) then
            tsumx   = asumx  + dble( (valtry-vali)*xx    )
            tsumy   = asumy  + dble( (valtry-vali)*yy    )
            tsumxx  = asumxx + dble( (valtry-vali)*xx*xx )
            tsumyy  = asumyy + dble( (valtry-vali)*yy*yy )
            tsumxy  = asumxy + dble( (valtry-vali)*xx*yy )
            corrtry = corrcalc(tsumx,tsumy,tsumxx,tsumyy,tsumxy)
            obtcorr = real((corr-corrtry)**2)
      end if
c
c Objective Function Component:        SMOOTHNESS
c
      if(lsmth) then
            obtsmth = objsmth
            do is=1,nsmooth
                  locx = ipertx + ixs(is)
                  locy = iperty + iys(is)
                  if(locx.ge.1.and.locx.le.nclsx.and.
     +               locy.ge.1.and.locy.le.nclsy) then
                        svali(is) = sval(locx,locy) + del
                        obtsmth = obtsmth 
     +                  - (val(locx,locy)-sval(locx,locy)*sfac)
     +                  * (val(locx,locy)-sval(locx,locy)*sfac)
     +                  + (val(locx,locy)-svali(is)      *sfac)
     +                  * (val(locx,locy)-svali(is)      *sfac)
                  end if
            end do
            obtsmth = obtsmth - (vali  -sval(ipertx,iperty)*sfac)
     +                        * (vali  -sval(ipertx,iperty)*sfac)
     +                        + (valtry-sval(ipertx,iperty)*sfac)
     +                        * (valtry-sval(ipertx,iperty)*sfac)
      end if
c
c Objective Function Component:        QUANTILES
c
      if(lquan) then
            do iqy=1,nqy
            do iqx=1,nqx
                  qtry(iqx,iqy) = qact(iqx,iqy)
            end do
            end do
            do iqy=1,nqy
            do iqx=1,nqx
                  if(ipertx.le.indqx(iqx).and.iperty.le.indqy(iqy)) 
     +               qtry(iqx,iqy) = qtry(iqx,iqy) - vali + valtry
            end do
            end do

            obtquan = 0.0
            do iqy=1,nqy
            do iqx=1,nqx
                  obtquan = obtquan + (  (qtry(iqx,iqy)-qval(iqx,iqy))
     +                                  *(qtry(iqx,iqy)-qval(iqx,iqy)) )
            end do
            end do
      end if
c
c Objective Function Component:        BIVARIATE HISTOGRAM
c
      if(lbhst) then
            do iqy=1,nqy+1
            do iqx=1,nqx+1
                  qtry(iqx,iqy) = qact(iqx,iqy)
            end do
            end do

            do iqy=1,nqy+1
                  if(iperty.gt.indqy(iqy-1).and.iperty.le.indqy(iqy))
     +            iiy = iqy
            end do
            do iqx=1,nqx+1
                  if(ipertx.gt.indqx(iqx-1).and.ipertx.le.indqx(iqx))
     +            iix = iqx
            end do
            qtry(iix,iiy) = qtry(iix,iiy) - vali + valtry

            obtbhst = 0.0
            do iqy=1,nqy+1
            do iqx=1,nqx+1
                  obtbhst = obtbhst + (  (qtry(iqx,iqy)-qval(iqx,iqy))
     +                                  *(qtry(iqx,iqy)-qval(iqx,iqy)) )
            end do
            end do
      end if
c
c Finished:
c
      return
      end



      subroutine getlag
c-----------------------------------------------------------------------
c            Establish the Smoothing Window to Consider
c            ******************************************
c
c
c
c
c
c-----------------------------------------------------------------------
      use        dec_dy
      parameter(RAD2DEG=180.0/3.14159265)
      include   'scatsmth.inc'
      real       distarr(MAXSMT)
c
c Initialize the lag arrays, angles, and anisotropy parameters:
c
      do is=1,nsmooth
            ixs(is) = 0
            iys(is) = 0
      end do
c
c Set a 45 of -45 angle depending on slope of regression line:
c
      ang2  = 0.0
      ang3  = 0.0
      anis2 = 1.0
      ang1  = RAD2DEG*atan(xslp*xinc/yinc)
      anis1 =  0.25
c
c Find the closest "lags":
c
      na  = 0
      do ix=-MAXSML,MAXSML
      do iy=-MAXSML,MAXSML
            if(ix.eq.0.and.iy.eq.0) go to 2
            dx = real(ix) * xinc
            dy = real(iy) * yinc
            dx = real(ix)
            dy = real(iy)
            thedist = sqdist(0.0,0.0,0.0,dx,dy,0.0,ang1,ang2,ang3,
     +                       anis1,anis2)
            if(na.eq.nsmooth.and.thedist.ge.distarr(na)) go to 2
c
c Consider this sample (it will be added in the correct location):
c
            if(na.lt.nsmooth) na = na + 1
            ixs(na)     = ix
            iys(na)     = iy
            distarr(na) = thedist
            if(na.eq.1) go to 2
c
c Sort samples found thus far in increasing order of distance:
c
            n1 = na-1
            do ii=1,n1
                  k=ii
                  if(thedist.lt.distarr(ii)) then
                        jk = 0
                        do jj=k,n1
                              j  = n1-jk
                              jk = jk+1
                              j1 = j+1
                              distarr(j1) = distarr(j)
                              ixs(j1)     = ixs(j)
                              iys(j1)     = iys(j)
                        end do
                        distarr(k) = thedist
                        ixs(k)     = ix
                        iys(k)     = iy
                        go to 2
                  endif
            end do
 2    continue
      end do
      end do
c
c Debugging information:
c
      write(*,100)    nsmooth,ang1,anis1
      write(ldbg,100) nsmooth,ang1,anis1
 100  format(/' Smoothing Window: ',i3,' lag vectors with an ',/,
     +        '           Orientation = ',f7.2,' and anis = ',f8.5,/)
      do is=1,nsmooth
            write(ldbg,101) is,ixs(is),iys(is)
 101        format('    lag number ',i3,' X offset ',i4,' Y offset ',i4)
      end do
c
c Return with the closest lags:
c
      return
      end
 
 
 
      real function sqdist(x1,y1,z1,x2,y2,z2,ang1,ang2,ang3,anis1,anis2)
c-----------------------------------------------------------------------
c                   Anisotropic Distance Calculation
c                   ********************************
c
c This routine calculates the anisotropic distance between two points
c  given the coordinates of each point and a definition of the
c  anisotropy. The components of the vector in the rotated coordinates
c  are calculated and then the squared anisotropic distance is
c  calculated.
c
c
c INPUT VARIABLES:
c
c   x1,y1,z1         Coordinates of first point
c   x2,y2,z2         Coordinates of second point
c   ang1             Azimuth angle for the principal direction of
c                      continuity (measured clockwise in degrees from Y)
c   ang2             Dip angle for the principal direction of continuity
c                      (measured in negative degrees down)
c   ang3             Third rotation angle to rotate the two minor
c                      directions around the principal direction.  A
c                      positive angle acts clockwise while looking
c                      in the principal direction.
c   anis1            Anisotropy (radius in minor direction at 90
c                      degrees from ang1 divided by the principal radius
c                      in direction ang1)
c   anis2            Anisotropy (radius in minor direction at 90 degrees
c                      vertical from "ang1" divided by the principal
c                      radius in direction "ang1")
c
c
c
c OUTPUT VARIABLES:
c
c   sqdist           The squared distance accounting for the anisotropy
c                      and the rotation of coordinates (if any).
c
c
c
c PROGRAM NOTES:
c
c     1. The program converts the input (ang1,dip,plg) to three angles
c      which make more mathematical sense:
c
c         alpha   angle between the major axis of anisotropy and the
c                 E-W axis. Note: Counter clockwise is positive.
c         beta    angle between major axis and the horizontal plane.
c                 (The dip of the ellipsoid measured positive down)
c         theta   Angle of rotation of minor axis about the major axis
c                 of the ellipsoid.
c
c
c
c NO EXTERNAL REFERENCES
c
c
c--------------------------------------------------------------------
      parameter(DEG2RAD=3.14159265/180.0)
      real      rmatrx(3,3)
      save      rmatrx,ang1o,ang2o,ang3o,anis1o,anis2o
c
c Compute rotation matrix only if required:
c
      if(ang1.ne.ang1o.or.ang2.ne.ang2o.or.ang3.ne.ang3o.or.
     +   anis1.ne.anis1o.or.anis2.ne.anis2o) then
            ang1o  = ang1
            ang2o  = ang2
            ang3o  = ang3
            anis1o = anis1
            anis2o = anis2
            if(ang1.ge.0.0.and.ang1.lt.270.0) then
                  alpha = (90.0   - ang1) * DEG2RAD
            else
                  alpha = (450.0  - ang1) * DEG2RAD
            endif
            beta  = -1.0 * ang2 * DEG2RAD
            theta =        ang3 * DEG2RAD
            cosa  = cos(alpha)
            cosb  = cos(beta)
            cost  = cos(theta)
            sina  = sin(alpha)
            sinb  = sin(beta)
            sint  = sin(theta)
            rmatrx(1,1) =             (cosb * cosa)
            rmatrx(1,2) =             (cosb * sina)
            rmatrx(1,3) =             (-sinb)
            rmatrx(2,1) = (1.0/anis1)*(-cost*sina + sint*sinb*cosa)
            rmatrx(2,2) = (1.0/anis1)*(cost*cosa + sint*sinb*sina)
            rmatrx(2,3) = (1.0/anis1)*( sint * cosb)
            rmatrx(3,1) = (1.0/anis2)*(sint*sina + cost*sinb*cosa)
            rmatrx(3,2) = (1.0/anis2)*(-sint*cosa + cost*sinb*sina)
            rmatrx(3,3) = (1.0/anis2)*(cost * cosb)
      endif
c
c Compute component distance vectors and the squared distance:
c
      dx = x1 - x2
      dy = y1 - y2
      dz = z1 - z2
      sqdist = 0.0
      do 1 i=1,3
            temp   = rmatrx(i,1)*dx + rmatrx(i,2)*dy + rmatrx(i,3)*dz
            sqdist = sqdist + temp*temp
 1    continue
      return
      end



      double precision function corrcalc(sumx,sumy,sumxx,sumyy,sumxy)
c-----------------------------------------------------------------------
c
c-----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      
      divisor = (sumxx-sumx*sumx)*(sumyy-sumy*sumy)
      if(divisor.le.0.0) then
            corrcalc = 0.0
      else
            corrcalc=             (sumxy-sumx*sumy)
     +              / dsqrt(((sumxx-sumx*sumx)*(sumyy-sumy*sumy)))
      endif

      return
      end



c-----------------------------------------------------------------------
c
c                   Point-in Polygon Subroutines
c                   ****************************
c
c
c Adopted from: Computers & Geosciences Vol4 - by K. Salomon(1978)
c-----------------------------------------------------------------------



      subroutine preply(x,y,nuvert,yintvl,intvls,swath,rslope)
c-----------------------------------------------------------------------
c
c  this routine prepares the polygon consisting of the nuvert vertices
c  (x(i),y(i)) by first sorting the segment y-end points into
c  decteasing order and forming an interval for each consecutive pair:
c  (yintvl(i,,yintvl(i+1)),i=1,intvls.  this is performed by calling
c  sort subroutine.
c
c     the code consisting of the first two do loops constructs,
c  for each interval i, the list of segments to be tested by inout.
c  this list is placed into the i-th row of swath.  the first entry,
c  swath(i,1) will be set to the number of segments in the row.  note
c  that as yintvl contains no redundancies, i.e. yintvl(i) is strictly
c  greater than yintvl(i+1), no horizontal segments will be placed in
c  the list.
c
c     the code consisting of the second do loop establishes the
c  reciprocal slope for each non-horizontal segment.  this is to be
c  used by inout.  finally, the segments within a row of swath are
c  ordered form left to right.
c
c-----------------------------------------------------------------------

      include 'scatsmth.inc'

      integer swath(MAXLIM,50)
      real x(MAXLIM),y(MAXLIM),yintvl(MAXLIM),rslope(MAXLIM)
      call sortmm(y,nuvert,yintvl,intvls)
      if(intvls.le. 0) then
            write(*,*) ' ERROR prep of polygon aborted - no intervals'
            stop
      end if
      x(nuvert+1) = x(1)
      y(nuvert+1) = y(1)
      do i=1,intvls
            swath(i,1)=0
      end do
      do i=1,intvls
            do j=1,nuvert
                  if(y(j).ge.yintvl(i).and.yintvl(i+1).ge.y(j+1).or.
     +               y(j+1).ge.yintvl(i).and.yintvl(i+1).ge.y(j))
     +               call includ(swath,i,j)
            end do
      end do
      do i=1,nuvert
         if(y(i).ne.y(i+1)) rslope(i)=(x(i+1)-x(i))/(y(i+1)-y(i))
      end do
      call order(x,y,yintvl,intvls,swath,rslope)
      return
      end



      subroutine sortmm(y,nuvert,yintvl,intvls)
c-----------------------------------------------------------------------
c
c  routine establishes the intervals of the y-axis defined by the
c  endpoints of the polygon's segments. the first do loop initializes
c  ysort form the segment y=end points.  the second do loop sorts ysort
c  into descending order.  the third do loop elininates redundancies in
c  ysort and places irredundant sorted y's into yintvl. it also sets
c  intvls to the true number of y intervals. just prior to returing
c  a final interval extending to '-infinity' is established.
c
c-----------------------------------------------------------------------

      include 'scatsmth.inc'

      real y(MAXLIM),yintvl(MAXLIM),ysort(MAXLIM)
      integer upper
      do i=1,nuvert
            ysort(i)=y(i)
      end do
      upper=nuvert-1
      do i=1,upper
            ipls1=i+1
            do j=ipls1,nuvert
                  if(ysort(i).lt.ysort(j)) then
                        temp=ysort(i)
                        ysort(i)=ysort(j)
                        ysort(j)=temp
                  end if
            end do
      end do
      yintvl(1)=ysort(1)
      intvls=0
      upper=nuvert-1
      do i=1,upper
            if(ysort(i).ne.ysort(i+1)) then
                  intvls=intvls+1
                  yintvl(intvls+1)=ysort(i+1)
            end if
      end do
      yintvl(intvls+2)=-1.0e32
      return
      end



      subroutine includ(swath,i,j)
c-----------------------------------------------------------------------
c
c  routine places the j-th polygon segment into the next avalilable
c  location in row i of swath.
c
c-----------------------------------------------------------------------

      include 'scatsmth.inc'

      integer swath(MAXLIM,50),pointr
      swath(i,1)=swath(i,1) + 1
      pointr=swath(i,1)
      swath(i,pointr+1)=j
      return
      end



      subroutine order(x,y,yintvl,intvls,swath,rslope)
c-----------------------------------------------------------------------
c
c  for each interval, a horixontal line is passed through the middle
c  (ymid) of the interval.  the do 100 loop places the x-intersection
c  of each segment in this swath so that these intersectons occur
c  from left to right.
c
c-----------------------------------------------------------------------

      include 'scatsmth.inc'

      real x(MAXLIM),y(MAXLIM),yintvl(MAXLIM)
      real rslope(MAXLIM),xintsc(50)
      integer swath(MAXLIM,50),pointr,segno,upper
      logical vertsg
      do 200 intval=1,intvls
         nmbseg=swath(intval,1)
         ymid=(yintvl(intval)+yintvl(intval+1))/2.0
      do 100 pointr=1,nmbseg
            segno=swath(intval,pointr+1)
            vertsg=abs(x(segno+1)-x(segno)) .lt. 1.0e-5
            if(vertsg) xintsc(pointr)=x(segno)
            if(.not.vertsg) xintsc(pointr)=x(segno)+
     1                      rslope(segno)*(ymid-y(segno))
100   continue
      if(nmbseg.lt.2.or.mod(nmbseg,2).ne.0) go to 300
      upper=nmbseg-1
      do 200 i=1,upper
         ipls1=i+1
         do 200 j=ipls1,nmbseg
            if(xintsc(i).le.xintsc(j)) go to 200
            temp=xintsc(i)
            xintsc(i)=xintsc(j)
            xintsc(j)=temp
            itemp=swath(intval,i+1)
            swath(intval,i+1)=swath(intval,j+1)
            swath(intval,j+1)=itemp
200   continue
      return
300   write(6,301) intval
301   format(' *** error *** prep of polygon aborted. interval ',
     1 i5,' has either less than two segments or an odd number',
     2 ' of them')
      stop
      end



      logical function inout(xp,yp)
c-----------------------------------------------------------------------
c
c  the four lines enclosed in dashes determine the interval containing
c  yp. the do 400 looop continues until the first segment within the
c  interval falls to the left of (xp,yp). in this event, inout is set
c  .true. if an even number of segments had been tested.
c
c-----------------------------------------------------------------------

      include 'scatsmth.inc'

      common x(MAXLIM),y(MAXLIM),yintvl(MAXLIM),intvls,
     +       swath(MAXLIM,50),rslope(MAXLIM)
      integer swath,segno
      inout=.false.
      intval=0
100   intval=intval+ 1
      if(yintvl(intval).gt.yp) go to 100
      intval=intval -1
      if(intval.lt.1.or.intval.gt.intvls) return
      nmbseg=swath(intval,1) + 1
      do ii=2,nmbseg
            segno=swath(intval,ii)
            if(xp-x(segno).le.(yp-y(segno))*rslope(segno)) then
                  inout=mod(ii,2) .eq.1
                  return
            end if
      end do
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
      open(lun,file='scatsmth.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for SCATSMTH',/,
     +       '                  ***********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('data.dat                         ',
     +       '-file with data')
      write(lun,12)
 12   format('3   5   0                        ',
     +       '-  columns for X, Y, weight')
      write(lun,13)
 13   format('histsmth.sx                      ',
     +       '-file with smoothed X distribution')
      write(lun,14)
 14   format('1   2                            ',
     +       '-  columns for variable, weight')
      write(lun,15)
 15   format('histsmth.sy                      ',
     +       '-file with smoothed Y distribution')
      write(lun,16)
 16   format('1   2                            ',
     +       '-  columns for variable, weight')
      write(lun,17)
 17   format('0   0                            ',
     +       '-log scaling flags')
      write(lun,18)
 18   format('scatsmth.dbg                     ',
     +       '-file for debug information')
      write(lun,19)
 19   format('scatsmth.xr                      ',
     +       '-file for final X distribution')
      write(lun,20)
 20   format('scatsmth.yr                      ',
     +       '-file for final Y distribution')
      write(lun,21)
 21   format('scatsmth.out                     ',
     +       '-file for output')
      write(lun,22)
 22   format('150.  1.0   0.0001   69069       ',
     +       '-maxpert, report, min obj, seed')
      write(lun,23)
 23   format('1  1  1  1                       ',
     +       '-1=on,0=off: marg,corr,smth,quan')
      write(lun,24)
 24   format('1  1  2  2                       ',
     +       '-weighting : marg,corr,smt,hquan')
      write(lun,25)
 25   format(' 25                              ',
     +       '-smoothing Window Size (number)')
      write(lun,26)
 26   format('-999.                            ',
     +       '-correlation (-999 take from data)')
      write(lun,27)
 27   format('9  9                             ',
     +       '-number of X and Y quantiles')
      write(lun,28)
 28   format('0                                ',
     +       '-points defining envelope (0=none)')
      write(lun,29)
 29   format('  0.0    0.0                     ',
     +       '-   x(i)   y(i)')
      write(lun,30)
 30   format('  0.0  999.0                     ',
     +       '-   x(i)   y(i)')
      write(lun,31)
 31   format('999.0  999.0                     ',
     +       '-   x(i)   y(i)')
      write(lun,32)
 32   format('999.0    0.0                     ',
     +       '-   x(i)   y(i)')

      close(lun)
      return
      end
