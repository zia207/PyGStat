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
c-----------------------------------------------------------------------
c
c The following Parameters control static dimensioning within sasim3d:
c
c   MAXX      maximum nodes in X
c   MAXY      maximum nodes in Y
c   MAXZ      maximum nodes in Z
c   MAXDAT    maximum number of cutoffs/data to define CDF
c   MAXHST    maximum number of classes in histogram objective function
c   MAXCUT    maximum number of indicator cutoffs
c   MAXLAG    maximum number of lags in variogram calculation
c   MAXNST    maximum number of structures for variogram model
c
c
c-----------------------------------------------------------------------
c
c User Adjustable Parameters:
c
      parameter (MAXNST =    2,MAXROT = MAXNST+1)
c
c Fixed Parameters:
c
      parameter(EPSLON=1.0e-10,BIGNUM=1.0e10,MAXOBJ=8,VERSION=3.000)
c
c ACORN parameters:
c
      parameter(KORDEI=12,MAXOP1=KORDEI+1,MAXINT=2**30)
c
c Variable declaration:
c
      integer   report,kasas,ksas,num,maxswap,nhist,ndhist,
     +          npricut,nseccut,nst(1),test
      real      tmin,tmax,omin,omin2,reltol,redfac,thnum,tnum,
     +          c0(1),userfac(MAXOBJ)
      real*8    objscl(MAXOBJ),sumsx,sumsy,sumsxx,sumsyy,sumsxy,
     +          sumtx,sumtxx,sumtxy
      character outfl*512
      logical   testhist,testvarg,testivar,testcorr,testcpdf,vertavg
c
c Common blocks:
c
      common /iaco/   ixv(MAXOP1)

      common /grid3d/ nx,ny,nz,xsiz,ysiz,zsiz,xmn,ymn,zmn

      common /genral/ lout,idbg,report,ldbg,nsim,t0,redfac,test,
     +                kasas,ksas,num,omin,omin2,reltol,maxswap,
     +                userfac,objscl,tmin,tmax,ilog,maxnochange

      common /secstf/ sumsx,sumsy,sumsxx,sumsyy,sumsxy,
     +                sumtx,sumtxx,sumtxy,corr,tnum

      common /variog/ nicut,isill,ndata,zmin,zmax,nlag,nhist,ndhist,
     +                thnum,nst,c0,cmax,npricut,nseccut

      common /charac/ outfl

      common /logics/ testhist,testvarg,testivar,testcorr,testcpdf,
     +                vertavg
