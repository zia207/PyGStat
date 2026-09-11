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
c Parameters:
c
      parameter(UNEST=-999.0,EPSLON=0.000001,MAXNST=4,MAXROT=MAXNST+ 1,
     +          VERSION=3.000)
c
c
c
      integer test
      real*8  unbias,rotmat(MAXROT,3,3)
c
c The data and other input variables:
c
      common /datcom/ nd,ktype,nvr,tmin,idbg,ldbg,lout,test,
     +                tmax,nx,ny,nz,xmn,ymn,zmn,xsiz,ysiz,zsiz
c
c Kriging parameters:
c
      common /krigcm/ ndmin,ndmaxp,ndmaxs,radiusp,radiuss,
     +                nxdis,nydis,nzdis
c
c Search variables and data for kriging:
c
      common /srccom/ sang1,sang2,sang3,sanisp1,sanisp2,isrot,saniss1,
     +                saniss2,radsqdp,radsqds,na,xas,yas,zas,vras,ndb
c
c Kriging systems (double precision arrays):
c
      common /krgsys/ unbias,rotmat
