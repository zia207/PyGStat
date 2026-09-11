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
c                       2-D Kriging Program
c                       *******************
c
c The following Parameters control static dimensioning within okb2d:
c
c   MAXX      maximum nodes in X
c   MAXY      maximum nodes in Y
c   MAXDAT    maximum number of data points
c   MAXSAM    maximum number of data points to use in one kriging system
c   MAXDIS    maximum number of discretization points per block
c   MAXNST    maximum number of nested structures
c
c-----------------------------------------------------------------------
c
c User Adjustable:
c
      parameter(MAXNST =    4)
c
c Fixed:
c
      parameter(UNEST=-999.,EPSLON=0.0000001,VERSION=3.000)
c
c Variable Declaration:
c
      integer   it(MAXNST),test
      real      aa(MAXNST),cc(MAXNST),ang(MAXNST),anis(MAXNST)
c
c Common Block:
c
      common /parmfk/ nd,nx,ny,xmn,ymn,xsiz,ysiz,nxdis,nydis,ndmin,
     +                ndmax,radius,nst,c0,it,aa,cc,ang,anis,ktype,
     +                skmean,idbg,lout,ldbg
