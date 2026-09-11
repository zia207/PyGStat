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

      real,allocatable    :: x(:),y(:),z(:),vr(:),close(:),
     +        sec1(:),sec2(:),sec3(:),tmp(:),vmean(:),c0(:),
     +        aa(:),ang1(:),ang2(:),ang3(:),anis1(:),anis2(:),
     +        cc(:),xa(:),ya(:),za(:),vra(:),xdb(:),ydb(:),zdb(:)
      real*8,allocatable  :: r(:),rr(:),s(:),a(:)
      integer,allocatable :: nst(:),it(:),iva(:),nisb(:),
     +        ixsbtosr(:),iysbtosr(:),izsbtosr(:)

      end module
c
c
c
      program main
c-----------------------------------------------------------------------
c
c               CoKriging of a 3-D Rectangular Grid
c               ***********************************
c
c This program estimates the value of a "primary" variable with primary
c and secondary data.  The program could be modified to jointly predict
c primary and secondary data.
c
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use       dec_dy      
      include  'cokb3d.inc'
c
c Read the Parameter File and the Data:
c
      call readparm(MAXVAR,MAXSBX,MAXSBY,MAXSBZ,MAXCOK)
c
c Call cokb3d to krige the grid:
c
      call cokb3d(MAXVAR,MAXSBX,MAXSBY,MAXSBZ,MAXCOK)
c
c Finished:
c
      write(*,9998) VERSION
 9998 format(/' COKB3D Version: ',f5.3, ' Finished'/)
      stop
      end



      subroutine readparm(MAXVAR,MAXSBX,MAXSBY,MAXSBZ,MAXCOK)
c-----------------------------------------------------------------------
c
c                  Initialization and Read Parameters
c                  **********************************
c
c The input parameters and data are read, some quick error checking is
c performed, and the statistics of all the variables being considered
c are written to standard output.
c
c
c
c-----------------------------------------------------------------------
c     use msflib
      use dec_dy
      include  'cokb3d.inc'
      parameter(MV=20)
      real      var(MV),av(MV),ss(MV)
      integer   ivrl(MV),nn(MV)
      character datafl*512,outfl*512,dbgfl*512,secfl*512,str*512
      logical   testfl,linmod,posdef
c
c I/O units:
c
      lin  = 1
      lout = 2
      ldbg = 3
c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' COKB3D Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'cokb3d.par          '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'cokb3d.par          ') then
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

      read(lin,*,err=98) nvr
      write(*,*) ' number of variables = ',nvr
      if(nvr.gt.4) stop 'can not use more than 3 secondary variables'

      read(lin,*,err=98) ixl,iyl,izl,(ivrl(i),i=1,nvr)
      write(*,*) ' columns = ',ixl,iyl,izl,(ivrl(i),i=1,nvr)

      read(lin,*,err=98) tmin,tmax
      write(*,*) ' trimming limits = ',tmin,tmax

      read(lin,*,err=98) icolloc
      write(*,*) ' co-located cokriging flag = ',icolloc
      if(icolloc.eq.1) then
            write(*,*)
            write(*,*)' The co-located cokriging flag does not work.'
            write(*,*)' Modify the search and ndmaxs for co-located.'
            write(*,*)' The original intent was for the program to'
            write(*,*)' establish the variograms using a Markov model.'
            write(*,*)' You can do that outside the program.'
            write(*,*)
            write(*,*)' Note: the collocated cokriging file is not used'
            write(*,*)
            stop
      end if

      read(lin,'(a512)',err=98) secfl
      call chknam(secfl,512)
      write(*,*) ' collocated cokriging file = ',secfl(1:40)

      read(lin,*,err=98) iclcol
      write(*,*) ' column for covariate = ',iclcol

      read(lin,*,err=98) idbg
      write(*,*) ' debug level = ',idbg

      read(lin,'(a512)',err=98) dbgfl
      call chknam(dbgfl,512)
      write(*,*) ' debug file = ',dbgfl(1:40)
      write(*,*)
      write(*,*) ' Some input parameters are now echoed to debug file'
      write(*,*)
      open(ldbg,file=dbgfl,status='UNKNOWN')

      read(lin,'(a512)',err=98) outfl
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,*,err=98) nx,xmn,xsiz
      write(*,*) ' nx, xmn, xsiz = ',nx,xmn,xsiz

      read(lin,*,err=98) ny,ymn,ysiz
      write(*,*) ' ny, ymn, ysiz = ',ny,ymn,ysiz

      read(lin,*,err=98) nz,zmn,zsiz
      write(*,*) ' nz, zmn, zsiz = ',nz,zmn,zsiz

      read(lin,*,err=98) nxdis,nydis,nzdis
      write(*,*) ' nxdis,nydis,nzdis = ',nxdis,nydis,nzdis

      read(lin,*,err=98) ndmin,ndmaxp,ndmaxs
      write(*,*) ' ndmin,ndmaxp,ndmaxs = ',ndmin,ndmaxp,ndmaxs

      read(lin,*,err=98) radiusp,radius1,radius2
      write(*,*) ' primary search radii = ',radiusp,radius1,radius2
      if(radiusp.lt.EPSLON) stop 'radius must be greater than zero'
      radsqdp = radiusp * radiusp
      sanisp1 = radius1 / radiusp
      sanisp2 = radius2 / radiusp

      read(lin,*,err=98) radiuss,radius1,radius2
      write(*,*) ' secondary search radii = ',radiuss,radius1,radius2
      if(radiuss.lt.EPSLON) stop 'radius must be greater than zero'
      radsqds = radiuss * radiuss
      saniss1 = radius1 / radiuss
      saniss2 = radius2 / radiuss

      read(lin,*,err=98) sang1,sang2,sang3
      write(*,*) ' search anisotropy angles = ',sang1,sang2,sang3

      read(lin,*,err=98) ktype
      write(*,*) ' kriging type = ',ktype
      if(ktype.lt.0.or.ktype.gt.2) stop ' ERROR: invalid kriging type'
c
c Find the needed parameters:
c
      MAXVAR = nvr
      MXVARG = MAXVAR * MAXVAR
c
c Allocate the needed memory:
c1
      allocate(nst(MXVARG),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c24
      allocate(vmean(MAXVAR),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
      read(lin,*,err=98) (vmean(i),i=1,nvr)
      write(*,*) ' variable means = ',(vmean(i),i=1,nvr)
c
c Now, initialize nst value to -1 to flag all missing variograms:
c
      do i=1,nvr
            do j=1,nvr
                  ind = i + (j-1)*(nvr)
                  nst(ind) = -1
            end do
      end do
      MAXDIS = nxdis * nydis * nzdis
      MAXSAM = ndmaxp + ndmaxs
      MAXCOK = MAXSAM * MAXVAR + MAXVAR
      MAXSBX = 1
      if(nx.gt.1)then
            MAXSBX = int(nx/2.00)
            if(MAXSBX.gt.50)MAXSBX=50
      end if
c
      MAXSBY = 1
      if(ny.gt.1)then
            MAXSBY = int(ny/2.00)
            if(MAXSBY.gt.50)MAXSBY=50
      end if
c
      MAXSBZ = 1
      if(nz.gt.1)then
            MAXSBZ = int(nz/2.00)
            if(MAXSBZ.gt.50)MAXSBZ=50
      end if
c
      MAXSB = MAXSBX * MAXSBY * MAXSBZ
c
c Allocate the needed memory:
c2
      allocate(it(MXVARG*MAXNST),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c3
      allocate(iva(MAXCOK),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c4
      allocate(nisb(MAXSB),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c5
      allocate(ixsbtosr(8*MAXSB),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c6
      allocate(iysbtosr(8*MAXSB),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c7
      allocate(izsbtosr(8*MAXSB),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c17
      allocate(xa(MAXCOK),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c18
      allocate(ya(MAXCOK),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c19
      allocate(za(MAXCOK),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c20
      allocate(vra(MAXCOK),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c21
      allocate(xdb(MAXDIS),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c22
      allocate(ydb(MAXDIS),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c23
      allocate(zdb(MAXDIS),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c25
      allocate(c0(MXVARG),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c26
      allocate(cc(MXVARG*MAXNST),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c27
      allocate(aa(MXVARG*MAXNST),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c28
      allocate(ang1(MXVARG*MAXNST),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c29
      allocate(ang2(MXVARG*MAXNST),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c30
      allocate(ang3(MXVARG*MAXNST),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c31
      allocate(anis1(MXVARG*MAXNST),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c32
      allocate(anis2(MXVARG*MAXNST),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c33
      allocate(r(MAXCOK),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c34
      allocate(rr(MAXCOK),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c35
      allocate(s(MAXCOK),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c36
      allocate(a(MAXCOK*MAXCOK),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
c Read as many variograms as are in the parameter file:
c
 3    read(lin,*,end=4,err=98) i,j
      ind = i + (j-1)*MAXVAR
      read(lin,*,err=98) nst(ind),c0(ind)
      write(ldbg,103) i,j,nst(ind),c0(ind)
      istart = 1 + (ind-1)*MAXNST
      do i=1,nst(ind)
            index = istart + i - 1
            read(lin,*,err=98) it(index),cc(index),ang1(index),
     +                         ang2(index),ang3(index)
            read(lin,*,err=98) aa(index),aa1,aa2
            anis1(index) = aa1 / max(aa(index),EPSLON)
            anis2(index) = aa2 / max(aa(index),EPSLON)
            if(it(index).eq.4.and.ktype.eq.0) 
     +                         stop 'No Power model with SK'
      end do
      write(ldbg,104) (it(istart+i-1),   i=1,nst(ind))
      write(ldbg,105) (aa(istart+i-1),   i=1,nst(ind))
      write(ldbg,106) (cc(istart+i-1),   i=1,nst(ind))
      write(ldbg,107) (ang1(istart+i-1), i=1,nst(ind))
      write(ldbg,108) (ang2(istart+i-1), i=1,nst(ind))
      write(ldbg,109) (ang3(istart+i-1), i=1,nst(ind))
      write(ldbg,110) (anis1(istart+i-1),i=1,nst(ind))
      write(ldbg,111) (anis2(istart+i-1),i=1,nst(ind))
 103  format(/,' USER input variogram for variables ',i2,' and ',i2,/,
     +       '      number of structures=',i2,' nugget effect=',f12.4)
 104  format('      types of structures: ',10i2)
 105  format('      aa values:           ',10f12.4)
 106  format('      cc values:           ',10f12.4)
 107  format('      ang1 values:         ',10f12.4)
 108  format('      ang2 values:         ',10f12.4)
 109  format('      ang3 values:         ',10f12.4)
 110  format('      anis1 values:        ',10f12.4)
 111  format('      anis2 values:        ',10f12.4)
      go to 3
 4    close(lin)
c
c Fill in cross variograms j=i if they have not been explicitly entered:
c
      do i=1,nvr
      do j=1,nvr
            ind1 = i + (j-1)*MAXVAR
            ind2 = j + (i-1)*MAXVAR
            if(nst(ind1).eq.-1.and.nst(ind2).eq.-1) then
                  write(*,*) ' Need variogram between variables ',i,j
                  stop
            end if
            if(nst(ind1).eq.-1) then
                  nst(ind1) = nst(ind2)
                  c0(ind1)  = c0(ind2)
                  istart1   = 1 + (ind1-1)*MAXNST
                  istart2   = 1 + (ind2-1)*MAXNST
                  do ist=1,nst(ind1)
                        index2        = istart2 + ist - 1
                        index1        = istart1 + ist - 1
                        it(index1)    = it(index2)
                        cc(index1)    = cc(index2)
                        aa(index1)    = aa(index2)
                        ang1(index1)  = ang1(index2)
                        ang2(index1)  = ang2(index2)
                        ang3(index1)  = ang3(index2)
                        anis1(index1) = anis1(index2)
                        anis2(index1) = anis2(index2)
                  end do
            else if(nst(ind2).eq.-1) then
                  nst(ind2) = nst(ind1)
                  c0(ind2)  = c0(ind1)
                  istart1   = 1 + (ind1-1)*MAXNST
                  istart2   = 1 + (ind2-1)*MAXNST
                  do ist=1,nst(ind2)
                        index2        = istart2 + ist - 1
                        index1        = istart1 + ist - 1
                        it(index2)    = it(index1)
                        cc(index2)    = cc(index1)
                        aa(index2)    = aa(index1)
                        ang1(index2)  = ang1(index1)
                        ang2(index2)  = ang2(index1)
                        ang3(index2)  = ang3(index1)
                        anis1(index2) = anis1(index1)
                        anis2(index2) = anis2(index1)
                  end do
            end if
      end do
      end do
c
c Has the linear model of coregionalization been used?
c
      linmod = .true.
      do i=1,nvr
      do j=1,nvr
            ind1 = i + (j-1)*MAXVAR
            do i2=1,nvr
            do j2=1,nvr
                  ind2 = i2 + (j2-1)*MAXVAR
                  if(nst(ind1).ne.nst(ind2)) linmod = .false.
                  istart1 = 1 + (ind1-1)*MAXNST
                  istart2 = 1 + (ind2-1)*MAXNST
                  do ist=1,nst(ind1)
                     index2 = istart2 + ist - 1
                     index1 = istart1 + ist - 1
                     if(it(index1).ne.it(index2).or.
     +                abs(aa(index1)    - aa(index2)).gt.EPSLON.or.
     +                abs(ang1(index1)  - ang1(index2)).gt.EPSLON.or.
     +                abs(ang2(index1)  - ang2(index2)).gt.EPSLON.or.
     +                abs(ang3(index1)  - ang3(index2)).gt.EPSLON.or.
     +                abs(anis1(index1) - anis1(index2)).gt.EPSLON.or.
     +                abs(anis2(index1) - anis2(index2)).gt.EPSLON)
     +                linmod = .false.
                  end do
            end do
            end do
      end do
      end do
      if(linmod) then
c
c Yes, the linear model of coregionalization has been used, now check
c to ensure positive definiteness:
c
            posdef = .true.
            do i=1,nvr
            do j=1,nvr
               if(i.ne.j) then
                  ii = i+(i-1)*MAXVAR
                  jj = j+(j-1)*MAXVAR
                  ij = i+(j-1)*MAXVAR
                  ji = j+(i-1)*MAXVAR
                  istartii = 1 + (ii-1)*MAXNST
                  istartjj = 1 + (jj-1)*MAXNST
                  istartij = 1 + (ij-1)*MAXNST
                  istartji = 1 + (ji-1)*MAXNST
c
c First check the nugget effects:
c
                  if(c0(ii).le.0.0.or.c0(jj).le.0.0.or.
     +              (c0(ii)*c0(jj)).lt.(c0(ij)*c0(ji)) ) then
                        posdef = .false.
                        write(ldbg,120) i,j
                  endif
                  do ist=1,nst(ii)
                        indexii = istartii + ist - 1
                        indexjj = istartjj + ist - 1
                        indexij = istartij + ist - 1
                        indexji = istartji + ist - 1
                        if(cc(indexii).le.0.0.or.cc(indexjj).le.0.0.or.
     +                    (cc(indexii)*cc(indexjj)).lt.
     +                    (cc(indexij)*cc(indexji)) ) then
                              posdef = .false.
                              write(ldbg,121) ist,i,j
                        endif
                  end do
               end if
            end do
            end do
 120        format(/,'Positive definiteness violation on nugget effects'
     +            ,/,' between ',i2,' and ',i2)
 121        format(/,'Positive definiteness violation on structure ',i2
     +            ,/,' between ',i2,' and ',i2)
c
c The model is not positive definite:
c
            if(.not.posdef) then
            write(*,*)
            write(*,*) ' The linear model of coregionalization is NOT'
            write(*,*) ' positive definite! This could lead to singular'
            write(*,*) ' matrices and unestimated points.'
            write(*,*)
            stop
c           write(*,*) ' Do you want to proceed? (y/n)'
c           read (*,'(a40)') str
c           if(str(1:1).ne.'y'.and.str(1:1).ne.'Y') stop
            end if
      else
c
c No linear model of coregionalization:
c
            write(*,*)
            write(*,*) ' A linear model of coregionalization has NOT'
            write(*,*) ' been used!!  This could lead to many singular'
            write(*,*) ' matrices and unestimated points.'
            write(*,*)
            stop
c           write(*,*) ' Do you want to proceed? (y/n)'
c           read (*,'(a40)') str
c           if(str(1:1).ne.'y'.and.str(1:1).ne.'Y') stop
      endif
c
c Perform some quick error checking:
c
      if(ndmin .le.0)      stop ' NDMIN too small'
      if((ndmaxs/2).le.nvr.and.ktype.eq.2) then
            write(*,100) nvr,ndmaxs
 100        format('WARNING: with traditional ordinary cokriging the ',
     +           /,'sum of the weights applied to EACH secondary data',
     +           /,'is zero.  With ndmaxs set low and nvr large the',
     +           /,'secondary data will not contribute to the estimate')
      endif
c
c Check to make sure the data file exists, then either read in the
c data or write an error message and stop:
c
      inquire(file=datafl,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR data file ',datafl,' does not exist!'
            stop
      endif
c
c The data file exists so open the file and read in the header
c information. Initialize the storage that will be used to summarize
c the data found in the file:
c
      open(lin,file=datafl,status='OLD')
      read(lin,*,err=99)
      read(lin,*,err=99)       nvari
      do i=1,nvari
            read(lin,*)
      end do
      maxdat = 0
 22   read(lin,*,end=44,err=99)(var(j),j=1,nvari)
      maxdat = maxdat + 1
      go to 22
 44   continue
 
c
c Allocate the needed memory:
c8
      allocate(x(maxdat),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c9
      allocate(y(maxdat),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c10
      allocate(z(maxdat),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c11
      allocate(vr(maxdat),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c12
      allocate(close(maxdat),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c13
      allocate(sec1(maxdat),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c14
      allocate(sec2(maxdat),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c15
      allocate(sec3(maxdat),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c16
      allocate(tmp(maxdat),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed',
     +                  ' due to insufficient memory.'
                  stop
            end if
c
      rewind(lin)
      read(lin,'(a40)',err=99) str
      read(lin,*,err=99)       nvari
      do i=1,nvari
            read(lin,'()',err=99)
      end do
      do i=1,nvr
            nn(i) = 0
            av(i) = 0.0
            ss(i) = 0.0
      end do
c
c Some tests on column numbers:
c
      if(ixl.gt.nvari.or.iyl.gt.nvari.or.izl.gt.nvari.or.
     +   ivrl(1).gt.nvari) then
            write(*,*) 'There are only ',nvari,' columns in input data'
            write(*,*) '  your specification is out of range'
            stop
      end if
c
c Read all the data until the end of the file:
c
      nd = 0
 7    read(lin,*,end=9,err=99) (var(j),j=1,nvari)
      nd = nd + 1
c
c Store data values (all secondary data must be transformed such that
c their mean is the same as the primary variable (if the first type of
c ordinary kriging is being used)):
c
      vr(nd) = var(ivrl(1))
      if(vr(nd).ge.tmin.and.vr(nd).lt.tmax) then
            nn(1) = nn(1) + 1
            av(1) = av(1) + vr(nd)
            ss(1) = ss(1) + vr(nd)*vr(nd)
      endif
      if(nvr.ge.2) then
            sec1(nd) = var(ivrl(2))
            if(sec1(nd).ge.tmin.and.sec1(nd).lt.tmax) then
                  nn(2) = nn(2) + 1
                  av(2) = av(2) + sec1(nd)
                  ss(2) = ss(2) + sec1(nd)*sec1(nd)
            endif
      end if
      if(nvr.ge.3) then
            sec2(nd) = var(ivrl(3))
            if(sec2(nd).ge.tmin.and.sec2(nd).lt.tmax) then
                  nn(3) = nn(3) + 1
                  av(3) = av(3) + sec2(nd)
                  ss(3) = ss(3) + sec2(nd)*sec2(nd)
            endif
      end if
      if(nvr.ge.4) then
            sec3(nd) = var(ivrl(4))
            if(sec3(nd).ge.tmin.and.sec3(nd).lt.tmax) then
                  nn(4) = nn(4) + 1
                  av(4) = av(4) + sec3(nd)
                  ss(4) = ss(4) + sec3(nd)*sec3(nd)
            endif
      end if
c
c Assign the coordinate location of this data:
c
      if(ixl.le.0) then
            x(nd) = xmn
      else
            x(nd) = var(ixl)
      endif
      if(iyl.le.0) then
            y(nd) = ymn
      else
            y(nd) = var(iyl)
      endif
      if(izl.le.0) then
            z(nd) = zmn
      else
            z(nd) = var(izl)
      endif
      go to 7
 9    close(lin)
c
c Compute the averages and variances as an error check for the user:
c
      do i=1,nvr
            av(i) = av(i) / max(real(nn(i)),1.0)
            ss(i) =(ss(i) / max(real(nn(i)),1.0)) - av(i) * av(i)
            write(*,*) 'COKB3D Variable ',i,' in data file: ',ivrl(i)
            write(*,*) '  Number   = ',nn(i)
            write(*,*) '  Average  = ',av(i)
            write(*,*) '  Variance = ',ss(i)
      end do
c
c Open output files and write headers:
c
      open(lout,file=outfl,status='UNKNOWN')
      write(lout,201) str
 201  format('COKB3D with:',a40)
      write(lout,202) 2,nx,ny,nz,xmn,ymn,zmn,xsiz,ysiz,zsiz,nreal
 202  format(i2,3(1x,i4),3(1x,g14.8),3(1x,g12.6),i4) 
      write(lout,203)
 203  format('estimate',/,'estimation variance')

      write(ldbg,204) str
 204  format(/,'DEBUGGING COKB3D with:',a40)
      return
c
c Error in an Input File Somewhere:
c
 98   stop 'ERROR in parameter file!'
 99   stop 'ERROR in data file!'
      end



      subroutine cokb3d(MAXVAR,MAXSBX,MAXSBY,MAXSBZ,MAXCOK)
c-----------------------------------------------------------------------
c
c                 CoKriging of a 3-D Rectangular Grid
c                 ***********************************
c
c This subroutine estimates point or block values of one variable by
c ordinary cokriging using up to MAXVAR variables.
c
c
c
c Original:  A.J. Desbarats                                         1984
c-----------------------------------------------------------------------
      use       dec_dy      
      include  'cokb3d.inc'
      parameter(PMX=999.)
c
c Set up the search and covariance rotation matrices:
c

      do is=1,nst(1)
            call setrot(ang1(is),ang2(is),ang3(is),anis1(is),anis2(is),
     +                  is,MAXROT,rotmat)
      end do
      isrot = MAXNST + 1
      call setrot(sang1,sang2,sang3,sanisp1,sanisp2,isrot,MAXROT,rotmat)
c
c Set up for super block searching:
c
      nsec = nvr - 1
      write(*,*) 'Setting up super block search strategy'
      call setsupr(nx,xmn,xsiz,ny,ymn,ysiz,nz,zmn,zsiz,nd,x,y,z,
     +             vr,tmp,nsec,sec1,sec2,sec3,MAXSBX,MAXSBY,MAXSBZ,nisb,
     +             nxsup,xmnsup,xsizsup,nysup,ymnsup,ysizsup,nzsup,
     +             zmnsup,zsizsup)
      call picksup(nxsup,xsizsup,nysup,ysizsup,nzsup,zsizsup,
     +             isrot,MAXROT,rotmat,radsqdp,nsbtosr,ixsbtosr,
     +             iysbtosr,izsbtosr)
c
c Set up the discretization points per block.  Figure out how many
c are needed, the spacing, and fill the xdb, ydb and zdb arrays with
c the offsets relative to the block center (this only gets done once):
c
      ndb  = nxdis * nydis * nzdis
      xdis = xsiz  / max(real(nxdis),1.0)
      ydis = ysiz  / max(real(nydis),1.0)
      zdis = zsiz  / max(real(nzdis),1.0)
      xloc = -0.5*(xsiz+xdis)
      i    = 0
c
      do ix =1,nxdis
            xloc = xloc + xdis
            yloc = -0.5*(ysiz+ydis)
            do iy=1,nydis
                  yloc = yloc + ydis
                  zloc = -0.5*(zsiz+zdis)
                  do iz=1,nzdis
                        zloc = zloc + zdis
                        i = i+1
                        xdb(i) = xloc
                        ydb(i) = yloc
                        zdb(i) = zloc
                  end do
            end do
      end do
c
c Initialize accumulators:
c
      uk   = 0.0
      vk   = 0.0
      nk   = 0
c
c Calculate Block Covariance. Check for point kriging.
c
      call cova3(0.,0.,0.,0.,0.,0.,1,nst,MAXNST,c0,it,cc,aa,
     +           1,MAXROT,rotmat,cmax,cova)
      unbias = dble(cova)
      unbias = 1.0
      if(ndb.le.1) then
            cbb = cova
      else
            cbb = 0.0
            do i=1,ndb
            do j=1,ndb
                  call cova3(xdb(i),ydb(i),zdb(i),xdb(j),ydb(j),zdb(j),
     +                       1,nst,MAXNST,c0,it,cc,aa,1,MAXROT,
     +                       rotmat,cmax,cova)
                  if(i.eq.j) cova = cmax - c0(1)
                  cbb = cbb + cova
            end do
            end do
            cbb = cbb/real(ndb*ndb)
      endif
      write(ldbg,*) 'Block average covariance ',cbb
c
c MAIN LOOP OVER ALL THE BLOCKS IN THE GRID:
c
      do 4 iz=1,nz
      zloc = zmn + (iz-1)*zsiz
      do 4 iy=1,ny
      yloc = ymn + (iy-1)*ysiz
      do 4 ix=1,nx
      xloc = xmn + (ix-1)*xsiz
c
c Find the nearest samples:
c
      call srchsupr(xloc,yloc,zloc,radsqdp,isrot,MAXROT,rotmat,nsbtosr,
     +              ixsbtosr,iysbtosr,izsbtosr,noct,nd,x,y,z,tmp,
     +              nisb,nxsup,xmnsup,xsizsup,nysup,ymnsup,ysizsup,
     +              nzsup,zmnsup,zsizsup,nclose,close,infoct)
c
c Load the nearest data in xa,ya,za,vra:
c
            np = 0
            ns = 0
            na = 0
            do i=1,nclose
                  if(np.eq.ndmaxp.and.ns.eq.ndmaxs) go to 32
                  ind = int(close(i)+0.5)
c
c Load primary data until there are enough:
c
                  if(vr(ind).ge.tmin.and.vr(ind).lt.tmax.
     +                         and.np.lt.ndmaxp) then
                        np = np + 1
                        na = na + 1
                        xa(na)  = x(ind)
                        ya(na)  = y(ind)
                        za(na)  = z(ind)
                        vra(na) = vr(ind)
                        iva(na) = 1
                  end if
c
c Load secondary data until maximum is met:
c
                  if(sec1(ind).ge.tmin.and.sec1(ind).lt.tmax.
     +               and.nvr.ge.2.and.ns.lt.ndmaxs) then
                        ns = ns + 1
                        na = na + 1
                        xa(na)  = x(ind)
                        ya(na)  = y(ind)
                        za(na)  = z(ind)
                        vra(na) = sec1(ind)
                        ivar    = 2
                        if(ktype.ne.2) 
     +                  vra(na) = vra(na) - vmean(ivar) + vmean(1)
                        iva(na) = 2
                  end if
                  if(sec2(ind).ge.tmin.and.sec2(ind).lt.tmax.
     +               and.nvr.ge.3.and.ns.lt.ndmaxs) then
                        ns = ns + 1
                        na = na + 1
                        xa(na)  = x(ind)
                        ya(na)  = y(ind)
                        za(na)  = z(ind)
                        vra(na) = sec2(ind)
                        ivar    = 3
                        if(ktype.ne.2) 
     +                  vra(na) = vra(na) - vmean(ivar) + vmean(1)
                        iva(na) = 3
                  end if
                  if(sec3(ind).ge.tmin.and.sec3(ind).lt.tmax.
     +               and.nvr.ge.4.and.ns.lt.ndmaxs) then
                        ns = ns + 1
                        na = na + 1
                        xa(na)  = x(ind)
                        ya(na)  = y(ind)
                        za(na)  = z(ind)
                        vra(na) = sec3(ind)
                        ivar    = 4
                        if(ktype.ne.2) 
     +                  vra(na) = vra(na) - vmean(ivar) + vmean(1)
                        iva(na) = 4
                  end if
            end do
 32         continue
c
c Solve the Kriging System:
c
            if(ktype.eq.0) neq = na
            if(ktype.eq.1) neq = na + 1
            if(ktype.eq.2) neq = na + nvr
            if((neq-na).gt.na.or.na.lt.ndmin) then
                  write(lout,100) UNEST,UNEST
                  go to 4
            end if
c
c Set up kriging matrices:
c
            do i=1,neq*neq
                  a(i) = 0.0
            end do
            do j=1,na
                  do i=1,j
                        ind   = iva(i) + (iva(j)-1)*MAXVAR
                        call cova3(xa(i),ya(i),za(i),xa(j),ya(j),za(j),
     +                             ind,nst,MAXNST,c0,it,cc,aa,1,MAXROT,
     +                             rotmat,cmax,cova)
                        a(neq*(i-1)+j) = dble(cova)
                        a(neq*(j-1)+i) = dble(cova)
                  end do
                  xx = xa(j) - xloc
                  yy = ya(j) - yloc
                  zz = za(j) - zloc
c
c Right hand side covariance:
c
                  iv  = 1
                  ind = iv + (iva(j)-1)*MAXVAR
                  if(ndb.le.1) then
                        call cova3(xx,yy,zz,xdb(1),ydb(1),zdb(1),
     +                             ind,nst,MAXNST,c0,it,cc,aa,1,MAXROT,
     +                             rotmat,cmax,cova)
                        cb = cova
                  else
                     cb  = 0.0
                     do j1=1,ndb
                        call cova3(xx,yy,zz,xdb(j1),ydb(j1),zdb(j1),
     +                             ind,nst,MAXNST,c0,it,cc,aa,1,MAXROT,
     +                             rotmat,cmax,cova)
                        dx = xx - xdb(j1)
                        dy = yy - ydb(j1)
                        dz = zz - zdb(j1)
                        if((dx*dx+dy*dy+dz*dz).lt.EPSLON) then
                              cb = cb + cova - c0(ind)
                        else
                              cb = cb + cova
                        end if
                     end do
                     cb = cb / real(ndb)
                  endif
                  r(j)  = dble(cb)
                  rr(j) = r(j)
            end do
c
c Set up for either simple or ordinary cokriging:
c
            if(ktype.eq.1) then
                  do i=1,na
                        a(neq*(i-1)+na+1) = unbias
                        a(neq*na+i)       = unbias
                  end do
            else if(ktype.eq.2) then
                  do i=1,nvr
                        lim = na + i
                        r(lim)  = 0.0
                        rr(lim) = 0.0
                        do j=1,lim
                              if(j.gt.na.or.iva(j).ne.i) then
                                    a(neq*(lim-1)+j) = 0.0
                                    a(neq*(j-1)+lim) = 0.0
                              else
                                    a(neq*(lim-1)+j) = unbias
                                    a(neq*(j-1)+lim) = unbias
                              endif
                        end do
                  end do
            endif
            r(na+1)  = unbias
            rr(na+1) = unbias
c
c Write out the kriging Matrix if Seriously Debugging:
c
            if(idbg.ge.3) then
                  is = 1 - neq
                  do i=1,neq
                        is = 1 + (i-1)*neq
                        ie = is + neq - 1
                        write(ldbg,103) i,r(i),(a(j),j=is,ie)
 103                    format('    r(',i2,') =',f7.4,'  a= ',9(10f7.4))
                  end do
            endif
            call ktsol(neq,1,1,a,r,s,ising,MAXCOK)
c
c Write a warning if the matrix is singular:
c
            if(ising.ne.0) then
                  write(ldbg,*) 'WARNING COKB3D: singular matrix'
                  write(ldbg,*) '        for block',ix,iy,iz
                  write(lout,100) UNEST,UNEST
                  go to 4
            endif
c
c Write the kriging weights and data if requested:
c
            if(idbg.ge.2) then
                  write(ldbg,*) '       '
                  write(ldbg,*) 'BLOCK: ',ix,iy,iz,' at ',xloc,yloc,zloc
                  write(ldbg,*) '    at ',xloc,yloc,zloc
                  write(ldbg,*) ' '
                  if(ktype.eq.1) then
                        write(ldbg,*) '  Lagrange multiplier: ',s(na+1)
                  else if(ktype.ge.2) then
                        do i=1,nvr
                        write(ldbg,*) '  Lagrange multiplier: ',s(na+i)
                        end do
                  endif
                  write(ldbg,*) '  BLOCK EST: x,y,z,vr,wt '
                  do i=1,na
                  write(ldbg,'(5f12.3)') xa(i),ya(i),za(i),vra(i),s(i)
                  end do
            endif
c
c Compute the estimate and the kriging variance:
c
            sumw = 0.0
            ook  = 0.0
            ookv = cbb
            do i=1,na
                        ookv = ookv - real(s(i)*rr(i))
                        sumw = sumw + real(s(i))
                        ook  = ook  + real(s(i))*vra(i)
            end do
            if(neq.gt.na) ookv = ookv - real(s(na+1))*unbias
c
c Add mean if SK:
c
            ook = ook + (1.0-sumw)*vmean(1)
c
c Write results:
c
            write(lout,100) ook,ookv
 100        format(g14.8,1x,g14.8)
c
c Accumulate statistics of kriged blocks:
c
            nk = nk + 1
            uk = uk + ook
            vk = vk + ook*ook
            if(idbg.ge.4) write(ldbg,*) ' estimate, variance  ',ook,ookv
c
c END OF MAIN LOOP OVER ALL THE BLOCKS:
c
 4    continue
c
c Write statistics of kriged values:
c
      if(nk.gt.0.and.idbg.gt.0) then
            vk = (vk-uk*uk/real(nk))/real(nk)
            uk = uk/real(nk)
            write(ldbg,*)
            write(ldbg,*) 'Estimated  ',nk,' blocks '
            write(ldbg,*) '  average  ',uk
            write(ldbg,*) '  variance ',vk
            write(*,*)
            write(*,*)    'Estimated  ',nk,' blocks '
            write(*,*)    '  average  ',uk
            write(*,*)    '  variance ',vk
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
      open(lun,file='cokb3d.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for COKB3D',/,
     +       '                  **********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('somedata.dat                      ',
     +       '-file with data')
      write(lun,12)
 12   format('3                                 ',
     +       '-   number of variables primary+other')
      write(lun,13)
 13   format('1   2   0   3   4   5             ',
     +       '-   columns for X,Y,Z and variables')
      write(lun,14)
 14   format('-0.01     1.0e21                  ',
     +       '-   trimming limits')
      write(lun,15)
 15   format('0                                 ',
     +       '-co-located cokriging? (0=no, 1=yes)')
      write(lun,16)
 16   format('somedata.dat                      ',
     +       '-   file with gridded covariate')
      write(lun,17)
 17   format('4                                 ',
     +       '-   column for covariate')
      write(lun,18)
 18   format('3                                 ',
     +       '-debugging level: 0,1,2,3')
      write(lun,19)
 19   format('cokb3d.dbg                        ',
     +       '-file for debugging output')
      write(lun,20)
 20   format('cokb3d.out                        ',
     +       '-file for output')
      write(lun,21)
 21   format('50   0.5   1.0                    ',
     +       '-nx,xmn,xsiz')
      write(lun,22)
 22   format('50   0.5   1.0                    ',
     +       '-ny,ymn,ysiz')
      write(lun,23)
 23   format('10   0.5   1.0                    ',
     +       '-nz,zmn,zsiz')
      write(lun,24)
 24   format('1    1     1                      ',
     +       '-x, y, and z block discretization')
      write(lun,25)
 25   format('1   12     8                      ',
     +       '-min primary,max primary,max all sec')
      write(lun,26)
 26   format('25.0  25.0  25.0                  ',
     +       '-maximum search radii: primary')
      write(lun,27)
 27   format('10.0  10.0  10.0                  ',
     +       '-maximum search radii: all secondary')
      write(lun,28)
 28   format(' 0.0   0.0   0.0                  ',
     +       '-angles for search ellipsoid')
      write(lun,29)
 29   format('2                                 ',
     +       '-kriging type (0=SK, 1=OK, 2=OK-trad)')
      write(lun,30)
 30   format('3.38  2.32  0.00  0.00            ',
     +       '-mean(i),i=1,nvar')
      write(lun,31)
 31   format('1     1                           ',
     +       '-semivariogram for "i" and "j"')
      write(lun,32)
 32   format('1   11.0                          ',
     +       '-   nst, nugget effect')
      write(lun,33)
 33   format('1   39.0  0.0   0.0   0.0         ',
     +       '-   it,cc,ang1,ang2,ang3')
      write(lun,34)
 34   format('         60.0  60.0  60.0         ',
     +       '-   a_hmax, a_hmin, a_vert')
      write(lun,35)
 35   format('1     2                           ',
     +       '-semivariogram for "i" and "j"')
      write(lun,36)
 36   format('1    0.0                          ',
     +       '-   nst, nugget effect')
      write(lun,37)
 37   format('1   14.5  0.0   0.0   0.0         ',
     +       '-   it,cc,ang1,ang2,ang3')
      write(lun,38)
 38   format('         60.0  60.0  60.0         ',
     +       '-   a_hmax, a_hmin, a_vert')
      write(lun,39)
 39   format('1     3                           ',
     +       '-semivariogram for "i" and "j"')
      write(lun,40)
 40   format('1    0.0                          ',
     +       '-   nst, nugget effect')
      write(lun,41)
 41   format('1    5.0  0.0   0.0   0.0         ',
     +       '-   it,cc,ang1,ang2,ang3')
      write(lun,42)
 42   format('         60.0  60.0  60.0         ',
     +       '-   a_hmax, a_hmin, a_vert')
      write(lun,43)
 43   format('2     2                           ',
     +       '-semivariogram for "i" and "j"')
      write(lun,44)
 44   format('1    9.0                          ',
     +       '-   nst, nugget effect')
      write(lun,45)
 45   format('1   15.0  0.0   0.0   0.0         ',
     +       '-   it,cc,ang1,ang2,ang3')
      write(lun,46)
 46   format('         60.0  60.0  60.0         ',
     +       '-   a_hmax, a_hmin, a_vert')
      write(lun,47)
 47   format('2     3                           ',
     +       '-semivariogram for "i" and "j"')
      write(lun,48)
 48   format('1    0.0                          ',
     +       '-   nst, nugget effect')
      write(lun,49)
 49   format('1    3.8  0.0   0.0   0.0         ',
     +       '-   it,cc,ang1,ang2,ang3')
      write(lun,50)
 50   format('         60.0  60.0  60.0         ',
     +       '-   a_hmax, a_hmin, a_vert')
      write(lun,51)
 51   format('3     3                           ',
     +       '-semivariogram for "i" and "j"')
      write(lun,52)
 52   format('1    1.1                          ',
     +       '-   nst, nugget effect')
      write(lun,53)
 53   format('1    1.8  0.0   0.0   0.0         ',
     +       '-   it,cc,ang1,ang2,ang3')
      write(lun,54)
 54   format('         60.0  60.0  60.0         ',
     +       '-   a_hmax, a_hmin, a_vert')

      close(lun)
      return
      end
      
      
      
      subroutine cova3(x1,y1,z1,x2,y2,z2,ivarg,nst,MAXNST,c0,it,cc,aa,
     +                 irot,MAXROT,rotmat,cmax,cova)
c-----------------------------------------------------------------------
c
c                    Covariance Between Two Points
c                    *****************************
c
c This subroutine calculated the covariance associated with a variogram
c model specified by a nugget effect and nested varigoram structures.
c The anisotropy definition can be different for each nested structure.
c
c
c
c INPUT VARIABLES:
c
c   x1,y1,z1         coordinates of first point
c   x2,y2,z2         coordinates of second point
c   nst(ivarg)       number of nested structures (maximum of 4)
c   ivarg            variogram number (set to 1 unless doing cokriging
c                       or indicator kriging)
c   MAXNST           size of variogram parameter arrays
c   c0(ivarg)        isotropic nugget constant
c   it(i)            type of each nested structure:
c                      1. spherical model of range a;
c                      2. exponential model of parameter a;
c                           i.e. practical range is 3a
c                      3. gaussian model of parameter a;
c                           i.e. practical range is a*sqrt(3)
c                      4. power model of power a (a must be gt. 0  and
c                           lt. 2).  if linear model, a=1,c=slope.
c                      5. hole effect model
c   cc(i)            multiplicative factor of each nested structure.
c                      (sill-c0) for spherical, exponential,and gaussian
c                      slope for linear model.
c   aa(i)            parameter "a" of each nested structure.
c   irot             index of the rotation matrix for the first nested 
c                    structure (the second nested structure will use
c                    irot+1, the third irot+2, and so on)
c   MAXROT           size of rotation matrix arrays
c   rotmat           rotation matrices
c
c
c OUTPUT VARIABLES:
c
c   cmax             maximum covariance
c   cova             covariance between (x1,y1,z1) and (x2,y2,z2)
c
c
c
c EXTERNAL REFERENCES: sqdist    computes anisotropic squared distance
c                      rotmat    computes rotation matrix for distance
c-----------------------------------------------------------------------
      parameter(PI=3.14159265,PMX=999.,EPSLON=1.e-5)
      integer   nst(*),it(*)
      real      c0(*),cc(*),aa(*)
      real*8    rotmat(MAXROT,3,3),hsqd,sqdist
c
c Calculate the maximum covariance value (used for zero distances and
c for power model covariance):
c
      istart = 1 + (ivarg-1)*MAXNST
      cmax   = c0(ivarg)
      do is=1,nst(ivarg)
            ist = istart + is - 1
            if(it(ist).eq.4) then
                  cmax = cmax + PMX
            else
                  cmax = cmax + cc(ist)
            endif
      end do
c
c Check for "zero" distance, return with cmax if so:
c
      hsqd = sqdist(x1,y1,z1,x2,y2,z2,irot,MAXROT,rotmat)
      if(real(hsqd).lt.EPSLON) then
            cova = cmax
            return
      endif
c
c Loop over all the structures:
c
      cova = 0.0
      do is=1,nst(ivarg)
            ist = istart + is - 1
c
c Compute the appropriate distance:
c
            if(ist.ne.1) then
                  ir = min((irot+is-1),MAXROT)
                  hsqd=sqdist(x1,y1,z1,x2,y2,z2,ir,MAXROT,rotmat)
            end if
            h = real(dsqrt(hsqd))
c
c Spherical Variogram Model?
c
            if(it(ist).eq.1) then
                  hr = h/aa(ist)
                  if(hr.lt.1.) cova=cova+cc(ist)*(1.-hr*(1.5-.5*hr*hr))
c
c Exponential Variogram Model?
c
            else if(it(ist).eq.2) then
                  cova = cova + cc(ist)*exp(-3.0*h/aa(ist))
c
c Gaussian Variogram Model?
c
            else if(it(ist).eq.3) then
                  cova = cova + cc(ist)*exp(-3.*(h/aa(ist))*(h/aa(ist)))
c
c Power Variogram Model?
c
            else if(it(ist).eq.4) then
                  cova = cova + cmax - cc(ist)*(h**aa(ist))
c
c Hole Effect Model?
c
            else if(it(ist).eq.5) then
c                 d = 10.0 * aa(ist)
c                 cova = cova + cc(ist)*exp(-3.0*h/d)*cos(h/aa(ist)*PI)
                  cova = cova + cc(ist)*cos(h/aa(ist)*PI)
            endif
      end do
c
c Finished:
c
      return
      end
      
