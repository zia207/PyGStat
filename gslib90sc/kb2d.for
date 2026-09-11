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
      module geostat
      
      real,allocatable    :: x(:),y(:),vr(:)

      real    UNEST,EPSLON,VERSION,aa(4),cc(4),ang(4),anis(4),
     +        xmn,ymn,xsiz,ysiz,radius,c0,skmean
      integer MAXNST,it(4),test,nd,nx,ny,nxdis,nydis,ndmin,
     +        ndmax,nst,ktype,idbg,lout,ldbg
      
      end module
c
c
c
      program main
c-----------------------------------------------------------------------
c
c                 Kriging of a 2-D Rectangular Grid
c                 *********************************
c
c The program is executed with no command line arguments.  The user
c will be prompted for the name of a parameter file.  The parameter
c file is described in the documentation (see the example kb2d.par)
c
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use geostat
      MAXNST  =   4
      UNEST   = -999.
      EPSLON  = 1.0e-10
      VERSION = 3.000
c
c Read the Parameter File and the Data:
c
      call readparm(MAXDAT,MAXDIS,MAXSAM,MAXKD,MAXKRG)
c
c Call kb2d to krige the grid:
c
      call kb2d(MAXDAT,MAXDIS,MAXSAM,MAXKD,MAXKRG)
c
c Finished:
c
      write(*,9998) VERSION
 9998 format(/' KB2D Version: ',f5.3, ' Finished'/)
      stop
      end
 
 
 
      subroutine readparm(MAXDAT,MAXDIS,MAXSAM,MAXKD,MAXKRG)
c-----------------------------------------------------------------------
c
c                  Initialization and Read Parameters
c                  **********************************
c
c The input parameters and data are read in from their files. Some quick
c error checking is performed and the statistics of all the variables
c being considered are written to standard output.
c
c
c
c-----------------------------------------------------------------------
      use       msflib
      use       geostat
      parameter(MV=20)
      real      var(MV)
      character datafl*512,outfl*512,dbgfl*512,str*512
      logical   testfl
c
c Unit numbers:
c
      lin  = 1
      lout = 2
      ldbg = 3
c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' KB2D Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'kb2d.par            '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'kb2d.par            ') then
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

      read(lin,*,err=98)       ixl,iyl,ivrl
      write(*,*) ' columns for X,Y, VR = ',ixl,iyl,ivrl

      read(lin,*,err=98)       tmin,tmax
      write(*,*) ' trimming limits = ',tmin,tmax

      read(lin,*,err=98)       idbg
      write(*,*) ' debugging level = ',idbg

      read(lin,'(a512)',err=98) dbgfl
      call chknam(dbgfl,512)
      write(*,*) ' debugging file = ',dbgfl(1:40)

      read(lin,'(a512)',err=98) outfl
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,*,err=98)       nx,xmn,xsiz
      write(*,*) ' nx, xmn, xsiz = ',nx,xmn,xsiz

      read(lin,*,err=98)       ny,ymn,ysiz
      write(*,*) ' ny, ymn, ysiz = ',ny,ymn,ysiz

      read(lin,*,err=98)       nxdis,nydis
      write(*,*) ' discretization = ',nxdis,nydis

      read(lin,*,err=98)       ndmin,ndmax
      if(ndmin.lt.0) ndmin = 0
      write(*,*) ' min max data = ',ndmin,ndmax

      read(lin,*,err=98)       radius
      write(*,*) ' isotropic radius = ',radius

      read(lin,*,err=98)       ktype,skmean
      write(*,*) ' ktype,skmea = ',ktype,skmean

      read(lin,*,err=98)       nst,c0
      write(*,*) ' nst, nugget = ',nst,c0

      if(nst.le.0) then
            nst     = 1
            it(1)   = 1
            cc(1)   = 0.0
            ang(1)  = 0.0
            aa(1)   = 0.0
            anis(1) = 0.0
      else
            do i=1,nst
                  read(lin,*,err=98) it(i),cc(i),ang(i),aa(i),a2
                  anis(i) = a2 / aa(i)
                  write(*,*) ' it,cc,ang,a_max,a_min = ',
     +                         it(i),cc(i),ang(i),aa(i),a2
                  if(it(i).eq.4) then
                        if(aa(i).lt.0.0) stop ' INVALID power variogram'
                        if(aa(i).gt.2.0) stop ' INVALID power variogram'
                  end if
                  if(it(i).eq.5) then
                        write(*,*) ' we do not support this variogram'
                        write(*,*) ' use kt3d instead '
                        stop                        
                  end if
            end do
      end if

      close(lin)
c
c Find the needed parameters:
c
      MAXSAM = ndmax + 1
      MAXDIS = nxdis * nydis
      MAXKD = MAXSAM + 1
      MAXKRG = MAXKD * MAXKD
c
      write(*,*)
      if(nst.gt.MAXNST)   stop 'nst is too big - modify .inc file'
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
      MAXDAT = 0
 22   read(lin,*,end=44,err=99)(var(j),j=1,nvari)
      if(var(ivrl).lt.tmin.or.var(ivrl).ge.tmax)go to 22
      MAXDAT = MAXDAT + 1
      go to 22
 44   continue
c
c Allocate the needed memory:
c
      allocate(x(MAXDAT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c
      allocate(y(MAXDAT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c
      allocate(vr(MAXDAT),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c
      rewind(lin)
      read(lin,'(a40)',err=99) str
      read(lin,*,err=99)       nvari
      av = 0.0
      ss = 0.0
      do i=1,nvari
            read(lin,'(a40)',err=99) str
      end do
c
c Read the data:
c
      nd = 0
 7    read(lin,*,end=8,err=99) (var(j),j=1,nvari)
      vrt = var(ivrl)
      if(vrt.lt.tmin.or.vrt.ge.tmax) go to 7
      nd = nd + 1
      x(nd)  = var(ixl)
      y(nd)  = var(iyl)
      vr(nd) = vrt
      av     = av + vrt
      ss     = ss + vrt*vrt
      go to 7
 8    close(lin)
c
c Open the output files:
c
      open(lout,file=outfl,status='UNKNOWN')
      write(lout,300)
 300  format('KB2D Output')
      zmn   = 0.
      zsiz  = 1. 
      nreal = 1
      write(lout,301) 2,nx,ny,nz,xmn,ymn,zmn,xsiz,ysiz,zsiz,nreal
 301  format(i2,3(1x,i4),3(1x,g14.8),3(1x,g12.6),i4) 
      write(lout,302)
 302  format(' Estimate',/,'Estimation Variance')

      open(ldbg,file=dbgfl,status='UNKNOWN')
c
c Compute the averages and variances as an error check for the user:
c
      av = av / max(real(nd),1.0)
      ss =(ss / max(real(nd),1.0)) - av * av
c
c Write Some of the Statistics to the screen:
c
      write(*,900) nd,av,sqrt(max(ss,0.0))
 900  format(/'   There are ',i8,' data with:',/,
     +        '   mean value          = ',f12.5,/,
     +        '   standard deviation  = ',f12.5,/)
      return
c
c Error in an Input File Somewhere:
c
 98   stop 'ERROR in parameter file!'
 99   stop 'ERROR in data file!'
      end



      subroutine kb2d(MAXDAT,MAXDIS,MAXSAM,MAXKD,MAXKRG)
c-----------------------------------------------------------------------
c
c           Ordinary/Simple Kriging of a 2-D Rectangular Grid
c           *************************************************
c
c This subroutine estimates point or block values of one variable by
c ordinary kriging.  All of the samples are rescanned for each block
c estimate; this makes the program simple but inefficient.  The data
c should NOT contain any missing values.  Unestimated points are
c returned as -1.0e21
c
c
c
c Original:  A.G. Journel                                           1978
c Revisions: B.E. Buxton                                       Apr. 1983
c-----------------------------------------------------------------------
      use geostat
      real,allocatable    :: xdb(:),ydb(:),xa(:),ya(:),vra(:),dist(:)
      real*8,allocatable  :: r(:),rr(:),s(:),a(:)
      integer,allocatable :: nums(:)
c
      logical   first
      data      first/.true./,PMX/9999.0/
c
c Echo the input parameters if debugging flag is >2:
c
c Allocate the needed memory:
c
      allocate(xdb(MAXDIS),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c
      allocate(ydb(MAXDIS),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c
      allocate(xa(MAXSAM),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c
      allocate(ya(MAXSAM),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c
      allocate(vra(MAXSAM),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c
      allocate(dist(MAXSAM),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c
      allocate(nums(MAXSAM),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c
      allocate(r(MAXKD),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c
      allocate(rr(MAXKD),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c
      allocate(s(MAXKD),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c
      allocate(a(MAXKRG),stat = test)
            if(test.ne.0)then
                  write(*,*)'ERROR: Allocation failed due to',
     +                        ' insufficient memory.'
                  stop
            end if
c
      if(idbg.gt.2) then
            write(ldbg,*) 'KB2D Parameters'
            write(ldbg,*)
            write(ldbg,*) 'Variogram Parameters for ',nst,' structures:'
            write(ldbg,*) '  Nugget effect:         ',c0
            write(ldbg,*) '  Types of variograms:   ',(it(i),i=1,nst)
            write(ldbg,*) '  Contribution cc        ',(cc(i),i=1,nst)
            write(ldbg,*) '  Ranges:                ',(aa(i),i=1,nst)
            write(ldbg,*) '  Angle for Continuity:  ',(ang(i),i=1,nst)
            write(ldbg,*) '  Anisotropy Factors:    ',(anis(i),i=1,nst)
            write(ldbg,*) ' '
            write(ldbg,*) 'Grid for Kriging:'
            write(ldbg,*) '  Number of X and Y Blocks:',nx,ny
            write(ldbg,*) '  Origin of X and Y Blocks:',xmn,ymn
            write(ldbg,*) '  Size   of X and Y Blocks:',xsiz,ysiz
            write(ldbg,*) ' '
            write(ldbg,*) 'Discretization of blocks:  ',nxdis,nydis
            write(ldbg,*) 'Search Radius:             ',radius
            write(ldbg,*) 'Minimum number of samples: ',ndmin
            write(ldbg,*) 'Maximum number of samples: ',ndmax
            write(ldbg,*) ' '
      endif
c
c Echo the input data if debugging flag >1:
c
      if(idbg.ge.4) then
            do id=1,nd
                  write(ldbg,99) id,x(id),y(id),vr(id)
 99               format('Data: ',i5,' at ',2f12.3,' value: ',f12.5)
            end do
      endif
c
c Set up the discretization points per block.  Figure out how many
c are needed, the spacing, and fill the xdb and ydb arrays with the
c offsets relative to the block center (this only gets done once):
c
      ndb  = nxdis * nydis
      if(ndb.gt.MAXDIS) then
            write(*,*) 'ERROR KB2D: Too many discretization points '
            write(*,*) '            Increase MAXDIS or lower n[xy]dis'
            stop
      endif
      xdis = xsiz  / max(real(nxdis),1.0)
      ydis = ysiz  / max(real(nydis),1.0)
      xloc = -0.5*(xsiz+xdis)
      i    = 0
      do ix =1,nxdis
            xloc = xloc + xdis
            yloc = -0.5*(ysiz+ydis)
            do iy=1,nydis
                  yloc = yloc + ydis
                  i = i+1
                  xdb(i) = xloc
                  ydb(i) = yloc
            end do
      end do
c
c Initialize accumulators:
c
      cbb  = 0.0
      rad2 = radius*radius
c
c Calculate Block Covariance. Check for point kriging.
c
      cov   = cova2(xdb(1),ydb(1),xdb(1),ydb(1),nst,c0,PMX,cc,
     +              aa,it,ang,anis,first)
c
c Keep this value to use for the unbiasedness constraint:
c
      unbias = cov
      first  = .false.
      if (ndb.le.1) then
            cbb = cov
      else
            do i=1,ndb
                  do j=1,ndb
                        cov = cova2(xdb(i),ydb(i),xdb(j),ydb(j),nst,c0,
     +                              PMX,cc,aa,it,ang,anis,first)
                        if(i.eq.j) cov = cov - c0
                        cbb = cbb + cov
                  end do
            end do
            cbb = cbb/real(ndb*ndb)
      endif
      if(idbg.gt.1) then
            write(ldbg,*) ' '
            write(ldbg,*) 'Block Covariance: ',cbb
            write(ldbg,*) ' '
      endif
c
c MAIN LOOP OVER ALL THE BLOCKS IN THE GRID:
c
      nk = 0
      ak = 0.0
      vk = 0.0
      do 4 iy=1,ny
      yloc = ymn + (iy-1)*ysiz
      do 4 ix=1,nx
            xloc = xmn + (ix-1)*xsiz
c
c Find the nearest samples within each octant: First initialize
c the counter arrays:
c
            na = 0
            do isam=1,ndmax
                  dist(isam) = 1.0e+20
                  nums(isam) = 0
            end do
c
c Scan all the samples (this is inefficient and the user with lots of
c data should move to ktb3d):
c
            do 6 id=1,nd
                  dx = x(id) - xloc
                  dy = y(id) - yloc
                  h2 = dx*dx + dy*dy
                  if(h2.gt.rad2) go to 6
c
c Do not consider this sample if there are enough close ones:
c
                  if(na.eq.ndmax.and.h2.gt.dist(na)) go to 6
c
c Consider this sample (it will be added in the correct location):
c
                  if(na.lt.ndmax) na = na + 1
                  nums(na)           = id
                  dist(na)           = h2
                  if(na.eq.1) go to 6
c
c Sort samples found thus far in increasing order of distance:
c
                  n1 = na-1
                  do ii=1,n1
                        k=ii
                        if(h2.lt.dist(ii)) then
                              jk = 0
                              do jj=k,n1
                                    j  = n1-jk
                                    jk = jk+1
                                    j1 = j+1
                                    dist(j1) = dist(j)
                                    nums(j1) = nums(j)
                              end do
                              dist(k) = h2
                              nums(k) = id
                              go to 6
                        endif
                  end do
 6          continue
c
c Is there enough samples?
c
            if(na.lt.ndmin) then
                  if(idbg.ge.2)
     +            write(ldbg,*) 'Block ',ix,iy, 'not estimated'
                  est  = UNEST
                  estv = UNEST
                  go to 1
            endif
c
c Put coordinates and values of neighborhood samples into xa,ya,vra:
c
            do ia=1,na
                  jj      = nums(ia)
                  xa(ia)  = x(jj)
                  ya(ia)  = y(jj)
                  vra(ia) = vr(jj)
            end do
c
c Handle the situation of only one sample:
c
            if(na.eq.1) then
                  cb1 = cova2(xa(1),ya(1),xa(1),ya(1),nst,c0,
     +                        PMX,cc,aa,it,ang,anis,first)
                  xx  = xa(1) - xloc
                  yy  = ya(1) - yloc
c
c Establish Right Hand Side Covariance:
c
                  if(ndb.le.1) then
                        cb = cova2(xx,yy,xdb(1),ydb(1),nst,c0,
     +                             PMX,cc,aa,it,ang,anis,first)
                  else
                        cb  = 0.0
                        do i=1,ndb
                              cb = cb + cova2(xx,yy,xdb(i),ydb(i),nst,
     +                                  c0,PMX,cc,aa,it,ang,anis,first)
                              dx = xx - xdb(i)
                              dy = yy - ydb(i)
                              if((dx*dx+dy*dy).lt.EPSLON)
     +                        cb = cb - c0
                        end do
                        cb = cb / real(ndb)
                  end if
                  if(ktype.eq.0) then
                        s(1) = cb/cbb
                        est  = s(1)*vra(1) + (1.0-s(1))*skmean
                        estv = cbb - s(1) * cb
                  else
                        est  = vra(1)
                        estv = cbb - 2.0*cb + cb1
                  end if
            else
c
c Solve the Kriging System with more than one sample:
c
                  neq = na + ktype
                  nn  = (neq + 1)*neq/2
c
c Set up kriging matrices:
c
                  in=0
                  do j=1,na
c
c Establish Left Hand Side Covariance Matrix:
c
                        do i=1,j
                              in = in + 1
                              a(in) = dble( cova2(xa(i),ya(i),xa(j),
     +                                      ya(j),nst,c0,PMX,cc,aa,it,
     +                                      ang,anis,first) )
                        end do
                        xx = xa(j) - xloc
                        yy = ya(j) - yloc
c
c Establish Right Hand Side Covariance:
c
                        if(ndb.le.1) then
                              cb = cova2(xx,yy,xdb(1),ydb(1),nst,c0,
     +                                   PMX,cc,aa,it,ang,anis,first)
                        else
                              cb  = 0.0
                              do j1=1,ndb
                                    cb = cb + cova2(xx,yy,xdb(j1),
     +                                   ydb(j1),nst,c0,PMX,cc,aa,
     +                                   it,ang,anis,first)
                                    dx = xx - xdb(j1)
                                    dy = yy - ydb(j1)
                                    if((dx*dx+dy*dy).lt.EPSLON)
     +                                    cb = cb - c0
                              end do
                              cb = cb / real(ndb)
                        end if
                        r(j)  = dble(cb)
                        rr(j) = r(j)
                  end do
c
c Set the unbiasedness constraint:
c
                  if(ktype.eq.1) then
                        do i=1,na
                              in    = in + 1
                              a(in) = dble(unbias)
                        end do
                        in      = in + 1
                        a(in)   = 0.0
                        r(neq)  = dble(unbias)
                        rr(neq) = r(neq)
                  end if
c
c Write out the kriging Matrix if Seriously Debugging:
c
                  if(idbg.ge.3) then
                        write(ldbg,101) ix,iy
                        is = 1
                        do i=1,neq
                              ie = is + i - 1
                              write(ldbg,102) i,r(i),(a(j),j=is,ie)
                              is = is + i
                        end do
 101                    format(/,'Kriging Matrices for Node: ',2i4,
     +                           ' RHS first')
 102                    format('  r(',i2,') =',f12.4,'  a= ',9(10f12.4))
                  endif
c
c Solve the Kriging System:
c
                  call ksol(1,neq,1,a,r,s,ising)
c
c Write a warning if the matrix is singular:
c
                  if(ising.ne.0) then
                        write(*,*) 'WARNING KB2D: singular matrix'
                        write(*,*) '              for block',ix,iy
                        est  = UNEST
                        estv = UNEST
                        go to 1
                  endif
c
c Write the kriging weights and data if requested:
c
                  if(idbg.ge.2) then
                        write(ldbg,*) '       '
                        write(ldbg,*) 'BLOCK: ',ix,iy
                        write(ldbg,*) '       '
                        if(ktype.eq.1) write(ldbg,*) 
     +                  '  Lagrange multiplier: ',s(neq)*unbias
                        write(ldbg,*) '  BLOCK EST: x,y,vr,wt '
                        do i=1,na
                        write(ldbg,'(4f12.3)') xa(i),ya(i),vra(i),s(i)
                        end do
                  endif
c
c Compute the estimate and the kriging variance:
c
                  est  = 0.0
                  estv = cbb
                  sumw = 0.0
                  if(ktype.eq.1) estv = estv - real(s(na+1))*unbias
                  do i=1,na
                        sumw = sumw + real(s(i))
                        est  = est  + real(s(i))*vra(i)
                        estv = estv - real(s(i)*rr(i))
                  end do
                  if(ktype.eq.0) est = est + (1.0-sumw)*skmean
            endif
            if(idbg.ge.2) then
                  write(ldbg,*) '  est  ',est
                  write(ldbg,*) '  estv ',estv
                  write(ldbg,*) ' '
            endif
c
c Write the result to the output file:
c
 1          write(lout,'(g14.8,1x,g14.8)') est,estv
            if(est.gt.UNEST) then
                  nk = nk + 1
                  ak = ak + est
                  vk = vk + est*est
            end if
c
c END OF MAIN LOOP OVER ALL THE BLOCKS:
c
 4    continue
      if(nk.ge.1) then
            ak = ak / real(nk)
            vk = vk/real(nk) - ak*ak
            write(ldbg,105) nk,ak,vk
            write(*,   105) nk,ak,vk
 105        format(/,'  Estimated   ',i8,' blocks ',/,
     +               '  average   ',f9.4,/,'  variance  ',f9.4,/)
      end if
      return
      end
 
 
 
      real function cova2(x1,y1,x2,y2,nst,c0,PMX,cc,aa,it,
     +                    ang,anis,first)
c-----------------------------------------------------------------------
c
c              Covariance Between Two Points (2-D Version)
c              *******************************************
c
c This function returns the covariance associated with a variogram model
c that is specified by a nugget effect and possibly four different
c nested varigoram structures.  The anisotropy definition can be
c different for each of the nested structures (spherical, exponential,
c gaussian, or power).
c
c
c
c INPUT VARIABLES:
c
c   x1,y1            Coordinates of first point
c   x2,y2            Coordinates of second point
c   nst              Number of nested structures (max. 4).
c   c0               Nugget constant (isotropic).
c   PMX              Maximum variogram value needed for kriging when
c                      using power model.  A unique value of PMX is
c                      used for all nested structures which use the
c                      power model.  therefore, PMX should be chosen
c                      large enough to account for the largest single
c                      structure which uses the power model.
c   cc(nst)          Multiplicative factor of each nested structure.
c   aa(nst)          Parameter "a" of each nested structure.
c   it(nst)          Type of each nested structure:
c                      1. spherical model of range a;
c                      2. exponential model of parameter a;
c                           i.e. practical range is 3a
c                      3. gaussian model of parameter a;
c                           i.e. practical range is a*sqrt(3)
c                      4. power model of power a (a must be gt. 0  and
c                           lt. 2).  if linear model, a=1,c=slope.
c   ang(nst)         Azimuth angle for the principal direction of
c                      continuity (measured clockwise in degrees from Y)
c   anis(nst)        Anisotropy (radius in minor direction at 90 degrees
c                      from "ang" divided by the principal radius in 
c                      direction "ang")
c   first            A logical variable which is set to true if the
c                      direction specifications have changed - causes
c                      the rotation matrices to be recomputed.
c
c
c
c OUTPUT VARIABLES: returns "cova2" the covariance obtained from the
c                   variogram model.
c
c
c
c-----------------------------------------------------------------------
      parameter(DTOR=3.14159265/180.0,EPSLON=0.0000001)
      real      aa(*),cc(*),ang(*),anis(*),rotmat(4,4),maxcov
      integer   it(*)
      logical   first
      save      rotmat,maxcov
c
c The first time around, re-initialize the cosine matrix for the
c variogram structures:
c
      if(first) then
            maxcov = c0
            do is=1,nst
                  azmuth       = (90.0-ang(is))*DTOR
                  rotmat(1,is) =  cos(azmuth)
                  rotmat(2,is) =  sin(azmuth)
                  rotmat(3,is) = -sin(azmuth)
                  rotmat(4,is) =  cos(azmuth)
                  if(it(is).eq.4) then
                        maxcov = maxcov + PMX
                  else
                        maxcov = maxcov + cc(is)
                  endif
            end do
      endif
c
c Check for very small distance:
c
      dx = x2-x1
      dy = y2-y1
      if((dx*dx+dy*dy).lt.EPSLON) then
            cova2 = maxcov
            return
      endif
c
c Non-zero distance, loop over all the structures:
c
      cova2 = 0.0
      do is=1,nst
c
c Compute the appropriate structural distance:
c
            dx1 = (dx*rotmat(1,is) + dy*rotmat(2,is))
            dy1 = (dx*rotmat(3,is) + dy*rotmat(4,is))/anis(is)
            h   = sqrt(max((dx1*dx1+dy1*dy1),0.0))
            if(it(is).eq.1) then
c
c Spherical model:
c
                  hr = h/aa(is)
                  if(hr.lt.1.0) cova2 = cova2 
     +                                + cc(is)*(1.-hr*(1.5-.5*hr*hr))
            else if(it(is).eq.2) then
c
c Exponential model:
c
                  cova2 = cova2 +cc(is)*exp(-3.0*h/aa(is))
            else if(it(is).eq. 3) then
c
c Gaussian model:
c
                  hh=-3.0*(h*h)/(aa(is)*aa(is))
                  cova2 = cova2 +cc(is)*exp(hh)
            else
c
c Power model:
c
                  cov1  = PMX - cc(is)*(h**aa(is))
                  cova2 = cova2 + cov1
            endif
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
      open(lun,file='kb2d.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for KB2D',/,
     +       '                  *******************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('../data/cluster.dat          ',
     +       '-file with data')
      write(lun,12)
 12   format('1   2   3                    ',
     +       '-   columns for X, Y, and variable')
      write(lun,13)
 13   format('-1.0e21   1.0e21             ',
     +       '-   trimming limits')
      write(lun,14)
 14   format('3                            ',
     +       '-debugging level: 0,1,2,3')
      write(lun,15)
 15   format('kb2d.dbg                     ',
     +       '-file for debugging output')
      write(lun,16)
 16   format('kb2d.out                     ',
     +       '-file for kriged output')
      write(lun,17)
 17   format('5    5.0  10.0               ',
     +       '-nx,xmn,xsiz')
      write(lun,18)
 18   format('5    5.0  10.0               ',
     +       '-ny,ymn,ysiz')
      write(lun,19)
 19   format('1    1                       ',
     +       '-x and y block discretization')
      write(lun,20)
 20   format('4    8                       ',
     +       '-min and max data for kriging')
      write(lun,21)
 21   format('20.0                         ',
     +       '-maximum search radius')
      write(lun,22)
 22   format('1    2.302                   ',
     +       '-0=SK, 1=OK,  (mean if SK)')
      write(lun,23)
 23   format('1   2.0                      ',
     +       '-nst, nugget effect')
      write(lun,24)
 24   format('1   8.0  0.0  10.0  10.0     ',
     +       '-it, c, azm, a_max, a_min')

      close(lun)
      return
      end
