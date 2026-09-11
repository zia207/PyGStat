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
      
      real,allocatable :: x(:),y(:),z(:),vr(:),sills(:),np(:,:,:,:),
     +                    gam(:,:,:,:),hm(:,:,:,:),tm(:,:,:,:), 
     +                    hv(:,:,:,:),tv(:,:,:,:)
      integer,allocatable :: ivtail(:),ivhead(:),ivtype(:)
      character*12,allocatable :: names(:)

      real    EPSLON,VERSION,xsiz,ysiz,zsiz,tmin,tmax,dxlag,dylag,dzlag
      integer igrid,nx,ny,nz,nxlag,nylag,nzlag,nd,isill,nvarg,minnp,test
      character outfl*512
      
      end module
c
c
c
      program main
c-----------------------------------------------------------------------
c
c                 Variogram Map/Volume Calculation
c                 ********************************
c
c The input data may be a regular grid (x cycles fastest, then y, 
c then z) or irregular / scattered locations.
c
c The program is executed with no command line arguments.  The user
c will be prompted for the name of a parameter file.  The parameter
c file is described in the documentation (see the example varmap.par)
c
c
c The output file will contain the variogram map/volume ordered by
c lag grid (the lags are a grid which cycle fastest by x then y then z)
c then by variogram number.  Each grid value has:
c
c        a) the "variogram" value
c        b) the number of pairs for the lag
c        c) the mean of the data contributing to the tail
c        d) the mean of the data contributing to the head
c
c
c
c AUTHOR: Clayton V. Deutsch                             DATE: 1989-1999
c-----------------------------------------------------------------------
      use geostat
      EPSLON  = 1.0e-20
      VERSION = 3.000
c
c Read the Parameter File:
c
      call readparm
c
c Call varmap to compute the required variograms:
c
      call varmap
c
c Write Results:
c
      call writeout
c
c Finished:
c
      write(*,9998) VERSION
 9998 format(/' VARMAP Version: ',f5.3, ' Finished'/)
      stop
      end
 
 
 
      subroutine readparm
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
      parameter(MV=500)
      real      var(MV),cut(MV)
      real*8    avg(MV),ssq(MV)
      integer   ivar(MV),num(MV),ivc(MV),indflag(MV)
      character datafl*512,str*512
      logical   testfl,testdat
      data      lin/1/,ncut/0/
      real,allocatable :: vrmin(:),vrmax(:)
c
c Note VERSION number:
c
      write(*,9999) VERSION
 9999 format(/' VARMAP Version: ',f5.3/)
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
      if(str(1:1).eq.' ') str(1:20) = 'varmap.par          '
      inquire(file=str,exist=testfl)
      if(.not.testfl) then
            write(*,*) 'ERROR - the parameter file does not exist,'
            write(*,*) '        check for the file and try again  '
            write(*,*)
            if(str(1:20).eq.'varmap.par          ') then
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

      read(lin,*,err=98) nvar
      write(*,*) ' number of variables = ',nvar
      backspace lin

      read(lin,*,err=98) j,(ivar(i),i=1,nvar)
      write(*,*) ' columns = ',(ivar(i),i=1,nvar)

      read(lin,*,err=98) tmin,tmax
      write(*,*) ' trimming limits = ',tmin,tmax

      read(lin,*,err=98) igrid
      if(igrid.eq.1) write(*,*) ' working with gridded data'
      if(igrid.eq.0) write(*,*) ' working with scattered data'

      read(lin,*,err=98) nx,ny,nz
      write(*,*) ' nx,ny,nz = ',nx,ny,nz

      read(lin,*,err=98) xsiz,ysiz,zsiz
      write(*,*) ' xsiz,ysiz,zsiz = ',xsiz,ysiz,zsiz

      read(lin,*,err=98) icolx,icoly,icolz
      write(*,*) ' columns for x, y, and z = ',icolx,icoly,icolz

      read(lin,'(a512)',err=98) outfl
      call chknam(outfl,512)
      write(*,*) ' output file = ',outfl(1:40)

      read(lin,*,err=98) nxlag,nylag,nzlag
      write(*,*) ' nxlag,nylag,nzlag = ',nxlag,nylag,nzlag

      read(lin,*,err=98) dxlag,dylag,dzlag
      write(*,*) ' dxlag,dylag,dzlag = ',dxlag,dylag,dzlag

      read(lin,*,err=98) minnp
      write(*,*) ' minimum number of pairs = ',minnp

      read(lin,*,err=98) isill
      write(*,*) ' flag to standardize sills = ',isill

      read(lin,*,err=98) nvarg
      write(*,*) ' number of variograms = ',nvarg
      if(nvarg.lt.1)      stop 'nvarg is too small: check parameters'
c
c  Allocate the storage needed for the array "names":
c
      allocate (ivtype(nvarg),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
      allocate (ivtail(nvarg),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
      allocate (ivhead(nvarg),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
      allocate (names(nvar+nvarg),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
c
      ncut = 0
      do i=1,nvarg
            read(lin,*,err=98) ivtail(i),ivhead(i),ivtype(i)
            write(*,*) ' tail,head,type = ',
     +                   ivtail(i),ivhead(i),ivtype(i)
            if(ivtype(i).eq.9.or.ivtype(i).eq.10) then
                   ncut = ncut + 1
                   if(tmin.gt.0.0)stop'tmin interferes with indicators!'
                   if(tmax.le.1.0)stop'tmax interferes with indicators!'
                   backspace lin
                   read(lin,*,err=98) ii,jj,kk,cut(ncut)
                   if(ivtype(i).eq.9)  indflag(ncut) = 1
                   if(ivtype(i).eq.10) indflag(ncut) = 0
                   ivc(ncut) = ivtail(i)
                   ivtail(i) = nvar + ncut
                   ivhead(i) = nvar + ncut
                   write(names(nvar+ncut),140) ncut
 140               format('Indicator ',i2)
                   write(*,*) ' indicator threshold = ',cut(ncut)
            endif
      end do
      write(*,*)
      close(lin)
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
      read(lin,*,err=99) nvari
      do i=1,nvari
            read(lin,*)
      end do
      maxdat = 0
 22   read(lin,*,end=44,err=99)(var(j),j=1,nvari)
      maxdat = maxdat + 1
      go to 22
 44   continue
      maxdim = nx * ny * nz
      if(maxdat.gt.maxdim) maxdim = maxdat
c
      nsize = maxdim*nvar
      if(ncut.ge.1) nsize = nsize * (ncut+1)
      allocate (vr(nsize),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
c
      allocate (vrmin(nvar+ncut),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
c
      allocate (vrmax(nvar+ncut),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
c
      allocate (sills(nvar+ncut),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
c
      allocate (x(maxdat),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
c
      allocate (y(maxdat),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
c
      allocate (z(maxdat),stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
c
      allocate (np(-nxlag:nxlag,-nylag:nylag,-nzlag:nzlag,nvarg),
     +          stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
c
      allocate (gam(-nxlag:nxlag,-nylag:nylag,-nzlag:nzlag,nvarg),
     +          stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
c
      allocate (hm(-nxlag:nxlag,-nylag:nylag,-nzlag:nzlag,nvarg),
     +          stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
c
      allocate (tm(-nxlag:nxlag,-nylag:nylag,-nzlag:nzlag,nvarg),
     +          stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
c
      allocate (hv(-nxlag:nxlag,-nylag:nylag,-nzlag:nzlag,nvarg),
     +          stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
c
      allocate (tv(-nxlag:nxlag,-nylag:nylag,-nzlag:nzlag,nvarg),
     +          stat = test)
      if (test.ne.0) then
            write(*,*) 'Error: Allocation failed due to ',
     +                 'insufficient memory!', test
            stop
      end if
c
      rewind(lin)
      read(lin,'(a40)',err=99) str
      read(lin,*,err=99) nvari
      do i=1,nvari
            read(lin,'(a40)',err=99) str
            do iv=1,nvar
                  j=ivar(iv)
                  if(i.eq.j) names(iv) = str(1:12)
            end do
            num(i) = 0
            avg(i) = 0.0
            ssq(i) = 0.0
      end do
c
c Read the regular grid information row wise (x cycles fastest):
c
      if(igrid.eq.1) then
            do iz=1,nz
            do iy=1,ny
            do ix=1,nx
                  read(lin,*,err=99) (var(i),i=1,nvari)
                  do iv=1,nvar
                        i = ivar(iv)
                        index=ix+(iy-1)*nx+(iz-1)*nx*ny+(iv-1)*nx*ny*nz
                        vr(index) = var(i)
                        if(var(i).ge.tmin.and.var(i).lt.tmax) then
                              num(iv) = num(iv) + 1
                              avg(iv) = avg(iv) + dble(var(i))
                              ssq(iv) = ssq(iv) + dble(var(i)*var(i))
                        end if
                  end do
            end do
            end do
            end do
      end if
c
c Read in Scattered Data points:
c
      if(igrid.eq.0) then
            nd = 0
 2          continue
            read(lin,*,end=9,err=99) (var(j),j=1,nvari)
            testdat = .false.
            do iv=1,nvar
                  j=ivar(iv)
                  if(var(j).ge.tmin.and.var(j).lt.tmax) testdat = .true.
            end do
            if(.not.testdat) go to 2
            nd = nd + 1
c
c Acceptable data, make sure there are not too many data:
c
            do iv=1,nvar
                  j=ivar(iv)
                  index = nd + (iv-1)*maxdim
                  vr(index) = var(j)
                  if(var(j).ge.tmin.and.var(j).lt.tmax) then
                        num(iv) = num(iv) + 1
                        avg(iv) = avg(iv) + dble(var(j))
                        ssq(iv) = ssq(iv) + dble(var(j)*var(j))
                  endif
            end do
            if(icolx.le.0) then
                  x(nd) = 0.0
            else
                  x(nd) = var(icolx)
            endif
            if(icoly.le.0) then
                  y(nd) = 0.0
            else
                  y(nd) = var(icoly)
            endif
            if(icolz.le.0) then
                  z(nd) = 0.0
            else
                  z(nd) = var(icolz)
            endif
            go to 2
 9          continue
      end if
      close(lin)
c
c Compute the averages and variances as an error check for the user:
c
      do iv=1,nvar
            sills(iv) = -999.
            if(num(iv).gt.0) then
                  avg(iv) = avg(iv) / dble(num(iv))
                  ssq(iv) =(ssq(iv) / dble(num(iv))) - avg(iv) * avg(iv)
                  sills(iv) = real(ssq(iv))
                  write(*,*) 'Variable number ',iv
                  write(*,*) '  Number   = ',num(iv)
                  write(*,*) '  Average  = ',real(avg(iv))
                  write(*,*) '  Variance = ',real(ssq(iv))
            endif
      end do
c
c Construct Indicator Variables if necessary:
c
      if(igrid.eq.1) then
            nloop = nx*ny*nz
      else
            nloop = nd
      end if
      do ic=1,ncut
            iv   = ivc(ic)
            jv   = nvar + ic
            ptot = 0.0
            p1   = 0.0
            if(igrid.eq.1) then
                  nloop = nx*ny*nz
            else
                  nloop = nd
            end if
            do iloop=1,nloop
                  if(igrid.eq.1) then
                        index = iloop + (iv-1)*nx*ny*nz
                        jndex = iloop + (jv-1)*nx*ny*nz
                  else
                        index = iloop + (iv-1)*maxdim
                        jndex = iloop + (jv-1)*maxdim
                  end if
                  if(vr(index).lt.tmin.or.vr(index).ge.tmax) then
                        vr(jndex) = tmin - EPSLON
                  else
                        if(indflag(ic).eq.1) then
                              if(vr(index).lt.cut(ic)) then
                                    vr(jndex) = 0.0
                              else
                                    vr(jndex) = 1.0
                              end if
                              ptot = ptot + 1.0
                              p1   = p1   + vr(jndex)
                        else
                              vr(jndex) = 0.0
                              if(int(vr(index)+0.5).eq.int(cut(ic)+0.5))
     +                        vr(jndex) = 1.0
                              ptot = ptot + 1.0
                              p1   = p1   + vr(jndex)
                        end if
                  end if
            end do
            p1        = p1 / max(ptot,1.0)
            sills(jv) = dble (p1*(1.0-p1))
            write(*,*) ' setup indicator ',ic
      end do
c
c Establish minimums and maximums:
c
      do i=1,nvar
            vrmin(i) =  1.0e21
            vrmax(i) = -1.0e21
      end do
      do iloop=1,nloop
            do iv=1,nvar+ncut
                  if(igrid.eq.1) then
                        index = iloop + (iv-1)*nx*ny*nz
                  else
                        index = iloop + (iv-1)*maxdim
                  end if
                  if(vr(index).ge.tmin.and.vr(index).lt.tmax) then
                        if(vr(index).lt.vrmin(iv)) vrmin(iv) = vr(index)
                        if(vr(index).gt.vrmax(iv)) vrmax(iv) = vr(index)
                  end if
            end do 
      end do 
c
c Check on the variogams that were requested:
c
      call check(vrmin,vrmax)
c
c Return:
c
      return
c
c Error in an Input File Somewhere:
c
 98   stop 'ERROR in parameter file!'
 99   stop 'ERROR in data file!'
      end



      subroutine varmap
c-----------------------------------------------------------------------
c
c                       Variogram Map/Volume
c                       ********************
c
c
c
c
c
c-----------------------------------------------------------------------
      use geostat
c
c Initialize the summation arrays:
c
      do iv=1,nvarg
            do iz=1,nzlag
                  do iy=1,nylag
                        do ix=1,nxlag
                              np(ix,iy,iz,iv)  = 0.
                              gam(ix,iy,iz,iv) = 0.0
                              hm(ix,iy,iz,iv)  = 0.0
                              tm(ix,iy,iz,iv)  = 0.0
                              hv(ix,iy,iz,iv)  = 0.0
                              tv(ix,iy,iz,iv)  = 0.0
                        end do
                  end do
            end do
      end do
c
c REGULAR GRID:
c
      if(igrid.eq.1) then
c
c Number of node spacings per lag:
c
            xnspl = xsiz / dxlag
            ynspl = ysiz / dylag
            znspl = zsiz / dzlag
            nxy   = nx*ny
            nxyz  = nx*ny*nz
c
c First fix the location of a seed point on the grid (ix,iy,iz):
c
            do iz=1,nz
            do iy=1,ny
            do ix=1,nx
c
c Second loop over the grid:
c
            do 11 jz=1,nz
            if(jz.ge.iz) then
                  izl =  int((jz-iz)*znspl+0.5)
            else
                  izl = -int((iz-jz)*znspl+0.5)
            end if
            if(izl.lt.-nzlag.or.izl.gt.nzlag) go to 11

            do 12 jy=1,ny
            if(jy.ge.iy) then
                  iyl =  int((jy-iy)*ynspl+0.5)
            else
                  iyl = -int((iy-jy)*ynspl+0.5)
            end if
            if(iyl.lt.-nylag.or.iyl.gt.nylag) go to 12

            do 13 jx=1,nx
            if(jx.ge.ix) then
                  ixl =  int((jx-ix)*xnspl+0.5)
            else
                  ixl = -int((ix-jx)*xnspl+0.5)
            end if
            if(ixl.lt.-nxlag.or.ixl.gt.nxlag) go to 13
c
c Loop over all variograms for this lag:
c
            do iv=1,nvarg
                  it = ivtype(iv)
c
c Get the head and tail values:
c
                  index = ix+(iy-1)*nx+(iz-1)*nxy+(ivhead(iv)-1)*nxyz
                  vrh   = vr(index)
                  index = jx+(jy-1)*nx+(jz-1)*nxy+(ivtail(iv)-1)*nxyz
                  vrt   = vr(index)
                  if(vrh.lt.tmin.or.vrh.ge.tmax.or.
     +               vrt.lt.tmin.or.vrt.ge.tmax) go to 5
c
c Need increment for the cross semivariogram only:
c
                  if(it.eq.2) then
                     index = ix+(iy-1)*nx+(iz-1)*nxy+(ivtail(iv)-1)*nxyz
                     vrhpr = vr(index)
                     index = jx+(jy-1)*nx+(jz-1)*nxy+(ivhead(iv)-1)*nxyz
                     vrtpr = vr(index)
                     if(vrhpr.lt.tmin.or.vrhpr.ge.tmax.or.
     +                  vrtpr.lt.tmin.or.vrtpr.ge.tmax) go to 5
                  endif
c
c We have an acceptable pair, update the variogram arrays:
c
                  call updtvarg(ixl,iyl,izl,iv,it,vrt,vrh,vrtpr,vrhpr)
 5                continue
            end do
 13   continue
 12   continue
 11   continue
      end do
      end do
      end do
c
c Finished regular grid:
c
      end if
c
c SCATTERED DATA:
c
      maxdim = nx * ny * nz
      if(igrid.eq.0) then
c
c First fix the location of a seed point:
c
            do i=1,nd
c
c Second loop over the data:
c
            do j=1,nd
c
c The lag:
c
            zdis = z(j) - z(i)
            if(zdis.ge.0.0) then
                  izl =  int( zdis/dzlag+0.5)
            else
                  izl = -int(-zdis/dzlag+0.5)
            end if
            if(izl.lt.-nzlag.or.izl.gt.nzlag) go to 15
            ydis = y(j) - y(i)
            if(ydis.ge.0.0) then
                  iyl =  int( ydis/dylag+0.5)
            else
                  iyl = -int(-ydis/dylag+0.5)
            end if
            if(iyl.lt.-nylag.or.iyl.gt.nylag) go to 15
            xdis = x(j) - x(i)
            if(xdis.ge.0.0) then
                  ixl =  int( xdis/dxlag+0.5)
            else
                  ixl = -int(-xdis/dxlag+0.5)
            end if
            if(ixl.lt.-nxlag.or.ixl.gt.nxlag) go to 15
c
c Loop over all variograms for this lag:
c
            do iv=1,nvarg
                  it = ivtype(iv)
c
c Get the head and tail values:
c
                  index = i+(ivhead(iv)-1)*maxdim
                  vrh   = vr(index)
                  index = j+(ivtail(iv)-1)*maxdim
                  vrt   = vr(index)
                  if(vrh.lt.tmin.or.vrh.ge.tmax.or.
     +               vrt.lt.tmin.or.vrt.ge.tmax) go to 16
c
c Need increment for the cross semivariogram only:
c
                  if(it.eq.2) then
                     index = i+(ivtail(iv)-1)*maxdim
                     vrhpr = vr(index)
                     index = j+(ivhead(iv)-1)*maxdim
                     vrtpr = vr(index)
                     if(vrhpr.lt.tmin.or.vrhpr.ge.tmax.or.
     +                  vrtpr.lt.tmin.or.vrtpr.ge.tmax) go to 16
                  endif
c
c We have an acceptable pair, update the variogram arrays:
c
                  call updtvarg(ixl,iyl,izl,iv,it,vrt,vrh,vrtpr,vrhpr)
 16               continue
            end do
 15   continue
      end do
      end do
c
c Finished regular grid:
c
      end if
c
c Get average values for gam, hm, tm, hv, and tv, then compute
c the correct "variogram" measure:
c
      do iv=1,nvarg
      do iz=-nzlag,nzlag
      do iy=-nylag,nylag
      do ix=-nxlag,nxlag
            if(np(ix,iy,iz,iv).le.minnp) then
                  gam(ix,iy,iz,iv) = -999.
                  hm(ix,iy,iz,iv)  = -999.
                  tm(ix,iy,iz,iv)  = -999.
                  hv(ix,iy,iz,iv)  = -999.
                  tv(ix,iy,iz,iv)  = -999.
                  go to 6
            end if
            rnum   = np(ix,iy,iz,iv)
            gam(ix,iy,iz,iv) = gam(ix,iy,iz,iv) / (rnum)
            hm(ix,iy,iz,iv)  = hm(ix,iy,iz,iv)  / (rnum)
            tm(ix,iy,iz,iv)  = tm(ix,iy,iz,iv)  / (rnum)
            hv(ix,iy,iz,iv)  = hv(ix,iy,iz,iv)  / (rnum)
            tv(ix,iy,iz,iv)  = tv(ix,iy,iz,iv)  / (rnum)
            it     = ivtype(iv)
c
c Attempt to standardize:
c
            if(isill.eq.1) then
                  if(ivtail(iv).eq.ivhead(iv)) then
                        iii = ivtail(iv)
                        if((it.eq.1.or.it.ge.9).and.sills(iii).gt.0.0)
     +                    gam(ix,iy,iz,iv) = gam(ix,iy,iz,iv)/sills(iii)
                  end if
            end if
c
c 1. report the semivariogram rather than variogram
c 2. report the cross-semivariogram rather than variogram
c 3. the covariance requires "centering"
c 4. the correlogram requires centering and normalizing
c 5. general relative requires division by lag mean
c 6. report the semi(pairwise relative variogram)
c 7. report the semi(log variogram)
c 8. report the semi(rodogram)
c 9. report the semi(madogram)
c
            if(it.eq.1.or.it.eq.2) then
                  gam(ix,iy,iz,iv) = 0.5 * gam(ix,iy,iz,iv)
            else if(abs(it).eq.3) then
                  gam(ix,iy,iz,iv) = gam(ix,iy,iz,iv) 
     +                             - hm(ix,iy,iz,iv)*tm(ix,iy,iz,iv)
                  if(it.lt.0) then
                        if(sills(ivtail(iv)).lt.0.0.or.
     +                     sills(ivhead(iv)).lt.0.0) then
                              gam(ix,iy,iz,iv) = 0.0
                        else
                              variance = ( sqrt(sills(ivtail(iv)))
     +                                 *   sqrt(sills(ivhead(iv))) )
                              gam(ix,iy,iz,iv)=variance-gam(ix,iy,iz,iv)
                        end if
                  end if
            else if(it.eq.4) then
                  hv(ix,iy,iz,iv) = hv(ix,iy,iz,iv) -
     +                              hm(ix,iy,iz,iv)*hm(ix,iy,iz,iv)
                  if(hv(ix,iy,iz,iv).lt.0.0) hv(ix,iy,iz,iv) = 0.0
                  hv(ix,iy,iz,iv)  = sqrt(hv(ix,iy,iz,iv))
                  tv(ix,iy,iz,iv) = tv(ix,iy,iz,iv) -
     +                              tm(ix,iy,iz,iv)*tm(ix,iy,iz,iv)
                  if(tv(ix,iy,iz,iv).lt.0.0) tv(ix,iy,iz,iv) = 0.0
                  tv(ix,iy,iz,iv)  = sqrt(tv(ix,iy,iz,iv))
                  if((hv(ix,iy,iz,iv)*tv(ix,iy,iz,iv)).lt.EPSLON) then
                        gam(ix,iy,iz,iv) = 0.0
                  else
                        gam(ix,iy,iz,iv) =(gam(ix,iy,iz,iv)
     +                         -  hm(ix,iy,iz,iv)*tm(ix,iy,iz,iv))
     +                         / (hv(ix,iy,iz,iv)*tv(ix,iy,iz,iv))
                  endif
c
c Square "hv" and "tv" so that we return the variance:
c
                  hv(ix,iy,iz,iv)  = hv(ix,iy,iz,iv)*hv(ix,iy,iz,iv)
                  tv(ix,iy,iz,iv)  = tv(ix,iy,iz,iv)*tv(ix,iy,iz,iv)
            else if(it.eq.5) then
                  htave  = 0.5*(hm(ix,iy,iz,iv)+tm(ix,iy,iz,iv))
                  htave  = htave   *   htave
                  if(htave.lt.EPSLON) then
                        gam(ix,iy,iz,iv) = 0.0
                  else
                        gam(ix,iy,iz,iv) = gam(ix,iy,iz,iv)/dble(htave)
                  endif
            else if(it.ge.6) then
                  gam(ix,iy,iz,iv) = 0.5 * gam(ix,iy,iz,iv)
            end if
 6    continue
      end do
      end do
      end do
      end do
c
c Finished here:
c
      return
      end
 
 
 
      subroutine updtvarg(ixl,iyl,izl,iv,it,vrt,vrh,vrtpr,vrhpr)
c-----------------------------------------------------------------------
c
c                     Update Variogram Arrays
c                     ***********************
c
c
c
c
c-----------------------------------------------------------------------
      use geostat
c
c We have an acceptable pair, therefore accumulate all the statistics
c that are required for the variogram:
c
      np(ixl,iyl,izl,iv)  = np(ixl,iyl,izl,iv) + 1.
      tm(ixl,iyl,izl,iv)  = tm(ixl,iyl,izl,iv) + vrt
      hm(ixl,iyl,izl,iv)  = hm(ixl,iyl,izl,iv) + vrh
c
c Choose the correct variogram type and keep relevant sums:
c
      if(it.eq.1.or.it.ge.9) then
            gam(ixl,iyl,izl,iv) = gam(ixl,iyl,izl,iv)
     +                          + ((vrh-vrt)*(vrh-vrt))
      else if(it.eq.2) then
            gam(ixl,iyl,izl,iv) = gam(ixl,iyl,izl,iv) 
     +                          + ((vrhpr-vrh)*(vrt-vrtpr))
      else if(abs(it).eq.3) then
            gam(ixl,iyl,izl,iv) = gam(ixl,iyl,izl,iv) +  (vrh*vrt)
      else if(it.eq.4) then
            gam(ixl,iyl,izl,iv) = gam(ixl,iyl,izl,iv) +  (vrh*vrt)
            hv(ixl,iyl,izl,iv)  = hv(ixl,iyl,izl,iv)  +  (vrh*vrh)
            tv(ixl,iyl,izl,iv)  = tv(ixl,iyl,izl,iv)  +  (vrt*vrt)
      else if(it.eq.5) then
            gam(ixl,iyl,izl,iv) = gam(ixl,iyl,izl,iv) 
     +                          + ((vrh-vrt)*(vrh-vrt))
      else if(it.eq.6) then
            if((vrt+vrh).lt.EPSLON) then
                  np(ixl,iyl,izl,iv)  = np(ixl,iyl,izl,iv) - 1.
                  tm(ixl,iyl,izl,iv)  = tm(ixl,iyl,izl,iv) - (vrt)
                  hm(ixl,iyl,izl,iv)  = hm(ixl,iyl,izl,iv) - (vrh)
            else
                  tempvar= 2.0*(vrt-vrh)/(vrt+vrh)
                  gam(ixl,iyl,izl,iv) = gam(ixl,iyl,izl,iv) 
     +                                + (tempvar*tempvar)
            endif
      else if(it.eq.7) then
            if(vrt.lt.EPSLON.or.vrh.lt.EPSLON) then
                  np(ixl,iyl,izl,iv)  = np(ixl,iyl,izl,iv) - 1.
                  tm(ixl,iyl,izl,iv)  = tm(ixl,iyl,izl,iv) - (vrt)
                  hm(ixl,iyl,izl,iv)  = hm(ixl,iyl,izl,iv) - (vrh)
            else
                  tempvar= alog(vrt)-alog(vrh)
                  gam(ixl,iyl,izl,iv) = gam(ixl,iyl,izl,iv) 
     +                                + (tempvar*tempvar)
            endif
      else if(it.eq.8) then
            gam(ixl,iyl,izl,iv) = gam(ixl,iyl,izl,iv) 
     +                          + (abs(vrt-vrh))
      endif
c
c Return to main variogram calculation subroutine:
c
      return
      end
 
 
 
      subroutine writeout
c-----------------------------------------------------------------------
c
c                  Write Out the Results of VARMAP
c                  *******************************
c
c
c
c
c-----------------------------------------------------------------------
      use geostat
      data      lout/1/
c
c Open the output file:
c
      open(lout,file=outfl,status='UNKNOWN')

      write(lout,101) nxlag+nxlag+1,nylag+nylag+1,nzlag+nzlag+1
 101  format('Variogram Volume: nx ',i3,' ny ',i3,'nz ',i3)
      xmn  = -real(nxlag)
      ymn  = -real(nylag)
      zmn  = -real(nzlag)      
      xsiz = 1.
      ysiz = 1.
      zsiz = 1.      
      nsim = 1
      write(lout,102) 6,nxlag+nxlag+1,nylag+nylag+1,nzlag+nzlag+1,
     +                xmn,ymn,zmn,xsiz,ysiz,zsiz,nsim
 102  format(i2,3(1x,i4),3(1x,g14.8),3(1x,g12.6),i4) 
      write(lout,103)
 103  format('variogram',/,'number of pairs',/,
     +       'head mean',/,'tail mean',/,'head variance',/,
     +       'tail variance')
c
c Loop over all the variograms that have been computed:
c
      do iv=1,nvarg
            do iz=-nzlag,nzlag
            do iy=-nylag,nylag
            do ix=-nxlag,nxlag
                  write(lout,104) gam(ix,iy,iz,iv),np(ix,iy,iz,iv),
     +                            hm(ix,iy,iz,iv), tm(ix,iy,iz,iv),
     +                            hv(ix,iy,iz,iv),tv(ix,iy,iz,iv)
 104              format(f12.5,1x,f10.0,4(1x,g14.8))
            end do
            end do
            end do
      end do
      close(lout)
      return
      end
 
 
 
      subroutine check(vrmin,vrmax)
c-----------------------------------------------------------------------
c
c                Error Check and Note Variogram Types
c                ************************************
c
c Go through each variogram type and note the type to the screen and
c report any possible errors.
c
c
c
c
c
c-----------------------------------------------------------------------
      use geostat
      real      vrmin(*),vrmax(*)
      character title*80
c
c Loop over all the variograms to be computed:
c
      write(*,*)
      do iv=1,nvarg
c
c Note the variogram type and the variables being used:
c
      it = abs(ivtype(iv))
      if(it.eq. 1) title(1:24) = 'Semivariogram          :'
      if(it.eq. 2) title(1:24) = 'Cross Semivariogram    :'
      if(it.eq. 3) title(1:24) = 'Covariance             :'
      if(it.eq. 4) title(1:24) = 'Correlogram            :'
      if(it.eq. 5) title(1:24) = 'General Relative       :'
      if(it.eq. 6) title(1:24) = 'Pairwise Relative      :'
      if(it.eq. 7) title(1:24) = 'Variogram of Logarithms:'
      if(it.eq. 8) title(1:24) = 'Semimadogram           :'
      if(it.eq. 9) title(1:24) = 'Indicator 1/2 Variogram:'
      if(it.eq.10) title(1:24) = 'Indicator 1/2 Variogram:'
      write(title(25:64),100) names(ivtail(iv)),names(ivhead(iv))
 100  format('  tail=',a12,' head=',a12)
      write(*,101) iv,title(1:64)
 101  format(' Variogram ',i2,1x,a64)
c
c Check for possible errors or inconsistencies:
c
      if(it.eq.2) then
            if(ivtail(iv).eq.ivhead(iv)) write(*,201)
 201        format('  WARNING: cross variogram with the same variable!')
      else if(it.eq.5) then
            if(ivtail(iv).ne.ivhead(iv)) write(*,501)
            if(vrmin(ivtail(iv)).lt.0.0.and.vrmax(ivtail(iv)).gt.0.0)
     +            write(*,502)
            if(vrmin(ivhead(iv)).lt.0.0.and.vrmax(ivhead(iv)).gt.0.0)
     +            write(*,502)
 501        format('  WARNING: cross general relative variogram are',
     +             ' difficult to interpret!')
 502        format('  WARNING: there are both positive and negative',
     +             ' values - lag mean could be zero!')
      else if(it.eq.6) then
            if(ivtail(iv).ne.ivhead(iv)) write(*,601)
            if(vrmin(ivtail(iv)).lt.0.0.and.vrmax(ivtail(iv)).gt.0.0)
     +            write(*,602)
            if(vrmin(ivhead(iv)).lt.0.0.and.vrmax(ivhead(iv)).gt.0.0)
     +            write(*,602)
 601        format('  WARNING: cross pairwise relative variogram are',
     +             ' difficult to interpret!')
 602        format('  WARNING: there are both positive and negative',
     +             ' values - pair means could be zero!')
      else if(it.eq.7) then
            if(ivtail(iv).ne.ivhead(iv)) write(*,701)
            if(vrmin(ivtail(iv)).lt.0.0.or.vrmin(ivhead(iv)).lt.0.0)
     +      write(*,702)
 701        format('  WARNING: cross logarithmic variograms may be',
     +             ' difficult to interpret!')
 702        format('  WARNING: there are zero or negative',
     +             ' values - logarithm undefined!')
      else if(it.eq.8) then
            if(ivtail(iv).ne.ivhead(iv)) write(*,901)
 901        format('  WARNING: cross madograms may be difficult to',
     +             ' interpret!')
      endif
c
c Loop over all variograms:
c
      end do
      return
      end
c
c
c
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
      open(lun,file='varmap.par',status='UNKNOWN')
      write(lun,10)
 10   format('                  Parameters for VARMAP',/,
     +       '                  *********************',/,/,
     +       'START OF PARAMETERS:')

      write(lun,11)
 11   format('../data/cluster.dat          ',
     +       '-file with data')
      write(lun,12)
 12   format('1   3                        ',
     +       '-   number of variables: column numbers')
      write(lun,13)
 13   format('-1.0e21     1.0e21           ',
     +       '-   trimming limits')
      write(lun,14)
 14   format('0                            ',
     +       '-1=regular grid, 0=scattered values')
      write(lun,15)
 15   format(' 50   50    1                ',
     +       '-if =1: nx,     ny,   nz')
      write(lun,16)
 16   format('1.0  1.0  1.0                ',
     +       '-       xsiz, ysiz, zsiz')
      write(lun,17)
 17   format('1   2   0                    ',
     +       '-if =0: columns for x,y, z coordinates')
      write(lun,18)
 18   format('varmap.out                   ',
     +       '-file for variogram output')
      write(lun,19)
 19   format(' 10    10     0              ',
     +       '-nxlag, nylag, nzlag')
      write(lun,20)
 20   format('5.0   5.0   1.0              ',
     +       '-dxlag, dylag, dzlag')
      write(lun,21)
 21   format('5                            ',
     +       '-minimum number of pairs')
      write(lun,22)
 22   format('0                            ',
     +       '-standardize sill? (0=no, 1=yes)')
      write(lun,23)
 23   format('1                            ',
     +       '-number of variograms')
      write(lun,24)
 24   format('1   1   1                    ',
     +       '-tail, head, variogram type')
      write(lun,40)
 40   format(//,'type 1 = traditional semivariogram',/,
     +          '     2 = traditional cross semivariogram',/,
     +          '     3 = covariance',/,
     +          '     4 = correlogram',/,
     +          '     5 = general relative semivariogram',/,
     +          '     6 = pairwise relative semivariogram',/,
     +          '     7 = semivariogram of logarithms',/,
     +          '     8 = semimadogram',/,
     +          '     9 = indicator semivariogram - continuous',/,
     +          '     10= indicator semivariogram - categorical')

      close(lun)
      return
      end
